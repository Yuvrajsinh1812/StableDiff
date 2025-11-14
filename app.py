# app.py
import os
import io
import base64
import asyncio
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from huggingface_hub import login
from diffusers import (
    StableDiffusion3Pipeline,
    StableDiffusion3InpaintPipeline,
    StableDiffusion3Img2ImgPipeline,
)

# utils
from utils.queue_manager import add_task, get_task, start_worker_if_not_running
from utils.image_tools import remove_background, upscale_image, restore_face

# basic logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pixfusion")

# Load env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("HF_MODEL_ID", "stabilityai/stable-diffusion-3.5-medium")

log.info("🔧 Model ID: %s", MODEL_ID)

# Hugging Face auth (optional but recommended for gated models)
if HF_TOKEN:
    log.info("🔑 Logging into Hugging Face")
    login(token=HF_TOKEN)
else:
    log.warning("⚠️ HF_TOKEN not set — gated models may fail to download")

# device info
device = "cuda" if torch.cuda.is_available() else "cpu"
log.info("🚀 Device: %s | GPUs: %d", device, torch.cuda.device_count())

# Helper: Clear GPU memory
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Robust pipeline loader — tries strategies until success
def load_pipeline_safe(pipeline_class, model_id, **kwargs):
    """
    Try to load with device_map='auto' first, then 'balanced' fallback.
    Apply memory optimizations where possible.
    Returns pipeline or None on failure.
    """
    strategies = ["auto", "balanced", "cuda"]
    last_error = None

    for strat in strategies:
        try:
            log.info("⏳ Loading %s with device_map=%s ...", pipeline_class.__name__, strat)
            pipe = pipeline_class.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=strat,
                **kwargs,
            )
            # Memory optimizations
            try:
                pipe.enable_attention_slicing()
            except Exception:
                log.debug("enable_attention_slicing() not available")

            # try xformers if available
            try:
                pipe.enable_xformers_memory_efficient_attention()
                log.info("✅ xFormers enabled for %s", pipeline_class.__name__)
            except Exception:
                log.info("⚠️ xFormers not available / failed to enable")

            # try CPU offload if balanced/cuda (only if no device_map='auto' restrictions)
            try:
                # if our pipe already has a device_map and supports cpu offload, use it
                if hasattr(pipe, "enable_model_cpu_offload"):
                    # If device_map was set to 'auto' or 'balanced', enable_model_cpu_offload
                    try:
                        pipe.enable_model_cpu_offload()
                        log.info("✅ CPU offload enabled for %s", pipeline_class.__name__)
                    except Exception as e:
                        # sometimes need to reset device map before offload; skip if fails
                        log.debug("Could not enable_model_cpu_offload: %s", e)
                else:
                    log.debug("enable_model_cpu_offload not present on pipeline")
            except Exception as e:
                log.debug("CPU offload attempt failed: %s", e)

            log.info("✅ Loaded %s (device_map=%s)", pipeline_class.__name__, strat)
            return pipe

        except Exception as e:
            last_error = e
            log.warning("Failed loading %s with device_map=%s: %s", pipeline_class.__name__, strat, e)

    log.error("❌ All strategies failed for %s. Last error: %s", pipeline_class.__name__, last_error)
    return None

# Load pipelines (may be large)
log.info("⏳ Loading Stable Diffusion pipelines...")
pipe_txt2img = load_pipeline_safe(StableDiffusion3Pipeline, MODEL_ID)
pipe_inpaint = load_pipeline_safe(StableDiffusion3InpaintPipeline, MODEL_ID)
pipe_img2img = load_pipeline_safe(StableDiffusion3Img2ImgPipeline, MODEL_ID)

if pipe_txt2img and pipe_inpaint and pipe_img2img:
    log.info("✅ All pipelines loaded successfully")
else:
    log.warning("⚠️ Some pipelines failed to load; endpoints will return model-not-loaded errors")

# FastAPI app
app = FastAPI(title="Stable Diffusion 3.5 API", version="1.0")

@app.on_event("startup")
async def startup_event():
    log.info("⚙️ Starting background worker (if not already started)...")
    start_worker_if_not_running()

# helper: convert PIL to base64
def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# --- Endpoints ---

@app.post("/txt2img")
async def txt2img(prompt: str = Form(...), num_images: int = Form(1), width: int = Form(512), height: int = Form(512), steps: int = Form(20)):
    if not pipe_txt2img:
        return JSONResponse({"error": "txt2img model not loaded"}, status_code=500)

    def generate():
        try:
            clear_gpu_memory()
            images = pipe_txt2img(
                prompt=prompt,
                num_images_per_prompt=max(1, int(num_images)),
                height=int(height),
                width=int(width),
                num_inference_steps=int(steps),
            ).images
            out = [pil_to_base64(img) for img in images]
            clear_gpu_memory()
            return {"images": out}
        except Exception as e:
            clear_gpu_memory()
            raise

    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.post("/img2img")
async def img2img(prompt: str = Form(...), strength: float = Form(0.7), file: UploadFile = File(...), steps: int = Form(20)):
    if not pipe_img2img:
        return JSONResponse({"error": "img2img model not loaded"}, status_code=500)

    def generate():
        try:
            clear_gpu_memory()
            init_img = Image.open(file.file).convert("RGB")
            images = pipe_img2img(prompt=prompt, image=init_img, strength=float(strength), num_inference_steps=int(steps)).images
            out = [pil_to_base64(img) for img in images]
            clear_gpu_memory()
            return {"images": out}
        except Exception as e:
            clear_gpu_memory()
            raise

    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.post("/inpaint")
async def inpaint(prompt: str = Form(...), image: UploadFile = File(...), mask: UploadFile = File(...), steps: int = Form(20)):
    if not pipe_inpaint:
        return JSONResponse({"error": "inpaint model not loaded"}, status_code=500)

    def generate():
        try:
            clear_gpu_memory()
            init_img = Image.open(image.file).convert("RGB")
            mask_img = Image.open(mask.file).convert("RGB")
            images = pipe_inpaint(prompt=prompt, image=init_img, mask_image=mask_img, num_inference_steps=int(steps)).images
            out = [pil_to_base64(img) for img in images]
            clear_gpu_memory()
            return {"images": out}
        except Exception as e:
            clear_gpu_memory()
            raise

    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    def generate():
        img_bytes = file.file.read()
        result = remove_background(img_bytes)
        return {"image_base64": base64.b64encode(result).decode("utf-8")}
    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.post("/upscale")
async def upscale(file: UploadFile = File(...), scale: int = Form(2)):
    def generate():
        img = Image.open(file.file).convert("RGB")
        result = upscale_image(img, scale=scale)
        return {"image_base64": pil_to_base64(result)}
    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.post("/restore-face")
async def restore_face_route(file: UploadFile = File(...)):
    def generate():
        img = Image.open(file.file).convert("RGB")
        result = restore_face(img)
        return {"image_base64": pil_to_base64(result)}
    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Return the task object: status, result (if done) or error"""
    return get_task(task_id)
