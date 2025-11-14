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

# Diffusers imports (we'll lazy-load pipelines)
from diffusers import (
    StableDiffusion3Pipeline,
    StableDiffusion3InpaintPipeline,
    StableDiffusion3Img2ImgPipeline,
)

# local utils
from utils.queue_manager import add_task, get_task, worker
from utils.image_tools import (
    remove_background,
    upscale_image_from_bytes,
    restore_face_from_bytes,
)

# logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pixfusion")

# load env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("HF_MODEL_ID") or "stabilityai/stable-diffusion-3.5-medium"

log.info("🔧 Using Model: %s", MODEL_ID)
if HF_TOKEN:
    log.info("🔑 Logging into Hugging Face...")
    login(token=HF_TOKEN)
else:
    log.warning("⚠️ No HF_TOKEN found: gated models may fail")

device = "cuda" if torch.cuda.is_available() else "cpu"
log.info("🚀 Device: %s | GPUs available: %d", device, torch.cuda.device_count())


# Memory-optimized pipeline loader
def load_pipeline(pipeline_class, model_id, device_map="balanced", torch_dtype=torch.float16, **kwargs):
    try:
        log.info("⏳ Loading %s ...", pipeline_class.__name__)
        pipe = pipeline_class.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )
        # memory optimizations
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

        log.info("✅ Loaded %s", pipeline_class.__name__)
        return pipe
    except Exception as e:
        log.exception("❌ Failed loading %s: %s", pipeline_class.__name__, e)
        return None


# Try to load pipelines. If running out of disk/VRAM, pipelines may be None.
log.info("⏳ Loading Stable Diffusion pipelines...")
pipe_txt2img = load_pipeline(StableDiffusion3Pipeline, MODEL_ID)
pipe_inpaint = load_pipeline(StableDiffusion3InpaintPipeline, MODEL_ID)
pipe_img2img = load_pipeline(StableDiffusion3Img2ImgPipeline, MODEL_ID)

if pipe_txt2img and pipe_inpaint and pipe_img2img:
    log.info("✅ All pipelines loaded successfully")
else:
    log.warning("⚠️ Some pipelines failed to load; respective endpoints will return errors.")


app = FastAPI(title="Stable Diffusion API (robust)", version="1.0")


@app.on_event("startup")
async def startup_event():
    log.info("⚙️ Starting background worker...")
    # start queue worker
    asyncio.create_task(worker())


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# --- ROUTES ---
@app.post("/txt2img")
async def txt2img(prompt: str = Form(...), num_images: int = Form(1), height: int = Form(512), width: int = Form(512), steps: int = Form(20)):
    if not pipe_txt2img:
        return JSONResponse({"error": "txt2img model not loaded"}, status_code=500)

    # task closure; synchronous/blocking
    def generate():
        try:
            clear_gpu_memory()
            imgs = pipe_txt2img(prompt=prompt, num_images_per_prompt=num_images, height=height, width=width, num_inference_steps=steps).images
            out = [pil_to_base64(img) for img in imgs]
            clear_gpu_memory()
            return out
        except Exception as e:
            clear_gpu_memory()
            raise

    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.post("/img2img")
async def img2img(prompt: str = Form(...), strength: float = Form(0.7), file: UploadFile = File(...), steps: int = Form(20)):
    if not pipe_img2img:
        return JSONResponse({"error": "img2img model not loaded"}, status_code=500)

    # read bytes immediately to avoid file-close I/O issues
    content = await file.read()

    def generate():
        try:
            clear_gpu_memory()
            init_img = Image.open(io.BytesIO(content)).convert("RGB")
            imgs = pipe_img2img(prompt=prompt, image=init_img, strength=strength, num_inference_steps=steps).images
            out = [pil_to_base64(img) for img in imgs]
            clear_gpu_memory()
            return out
        except Exception as e:
            clear_gpu_memory()
            raise

    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.post("/inpaint")
async def inpaint(prompt: str = Form(...), image: UploadFile = File(...), mask: UploadFile = File(...), steps: int = Form(20)):
    if not pipe_inpaint:
        return JSONResponse({"error": "inpaint model not loaded"}, status_code=500)

    image_bytes = await image.read()
    mask_bytes = await mask.read()

    def generate():
        try:
            clear_gpu_memory()
            init_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert("RGB")
            imgs = pipe_inpaint(prompt=prompt, image=init_img, mask_image=mask_img, num_inference_steps=steps).images
            out = [pil_to_base64(img) for img in imgs]
            clear_gpu_memory()
            return out
        except Exception:
            clear_gpu_memory()
            raise

    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    content = await file.read()

    def generate():
        try:
            result_bytes = remove_background(content)
            return base64.b64encode(result_bytes).decode("utf-8")
        except Exception as e:
            raise

    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.post("/upscale")
async def upscale(file: UploadFile = File(...), scale: int = Form(2)):
    """
    scale: 2 or 4 (defaults to 2). This endpoint returns base64 PNG string as the task result.
    """
    if scale not in (2, 4):
        return JSONResponse({"error": "scale must be 2 or 4"}, status_code=400)

    content = await file.read()

    def generate():
        try:
            result_bytes = upscale_image_from_bytes(content, scale=scale)
            return base64.b64encode(result_bytes).decode("utf-8")
        except Exception:
            raise

    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.post("/restore-face")
async def restore_face_route(file: UploadFile = File(...)):
    content = await file.read()

    def generate():
        try:
            result_bytes = restore_face_from_bytes(content)
            return base64.b64encode(result_bytes).decode("utf-8")
        except Exception:
            raise

    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    return get_task(task_id)
