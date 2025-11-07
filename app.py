# app.py
import os
import io
import base64
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import logging

from diffusers import (
    StableDiffusion3Pipeline,
    StableDiffusion3InpaintPipeline,
    StableDiffusion3Img2ImgPipeline,
)
import torch
from huggingface_hub import login
from PIL import Image

# Local utils
from utils.queue_manager import add_task, get_task, worker
from utils.image_tools import remove_background, upscale_image, restore_face

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pixfusion")

# --- Env & Config ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("HF_MODEL_ID", "stabilityai/stable-diffusion-3.5-medium")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

logger.info(f"🔧 Model ID: {MODEL_ID}")

if HF_TOKEN:
    logger.info("🔑 Logging into Hugging Face...")
    login(token=HF_TOKEN)
else:
    logger.warning("⚠️ No HF_TOKEN found — gated models may not load.")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"🚀 Device: {device} | GPUs available: {torch.cuda.device_count()}")

# --- GPU Memory Tools ---
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def optimize_pipeline(pipe):
    try:
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pass
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        logger.info("xformers not available — continuing without it")

# --- Pipeline Loader ---
def load_pipeline_safe(pipeline_class, model_id_or_path):
    try:
        logger.info(f"⏳ Loading {pipeline_class.__name__} ...")
        pipe = pipeline_class.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        optimize_pipeline(pipe)
        logger.info(f"✅ {pipeline_class.__name__} loaded successfully.")
        return pipe
    except Exception as e:
        logger.error(f"❌ Failed loading {pipeline_class.__name__}: {e}")
        return None

# --- Load Pipelines ---
pipe_txt2img = load_pipeline_safe(StableDiffusion3Pipeline, MODEL_ID)
pipe_inpaint = load_pipeline_safe(StableDiffusion3InpaintPipeline, MODEL_ID)
pipe_img2img = load_pipeline_safe(StableDiffusion3Img2ImgPipeline, MODEL_ID)

logger.info("✅ Pipelines loaded successfully!\n")

# --- FastAPI App ---
app = FastAPI(title="PixFusion AI - Stable Diffusion 3.5", version="1.0")

@app.on_event("startup")
async def startup_event():
    logger.info("⚙️ Starting background worker...")
    asyncio.create_task(worker())

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# --- ROUTES ---

@app.post("/txt2img")
async def txt2img(prompt: str = Form(...), num_images: int = Form(1),
                  width: int = Form(512), height: int = Form(512), steps: int = Form(20)):
    if not pipe_txt2img:
        return JSONResponse({"error": "txt2img model not loaded"}, status_code=500)

    if width > 1024 or height > 1024:
        return JSONResponse({"error": "Max size 1024x1024"}, status_code=400)

    def generate():
        clear_gpu_memory()
        images = pipe_txt2img(
            prompt=prompt,
            num_images_per_prompt=num_images,
            height=height,
            width=width,
            num_inference_steps=steps
        ).images
        results = [pil_to_base64(img) for img in images]
        clear_gpu_memory()
        return results

    task_id = await add_task(generate)
    return {"task_id": task_id}

@app.post("/img2img")
async def img2img(prompt: str = Form(...), strength: float = Form(0.7), file: UploadFile = File(...),
                  steps: int = Form(20)):
    if not pipe_img2img:
        return JSONResponse({"error": "img2img model not loaded"}, status_code=500)

    init_img = Image.open(file.file).convert("RGB")

    def generate():
        clear_gpu_memory()
        images = pipe_img2img(prompt=prompt, image=init_img, strength=strength, num_inference_steps=steps).images
        results = [pil_to_base64(img) for img in images]
        clear_gpu_memory()
        return results

    task_id = await add_task(generate)
    return {"task_id": task_id}

@app.post("/inpaint")
async def inpaint(prompt: str = Form(...), image: UploadFile = File(...), mask: UploadFile = File(...),
                  steps: int = Form(20)):
    if not pipe_inpaint:
        return JSONResponse({"error": "inpaint model not loaded"}, status_code=500)

    init_img = Image.open(image.file).convert("RGB")
    mask_img = Image.open(mask.file).convert("RGB")

    def generate():
        clear_gpu_memory()
        images = pipe_inpaint(prompt=prompt, image=init_img, mask_image=mask_img, num_inference_steps=steps).images
        results = [pil_to_base64(img) for img in images]
        clear_gpu_memory()
        return results

    task_id = await add_task(generate)
    return {"task_id": task_id}

@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    data = await file.read()
    def generate():
        result = remove_background(data)
        return base64.b64encode(result).decode("utf-8")

    task_id = await add_task(generate)
    return {"task_id": task_id}

@app.post("/upscale")
async def upscale(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    def generate():
        result = upscale_image(img)
        return pil_to_base64(result)

    task_id = await add_task(generate)
    return {"task_id": task_id}

@app.post("/restore-face")
async def restore_face_route(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    def generate():
        result = restore_face(img)
        return pil_to_base64(result)

    task_id = await add_task(generate)
    return {"task_id": task_id}

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    return get_task(task_id)
