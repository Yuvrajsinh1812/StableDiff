import os
import io
import base64
import asyncio
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
from huggingface_hub import login
from diffusers import (
    StableDiffusion3Pipeline,
    StableDiffusion3InpaintPipeline,
    StableDiffusion3Img2ImgPipeline,
)

# Utils
from utils.queue_manager import add_task, get_task, worker
from utils.image_tools import remove_background, upscale_image, restore_face

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("HF_MODEL_ID")

# Default fallback
if not MODEL_ID or MODEL_ID.strip() in ["", "/", ".", "-", "_"]:
    MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"

print(f"\n🔧 Using Model: {MODEL_ID}")

# Hugging Face login
if HF_TOKEN:
    print("🔑 Logging into Hugging Face...")
    login(token=HF_TOKEN)
else:
    print("⚠️ Warning: No HF_TOKEN found (some models may be restricted)")

# Device info
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Device: {device} | GPUs available: {torch.cuda.device_count()}")

# -----------------------------------------------------------------------------
# MEMORY-OPTIMIZED PIPELINE LOADER (final stable version)
# -----------------------------------------------------------------------------
def load_pipeline_balanced(pipeline_class, model_id, **kwargs):
    try:
        print(f"⏳ Loading {pipeline_class.__name__} ...")

        pipe = pipeline_class.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="balanced",  # ✅ Works well for multi-GPU
            **kwargs
        )

        # ✅ Enable memory optimization (safe version)
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            print("⚠️ xFormers not available, continuing without it.")

        print(f"✅ {pipeline_class.__name__} loaded successfully.")
        return pipe

    except Exception as e:
        print(f"❌ Failed loading {pipeline_class.__name__}: {e}")
        return None

# -----------------------------------------------------------------------------
# LOAD PIPELINES
# -----------------------------------------------------------------------------
print("\n⏳ Loading Stable Diffusion 3.5 Pipelines...")
pipe_txt2img = load_pipeline_balanced(StableDiffusion3Pipeline, MODEL_ID)
pipe_inpaint = load_pipeline_balanced(StableDiffusion3InpaintPipeline, MODEL_ID)
pipe_img2img = load_pipeline_balanced(StableDiffusion3Img2ImgPipeline, MODEL_ID)

if pipe_txt2img and pipe_inpaint and pipe_img2img:
    print("✅ All pipelines loaded successfully!\n")
else:
    print("⚠️ Some pipelines failed to load. Check VRAM or model paths.\n")

# -----------------------------------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------------------------------
app = FastAPI(title="Stable Diffusion 3.5 API", version="1.0")

@app.on_event("startup")
async def startup_event():
    print("⚙️ Starting background worker...")
    asyncio.create_task(worker())

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def pil_to_base64(img: Image.Image) -> str:
    """Convert PIL image to base64 string"""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# -----------------------------------------------------------------------------
# ROUTES
# -----------------------------------------------------------------------------
@app.post("/txt2img")
async def txt2img(prompt: str = Form(...), num_images: int = Form(1)):
    if not pipe_txt2img:
        return JSONResponse({"error": "txt2img model not loaded"}, status_code=500)

    def generate():
        try:
            clear_gpu_memory()
            images = pipe_txt2img(
                prompt=prompt,
                num_images_per_prompt=num_images,
                height=512,
                width=512,
                num_inference_steps=20
            ).images
            result = [pil_to_base64(img) for img in images]
            clear_gpu_memory()
            return result
        except Exception as e:
            clear_gpu_memory()
            raise e

    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.post("/img2img")
async def img2img(prompt: str = Form(...), strength: float = Form(0.7), file: UploadFile = File(...)):
    if not pipe_img2img:
        return JSONResponse({"error": "img2img model not loaded"}, status_code=500)

    def generate():
        clear_gpu_memory()
        init_img = Image.open(file.file).convert("RGB")
        images = pipe_img2img(prompt=prompt, image=init_img, strength=strength).images
        result = [pil_to_base64(img) for img in images]
        clear_gpu_memory()
        return result

    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.post("/inpaint")
async def inpaint(prompt: str = Form(...), image: UploadFile = File(...), mask: UploadFile = File(...)):
    if not pipe_inpaint:
        return JSONResponse({"error": "inpaint model not loaded"}, status_code=500)

    def generate():
        clear_gpu_memory()
        init_img = Image.open(image.file).convert("RGB")
        mask_img = Image.open(mask.file).convert("RGB")
        images = pipe_inpaint(prompt=prompt, image=init_img, mask_image=mask_img).images
        result = [pil_to_base64(img) for img in images]
        clear_gpu_memory()
        return result

    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.post("/remove-bg")
async def remove_bg(file: UploadFile = File(...)):
    def generate():
        img_bytes = file.file.read()
        result = remove_background(img_bytes)
        return base64.b64encode(result).decode("utf-8")

    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.post("/upscale")
async def upscale(file: UploadFile = File(...)):
    def generate():
        img = Image.open(file.file).convert("RGB")
        result = upscale_image(img)
        return pil_to_base64(result)

    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.post("/restore-face")
async def restore_face_route(file: UploadFile = File(...)):
    def generate():
        img = Image.open(file.file).convert("RGB")
        result = restore_face(img)
        return pil_to_base64(result)

    task_id = await add_task(generate)
    return {"task_id": task_id}


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    return get_task(task_id)
