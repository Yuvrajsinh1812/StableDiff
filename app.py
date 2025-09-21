import os
import io
import base64
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionImg2ImgPipeline,
)
import torch
from huggingface_hub import login
from PIL import Image

# Your utility modules (assuming they are in the `utils` folder)
from utils.queue_manager import add_task, get_task, worker
from utils.image_tools import remove_background, upscale_image, restore_face

# --- Configuration & Setup ---

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("HF_MODEL_ID", "stabilityai/stable-diffusion-3.5-medium")
# Local path to store the model for fast loading
LOCAL_MODEL_PATH = f"/workspace/models/{MODEL_ID}"

# Log in to Hugging Face (essential for gated models like SD 3.5)
if HF_TOKEN:
    print("Logging in to Hugging Face...")
    login(token=HF_TOKEN)
else:
    print("WARNING: HF_TOKEN not found. You may not be able to download gated models.")
    # Fallback to local
    MODEL_ID = LOCAL_MODEL_PATH

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Function to load pipelines, handling local vs. Hugging Face paths
def load_pipeline(pipeline_class, model_id_or_path, **kwargs):
    if os.path.exists(model_id_or_path):
        print(f"Loading pipeline from local path: {model_id_or_path}")
        return pipeline_class.from_pretrained(model_id_or_path, **kwargs).to(device)
    else:
        print(f"Local model not found. Downloading/loading from Hugging Face: {model_id_or_path}")
        return pipeline_class.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float16,
            **kwargs
        ).to(device)

# Load pipelines using the helper function
try:
    pipe_txt2img = load_pipeline(StableDiffusionPipeline, MODEL_ID)
    pipe_inpaint = load_pipeline(StableDiffusionInpaintPipeline, MODEL_ID)
    pipe_img2img = load_pipeline(StableDiffusionImg2ImgPipeline, MODEL_ID)
except Exception as e:
    print(f"Error loading a pipeline: {e}")
    print("Check your model path, Hugging Face token, and RunPod disk space.")
    pipe_txt2img, pipe_inpaint, pipe_img2img = None, None, None

# FastAPI app
app = FastAPI(title="Stable Diffusion 3.5 API (Local)", version="1.0")

# Background worker
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker())

# Utility: encode PIL -> base64
def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# --- Routes ---

@app.post("/txt2img")
async def txt2img(prompt: str = Form(...), num_images: int = Form(1)):
    if not pipe_txt2img:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)
    def generate():
        images = pipe_txt2img(prompt=prompt, num_images_per_prompt=num_images).images
        return [pil_to_base64(img) for img in images]
    task_id = await add_task(generate)
    return {"task_id": task_id}

@app.post("/img2img")
async def img2img(prompt: str = Form(...), strength: float = Form(0.7), file: UploadFile = File(...)):
    if not pipe_img2img:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)
    def generate():
        init_img = Image.open(file.file).convert("RGB")
        images = pipe_img2img(prompt=prompt, image=init_img, strength=strength).images
        return [pil_to_base64(img) for img in images]
    task_id = await add_task(generate)
    return {"task_id": task_id}

@app.post("/inpaint")
async def inpaint(prompt: str = Form(...), image: UploadFile = File(...), mask: UploadFile = File(...)):
    if not pipe_inpaint:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)
    def generate():
        init_img = Image.open(image.file).convert("RGB")
        mask_img = Image.open(mask.file).convert("RGB")
        images = pipe_inpaint(prompt=prompt, image=init_img, mask_image=mask_img).images
        return [pil_to_base64(img) for img in images]
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
async def restore(file: UploadFile = File(...)):
    def generate():
        img = Image.open(file.file).convert("RGB")
        result = restore_face(img)
        return pil_to_base64(result)
    task_id = await add_task(generate)
    return {"task_id": task_id}

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    return get_task(task_id)
