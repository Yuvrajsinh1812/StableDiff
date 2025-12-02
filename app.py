# app.py
import os
import io
import base64
import asyncio
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageChops
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


app = FastAPI(title="Stable Diffusion API (robust)", version="1.1")


@app.on_event("startup")
async def startup_event():
    log.info("⚙️ Starting background worker...")
    # start queue worker
    asyncio.create_task(worker())


def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def bytes_to_pil(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# --- ROUTES (existing) ---
@app.post("/txt2img")
async def txt2img(prompt: str = Form(...), num_images: int = Form(1), height: int = Form(512), width: int = Form(512), steps: int = Form(20)):
    if not pipe_txt2img:
        return JSONResponse({"error": "txt2img model not loaded"}, status_code=500)

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

    content = await file.read()

    def generate():
        try:
            clear_gpu_memory()
            init_img = bytes_to_pil(content)
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
            init_img = bytes_to_pil(image_bytes)
            mask_img = bytes_to_pil(mask_bytes)
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


# --- NEW: Beauty Retouch endpoint ---
@app.post("/beauty-retouch")
async def beauty_retouch(
    file: UploadFile = File(...),
    skin_smoothing: float = Form(0.4),   # 0.0 - 1.0
    brightness: float = Form(0.0),       # -1.0 - +1.0
    contrast: float = Form(0.0)          # -1.0 - +1.0
):
    """
    Applies lightweight beauty retouching:
    - skin_smoothing: float 0..1 (higher = more smoothing)
    - brightness: -1..1 (0 = no change)
    - contrast: -1..1 (0 = no change)

    Returns a task_id. Task result: base64 PNG bytes.
    """
    content = await file.read()

    def generate():
        try:
            img = bytes_to_pil(content).convert("RGB")

            # 1) Skin smoothing - using a selective blur approach (fast)
            # Create a blurred version and blend based on skin_smoothing.
            try:
                # small bilateral-like blur: use Gaussian + edge mask
                blurred = img.filter(ImageFilter.GaussianBlur(radius=6 * float(max(0.0, min(1.0, skin_smoothing)))))
                # Create edge mask to preserve details where edges exist
                edges = img.convert("L").filter(ImageFilter.FIND_EDGES).point(lambda p: 255 if p > 30 else 0).convert("L")
                edge_invert = ImageOps.invert(edges).convert("L")  # smooth areas have high values in edge_invert
                # Blend blurred and original: more smoothing on smooth areas
                blend_strength = max(0.0, min(1.0, float(skin_smoothing)))
                img = Image.composite(blurred, img, ImageChops.multiply(edge_invert, Image.new('L', img.size, int(255 * blend_strength))))
            except Exception:
                # fallback: simple gaussian blend
                if skin_smoothing > 0.01:
                    img = Image.blend(img, img.filter(ImageFilter.GaussianBlur(radius=4 * skin_smoothing)), alpha=skin_smoothing)

            # 2) Brightness and contrast adjustments (Pillow ImageEnhance expects factor, convert from -1..1)
            try:
                if abs(float(brightness)) > 0.001:
                    factor = 1.0 + float(brightness)  # brightness -1 => 0, 0 =>1, +1 =>2
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(factor)
                if abs(float(contrast)) > 0.001:
                    factor = 1.0 + float(contrast)  # contrast -1 =>0, +1 =>2
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(factor)
                # Slight sharpen to keep edges after smoothing
                img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3))
            except Exception:
                pass

            out_bytes = pil_to_bytes(img, fmt="PNG")
            return base64.b64encode(out_bytes).decode("utf-8")
        except Exception as e:
            log.exception("Beauty retouch failed: %s", e)
            raise

    task_id = await add_task(generate)
    return {"task_id": task_id}


# --- NEW: Effects Lab endpoint ---
@app.post("/effects-lab")
async def effects_lab(
    file: UploadFile = File(...),
    effect: str = Form(...),           # e.g., 'photo_enhance', 'vintage', 'cyberpunk', 'watercolor', 'oil', 'pencil', 'custom_prompt'
    prompt: str | None = Form(None),   # used only for custom_prompt (img2img)
    strength: float = Form(0.7),       # used for custom_prompt (img2img strength)
    steps: int = Form(20),             # used for custom_prompt (img2img steps)
):
    """
    Apply an effect to the uploaded image.
    For 'custom_prompt' effect, the prompt param is required and pipe_img2img must be loaded.
    Returns task_id. Result stored as base64 PNG string.
    """
    content = await file.read()

    def _photo_enhance(img: Image.Image) -> Image.Image:
        # Simple automatic enhancement pipeline
        img = ImageOps.autocontrast(img)
        img = ImageEnhance.Color(img).enhance(1.08)
        img = ImageEnhance.Sharpness(img).enhance(1.05)
        img = ImageEnhance.Contrast(img).enhance(1.06)
        img = ImageEnhance.Brightness(img).enhance(1.03)
        return img

    def _vintage(img: Image.Image) -> Image.Image:
        # vintage: warm tone + slight desaturation + vignette
        img = ImageEnhance.Color(img).enhance(0.85)
        img = ImageEnhance.Brightness(img).enhance(1.02)
        # warm tone by overlay
        overlay = Image.new("RGB", img.size, (230, 200, 170))
        img = ImageChops.multiply(Image.blend(img, overlay, 0.08), Image.new('RGB', img.size, (1,1,1)))
        # slight film curve via point map
        lut = [min(255, int((i / 255.0) ** 0.9 * 255)) for i in range(256)]
        img = img.point(lut * 3)
        return img

    def _cyberpunk(img: Image.Image) -> Image.Image:
        # cyberpunk: boost magenta/cyan and add glow
        r, g, b = img.split()
        # push R slightly higher, reduce G, boost B for neon
        r = r.point(lambda i: min(255, int(i * 1.05 + 10)))
        g = g.point(lambda i: int(i * 0.9))
        b = b.point(lambda i: min(255, int(i * 1.08 + 5)))
        img = Image.merge("RGB", (r, g, b))
        # add glow
        glow = img.filter(ImageFilter.GaussianBlur(radius=6)).point(lambda p: min(255, int(p * 0.6)))
        img = ImageChops.screen(img, glow)
        img = ImageEnhance.Contrast(img).enhance(1.08)
        img = ImageEnhance.Color(img).enhance(1.15)
        return img

    def _watercolor(img: Image.Image) -> Image.Image:
        # watercolor: posterize + edge-smooth
        small = img.resize((max(200, img.width // 2), max(200, img.height // 2)), Image.BILINEAR)
        blurred = small.filter(ImageFilter.MedianFilter(size=3)).filter(ImageFilter.GaussianBlur(radius=1.5))
        up = blurred.resize(img.size, Image.BILINEAR)
        up = ImageOps.posterize(up, bits=5)
        return up

    def _oil_paint(img: Image.Image) -> Image.Image:
        # oil painting-like effect: edge preserve + smooth + detail
        try:
            # mimic with multiple filters
            o = img.filter(ImageFilter.ModeFilter(size=5))
            o = o.filter(ImageFilter.SMOOTH_MORE)
            o = ImageEnhance.Sharpness(o).enhance(1.1)
            return o
        except Exception:
            return img.filter(ImageFilter.DETAIL)

    def _pencil(img: Image.Image) -> Image.Image:
        gray = img.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        inv = ImageOps.invert(edges)
        # blend with gray to produce pencil look
        pencil = Image.blend(gray, inv, 0.6).convert("RGB")
        return pencil

    def generate():
        try:
            # If custom_prompt -> use img2img pipeline (GPU)
            if effect == "custom_prompt":
                if prompt is None or prompt.strip() == "":
                    raise ValueError("custom_prompt requires 'prompt' parameter")
                if not pipe_img2img:
                    raise RuntimeError("img2img pipeline not loaded on server")
                # run stable-diffusion img2img (GPU)
                clear_gpu_memory()
                init_img = bytes_to_pil(content)
                imgs = pipe_img2img(prompt=prompt, image=init_img, strength=float(strength), num_inference_steps=int(steps)).images
                out = [pil_to_base64(img) for img in imgs]
                clear_gpu_memory()
                # return list as other endpoints do
                return out

            # CPU-only simple effects
            img = bytes_to_pil(content)

            if effect == "photo_enhance":
                out_img = _photo_enhance(img)
            elif effect == "vintage":
                out_img = _vintage(img)
            elif effect == "cyberpunk":
                out_img = _cyberpunk(img)
            elif effect == "watercolor":
                out_img = _watercolor(img)
            elif effect == "oil":
                out_img = _oil_paint(img)
            elif effect == "pencil":
                out_img = _pencil(img)
            elif effect == "photo_enhance_with_face":
                # a combined effect: enhance + face restore (if available)
                out_img = _photo_enhance(img)
                try:
                    # attempt to restore face using your image_tools function
                    restored_bytes = restore_face_from_bytes(pil_to_bytes(out_img))
                    out_img = bytes_to_pil(restored_bytes)
                except Exception:
                    pass
            else:
                raise ValueError(f"Unknown effect: {effect}")

            out_bytes = pil_to_bytes(out_img, fmt="PNG")
            return base64.b64encode(out_bytes).decode("utf-8")

        except Exception as e:
            log.exception("Effects lab failed: %s", e)
            raise

    task_id = await add_task(generate)
    return {"task_id": task_id}


# --- existing task check route ---
@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    return get_task(task_id)
