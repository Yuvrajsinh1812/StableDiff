# utils/image_tools.py
from PIL import Image
import io
import base64
import logging

log = logging.getLogger("pixfusion.image_tools")

# Try to import advanced libraries if available
try:
    from rembg import remove as rembg_remove
    _has_rembg = True
except Exception:
    _has_rembg = False
    log.info("rembg not installed; remove_background will use fallback (no-op)")

try:
    # Real-ESRGAN wrapper packages vary; try realesrgan
    from realesrgan import RealESRGAN
    _has_realesrgan = True
except Exception:
    _has_realesrgan = False
    log.info("realesrgan not installed; upscale_image will use PIL resize")

try:
    # GFPGAN import
    from gfpgan import GFPGANer
    _has_gfpgan = True
except Exception:
    _has_gfpgan = False
    log.info("gfpgan not installed; restore_face will be a no-op")


def remove_background(image_bytes: bytes) -> bytes:
    """
    Remove background using rembg if available; otherwise return original bytes.
    Returns bytes of PNG.
    """
    if _has_rembg:
        try:
            result = rembg_remove(image_bytes)
            if isinstance(result, (bytes, bytearray)):
                return bytes(result)
            # sometimes returns PIL.Image
            if hasattr(result, "save"):
                buf = io.BytesIO()
                result.save(buf, format="PNG")
                return buf.getvalue()
        except Exception as e:
            log.exception("rembg failed: %s", e)
    # fallback: just return original bytes
    return image_bytes


def upscale_image(img: Image.Image, scale: int = 2) -> Image.Image:
    """
    Upscale using Real-ESRGAN if available, otherwise simple PIL resize with Lanczos
    scale: 2, 4, etc.
    """
    if _has_realesrgan:
        try:
            # The RealESRGAN usage differs by package. This is a common pattern:
            model = RealESRGAN(".", scale=scale)
            model.load_weights("RealESRGAN_x2")  # ensure correct weights; may need adjustment
            arr = model.predict(img)
            return arr
        except Exception as e:
            log.exception("realesrgan upscale failed: %s", e)

    # fallback: PIL resizing
    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    return img.resize((new_w, new_h), resample=Image.LANCZOS)


def restore_face(img: Image.Image) -> Image.Image:
    """
    Try to restore face using GFPGAN; fallback returns original image.
    """
    if _has_gfpgan:
        try:
            # typical GFPGAN usage (may require different init args depending on package)
            # weights & model params may need to be adapted
            gfpganer = GFPGANer(model_path=None, upscale=1, arch="clean", channel_multiplier=2)
            cropped_faces, restored_faces, restored_img = gfpganer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            return restored_img
        except Exception as e:
            log.exception("gfpgan failed: %s", e)

    # fallback
    return img
