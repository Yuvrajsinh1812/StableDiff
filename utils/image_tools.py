# utils/image_tools.py
from rembg import remove as rembg_remove
from PIL import Image, ImageFilter
import io
import traceback

# Optional modules
_REALSRGAN_AVAILABLE = False
_GFPGAN_AVAILABLE = False

try:
    from realesrgan import RealESRGANer
    _REALSRGAN_AVAILABLE = True
except Exception:
    try:
        from realesrgan import RealESRGAN
        _REALSRGAN_AVAILABLE = True
    except Exception:
        _REALSRGAN_AVAILABLE = False

try:
    from gfpgan import GFPGANer
    _GFPGAN_AVAILABLE = True
except Exception:
    _GFPGAN_AVAILABLE = False

_upscaler = None
_face_restorer = None


def remove_background(image_bytes: bytes) -> bytes:
    """Remove background using rembg."""
    try:
        return rembg_remove(image_bytes)
    except Exception as e:
        print("rembg failed:", e)
        return image_bytes


def _pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def _pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def upscale_image_from_bytes(image_bytes: bytes, scale: int = 2) -> bytes:
    """Upscale using Real-ESRGAN (if available), else Pillow."""
    try:
        img = _pil_from_bytes(image_bytes)
        out = upscale_image(img, scale)
        return _pil_to_bytes(out, fmt="PNG")
    except Exception as e:
        print("upscale_image_from_bytes failed:", e)
        print(traceback.format_exc())
        return image_bytes


def upscale_image(img: Image.Image, scale: int = 2) -> Image.Image:
    """Uses Real-ESRGAN or fallback."""
    global _upscaler

    if _REALSRGAN_AVAILABLE:
        try:
            if _upscaler is None:
                try:
                    _upscaler = RealESRGANer(model_name="RealESRGAN_x4plus", scale=4)
                except Exception:
                    try:
                        _upscaler = RealESRGANer("RealESRGAN_x4plus", 4)
                    except Exception:
                        try:
                            _upscaler = RealESRGAN()
                        except:
                            _upscaler = None

            if _upscaler:
                if hasattr(_upscaler, "enhance"):
                    out, _ = _upscaler.enhance(img, outscale=scale)
                    return out
                else:
                    return _upscaler(img, scale=scale)

        except Exception as e:
            print("RealESRGAN failed:", e)

    # fallback
    w, h = img.size
    return img.resize((w * scale, h * scale), Image.LANCZOS)


def restore_face_from_bytes(image_bytes: bytes) -> bytes:
    try:
        img = _pil_from_bytes(image_bytes)
        out = restore_face(img)
        return _pil_to_bytes(out)
    except Exception as e:
        print("GFPGAN failed:", e)
        return image_bytes


def restore_face(img: Image.Image) -> Image.Image:
    global _face_restorer

    if _GFPGAN_AVAILABLE:
        try:
            if _face_restorer is None:
                _face_restorer = GFPGANer(
                    model_path=None,
                    upscale=1,
                    arch="clean"
                )
            restored = _face_restorer.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True
            )
            if isinstance(restored, tuple):
                return restored[-1]
            return restored
        except Exception:
            pass

    return img
