# utils/image_tools.py
from rembg import remove as rembg_remove
from PIL import Image
import io
import traceback

# Optional high-quality modules (we will use them if available; otherwise fallback)
_REALSRGAN_AVAILABLE = False
_GFPGAN_AVAILABLE = False

# Try to import Real-ESRGAN (may fail depending on environment)
try:
    # newer realesrgan wrapper names vary across versions; attempt import
    from realesrgan import RealESRGANer  # v0.3-ish API
    _REALSRGAN_AVAILABLE = True
except Exception:
    try:
        # older package/class names
        from realesrgan import RealESRGAN
        _REALSRGAN_AVAILABLE = True
    except Exception:
        _REALSRGAN_AVAILABLE = False

# Try to import GFPGAN face restorer
try:
    from gfpgan import GFPGANer
    _GFPGAN_AVAILABLE = True
except Exception:
    _GFPGAN_AVAILABLE = False

# Singletons for heavy models (created lazily)
_upscaler = None
_face_restorer = None


def remove_background(image_bytes: bytes) -> bytes:
    """
    Uses rembg to remove background. Input: bytes. Output: png bytes (RGBA).
    """
    try:
        result = rembg_remove(image_bytes)
        return result
    except Exception as e:
        # Return original bytes if rembg fails
        print("rembg failed:", e)
        return image_bytes


def _pil_from_bytes(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def _pil_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def upscale_image_from_bytes(image_bytes: bytes, scale: int = 2) -> bytes:
    """
    Upscale input bytes by scale factor (2 or 4). Prefer Real-ESRGAN if available,
    otherwise fallback to Pillow LANCZOS resizing.
    Returns PNG bytes.
    """
    try:
        img = _pil_from_bytes(image_bytes)
        out_img = upscale_image(img, scale=scale)
        return _pil_to_bytes(out_img, fmt="PNG")
    except Exception as e:
        print("upscale_image_from_bytes failed:", e)
        print(traceback.format_exc())
        # fallback: return original
        return image_bytes


def upscale_image(img: Image.Image, scale: int = 2) -> Image.Image:
    """
    Upscales a PIL image. If Real-ESRGAN is available we try to use it (lazy init),
    otherwise fallback to Pillow resize.
    scale: 2 or 4 recommended.
    """
    global _upscaler
    if scale not in (2, 4):
        scale = int(scale)

    if _REALSRGAN_AVAILABLE:
        try:
            if _upscaler is None:
                # Attempt to create RealESRGANer with safe defaults.
                # Different versions of 'realesrgan' expose different constructors;
                # we'll try a couple of possibilities.
                try:
                    # Most modern wrappers want model_name or model_path; we try safe sensible defaults
                    _upscaler = RealESRGANer(model_name="RealESRGAN_x4plus", scale=4)
                except Exception:
                    try:
                        _upscaler = RealESRGANer("RealESRGAN_x4plus", 4)
                    except Exception:
                        # older RealESRGAN usage RealESRGAN(...)
                        try:
                            _upscaler = RealESRGAN()
                        except Exception:
                            _upscaler = None

            if _upscaler is not None:
                if scale == 4:
                    if hasattr(_upscaler, "enhance"):
                        # many Real-ESRGAN APIs provide enhance()
                        out, _ = _upscaler.enhance(img, outscale=4)
                        return out
                    else:
                        # try call in other common signature
                        return _upscaler(img, scale=4)
                elif scale == 2:
                    # if only x4 model exists, downsample result or call with 2 if supported
                    if hasattr(_upscaler, "enhance"):
                        out, _ = _upscaler.enhance(img, outscale=2)
                        return out
                    else:
                        out = _upscaler(img, scale=2)
                        return out

        except Exception as e:
            print("RealESRGAN upscale failed, falling back to PIL resize:", e)
            print(traceback.format_exc())

    # Fallback: Pillow high-quality resize
    w, h = img.size
    new_size = (int(round(w * scale)), int(round(h * scale)))
    return img.resize(new_size, Image.LANCZOS)


def restore_face_from_bytes(image_bytes: bytes) -> bytes:
    """
    Restore face(s) using GFPGAN if available, else returns original bytes.
    """
    try:
        img = _pil_from_bytes(image_bytes)
        out_img = restore_face(img)
        return _pil_to_bytes(out_img, fmt="PNG")
    except Exception as e:
        print("restore_face_from_bytes failed:", e)
        print(traceback.format_exc())
        return image_bytes


def restore_face(img: Image.Image) -> Image.Image:
    """
    Use GFPGANer if available; otherwise return input.
    """
    global _face_restorer
    if _GFPGAN_AVAILABLE:
        try:
            if _face_restorer is None:
                # Default parameters frequently used in examples
                _face_restorer = GFPGANer(model_path=None, upscale=1, arch="clean")
            # gfpg an API often returns tuple (cropped_faces, restored_faces, restored_img)
            try:
                restored, _ = _face_restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
                # Some GFPGAN versions return different things; ensure PIL returned
                if isinstance(restored, tuple):
                    # restored image usually last element
                    return restored[-1]
                return restored
            except Exception:
                # alternative interface
                res = _face_restorer.enhance(img)
                if isinstance(res, tuple):
                    return res[0]
                return res
        except Exception as e:
            print("GFPGAN failed:", e)
            print(traceback.format_exc())

    # fallback: return input
    return img
