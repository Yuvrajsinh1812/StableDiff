from rembg import remove
from PIL import Image
import io

def remove_background(image_bytes: bytes) -> bytes:
    """Remove background from an image."""
    return remove(image_bytes)

def upscale_image(img: Image.Image) -> Image.Image:
    """Simple 2x upscale (placeholder for Real-ESRGAN)."""
    return img.resize((img.width * 2, img.height * 2))

def restore_face(img: Image.Image) -> Image.Image:
    """Placeholder for face restoration (GFPGAN/CodeFormer)."""
    return img
