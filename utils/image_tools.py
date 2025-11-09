import torch
import numpy as np
from io import BytesIO
from PIL import Image
from rembg import remove
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# BACKGROUND REMOVAL
# -----------------------------------------------------------------------------
def remove_background(image_bytes: bytes) -> bytes:
    """Remove background using rembg."""
    try:
        output = remove(image_bytes)
        return output
    except Exception as e:
        print(f"❌ Background removal failed: {e}")
        return image_bytes

# -----------------------------------------------------------------------------
# IMAGE UPSCALING (RealESRGAN)
# -----------------------------------------------------------------------------
def upscale_image(pil_img: Image.Image) -> Image.Image:
    """Upscale image using RealESRGAN (x4)."""
    try:
        print("⏫ Initializing RealESRGAN upscaler...")

        # Load ESRGAN model
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path=None,  # Automatically fetch model weights
            model=model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=True if device == "cuda" else False,
        )

        # Convert PIL to NumPy
        img_np = np.array(pil_img.convert("RGB"))[:, :, ::-1]  # RGB → BGR
        output, _ = upsampler.enhance(img_np, outscale=4)
        result_img = Image.fromarray(output[:, :, ::-1])  # BGR → RGB
        print("✅ Image upscaled successfully.")
        return result_img

    except Exception as e:
        print(f"❌ Upscaling failed: {e}")
        return pil_img

# -----------------------------------------------------------------------------
# FACE RESTORATION (GFPGAN)
# -----------------------------------------------------------------------------
def restore_face(pil_img: Image.Image) -> Image.Image:
    """Restore faces using GFPGAN."""
    try:
        print("😊 Initializing GFPGAN face restorer...")

        restorer = GFPGANer(
            model_path=None,  # Downloads GFPGANv1.4 automatically
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=device,
        )

        img_np = np.array(pil_img.convert("RGB"))[:, :, ::-1]
        _, _, restored_img = restorer.enhance(
            img_np, has_aligned=False, only_center_face=False, paste_back=True
        )
        result_img = Image.fromarray(restored_img[:, :, ::-1])
        print("✅ Face restored successfully.")
        return result_img

    except Exception as e:
        print(f"❌ Face restoration failed: {e}")
        return pil_img
