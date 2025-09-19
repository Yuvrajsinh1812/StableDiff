import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_ID = os.getenv("HF_MODEL_ID")

client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

# Simple test prompt
prompt = "A warrior in samurai armor standing under cherry blossom trees, traditional Japanese ink painting style"

# Generate image
image = client.text_to_image(prompt, height=512, width=512)

# Save properly
image.save("test_output.png")
print("✅ Image generated and saved as test_output.png")
