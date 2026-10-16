# ==============================================================================
# Practical: Text-to-Image Generation using Latent Diffusion Models
# Framework: PyTorch, Hugging Face Diffusers
# ==============================================================================

# Step 1: Install required libraries (run in Colab)
# !pip install -q diffusers transformers accelerate torch torchvision

import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt

# Step 2: Check for GPU acceleration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {device}")

# Step 3: Load the pretrained Latent Diffusion model pipeline
# Using float16 precision for fast execution within free GPU VRAM limits
model_id = "runwayml/stable-diffusion-v1-5"

print("Downloading and caching diffusion model pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None  # Disables safety checker to avoid excessive RAM overhead
)
pipe = pipe.to(device)

# Step 4: Define prompt and sampling parameters
prompt = "A futuristic cyberpunk city with neon lights, flying vehicles, and wet reflective streets, digital art, highly detailed"
negative_prompt = "blurry, low quality, distorted, extra limbs, dull colors"

# Step 5: Execute Reverse Diffusion Denoising Loop
generator = torch.Generator(device=device).manual_seed(42)  # For reproducible outputs

print("Starting reverse diffusion process...")
with torch.autocast(device):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,      # Denoising iterations (trade-off between quality and speed)
        guidance_scale=7.5,           # Classifier-Free Guidance (CFG): fidelity to text prompt
        generator=generator
    ).images[0]

# Step 6: Save and Display the synthesized output
output_filename = "generated_cyberpunk_city.png"
image.save(output_filename)

plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title(f"Prompt: {prompt[:50]}...", fontsize=10)
plt.axis("off")
plt.show()

print(f"Image generation completed! Output saved as '{output_filename}'.")
