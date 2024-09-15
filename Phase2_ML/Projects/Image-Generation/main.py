import torch
from diffusers import StableDiffusionPipeline

# Check if MPS is available
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load pre-trained Stable Diffusion model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

# Define the text prompt
prompt = "A Man Smiling"

# Generate the image
image = pipe(prompt).images[0]

# Save the image
image.save("generated_image.png")
