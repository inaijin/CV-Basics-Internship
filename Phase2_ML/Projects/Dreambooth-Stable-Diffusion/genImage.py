import torch
from diffusers import StableDiffusionPipeline

def genImg(prompt, outpath, modelPath):
    # Load the fine-tuned model
    print(modelPath)
    model_id = modelPath  # Path to the directory where the fine-tuned model is saved
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

    # Generate an image
    image = pipe(prompt).images[0]

    # Save the image
    image.save(outpath)
