import io
import torch
import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline

# Set page layout
st.set_page_config(layout="wide")
# App title
st.title("Stable Diffusion Image Generator")

# Sidebar: Device selection
device_option = st.sidebar.selectbox("Select Device", ["CPU", "MPS (Mac)", "CUDA (GPU)"])
device = {
    "CPU": "cpu",
    "MPS (Mac)": "mps" if torch.backends.mps.is_available() else "cpu",
    "CUDA (GPU)": "cuda" if torch.cuda.is_available() else "cpu"
}[device_option]

# Sidebar: Prompt input
prompt = st.sidebar.text_input("Enter your image prompt", value="A Man Smiling")

# Button to trigger image generation
if st.sidebar.button("Generate Image"):
    with st.spinner("Generating image..."):
        # Load pre-trained Stable Diffusion model
        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)

        # Generate the image
        image = pipe(prompt).images[0]

        # Display generated image
        st.image(image, caption="Generated Image", use_column_width=True)

        # Option to save image
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.sidebar.download_button(
            label="Download Image",
            data=byte_im,
            file_name="generated_image.png",
            mime="image/png"
        )

# Instructions
st.write("Enter a text prompt on the left and click 'Generate Image' to create your artwork!")
