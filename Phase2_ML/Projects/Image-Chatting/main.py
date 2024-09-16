import torch
from PIL import Image
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration

# Set page layout
st.set_page_config(layout="wide")

# App title
st.title("Image Caption Generator")

# Sidebar: Device selection
device_option = st.sidebar.selectbox("Select Device", ["CPU", "MPS (Mac)", "CUDA (GPU)"])
device = {
    "CPU": "cpu",
    "MPS (Mac)": "mps" if torch.backends.mps.is_available() else "cpu",
    "CUDA (GPU)": "cuda" if torch.cuda.is_available() else "cpu"
}[device_option]

# Sidebar: Image uploader
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

# Load the model and processor only once when the app starts
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, model

# Process and display the image if uploaded
if uploaded_file is not None:
    st.sidebar.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Button to trigger caption generation
    if st.sidebar.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            # Load image and processor model
            processor, model = load_model()
            
            # Load the uploaded image
            image = Image.open(uploaded_file)

            # Prepare the image and generate a description
            inputs = processor(image, return_tensors="pt").to(device)
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # Display the caption and image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.subheader("Generated Caption")
            st.write(caption)

# Instructions
st.write("Upload an image on the left and click 'Generate Caption' to see the description!")
