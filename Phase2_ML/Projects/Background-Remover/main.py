import os
import cv2
import base64
import numpy as np
from PIL import Image
import streamlit as st
from utils import remove_background
from streamlit_image_coordinates import streamlit_image_coordinates as im_coordinates

st.set_page_config(layout='wide')

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-color: transparent;
        }}
        [data-testid="stVerticalBlock"] > [data-testid="stBlock"] > div:nth-child(1) {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            background-repeat: no-repeat;
            background-position: left;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

# Set the background in the first column
set_background('./bg.jpg')

# Introduce a hyperparameter for resizing
resize_width = st.sidebar.slider("Resize image width", min_value=300, max_value=1200, value=880, step=10)

api_endpoint = None # Enter Your API

# Create two columns
col01, col02 = st.columns(2)

# File uploader in the second column
file = col02.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Read image if a file is uploaded
if file is not None:
    image = Image.open(file).convert('RGB')

    # Resize the image based on the selected width
    image = image.resize((resize_width, int(image.height * resize_width / image.width)))

    # Create buttons for actions
    col1, col2 = col02.columns(2)

    # Visualize image and capture coordinates
    placeholder0 = col02.empty()
    with placeholder0:
        value = im_coordinates(image)
        if value is not None:
            print(value)

    # Button to display original image
    if col1.button('Original', use_container_width=True):
        placeholder0.empty()
        placeholder1 = col02.empty()
        with placeholder1:
            col02.image(image, use_column_width=True)

    # Button to remove background
    if col2.button('Remove background', type='primary', use_container_width=True):
        # Call API or use local utility function
        placeholder0.empty()
        placeholder2 = col02.empty()

        filename = '{}_{}_{}.png'.format(file.name, value['x'], value['y'])

        # Check if result image already exists locally
        if os.path.exists(filename):
            result_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        else:
            # API Version

            # _, image_bytes = cv2.imencode('.png', np.asarray(image))

            # image_bytes = image_bytes.tobytes()

            # image_bytes_encoded_base64 = base64.b64encode(image_bytes).decode('utf-8')

            # api_data = {"data": [image_bytes_encoded_base64, value['x'], value['y']]}
            # response = requests.post(api_endpoint, json=api_data)

            # result_image = response.json()['data']

            # result_image_bytes = base64.b64decode(result_image)

            # result_image = cv2.imdecode(np.frombuffer(result_image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            # Local Version

            _, image_bytes = cv2.imencode('.png', np.asarray(image))

            image_bytes = image_bytes.tobytes()

            image_bytes_encoded_base64 = base64.b64encode(image_bytes).decode('utf-8')

            result_image = remove_background(image_bytes_encoded_base64, value['x'], value['y'])

            result_image_bytes = base64.b64decode(result_image)

            result_image = cv2.imdecode(np.frombuffer(result_image_bytes, dtype=np.uint8),
                                        cv2.IMREAD_UNCHANGED)

            # Save the result locally
            cv2.imwrite(filename, result_image)

        # Display the result image
        with placeholder2:
            col02.image(result_image, use_column_width=True)
