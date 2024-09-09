import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

project_root = Path(__file__).resolve().parents[3]

image_path = project_root / "Datas" / "Images" / "burger.webp"
model_path = 'runs/segment/train/weights/last.pt'

# Read the image
img = cv2.imread(str(image_path))  # Ensure the path is a string
H, W, _ = img.shape  # Height and Width

# Load the model
model = YOLO(model_path)

# Perform inference on the image
results = model(img)

# Iterate through the results
for i, result in enumerate(results):
    for j, mask in enumerate(result.masks.data):
        # Convert mask to numpy array and scale to 255
        mask = (mask.numpy() * 255).astype(np.uint8)
        
        # Resize the mask to match the image dimensions
        mask = cv2.resize(mask, (W, H))
        
        # Apply the mask to the image (keep the original image)
        masked_image = cv2.bitwise_and(img, img, mask=mask)
        
        # Save each masked part of the image with a unique filename
        cv2.imwrite(f'./output_image_{i}_{j}.png', masked_image)
