import cv2
import numpy as np
from ultralytics import YOLO

model_path = 'runs/segment/train/weights/last.pt'
image_path = 'Flood.jpg'

# Read the image
img = cv2.imread(image_path)

H, W, _ = img.shape # Height And Width

# Load the model
model = YOLO(model_path)

# Perform inference on the image
results = model(img)

# Initialize an empty mask (same size as the image) to combine all masks
combined_mask = np.zeros((H, W), dtype=np.uint8)

# Iterate through the results and combine the masks
for result in results:
    for mask in result.masks.data:
        # Convert mask to numpy array and scale to 255
        mask = (mask.numpy() * 255).astype(np.uint8)

        # Resize the mask to match the image dimensions
        mask = cv2.resize(mask, (W, H))

        # Combine the current mask with the combined_mask (using logical OR)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

# Save the combined mask as an image
cv2.imwrite('./combined_mask.png', combined_mask)

# Or If You Want To Have The Masks In Different Outputs

# # Iterate through the results
# for i, result in enumerate(results):
#     for j, mask in enumerate(result.masks.data):
#         # Convert mask to numpy array and scale to 255
#         mask = (mask.numpy() * 255).astype(np.uint8)
        
#         # Resize the mask to match the image dimensions
#         mask = cv2.resize(mask, (W, H))
        
#         # Save each mask with a unique filename
#         cv2.imwrite(f'./output_mask_{i}_{j}.png', mask)
