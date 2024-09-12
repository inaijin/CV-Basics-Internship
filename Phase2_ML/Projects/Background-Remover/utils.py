import cv2
import base64
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamPredictor, sam_model_registry

model_path = './sam_vit_b_01ec64.pth'

# Load the SAM model
sam = sam_model_registry["vit_b"](checkpoint=model_path)
predictor = SamPredictor(sam)

def remove_background(image_base64_encoding, x, y):
    # Decode the base64 image
    image_bytes = base64.b64decode(image_base64_encoding)
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    # Set image in the predictor
    predictor.set_image(image)

    # Perform the prediction
    masks, scores, logits = predictor.predict(
        point_coords=np.asarray([[x, y]]),
        point_labels=np.asarray([1]),
        multimask_output=True
    )

    # Create a mask from the predictions
    C, H, W = masks.shape
    result_mask = np.zeros((H, W), dtype=bool)
    for j in range(C):
        result_mask |= masks[j, :, :]

    # Create an alpha channel for the result
    result_mask = result_mask.astype(np.uint8)
    alpha_channel = np.ones(result_mask.shape, dtype=result_mask.dtype) * 255
    alpha_channel[result_mask == 0] = 0

    # Merge the original image with the alpha channel to remove the background
    result_image = cv2.merge((image, alpha_channel))

    # Encode the result image to PNG format
    _, result_image_bytes = cv2.imencode('.png', result_image)
    result_image_bytes = result_image_bytes.tobytes()

    # Return the base64-encoded result image
    result_image_bytes_encoded_base64 = base64.b64encode(result_image_bytes).decode('utf-8')
    
    return result_image_bytes_encoded_base64
