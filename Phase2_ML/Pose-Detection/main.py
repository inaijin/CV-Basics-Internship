import cv2
from ultralytics import YOLO

imagePath = "Data/YOLO/val/images/Ham MM (25).jpg"
model_path = 'runs/pose/train2/weights/last.pt'

img = cv2.imread(str(imagePath))
model = YOLO(model_path)

results = model(imagePath)[0]

for result in results:
    keypoints = result.keypoints.xy[0].numpy()
    for keypoint_indx, keypoint in enumerate(keypoints):
        x, y = int(keypoint[0]), int(keypoint[1])
        cv2.putText(img, str(keypoint_indx), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
