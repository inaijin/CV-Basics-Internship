import cv2
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
imagePath = project_root / "Datas" / "Images" / "cow.jpeg"
vidoePath = project_root / "Datas" / "Vidoes" / "Tomatos.mp4"

image = cv2.imread(imagePath)
imageEdge = cv2.Canny(image, 150, 250)
# Making The White Borders Thicker
imageDilate = cv2.dilate(imageEdge, np.ones((3, 3), dtype=np.int8))
# Making The White Borders Slimmer (Oposite Of Above)
imageErode = cv2.erode(imageDilate, np.ones((3, 3), dtype=np.int8))

cv2.imshow("img", image)
cv2.imshow("imgEdge", imageEdge)
cv2.imshow("imgDilate", imageDilate)
cv2.imshow("imageErode", imageErode)
cv2.waitKey(0)
