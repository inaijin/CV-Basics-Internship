import cv2
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
imagePath = project_root / "Datas" / "Images" / "StockImage.jpeg"
vidoePath = project_root / "Datas" / "Vidoes" / "Tomatos.mp4"

# BGR Color System (Original) To RGB Color Space (Converted)
image = cv2.imread(imagePath)

imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ImageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # Mainly Use For Color Detection

cv2.imshow("img", image)
cv2.imshow("imgRGB", imageRGB)
cv2.imshow("imgGray", imageGray)
print(imageGray.shape) # Making It Only One Channel
cv2.imshow("imgHSV", ImageHSV)
cv2.waitKey(0)
