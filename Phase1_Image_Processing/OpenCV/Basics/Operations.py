import cv2
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
imagePath = project_root / "Datas" / "Images" / "StockImage.jpeg"
vidoePath = project_root / "Datas" / "Vidoes" / "Tomatos.mp4"

image = cv2.imread(imagePath)
print(image.shape)

cv2.imshow("image", image)
cv2.waitKey(0)

# Resizing An Image
imageResized = cv2.resize(image, (2113, 1423)) # Width / Height
cv2.imwrite("resized.jpeg", imageResized)
print(imageResized.shape)

cv2.imshow("image resized", imageResized)
cv2.waitKey(0)

# Cropping An Image
croppedImage = image[1000:2000, 3000:4000]
cv2.imwrite("cropped.jpeg", croppedImage)
print(croppedImage.shape)

cv2.imshow("image cropped", croppedImage)
cv2.waitKey(0)
