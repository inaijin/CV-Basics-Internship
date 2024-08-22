import cv2
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
imagePath = project_root / "Datas" / "Images" / "whiteBoard.png"
vidoePath = project_root / "Datas" / "Vidoes" / "Tomatos.mp4"

image = cv2.imread(imagePath)
print(image.shape)

# Line First Cordinates Second Color Third Thickness
cv2.line(image, (0, 0), (1000, 500), (255, 0, 0), 5)

# Rectangle Negative Value For Thickness Will Result In Filling It
cv2.rectangle(image, (0, 250), (200, 500), (0, 0, 255), 5)

# Cricle
cv2.circle(image, (800, 200), 100, (0, 255, 0), -1)

# Text
cv2.putText(image, "Hello World !", (400, 400), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 3)

cv2.imshow("img", image)
cv2.waitKey(0)
