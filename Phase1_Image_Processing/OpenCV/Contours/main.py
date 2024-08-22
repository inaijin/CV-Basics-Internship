import cv2
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
imagePath = project_root / "Datas" / "Images" / "birds.jpg"
vidoePath = project_root / "Datas" / "Vidoes" / "Tomatos.mp4"

image = cv2.imread(imagePath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Objects That We Want To Detect Should Be White When Using Contours
ret, invBinaryImg = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

# Idealy Each Contour Should Be One Of The Birds
contours, hierarchy = cv2.findContours(invBinaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

birdCount = 0
for contour in contours:
    if cv2.contourArea(contour) > 200:
        birdCount += 1
        # Not That Visible
        cv2.drawContours(image, contour, -1, (255, 0, 0), 1)
        x1, y1, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
print(birdCount)

cv2.imshow("img", image)
cv2.imshow("imgInvBinary", invBinaryImg)
cv2.waitKey(0)
