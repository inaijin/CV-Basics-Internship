import cv2
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
imagePath = project_root / "Datas" / "Images" / "Cow.jpeg"
imageCPath = project_root / "Datas" / "Images" / "table.png"
vidoePath = project_root / "Datas" / "Vidoes" / "Tomatos.mp4"

threshold = 100
image = cv2.imread(imagePath)
imageComplex = cv2.imread(imageCPath)

# Thresholding Mainly Used To Make An Image Binary Useful For Segmentating
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Below Num1 Becoms 0 Upper Becoms Num2
ret, binaryImage = cv2.threshold(imageGray, threshold, 255, cv2.THRESH_BINARY)

# Making It Clearer
binaryImage = cv2.blur(binaryImage, (10, 10))
ret, binaryImage = cv2.threshold(binaryImage, threshold, 255, cv2.THRESH_BINARY)

binaryImage = cv2.blur(binaryImage, (20, 20))
ret, binaryImage = cv2.threshold(binaryImage, threshold, 255, cv2.THRESH_BINARY)

cv2.imshow("img", image)
cv2.imshow("imgBinary", binaryImage)
cv2.waitKey(0)

# Sometimes Simple Thresholding Will Not Work And We Need Adaptive Threshold
cv2.imshow("img", imageComplex)

imageGray = cv2.cvtColor(imageComplex, cv2.COLOR_BGR2GRAY)
ret, binaryImage = cv2.threshold(imageGray, threshold, 255, cv2.THRESH_BINARY)
cv2.imshow("imgBad", binaryImage)

# First Value Is For The Strength Of Thresholding The Second Is the Sliding Window
binaryImgGood = cv2.adaptiveThreshold(imageGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 21, 4)
cv2.imshow("imgGood", binaryImgGood)
cv2.waitKey(0)
