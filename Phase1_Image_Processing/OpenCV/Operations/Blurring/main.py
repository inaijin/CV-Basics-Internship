import cv2
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
imagePath = project_root / "Datas" / "Images" / "StockImage.jpeg"
noisyImage = project_root / "Datas" / "Images" / "NoisyDog.png"
vidoePath = project_root / "Datas" / "Vidoes" / "Tomatos.mp4"

image = cv2.imread(imagePath)
noisyImage = cv2.imread(noisyImage)

# Blurring Is Usefull When Dealing With Noise
# Different Types Of Blurr But They All Are Convulotions
convSize = 7
blurredImage = cv2.blur(image, (convSize, convSize))
moreBlurred = cv2.blur(image, (convSize * 10, convSize * 10))
gaussianBlurredImage = cv2.GaussianBlur(image, (convSize, convSize), 3)
medianBlurredImage = cv2.medianBlur(image, convSize)

cv2.imshow("img", image)
cv2.imshow("imgBlurred", blurredImage)
cv2.imshow("imgMoreBlurred", moreBlurred)
cv2.imshow("gaussianBlurr", gaussianBlurredImage)
cv2.imshow("medianBlurred", medianBlurredImage)
cv2.waitKey(0)

# Removing Noise
blurredImageNoise = cv2.blur(noisyImage, (convSize, convSize))
moreBlurredNoise = cv2.blur(noisyImage, (convSize * 10, convSize * 10))
gaussianBlurredImageNoise = cv2.GaussianBlur(noisyImage, (convSize, convSize), 3)
medianBlurredImageNoise = cv2.medianBlur(noisyImage, convSize)

cv2.imshow("noisyImg", noisyImage)
cv2.imshow("imgBlurred", blurredImageNoise)
cv2.imshow("imgMoreBlurred", moreBlurredNoise)
cv2.imshow("gaussianBlurr", gaussianBlurredImageNoise)
cv2.imshow("medianBlurred", medianBlurredImageNoise) # Works Best
cv2.waitKey(0)
