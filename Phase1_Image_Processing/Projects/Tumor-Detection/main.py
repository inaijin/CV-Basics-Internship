import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def loadImagesFromDirectory(directoryPath):
    imageList = []
    for filename in os.listdir(directoryPath):
        img = cv2.imread(os.path.join(directoryPath, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            imageList.append(img)
    return imageList

def splitBrainHalves(image):
    _, width = image.shape
    midPoint = width // 2
    leftHalf = image[:, :midPoint]
    rightHalf = image[:, midPoint:]
    return leftHalf, rightHalf

def isTumorPresent(leftHalf, rightHalf):
    score, _ = ssim(leftHalf, rightHalf, full=True)

    diffImage = cv2.absdiff(leftHalf, rightHalf)

    _, diffThresh = cv2.threshold(diffImage, 25, 255, cv2.THRESH_BINARY)

    nonZeroCount = cv2.countNonZero(diffThresh)

    if score < 0.95 and nonZeroCount > 50:
        return True
    return False

def detectTumor(image):
    blurredImage = cv2.GaussianBlur(image, (5, 5), 0)
    _, binaryImage = cv2.threshold(blurredImage, 111, 255, cv2.THRESH_BINARY_INV)

    binaryImage = cv2.dilate(binaryImage, np.ones((5, 5), dtype=np.int8))
    binaryImage = cv2.dilate(binaryImage, np.ones((5, 5), dtype=np.int8))

    contours, _ = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    tumorContours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if 300 < area < 5000:
            tumorContours.append(contour)

    if tumorContours:
        return tumorContours

    return None

def drawTumorContour(image, contour):
    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    return image

dataDirectory = './Datas'
images = loadImagesFromDirectory(dataDirectory)

for img in images:
    leftHalf, rightHalf = splitBrainHalves(img)
    if isTumorPresent(leftHalf, rightHalf):
        tumorContours = detectTumor(img)
        if tumorContours is not None:
            outputImage = img.copy()
            for tumor in tumorContours:
                drawTumorContour(outputImage, tumor)
            cv2.imshow('Tumor Detected', outputImage)
            cv2.waitKey(0)
        else:
            print('Tumor present, but could not be precisely located.')
    else:
        print('No tumor detected.')

cv2.destroyAllWindows()
