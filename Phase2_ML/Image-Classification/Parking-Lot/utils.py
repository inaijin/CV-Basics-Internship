import cv2
import pickle
import numpy as np
from skimage.transform import resize

EMPTY = True
NOT_EMPTY = False

# Load Our Model Which Was Made By ModelTrainer.py
MODEL = pickle.load(open("model.p", "rb"))

def emptyOrNot(spotBgr):

    flatData = []

    imgResized = resize(spotBgr, (15, 15, 3))
    flatData.append(imgResized.flatten())
    flatData = np.array(flatData)

    y_output = MODEL.predict(flatData)

    if y_output == 0:
        return EMPTY
    else:
        return NOT_EMPTY


def getParkingSpotsBoundaryBoxes(connectedComponents):
    (totalLabels, _, values, _) = connectedComponents

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots

def calcDiff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))
