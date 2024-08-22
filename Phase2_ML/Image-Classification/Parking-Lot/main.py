import os
import cv2
import numpy as np
from utils import emptyOrNot, getParkingSpotsBoundaryBoxes, calcDiff

# We Use A Pre Prepare'd Mask To Show Cars Location
videoPath = os.path.join("data", "parking_1920_1080_loop.mp4")
maskPath = os.path.join("data", "mask_1920_1080.png")

video = cv2.VideoCapture(videoPath)
mask = cv2.imread(maskPath, 0) # Loading It In GrayScale

connectedComponents = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
spots = getParkingSpotsBoundaryBoxes(connectedComponents)
spotsStatus = [None for _ in spots]
diffs = [None for _ in spots]

# Checking For Spots In Every Frame Isn't Efficient Instead We Do It Every 5 - 10 Seconds
step = 30
# Just Checking The Frames That Are Checking
previousFrame = None

ret = True
frameNum = 0
while ret:
    ret, frame = video.read()

    if frameNum % step == 0 and previousFrame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs[spot_indx] = calcDiff(spot_crop, previousFrame[y1:y1 + h, x1:x1 + w, :])

    if frameNum % step == 0:
        if previousFrame is None:
            arr_ = range(len(spots))
        else:
            arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for index in arr_:
            spot = spots[index]
            x1, y1, w, h = spot
            spotCrop = frame[y1:y1 + h, x1:x1 + w, :]
            spotStatus = emptyOrNot(spotCrop)
            spotsStatus[index] = spotStatus

    if frameNum % step == 0:
        previousFrame = frame.copy()

    for index, spot in enumerate(spots):
        spotStatus = spotsStatus[index]
        x1, y1, w, h = spots[index]
        if spotStatus:
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 3)
        else:
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 3)

    if not ret:
        break
    cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spotsStatus)), str(len(spotsStatus))),
                (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow("vid", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frameNum += 1

video.release()
cv2.destroyAllWindows()
