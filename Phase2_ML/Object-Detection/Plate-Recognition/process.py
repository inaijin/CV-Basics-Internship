import os
import cv2
from sort.sort import *
from ultralytics import YOLO
from utils import get_car, read_license_plate, write_csv

# This File Makes A CSV Containing BBX For Cars And Its Plates And The Plate Numbers

vehicles = [2, 3, 5, 7] # Cars Class Id For cocoModel
motTracker = Sort()
results = {}

cocoModel = YOLO('yolov8n.pt') # Pre Trained Model For Car Detection
licensePlateDetector = YOLO('./weights/last.pt') # Self Trained License Plate

videoPath = os.path.join("data", "Cars.mp4")
video = cv2.VideoCapture(videoPath)

frameNum = -1
ret = True
while ret:
    ret, frame = video.read()
    frameNum += 1

    if ret:
        results[frameNum] = {}

        carDetections = []
        detections = cocoModel(frame)[0] # Detect Vehicles
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, classId = detection
            if int(classId) in vehicles:
                carDetections.append([x1, y1, x2, y2, score])

        trackIds = motTracker.update(np.asarray(carDetections))

        licensePlates = licensePlateDetector(frame)[0]
        for licensePlate in licensePlates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = licensePlate
            xcar1, ycar1, xcar2, ycar2, carId = get_car(licensePlate, trackIds)

        if carId != -1:
            licensePlateCrop = frame[int(y1):int(y2), int(x1): int(x2), :]
            licensePlateCropGray = cv2.cvtColor(licensePlateCrop, cv2.COLOR_BGR2GRAY)
            _, licensePlateCropThresh = cv2.threshold(licensePlateCropGray, 64, 255, cv2.THRESH_BINARY_INV)

            licensePlateText, licensePlateTextScore = read_license_plate(licensePlateCropThresh)

            if licensePlateText is not None:
                results[frameNum][carId] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                'text': licensePlateText,
                                                                'bbox_score': score,
                                                                'text_score': licensePlateTextScore}}

write_csv(results, './PlatesDetected.csv')
