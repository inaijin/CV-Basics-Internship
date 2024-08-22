import os
from ultralytics import YOLO
# from utils import split_data

rootPath = os.path.join("data")
categories = ["Cloudy", "Rainy", "Shiny", "Sunrise"]

trainPath = os.path.join(rootPath, "train")
valPath = os.path.join(rootPath, "val")

# Init For Data (It Must Be In This Format For YOLO To Work)
# for category in categories:
#     split_data(category, rootPath, trainPath, valPath)

model = YOLO('yolov8n-cls.pt')  # load a pretrained model

model.train(data= "/Users/kourosh/GitHub/CV-Basics-Internship/Phase2_ML/Image-Classification/Wheather/data",
            epochs=20, imgsz=64)
