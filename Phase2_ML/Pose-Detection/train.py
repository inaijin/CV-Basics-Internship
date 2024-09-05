from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)

model.train(data='Data/YOLO/config.yaml', epochs=33, imgsz=640)
