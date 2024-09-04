from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

model.train(data='datas/config.yaml', epochs=20, imgsz=640)
