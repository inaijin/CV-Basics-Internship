from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # Load a pretrained model

# Train the model on the dataset with the simplified "food" category
model.train(data='FoodSeg/config.yaml', epochs=1, imgsz=640)
