from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  # Load a larger pretrained model

# Use the model
results = model.train(data="Prepare-Data/data/config.yaml", epochs=10, augment=True, lr0=0.01)
