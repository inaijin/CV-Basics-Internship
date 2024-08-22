import numpy as np
from ultralytics import YOLO

model = YOLO('./runs/classify/train2/weights/last.pt')  # load a custom model
results = model('rainy.jpg')  # predict on an image

names_dict = results[0].names
probs = results[0].probs.data.tolist()

print(names_dict)
print(probs)
print(names_dict[np.argmax(probs)])
