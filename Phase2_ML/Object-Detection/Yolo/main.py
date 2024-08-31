import os
import cv2
from pathlib import Path
from ultralytics import YOLO

project_root = Path(__file__).resolve().parents[3]
VIDEOS_DIR = project_root / "Datas" / "Vidoes"

video_path = os.path.join(VIDEOS_DIR, 'alpaca.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape

model_path = os.path.join(Path(__file__).resolve().parents[0],
                          'runs', 'detect', 'train2', 'weights', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("vid", frame)
    cv2.waitKey(40)

    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
