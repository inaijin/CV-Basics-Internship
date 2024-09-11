import cv2
from utils import *
from pathlib import Path

projectRoot = Path(__file__).resolve().parents[1]
dbImagePath = projectRoot / "Datas" / "Face-Analysis"
imagePath = projectRoot / "Datas" / "Face-Analysis" / "aks.jpeg"

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Create a copy of the frame for displaying results
    display_frame = frame.copy()

    # Create buttons on the frame
    cv2.rectangle(display_frame, (0, 0), (625, 300), (255, 255, 255), -1)
    draw_text(display_frame, "Match Faces (M)", (40, 55), color=(0, 0, 0), font_scale=2, thickness=2)
    draw_text(display_frame, "Match with DB (D)", (40, 155), color=(0, 0, 0), font_scale=2, thickness=2)
    draw_text(display_frame, "Analyze Face (A)", (40, 255), color=(0, 0, 0), font_scale=2, thickness=2)

    # Display the frame with buttons
    cv2.imshow("Webcam", display_frame)

    # Wait for a key press or mouse click
    key = cv2.waitKey(1) & 0xFF

    if key == ord('m'):
        process_frame("face matching", frame, img_path=imagePath)
    elif key == ord('d'):
        process_frame("match with database", frame, db_path=dbImagePath)
    elif key == ord('a'):
        process_frame("face analysis", frame)
    elif key == ord('q'):
        print("Quitting...")
        break

cap.release()
cv2.destroyAllWindows()
