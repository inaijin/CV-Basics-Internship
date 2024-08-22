import cv2
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
imagePath = project_root / "Datas" / "Images" / "StockImage.jpeg"
vidoePath = project_root / "Datas" / "Vidoes" / "Tomatos.mp4"

# Reading An Image
image = cv2.imread(imagePath)

# Images In OpenCV Are In Numpy Form
print(type(image))

# They Have Three Dimentions Height, Width, NumOfChannels
print(image.shape)

# Writing An Image
cv2.imwrite("StockImageOut.jpeg", image)

# Visualazing It
cv2.imshow("Image", image)
# Waiting Until A Key Press
cv2.waitKey(5000) # 0 for indefinit

# Reading A Video
video = cv2.VideoCapture(vidoePath)

# Visualazing It
ret = True # If Any Frame Remains
while ret:
    ret, frame = video.read()
    if ret:
        cv2.imshow("Video", frame)
        # Video is 24 frampes per second so 1 / 24 = 0.041
        cv2.waitKey(41)

video.release()
cv2.destroyAllWindows()

# Capturing The Webcam
webcam = cv2.VideoCapture(0) # The Numbers Id Of The Webcam

# Visualazing It
while True:
    ret, frame = webcam.read()
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'): # Bitwise And Cause Of The waitKey Output
        break

webcam.release()
cv2.destroyAllWindows()
