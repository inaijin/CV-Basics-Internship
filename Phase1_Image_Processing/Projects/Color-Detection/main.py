import cv2
from PIL import Image
from util import get_limits

color = [255, 255, 0]
webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()

    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Gets The Range In The HSV Color Space According To The Color (BGR)
    lowerLimit, upperLimit = get_limits(color=color)

    mask = cv2.inRange(hsvFrame, lowerLimit, upperLimit)
    maskConverted = Image.fromarray(mask)

    boundryBox = maskConverted.getbbox()
    if boundryBox is not None:
        x1, y1, x2, y2 = boundryBox

        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow("webcam", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
