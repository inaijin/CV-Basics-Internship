import cv2
import easyocr
from pathlib import Path

projectRoot = Path(__file__).resolve().parents[3]
imagePath = projectRoot / "Datas" / "Images" / "signSmall.jpeg"

# Read the image
img = cv2.imread(imagePath.as_posix())

# Init The OCR Reader
reader = easyocr.Reader(['en'], gpu=False)

# Read The Text From Our Image
readedText = reader.readtext(img)
threshold = 0.25

# Draw bbox and text
for _, t in enumerate(readedText):
    bbox, text, score = t

    if score > threshold:
        cv2.rectangle(img, bbox[0], bbox[2], (0, 0, 255), 3)
        cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (0, 255, 0), 1)

# Display the image in a window with a custom title
cv2.imshow("Annotated Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
