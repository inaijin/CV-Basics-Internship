import cv2
from pathlib import Path
import mediapipe as mp

project_root = Path(__file__).resolve().parents[3]
imagePath = project_root / "Datas" / "Images" / "person.jpeg"
vidoePath = project_root / "Datas" / "Vidoes" / "person.mp4"

image = cv2.imread(imagePath)
video = cv2.VideoCapture(vidoePath)
webcam = cv2.VideoCapture(0)

def blurFaces(image, model_selection=0, min_detection_confidence=0.5, kernel_size=50):

    """
    Detects faces in the input image and applies a blur to the detected face regions.

    :param image: Input image in which faces will be detected and blurred.
    :param model_selection: Model selection for face detection. 
                            0 for images where faces are within 2 meters.
                            1 for images where faces are farther away.
    :param min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for detecting a face.
    :param kernel_size: Size of the kernel used for blurring the face regions.
    :return: The image with blurred faces.
    """

    H, W, _ = image.shape
    faceDetectionMP = mp.solutions.face_detection

    # Initialize the face detection model
    with faceDetectionMP.FaceDetection(model_selection=model_selection, 
                                       min_detection_confidence=min_detection_confidence) as faceDetection:
        # Convert image to RGB as required by MediaPipe
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processedFace = faceDetection.process(imageRGB)

        if processedFace.detections is not None:
            for detection in processedFace.detections:
                locationData = detection.location_data
                boundaryBox = locationData.relative_bounding_box
                x1, y1, w, h = boundaryBox.xmin, boundaryBox.ymin, boundaryBox.width, boundaryBox.height

                # Convert relative coordinates to absolute pixel values
                x1 = int(x1 * W)
                y1 = int(y1 * H)
                w = int(w * W)
                h = int(h * H)

                # Apply blur to the detected face region
                image[y1 : y1 + h, x1 : x1 + w, :] = cv2.blur(
                    image[y1 : y1 + h, x1 : x1 + w, :], (kernel_size, kernel_size))

    return image

# Blurring Image
cv2.imshow("img", image)
blurredImage = blurFaces(image)
cv2.imshow("blurredImg", blurredImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Blurring Video
ret = True
while ret:
    ret, frame = video.read()
    if not ret:
        break
    blurFaces(frame)
    cv2.imshow("video", frame)
    cv2.waitKey(40)

video.release()
cv2.destroyAllWindows()

# Blurring Webcam
while True:
    ret, frame = webcam.read()
    blurFaces(frame)
    cv2.imshow("webcam", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
