import cv2
from pathlib import Path
import mediapipe as mp

project_root = Path(__file__).resolve().parents[3]
imagePath = project_root / "Datas" / "Images" / "person.jpeg"
vidoePath = project_root / "Datas" / "Vidoes" / "woman.mp4"

# Load the pre-trained models for age and gender prediction
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

def predictAgeAndGender(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Predict gender
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]

    # Predict age
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]

    return gender, age

def annotateFacesWithAge(image, model_selection=0, min_detection_confidence=0.5):
    H, W, _ = image.shape
    faceDetectionMP = mp.solutions.face_detection

    with faceDetectionMP.FaceDetection(model_selection=model_selection, 
                                       min_detection_confidence=min_detection_confidence) as faceDetection:
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processedFace = faceDetection.process(imageRGB)

        if processedFace.detections is not None:
            for detection in processedFace.detections:
                locationData = detection.location_data
                boundaryBox = locationData.relative_bounding_box
                x1, y1, w, h = boundaryBox.xmin, boundaryBox.ymin, boundaryBox.width, boundaryBox.height

                x1 = int(x1 * W)
                y1 = int(y1 * H)
                w = int(w * W)
                h = int(h * H)

                face = image[max(0, y1):min(y1 + h, H), max(0, x1):min(x1 + w, W)]
                gender, age = predictAgeAndGender(face)

                text = f'{gender}, {age}'
                cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)

    return image

# Testing on Image
image = cv2.imread(imagePath)
annotatedImage = annotateFacesWithAge(image)
cv2.imshow("Annotated Image", annotatedImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Testing on Video
video = cv2.VideoCapture(vidoePath)
ret = True
while ret:
    ret, frame = video.read()
    if not ret:
        break
    annotateFacesWithAge(frame)
    cv2.imshow("Video", frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# Testing on Webcam
webcam = cv2.VideoCapture(0)
while True:
    ret, frame = webcam.read()
    annotateFacesWithAge(frame)
    cv2.imshow("Webcam", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
