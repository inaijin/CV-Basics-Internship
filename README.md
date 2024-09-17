# Computer Vision Internship Report

This repository contains the projects completed as part of my Computer Vision internship. The internship was divided into two phases, each focusing on different aspects of image processing and computer vision. Below is a detailed breakdown of the first phase.

## Phase 1: Image Processing with OpenCV

In this phase, I focused on the fundamentals of image processing, learning to work with OpenCV in Python. The main topics covered were:

- **Image Input/Output**: Loading, saving, and displaying images.
- **Operations**: Blurring, drawing shapes, edge detection, and thresholding.
- **Color Spaces**: Conversion between color spaces like RGB, HSV, etc.
- **Contours**: Detecting and working with contours in images.

### Projects

1. [Age Detection](#age-detection)
2. [Color Detection](#color-detection)
3. [Face Anonymizer](#face-anonymizer)
4. [Text Detection](#text-detection)
5. [Tumor Detection](#tumor-detection)

### [Age Detection](https://github.com/inaijin/CV-Basics-Internship/tree/main/Phase1_Image_Processing/Projects/Age-Detection)

**Description**:  
In this project, I developed an age-detection system using the **age_net** model from OpenCV's deep learning module (`cv2.dnn.readNet`). The model predicts the age range of a person from an image, using pre-trained weights.

**Steps**:
- Load the **age_net** model.
- Perform face detection using OpenCV's `cv2.CascadeClassifier`.
- Apply the age model to the detected face regions to predict age.

**Results**:  
<img src="Datas/Results/Age-Detection/age-man.png" alt="Age-Detection Results" width="555"/>

### [Color Detection](https://github.com/inaijin/CV-Basics-Internship/tree/main/Phase1_Image_Processing/Projects/Color-Detection)

**Description**:  
This project focused on detecting a specific color in an image by converting it to the **HSV color space**. A bounding box was drawn around the detected object to highlight the color.

**Steps**:
- Convert the image to the HSV color space using `cv2.cvtColor()`.
- Use a color range mask to filter the target color.
- Draw a bounding box around the detected object.

**Results**:  
<img src="Datas/Results/Color-Detection/color-detected.png" alt="Color-Detection Results" width="400"/>

### [Face Anonymizer](https://github.com/inaijin/CV-Basics-Internship/tree/main/Phase1_Image_Processing/Projects/Face-Anonymizer)

**Description**:  
In this project, I implemented a face anonymization system where detected faces were blurred to hide identities. I used **MediaPipe's Face Detection** (`faceDetectionMP.FaceDetection`) for detecting faces.

**Steps**:
- Detect faces in the image using MediaPipe.
- Apply Gaussian blur to the detected face regions.

**Results**:  
<img src="Datas/Results/Face-Anonymizer/Face-Anonymizer.png" alt="Face-Anonymizer Results" width="900"/>

### [Text Detection](https://github.com/inaijin/CV-Basics-Internship/tree/main/Phase1_Image_Processing/Projects/Text-Detection)

**Description**:  
This project aimed at detecting text in images using the **easyOCR** library. The detected text was highlighted with bounding boxes, and the recognized characters were displayed.

**Steps**:
- Load the image and pass it through the easyOCR model.
- Draw bounding boxes around the detected text.
- Display the recognized text.

**Results**:  
<img src="Datas/Results/Text-Detection/Text-Detection.png" alt="Text-Detection Results" width="555"/>

### [Tumor Detection](https://github.com/inaijin/CV-Basics-Internship/tree/main/Phase1_Image_Processing/Projects/Tumor-Detection)

**Description**:  
This project involved detecting tumors in brain scans using image processing techniques. The image was split in two, and differences between the two halves were used to locate the tumor. **Contours** and **thresholding** techniques helped in highlighting the tumor region.

**Steps**:
- Split the image into two halves.
- Compute differences between the halves to identify abnormalities.
- Use contour detection and thresholding to locate the tumor.

**Results**:  
<img src="Datas/Results/Tumor-Detection/Tumor-Detection.png" alt="Tumor-Detection Results" width="400"/>
