import os
import shutil
from sklearn.model_selection import train_test_split

# Paths to the images and labels directories
imagesDir = "datas/Image"
labelsDir = "datas/labels"

# Create train and val directories for images and labels
trainImagesDir = os.path.join(imagesDir, "train")
valImagesDir = os.path.join(imagesDir, "val")
trainLabelsDir = os.path.join(labelsDir, "train")
valLabelsDir = os.path.join(labelsDir, "val")

os.makedirs(trainImagesDir, exist_ok=True)
os.makedirs(valImagesDir, exist_ok=True)
os.makedirs(trainLabelsDir, exist_ok=True)
os.makedirs(valLabelsDir, exist_ok=True)

# Get a list of image filenames (without extensions)
imageFiles = [os.path.splitext(f)[0] for f in os.listdir(imagesDir) if os.path.isfile(os.path.join(imagesDir, f))]

# Split the data into training and validation sets (80% train, 20% val)
trainFiles, valFiles = train_test_split(imageFiles, test_size=0.2, random_state=42)

# Function to move files
def moveFiles(fileList, sourceDir, destDir, extension):
    for fileName in fileList:
        srcFile = os.path.join(sourceDir, fileName + extension)
        destFile = os.path.join(destDir, fileName + extension)
        if os.path.exists(srcFile):
            shutil.move(srcFile, destFile)

# Move the image and label files to their respective train and val directories
moveFiles(trainFiles, imagesDir, trainImagesDir, ".jpg")
moveFiles(trainFiles, labelsDir, trainLabelsDir, ".txt")

moveFiles(valFiles, imagesDir, valImagesDir, ".jpg")
moveFiles(valFiles, labelsDir, valLabelsDir, ".txt")
