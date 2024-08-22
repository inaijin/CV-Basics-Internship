import os
import shutil
from sklearn.model_selection import train_test_split

# Function to split data into train and validation sets
def split_data(category, rootPath, trainPath, valPath, train_ratio=0.8):
    categoryPath = os.path.join(rootPath, category)
    images = os.listdir(categoryPath)

    train_images, val_images = train_test_split(images, train_size=train_ratio, random_state=42)

    trainCategoryPath = os.path.join(trainPath, category)
    valCategoryPath = os.path.join(valPath, category)

    os.makedirs(trainCategoryPath, exist_ok=True)
    os.makedirs(valCategoryPath, exist_ok=True)

    for image in train_images:
        shutil.move(os.path.join(categoryPath, image), os.path.join(trainCategoryPath, image))

    for image in val_images:
        shutil.move(os.path.join(categoryPath, image), os.path.join(valCategoryPath, image))
