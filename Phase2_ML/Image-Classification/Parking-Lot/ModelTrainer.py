import os
import pickle
import numpy as np
from sklearn.svm import SVC
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Eeach ML Project Composes Of Four Main Steps

# Preparing Our Data (Inputting And Preprocessing)
currentDirectory = os.path.dirname(os.path.abspath(__file__))
inputDir = os.path.join(currentDirectory, "clf-data")
categories = ["empty", "not_empty"]

datas = []
lables = []

for index, category in enumerate(categories):
    for file in os.listdir(os.path.join(inputDir, category)):
        imagePath = os.path.join(inputDir, category, file)
        image = imread(imagePath)
        datas.append(resize(image, (15, 15)).flatten())
        lables.append(index)

datas = np.asarray(datas)
lables = np.asarray(lables)

# Splitting The Data Up Into Train And Test
x_train, x_test, y_train, y_test = train_test_split(datas, lables, test_size = 0.2,
                                                    shuffle = True, stratify = lables)

# Training Our Classifier We Are Using A Support Vector Machine (Classifier)
classifier = SVC()

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}] # For GridSearch
gridSearch = GridSearchCV(classifier, parameters)

gridSearch.fit(x_train, y_train)

# Testing Our Model's Performance
bestClassifier = gridSearch.best_estimator_
y_prediction = bestClassifier.predict(x_test)
print(f"Our Accuracy Is : {accuracy_score(y_prediction, y_test) * 100:.2f}% !")

# Saving Our Model To Use It Again To Visualize The Empty Parking Spots
pickle.dump(bestClassifier, open('./model.p', 'wb'))
