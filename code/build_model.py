import cv2 as cv
import numpy as np
from solve_sudoku import *
from scanimg import *
import os
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument('--knc_filename', help = 'Path to pickle file storing knc model.')
parser.add_argument('--rfc_filename', help = 'Path to pickle file storing rfc model.')

args = parser.parse_args()

images = []
digits = []

# load dataset
for i in range(1, 10):
    directory = 'dataset/' + str(i) + '/output/'
    c = 0
    for filename in os.listdir(directory):
        if(filename.endswith('.jpg')):
            train_img = cv.imread(directory + filename)

            # feature extraction
            df = hog(train_img, orientations=8, pixels_per_cell=(
                4, 4), cells_per_block=(7, 7))

            images.append(df)
            digits.append(i)
            
            c = c + 1
            if(i == 8 or i == 9):
                c = c - 0.5
            if(c == 2000):
                break

    print('Images for digit', i, 'loaded.')

images = np.array(images, 'float64')

X_train, X_test, y_train, y_test = train_test_split(images, digits)

knn = KNeighborsClassifier(n_neighbors=1)
rfc = RandomForestClassifier(n_estimators=13)

knn.fit(X_train, y_train)
rfc.fit(X_train, y_train)

model_score = knn.score(X_test, y_test)
print('K Neighbors Classifier score:', model_score)

model_score = rfc.score(X_test, y_test)
print('Random Forest Classifier score:', model_score)

joblib.dump(knn, args.knc_filename)
joblib.dump(rfc, args.rfc_filename)