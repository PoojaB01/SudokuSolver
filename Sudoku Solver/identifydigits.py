import numpy
import cv2 as cv
import numpy as np
from solver import *
from scanimg import *
import os
import scipy.ndimage
from skimage.feature import hog
from skimage import data, color, exposure
from sklearn.model_selection import  train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

images = []
number = []

for i in range (1,10):
    directory =  'dataset/' + str(i) + '/output/'
    c = 0
    for filename in os.listdir(directory):
        if(filename.endswith('.jpg')):
            train_img = cv.imread(directory + filename)
            #contours, hierarchy = cv.findContours(train_img, cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
            #cv.drawContours(train_img, contours, -1, (0, 0, 255), 2) 
            df = hog(train_img, orientations=8, pixels_per_cell=(4,4), cells_per_block=(7, 7))
            images.append(df)
            number.append(i)
            c = c+1
            if(i==8 or i==9):
                c = c-0.5
            if(c==2000):
                break
    print(str(i)+ " DONE!")

images = np.array(images, 'float64')

X_train, X_test, y_train, y_test = train_test_split(images, number)

knn = KNeighborsClassifier(n_neighbors=1)
rfc = RandomForestClassifier(n_estimators=13)
knn.fit(images, number)
#rfc.fit(X_train, y_train)
model_score = knn.score(X_test, y_test)

print(model_score)

joblib.dump(knn, 'models/knn_model1.pkl')