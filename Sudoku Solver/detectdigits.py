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
import joblib


sudoku = [
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
]
img = extractsudoku("./images/sudoku3.jpg")
thresh = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                cv.THRESH_BINARY_INV,39,10)

cv.imshow('dst_rt', thresh)
cv.waitKey(0)
cv.destroyAllWindows()
knn = joblib.load('models/knn_model.pkl')
def feature_extraction(image):
    return hog(color.rgb2gray(image), orientations=8, pixels_per_cell=(4, 4), cells_per_block=(7, 7))

def predict(img):
    df = feature_extraction(img)
    predict = knn.predict(df.reshape(1,-1))[0]
    predict_proba = knn.predict_proba(df.reshape(1,-1))
    return predict, predict_proba[0][predict-1]
t = 300
for i in range(9):
    for j in range(9):
        cropped = thresh[j*40:j*40+40, i*40:i*40+40]
        cropped = cv.bitwise_not(cropped)
        contours, hierarchy = cv.findContours(cropped, cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
        cropped = cv.cvtColor(cropped, cv.COLOR_GRAY2RGB)
        #cv.drawContours(cropped, contours, -1, (0, 255, 0), 2) 
        blank = np.zeros(cropped.shape, np.uint8)
        for c in contours:
            rect = cv.boundingRect(c)
            if(cv.contourArea(c)>50 and cv.contourArea(c)<1300):
                if(rect[2] < 35 and rect[3] <35 and rect[2]*rect[3]>100):
                    print(j,i)
                    x,y,w,h = rect
                    blank[y: y+h, x: x+w] = cropped[y: y+h, x: x+w]
                    print(blank.shape)
                    #blank = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
                    cv.drawContours(blank, [c], -1, (0, 0, 255), 1) 
                    #cv.imwrite("dataset/digit"+str(t)+'.jpg', blank)
                    t = t+1
                    ans = predict(blank)
                    sudoku[j][i] = ans[0]
                    #cv.imshow("Show Boxes", blank)
                    #cv.waitKey(0)
                    #cv.destroyAllWindows()
print(sudoku)
cv.imshow('dst_rt', thresh)
cv.waitKey(0)
cv.destroyAllWindows()
solved = solve(sudoku,0,0)
if(solved!=0):
    blank = np.zeros(img.shape, np.uint8)
    for i in range(9):
        for j in range(9):
            cv.putText(blank, str(sudoku[i][j]), (j*40 + 10, i*40 + 30), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    cv.imshow('original', img)
    cv.imshow('solved',blank)
    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("Inavlid!")
