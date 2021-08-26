import cv2 as cv
import numpy as np
from solve_sudoku import *
from skimage.feature import hog
from skimage import data, color, exposure
import joblib
import operator
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument('--model_filename', help = 'Path to pickle file storing the model.')
parser.add_argument('--sudoku_image', help = 'Path to image of sudoku puzzle.')

args = parser.parse_args()

model_filename = args.model_filename

model = joblib.load(model_filename)
print('Model loaded from', model_filename)


def extract_sudoku(filename):
    img = cv.imread(filename)
    original = img.copy()

    # convert image to greyscale
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # convert each pixel to either complete black or white
    th = cv.adaptiveThreshold(
        grey, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 45, 10)

    # find contours
    contours, heirarchy = cv.findContours(
        th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # find largest contour
    maxarea = 0
    largest_contour = contours[0]
    for contour in contours:
        if(cv.contourArea(contour) > maxarea):
            largest_contour = contour
            maxarea = cv.contourArea(contour)

    largest_contour = [largest_contour]
    cv.drawContours(img, largest_contour, -1, (0, 255, 0), 2)
    polygon = largest_contour[0]
    points = []
    point, _ = max(enumerate([pt[0][0] + pt[0][1]
                   for pt in polygon]), key=operator.itemgetter(1))
    points.append(tuple(int(x) for x in polygon[point][0]))
    point, _ = min(enumerate([pt[0][0] + pt[0][1]
                   for pt in polygon]), key=operator.itemgetter(1))
    points.append(tuple(int(x) for x in polygon[point][0]))
    point, _ = min(enumerate([pt[0][0] - pt[0][1]
                   for pt in polygon]), key=operator.itemgetter(1))
    points.append(tuple(int(x) for x in polygon[point][0]))
    point, _ = max(enumerate([pt[0][0] - pt[0][1]
                   for pt in polygon]), key=operator.itemgetter(1))
    points.append(tuple(int(x) for x in polygon[point][0]))
    points.sort()
    if (points[0][1] > points[1][1]):
        points[0], points[1] = points[1], points[0]
    if (points[-1][1] < points[-2][1]):
        points[-1], points[-2] = points[-2], points[-1]

    points[1], points[2] = points[2], points[1]

    # crop image
    pts1 = np.float32(points)
    size = 360
    pts2 = np.float32([[0, 0], [size, 0], [0, size], [size, size]])
    print(points)
    M = cv.getPerspectiveTransform(pts1, pts2)
    img = cv.warpPerspective(th, M, (size, size))
    return img, [pts2, pts1]


def predict_digit(image):
    # feature extraction
    image = hog(color.rgb2gray(image), orientations=8,
                pixels_per_cell=(4, 4), cells_per_block=(7, 7))
    predict = model.predict(image.reshape(1, -1))[0]
    predict_proba = model.predict_proba(image.reshape(1, -1))
    return predict, predict_proba[0][predict-1]


def solveit(filename):
    unsolved = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    sudoku = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    img, pts = extract_sudoku(filename)
    print('Puzzle extracted from', filename)
    thresh = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                  cv.THRESH_BINARY_INV, 39, 10)

    for i in range(9):
        for j in range(9):

            # extract digit from ith row and jth column
            cropped = thresh[j * 40: j * 40 + 40, i * 40: i * 40 + 40]
            cropped = cv.bitwise_not(cropped)
            contours, _ = cv.findContours(
                cropped, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            cropped = cv.cvtColor(cropped, cv.COLOR_GRAY2RGB)
            solved_sudoku = np.zeros(cropped.shape, np.uint8)

            # identify contour containing digit
            for c in contours:
                rect = cv.boundingRect(c)
                if(cv.contourArea(c) > 50 and cv.contourArea(c) < 1300):
                    if(rect[2] < 35 and rect[3] < 35 and rect[2] * rect[3] > 100):
                        x, y, w, h = rect

                        # super impose the digit on a blank background
                        solved_sudoku[y: y + h, x: x +
                                      w] = cropped[y: y + h, x: x + w]

                        # predict digit from image
                        digit, _ = predict_digit(solved_sudoku)
                        sudoku[j][i] = digit
                        unsolved[j][i] = digit
                        break

    print('Unsolved sudoku:', sudoku)

    solved = solve_sudoku(sudoku, 0, 0)

    # check if solution exists
    if(solved != 0):
        print('Solved sudoku:', solved)
        solved_sudoku = np.zeros(img.shape, np.uint8)
        solved_sudoku = cv.cvtColor(solved_sudoku, cv.COLOR_GRAY2RGB)

        for i in range(9):
            for j in range(9):
                if(unsolved[i][j] == 0):
                    cv.putText(solved_sudoku, str(
                        sudoku[i][j]), (j * 40 + 10, i * 40 + 30), cv.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

        # augument the solution on the original picture
        transform = cv.getPerspectiveTransform(pts[0], pts[1])
        original = cv.imread(filename)
        img = cv.warpPerspective(
            solved_sudoku, transform, (original.shape[1], original.shape[0]))
        img = cv.bitwise_not(img)
        image = cv.bitwise_and(img, original)
        
        cv.imshow('Solution', image)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print('Inavlid!')

solveit(args.sudoku_image)