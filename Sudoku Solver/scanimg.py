import numpy
import cv2 as cv
import numpy as np
import operator

def extractsudoku(filename):
    img  = cv.imread(filename)
    original = img.copy()

    #convert image to greyscale
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    grey = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    #convert each pixel to either complete black or white
    th = cv.adaptiveThreshold(grey,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY_INV,45,10)

    #find contours
    contours, heirarchy = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #cv.drawContours(img, contours, -1, (0, 255, 0), 3) 

    #find largest contour
    maxarea  = 0
    cnt = contours[0]
    for contour in contours:
        if(cv.contourArea(contour)>maxarea):
            cnt = contour
            maxarea = cv.contourArea(contour)

    cnt = [cnt]
    cv.drawContours(img, cnt, -1,(0, 255, 0), 2)
    polygon = cnt[0]
    points = []
    point, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    points.append(tuple(int(x) for x in polygon[point][0]))
    point, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    points.append(tuple(int(x) for x in polygon[point][0]))
    point, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    points.append(tuple(int(x) for x in polygon[point][0]))
    point, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    points.append(tuple(int(x) for x in polygon[point][0]))
    print(points)

    """
    #filter out lines
    blank = np.zeros(img.shape,np.uint8)
    cv.drawContours(blank,cnt,-1,(255,255,255),2)
    edges = cv.Canny(blank, 40, 150, apertureSize = 3)
    lines = cv.HoughLines(edges, 1, np.pi/180, int(img.shape[1]/6))

    hor = []
    ver = []
    print(lines)
    for line in lines:
        for (rho,theta) in line:
            f = 0 
            for (_rho,_theta) in hor:
                if abs(rho-_rho)<20 and abs(theta-_theta)<10:
                    print(rho,theta,_rho,_theta)
                    f = 1
            for (_rho,_theta) in ver:
                if abs(rho-_rho)<20 and abs(theta-_theta)<10:
                    print(rho,theta,_rho,_theta)
                    f = 1
            if f == 0:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                d = np.linalg.norm(np.array((x1,y1,0))-np.array((x2,y2,0)))
                cv.line(img,(x1,y1),(x2,y2),(255,0,0),1)
                m = abs(1/np.tan(theta))
                if(m<1):
                    hor.append((rho,theta))
                else:
                    ver.append((rho,theta))
    points = []
    print(hor)
    print(ver)

    #find intersection points
    for (rho, theta) in hor:
        for (_rho, _theta) in ver:
            if((rho,theta)!=(_rho,_theta)):
                a = [[np.cos(theta),np.sin(theta)],[np.cos(_theta),np.sin(_theta)]]
                b = [rho, _rho]
                cor = np.linalg.solve(a,b)
                cor = list(cor)
                if cor not in points:
                    points.append(cor)
    """
    points.sort()
    print(points)
    if (points[0][1]>points[1][1]):
        points[0],points[1]=points[1],points[0]
    if (points[-1][1]<points[-2][1]):
        points[-1],points[-2]=points[-2],points[-1]

    points[1],points[2]=points[2],points[1]

    #crop image
    pts1 = np.float32(points)
    size = 360
    pts2 = np.float32([[0,0],[size,0],[0,size],[size,size]])
    print(points)
    M = cv.getPerspectiveTransform(pts1, pts2)

    img = cv.warpPerspective(th,M,(size,size))

    """
    cv.imshow('dst_rt', img)
    #cv.imshow('dst_rt1', th)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """
    return img, [pts2, pts1]

