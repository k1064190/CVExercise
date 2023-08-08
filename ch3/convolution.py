import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Ggol.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

cv.putText(gray, 'Ggolangee', (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
cv.imshow('Ggol', gray)

smooth = np.hstack((cv.GaussianBlur(gray, (5, 5), 0.0), cv.GaussianBlur(gray, (9, 9), 0.0), cv.GaussianBlur(gray, (15, 15), 0.0)))
cv.imshow('Gaussian', smooth)

femboss = np.array([[-1.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0]])

gray16 = np.int16(gray)
emboss = np.int8(np.clip(cv.filter2D(gray16, -1, femboss) + 128.0, 0, 255))
cv.imshow('Emboss', emboss)

cv.waitKey()
cv.destroyAllWindows()
