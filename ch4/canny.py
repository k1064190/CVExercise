import cv2 as cv
import numpy as np

img = cv.imread('son.jpg')
img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(gray.shape)

canny1 = cv.Canny(gray, 50, 150)
print(canny1.shape)
canny2 = cv.Canny(gray, 100, 200)
print(canny2.shape)

cv.imshow('Canny', np.hstack((gray, canny1, canny2)))

cv.waitKey()
cv.destroyAllWindows()

