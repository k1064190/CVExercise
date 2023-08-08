import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Ggol.jpg')
img = img[25:125, 25:125, :]

# nearest interpolation
nearest = cv.resize(img, (0, 0), fx=5, fy=5, interpolation=cv.INTER_NEAREST)

# bilinear interpolation
bilinear = cv.resize(img, (0, 0), fx=5, fy=5, interpolation=cv.INTER_LINEAR)

# bicubic interpolation
bicubic = cv.resize(img, (0, 0), fx=5, fy=5, interpolation=cv.INTER_CUBIC)

# hstack
res = np.hstack((nearest, bilinear, bicubic))

cv.imshow('res', res)

cv.waitKey(0)
cv.destroyAllWindows()
