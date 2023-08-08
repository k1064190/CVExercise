import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Ggol.jpg')
t, bin_img = cv.threshold(img[:,:,2], 0, 255, cv.THRESH_OTSU)
bin_img2 = cv.adaptiveThreshold(img[:,:,2], 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 31, 2)

print(t)
# cv.imshow('img', img[:,:,2])
# cv.imshow('bin_img', bin_img)
cv.imshow('bin_img2', bin_img2)

se = cv.getStructuringElement(cv.MORPH_DILATE, (5,5))
print(se)
b_dilation = cv.dilate(bin_img2, se, iterations=1)
cv.imshow('b_dilation', b_dilation)

b_erosion = cv.erode(bin_img2, se, iterations=1)
cv.imshow('b_erosion', b_erosion)

b_closing = cv.erode(b_dilation, se, iterations=1)
cv.imshow('b_closing', b_closing)

cv.waitKey()
cv.destroyAllWindows()

