import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread('Ggol.jpg')

plt.subplots(1, 4, figsize=(30, 5))
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
axs =plt.subplot(1, 4, 1)
axs.set_title('Original')
plt.imshow(gray, cmap='gray')
plt.xticks([])
plt.yticks([])
h = cv.calcHist([gray], [0], None, [256], [0, 256])
plt.subplot(1, 4, 2)
plt.title('Histogram')
plt.plot(h, color='r')
plt.xlim([0, 256])
equal = cv.equalizeHist(gray)
plt.subplot(1, 4, 3)
plt.title('Equalized')
plt.imshow(equal, cmap='gray')
plt.xticks([])
plt.yticks([])
h = cv.calcHist([equal], [0], None, [256], [0, 256])
plt.subplot(1, 4, 4)
plt.title('Histogram')
plt.plot(h, color='r')
plt.xlim([0, 256])
plt.show()

