import cv2 as cv
import numpy as np

img = cv.imread('Ggol.jpg')

def gamma(img, gamma):
    gamma_img = img / 255.0
    gamma_img = np.uint8(255 * (gamma_img**gamma))
    return gamma_img

cv.imshow('gamma', np.hstack((img, gamma(img, 0.3), gamma(img, 0.5), gamma(img, 1.0), gamma(img, 2.0))))

cv.waitKey()
cv.destroyAllWindows()
