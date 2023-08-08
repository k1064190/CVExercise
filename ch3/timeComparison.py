import cv2 as cv
import numpy as np
import time

img = cv.imread('Ggol.jpg')
img = cv.resize(img, (0, 0), fx=5, fy=5, interpolation=cv.INTER_LINEAR)

def my_cvtGray1(bgr_img):
    g = np.zeros((bgr_img.shape[0], bgr_img.shape[1]))
    for r in range(bgr_img.shape[0]):
        for c in range(bgr_img.shape[1]):
            g[r, c] = 0.114 * bgr_img[r, c, 0] + 0.587 * bgr_img[r, c, 1] + 0.299 * bgr_img[r, c, 2]
    return np.uint8(g)


def my_cvtGray2(bgr_img):
    g = np.zeros((bgr_img.shape[0], bgr_img.shape[1]))
    g = bgr_img[:, :, 0] * 0.114 + bgr_img[:, :, 1] * 0.587 + bgr_img[:, :, 2] * 0.299
    return g.astype(np.uint8)


start = time.time()
gray1 = my_cvtGray1(img)
end = time.time()
print('my_cvtGray1: ', end - start)

start = time.time()
gray2 = my_cvtGray2(img)
end = time.time()
print('my_cvtGray2: ', end - start)

start = time.time()
gray3 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
end = time.time()
print('cv.cvtColor: ', end - start)

cv.waitKey(0)
cv.destroyAllWindows()
