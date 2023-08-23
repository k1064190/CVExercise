import cv2 as cv
import numpy as np
import skimage

img = skimage.data.coffee()
cv.imshow('original', cv.cvtColor(img, cv.COLOR_BGR2RGB))

slic1 = skimage.segmentation.slic(img, n_segments=600, compactness=20)
sp_img1 = skimage.segmentation.mark_boundaries(img, slic1)
sp_img1 = np.uint8(sp_img1 * 255.0)

slic2 = skimage.segmentation.slic(img, n_segments=600, compactness=40)
sp_img2 = skimage.segmentation.mark_boundaries(img, slic2)
sp_img2 = np.uint8(sp_img2 * 255.0)

cv.imshow('slic1', cv.cvtColor(sp_img1, cv.COLOR_BGR2RGB))
cv.imshow('slic2', cv.cvtColor(sp_img2, cv.COLOR_BGR2RGB))

cv.waitKey()
cv.destroyAllWindows()
