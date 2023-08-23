import cv2 as cv

img = cv.imread('apples.webp')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

apples = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 200, param1=150, param2=50, minRadius=20, maxRadius=120)

for apple in apples[0]:
    cv.circle(img, (int(apple[0]), int(apple[1])), int(apple[2]), (255, 0, 0), 2)

cv.imshow('Apples', img)

cv.waitKey(0)
cv.destroyAllWindows()
