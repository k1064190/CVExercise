import cv2 as cv
from PyQt5.QtWidgets import *
import sys
import winsound
import numpy as np


class TrafficWeak(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Weak")
        self.setGeometry(200, 200, 700, 200)

        signButton = QPushButton("Register sign", self)
        roadButton = QPushButton("Load road", self)
        recognitionButton = QPushButton("Recognition", self)
        quitButton = QPushButton("Quit", self)

        self.label = QLabel("Welcome!", self)

        signButton.setGeometry(10, 10, 100, 30)
        roadButton.setGeometry(110, 10, 100, 30)
        recognitionButton.setGeometry(210, 10, 100, 30)
        quitButton.setGeometry(510, 10, 100, 30)
        self.label.setGeometry(10, 40, 600, 170)

        signButton.clicked.connect(self.signFunction)
        roadButton.clicked.connect(self.roadFunction)
        recognitionButton.clicked.connect(self.recognitionFunction)
        quitButton.clicked.connect(self.quitFunction)

        self.signFiles = [['child.png', 'child'], ['elder.png', 'elder'], ['disabled.png', 'disabled']]
        self.signImgs = []


    def signFunction(self):
        self.label.clear()
        self.label.setText("Register Transportation Disabled Sign")

        for fname, _ in self.signFiles:
            self.signImgs.append(cv.imread(fname))
            cv.imshow(fname, self.signImgs[-1])


    def roadFunction(self):
        if self.signImgs == []:
            self.label.setText("Register Sign first")
            return
        else:
            fname = QFileDialog.getOpenFileName(self, 'Open file', './')
            self.roadImg = cv.imread(fname[0])
            if self.roadImg is None:
                print("No image file")
                return
            cv.imshow('Road scene', self.roadImg)


    def recognitionFunction(self):
        if self.roadImg is None:
            self.label.setText("Load road first")
            return
        else:
            sift = cv.SIFT_create()

            KD = []
            for img in self.signImgs:
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                KD.append(sift.detectAndCompute(gray, None))
            print("Got feature points and descriptors of signs")

            grayRoad = cv.cvtColor(self.roadImg, cv.COLOR_BGR2GRAY)
            road_kp, road_des = sift.detectAndCompute(grayRoad, None)
            print("Got feature points and descriptors of road")

            matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
            GM = []
            for sign_kp, sign_des in KD:
                knn_matches = matcher.knnMatch(sign_des, road_des, k=2)
                T = 0.7
                good_matches = []
                for nearest1, nearest2 in knn_matches:
                    if nearest1.distance / nearest2.distance < T:
                        good_matches.append(nearest1)
                GM.append(good_matches)

            best = GM.index(max(GM, key=len))
            print("Best sign is", self.signFiles[best][1])

            if len(GM[best]) < 4:
                self.label.setText("No sign is detected")
                print("No sign is detected")
                return
            else:
                sign_kp = KD[best][0]
                good_match = GM[best]
                print("Got good match")

                points1 = np.float32([sign_kp[gm.queryIdx].pt for gm in good_match])
                points2 = np.float32([road_kp[gm.trainIdx].pt for gm in good_match])
                print("Get points from good match")

                H, _ = cv.findHomography(points1, points2, cv.RANSAC)
                print("Made homography matrix")

                h1, w1 = self.signImgs[best].shape[0], self.signImgs[best].shape[1]
                h2, w2 = self.roadImg.shape[0], self.roadImg.shape[1]

                box1 = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(4, 1, 2)
                box2 = cv.perspectiveTransform(box1, H)
                print("Made perspective transform")

                self.roadImg = cv.polylines(self.roadImg, [np.int32(box2)], True, (0, 255, 0), 4)
                img_match = np.empty((max(h1, h2), w1+w2, 3), dtype=np.uint8)
                cv.drawMatches(self.signImgs[best], sign_kp, self.roadImg, road_kp, good_match, img_match,\
                               flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                print("Drew matches")

                cv.imshow('Matched', img_match)
                self.label.setText(self.signFiles[best][1] + " is detected")
                winsound.Beep(3000, 500)


    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()


app = QApplication(sys.argv)
trafficWeak = TrafficWeak()
trafficWeak.show()
app.exec_()
