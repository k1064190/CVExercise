import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys
import winsound


class Panorama(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Panorama")
        self.setGeometry(200, 200, 700, 200)

        self.collectButton = QPushButton("Collect images", self)
        self.showButton = QPushButton("Show panorama", self)
        self.stitchButton = QPushButton("Stitch images", self)
        self.saveButton = QPushButton("Save panorama", self)
        self.quitButton = QPushButton("Quit", self)
        self.label = QLabel("Welcome!", self)

        self.collectButton.setGeometry(10, 25, 100, 30)
        self.showButton.setGeometry(110, 25, 100, 30)
        self.stitchButton.setGeometry(210, 25, 100, 30)
        self.saveButton.setGeometry(310, 25, 100, 30)
        self.quitButton.setGeometry(510, 25, 100, 30)
        self.label.setGeometry(10, 70, 600, 170)

        self.showButton.setEnabled(False)
        self.stitchButton.setEnabled(False)
        self.saveButton.setEnabled(False)

        self.collectButton.clicked.connect(self.collectFunction)
        self.showButton.clicked.connect(self.showFunction)
        self.stitchButton.clicked.connect(self.stitchFunction)
        self.saveButton.clicked.connect(self.saveFunction)
        self.quitButton.clicked.connect(self.quitFunction)

    def collectFunction(self):
        self.showButton.setEnabled(False)
        self.stitchButton.setEnabled(False)
        self.saveButton.setEnabled(False)
        self.label.setText("press C to capture image, press Q to quit")

        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not self.cap.isOpened():
            print("Cannot open camera")
            sys.exit()

        self.imgs = []
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            cv.imshow('Video display', frame)
            key = cv.waitKey(1)
            if key == ord('c'):
                self.imgs.append(frame)
                winsound.Beep(1000, 100)
            if key == ord('q'):
                cv.destroyWindow('Video display')
                break

        if len(self.imgs) >= 2:
            self.showButton.setEnabled(True)
            self.stitchButton.setEnabled(True)
            self.saveButton.setEnabled(True)


    def showFunction(self):
        self.label.setText(str(len(self.imgs)) + " images are collected")
        stack = cv.resize(self.imgs[0], dsize=(0, 0), fx=0.25, fy=0.25)
        for i in range(1, len(self.imgs)):
            stack = np.hstack((stack, cv.resize(self.imgs[i], dsize=(0, 0), fx=0.25, fy=0.25)))
        cv.imshow('Collected images', stack)

    def stitchFunction(self):
        stitcher = cv.Stitcher_create()
        status, self.img_stitched = stitcher.stitch(self.imgs)
        if status == cv.STITCHER_OK:
            cv.imshow("Image stitched panorama", self.img_stitched)
        else:
            print("Stitching failed!")
            winsound.Beep(3000, 500)

    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self, 'Save file', './')
        if fname[0] != '':
            cv.imwrite(fname[0], self.img_stitched)

    def quitFunction(self):
        self.cap.release()
        cv.destroyAllWindows()
        self.close()


app = QApplication(sys.argv)
win = Panorama()
win.show()
app.exec_()

