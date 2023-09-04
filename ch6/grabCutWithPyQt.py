import cv2 as cv
from PyQt5.QtWidgets import *
import numpy as np
import sys


class Orim(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Orim")
        self.setGeometry(200, 200, 700, 200)
        self.LColor, self.RColor = (255, 0, 0), (0, 0, 255)  # Blue, Red
        self.brushSize = 5

        fileButton = QPushButton("File", self)
        paintButton = QPushButton("Paint", self)
        cutButton = QPushButton("Cut", self)
        incButton = QPushButton("Increase", self)
        decButton = QPushButton("Decrease", self)
        saveButton = QPushButton("Save", self)
        quitButton = QPushButton("Quit", self)

        fileButton.setGeometry(10, 10, 100, 30)
        paintButton.setGeometry(110, 10, 100, 30)
        cutButton.setGeometry(210, 10, 100, 30)
        incButton.setGeometry(310, 10, 100, 30)
        decButton.setGeometry(410, 10, 100, 30)
        saveButton.setGeometry(510, 10, 100, 30)
        quitButton.setGeometry(610, 10, 100, 30)

        fileButton.clicked.connect(self.fileFunction)
        paintButton.clicked.connect(self.paintFunction)
        cutButton.clicked.connect(self.cutFunction)
        incButton.clicked.connect(self.incFunction)
        decButton.clicked.connect(self.decFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)


    def fileFunction(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')
        if fname[0] != '':
            self.img = cv.imread(fname[0])
            if self.img is None:
                print("No image file")
                return

            self.img_show = np.copy(self.img)
            cv.imshow('Painting', self.img_show)

            self.mask = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)
            self.mask[:, :] = cv.GC_PR_BGD


    def paintFunction(self):
        cv.setMouseCallback('Painting', self.painting)


    def painting(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(self.img_show, (x, y), self.brushSize, self.LColor, -1)
            cv.circle(self.mask, (x, y), self.brushSize, cv.GC_FGD, -1)
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:
            cv.circle(self.img_show, (x, y), self.brushSize, self.LColor, -1)
            cv.circle(self.mask, (x, y), self.brushSize, cv.GC_FGD, -1)
        elif event == cv.EVENT_RBUTTONDOWN:
            cv.circle(self.img_show, (x, y), self.brushSize, self.RColor, -1)
            cv.circle(self.mask, (x, y), self.brushSize, cv.GC_BGD, -1)
        elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_RBUTTON:
            cv.circle(self.img_show, (x, y), self.brushSize, self.RColor, -1)
            cv.circle(self.mask, (x, y), self.brushSize, cv.GC_BGD, -1)

        cv.imshow('Painting', self.img_show)


    def cutFunction(self):
        background = np.zeros((1, 65), np.float64)
        foreground = np.zeros((1, 65), np.float64)
        print("Set background and foreground")
        cv.grabCut(self.img, self.mask, None, background, foreground, 5, cv.GC_INIT_WITH_MASK)
        print("GrabCut finished")
        mask2 = np.where((self.mask == cv.GC_BGD) | (self.mask == cv.GC_PR_BGD), 0, 1).astype('uint8')
        self.grabImg = self.img * mask2[:, :, np.newaxis]
        cv.imshow('Scissoring', self.grabImg)


    def incFunction(self):
        self.brushSize = min(20, self.brushSize + 1)

    def decFunction(self):
        self.brushSize = max(1, self.brushSize - 1)

    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self, 'Save file', './')
        if fname[0] != '':
            cv.imwrite(fname[0], self.grabImg)

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()


app = QApplication(sys.argv)
win = Orim()
win.show()
app.exec_()

