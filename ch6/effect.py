import cv2 as cv
import numpy as np
from PyQt5.QtWidgets import *
import sys

class SpecialEffect(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Special Effect")
        self.setGeometry(200, 200, 800, 100)

        pictureButton = QPushButton("Load picture", self)
        embossButton = QPushButton("Embossing", self)
        carttonButton = QPushButton("Cartoonizing", self)
        sketchButton = QPushButton("Sketching", self)
        oilButton = QPushButton("Oil painting", self)
        saveButton = QPushButton("Save", self)
        quitButton = QPushButton("Quit", self)
        self.pickCombo = QComboBox(self)
        self.pickCombo.addItems(["Embossing", "Cartoonizing", "Sketching", "Color Sketching", "Oil painting"])
        self.label = QLabel("Welcome!", self)

        pictureButton.setGeometry(10, 10, 100, 30)
        embossButton.setGeometry(110, 10, 100, 30)
        carttonButton.setGeometry(210, 10, 100, 30)
        sketchButton.setGeometry(310, 10, 100, 30)
        oilButton.setGeometry(410, 10, 100, 30)
        saveButton.setGeometry(510, 10, 100, 30)
        self.pickCombo.setGeometry(510, 40, 110, 30)
        quitButton.setGeometry(620, 10, 100, 30)
        self.label.setGeometry(10, 40, 500, 170)

        pictureButton.clicked.connect(self.pictureFunction)
        embossButton.clicked.connect(self.embossFunction)
        carttonButton.clicked.connect(self.cartoonFunction)
        sketchButton.clicked.connect(self.sketchFunction)
        oilButton.clicked.connect(self.oilFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)


    def pictureFunction(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', './')
        if fname != '':
            self.img = cv.imread(fname[0])
            if self.img is None:
                print("No image file")
                sys.exit()
            cv.imshow('Painting', self.img)

    def embossFunction(self):
        femboss = np.array([[-1.0, 0.0, 0.0],[0.0, 0.0, 0.0],[0.0, 0.0, 1.0]])

        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        gray16 = np.int16(gray)
        self.emboss = np.uint8(np.clip(cv.filter2D(gray16, -1, femboss)+128,0, 255))

        cv.imshow('Emboss', self.emboss)

    def cartoonFunction(self):
        self.cartoon = cv.stylization(self.img, sigma_s=60, sigma_r=0.45)
        cv.imshow('Cartoon', self.cartoon)

    def sketchFunction(self):
        self.sketch_gray, self.sketch_color = cv.pencilSketch(self.img, sigma_s=60, sigma_r=0.07, shade_factor=0.02)
        cv.imshow('Pencil Sketch(gray)', self.sketch_gray)
        cv.imshow('Pencil Sketch(color)', self.sketch_color)

    def oilFunction(self):
        self.oil = cv.xphoto.oilPainting(self.img, 10, 1, cv.COLOR_BGR2Lab)
        cv.imshow('Oil painting', self.oil)

    def saveFunction(self):
        fname = QFileDialog.getSaveFileName(self, 'Save file', './')
        if fname != '':
            i = self.pickCombo.currentIndex()
            if i == 0: cv.imwrite(fname[0], self.emboss)
            elif i == 1: cv.imwrite(fname[0], self.cartoon)
            elif i == 2: cv.imwrite(fname[0], self.sketch_gray)
            elif i == 3: cv.imwrite(fname[0], self.sketch_color)
            elif i == 4: cv.imwrite(fname[0], self.oil)

    def quitFunction(self):
        cv.destroyAllWindows()
        self.close()


app = QApplication(sys.argv)
window = SpecialEffect()
window.show()
app.exec_()
