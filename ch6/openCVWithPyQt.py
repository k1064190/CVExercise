import cv2 as cv
from PyQt5.QtWidgets import *
import sys


class Video(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video")
        self.setGeometry(200, 200, 500, 100)
        self.frame = None

        videoButton = QPushButton("Turn on video", self)
        captureButton = QPushButton("Capture the frame", self)
        saveButton = QPushButton("Save the frame", self)
        quitButton = QPushButton("Quit", self)

        videoButton.setGeometry(10, 10, 100, 30)
        captureButton.setGeometry(110, 10, 100, 30)
        saveButton.setGeometry(210, 10, 100, 30)
        quitButton.setGeometry(310, 10, 100, 30)

        videoButton.clicked.connect(self.videoFunction)
        captureButton.clicked.connect(self.captureFunction)
        saveButton.clicked.connect(self.saveFunction)
        quitButton.clicked.connect(self.quitFunction)


    def videoFunction(self):
        self.cap = cv.VideoCapture(0, cv.CAP_DSHOW)
        if not self.cap.isOpened():
            self.close()

        while True:
            ret, self.frame = self.cap.read()
            if not ret: break
            cv.imshow('Video display', self.frame)
            if cv.waitKey(1) == ord('q'):
                cv.destroyWindow('Video display')
                break


    def captureFunction(self):
        if self.frame is not None:
            self.capturedFrame = self.frame
            cv.imshow('Captured frame', self.capturedFrame)


    def saveFunction(self):
        if self.frame is None:
            print("No frame is captured")
            return
        fname = QFileDialog.getSaveFileName(self, 'Save file', './')
        if fname[0] != '':
            cv.imwrite(fname[0], self.capturedFrame)


    def quitFunction(self):
        self.cap.release()
        cv.destroyAllWindows()
        self.close()


app = QApplication(sys.argv)
win = Video()
win.show()
app.exec_()
