from PyQt5.QtWidgets import *
import sys
import os
import winsound


class BeepSound(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beep Sound")
        self.setGeometry(200, 200, 500, 100)

        shortBeepButton = QPushButton("Short Beep", self)
        longBeepButton = QPushButton("Long Beep", self)
        quitButton = QPushButton("Quit", self)
        self.label = QLabel("Press a button to hear a beep sound", self)

        shortBeepButton.setGeometry(10, 10, 100, 30)
        longBeepButton.setGeometry(110, 10, 100, 30)
        quitButton.setGeometry(210, 10, 100, 30)
        self.label.setGeometry(10, 40, 500, 70)

        shortBeepButton.clicked.connect(self.shortBeepFunction)
        longBeepButton.clicked.connect(self.longBeepFunction)
        quitButton.clicked.connect(self.quitFunction)


    def shortBeepFunction(self):
        self.label.setText("Short Beep will sound with frequency of 1000Hz for 500ms")
        winsound.Beep(1000, 500)


    def longBeepFunction(self):
        self.label.setText("Long Beep will sound with frequency of 1000Hz for 3000ms")
        winsound.Beep(1000, 3000)


    def quitFunction(self):
        self.close()


app = QApplication(sys.argv)
win = BeepSound()
win.show()
app.exec_()






