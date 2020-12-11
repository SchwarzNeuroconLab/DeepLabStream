"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""

from utils.configloader import RESOLUTION

import cv2
from PySide2.QtCore import QObject
from PySide2.QtCore import Signal as pyqtSignal
from PySide2.QtCore import Slot as pyqtSlot
from PySide2.QtWidgets import QWidget, QLabel
from PySide2.QtGui import QImage, QPixmap
#from PyQt5.QtCore import QObject, pyqtSignal, Qt, pyqtSlot
#from PyQt5.QtWidgets import QWidget, QLabel
#from PyQt5.QtGui import QImage, QPixmap

width, height = RESOLUTION


class ImageWindow(QWidget):
    """
    Image window is used to show an image within PyQt GUI
    This example is hardcoded for width and height of currently defined resolution
    """
    def __init__(self, name):
        super().__init__()
        self.title = name
        self.left = 0
        self.top = 0
        self.width, self.height = width, height
        self.label = QLabel(self)
        self.init_ui()

    @pyqtSlot(QImage)
    def set_image(self, image):
        """
        This is a slot for QImage widget
        We use this to "catch" an emitted image
        :param image: QImage
        """
        # using label to output the image
        self.label.setPixmap(QPixmap.fromImage(image))

    def init_ui(self):
        """
        Creating a UI itself
        """
        # title
        self.setWindowTitle(self.title)
        # geometric parameters
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(self.width, self.height)
        # create a label
        self.label.resize(self.width, self.height)

    def closeEvent(self, event):
        # keeps window from closing on accident
        event.ignore()


class QFrame(QObject):
    """
    This is a dummy class to workaround PyQt5 restrictions on declaring signals
    Basically, it does not allow to be declared in an instance of a class, or it will become bounded
    You need to define it as a class variable, but that does not work every time, especially for dynamic tasks
    In short -
    Okay:
        class A:
            smth = pyqtSignal()
    Not okay:
        class A:
            def __init__()
            self.smth = pyqtSignal()

    But both of the examples provide the same functionality, A.smth is a pyqtSignal
    So we workaround it, making a dummy class and going with the "okay" route
    This is essential for multicam support

    ! Probably have a tax on the performance
    """
    signal = pyqtSignal(QImage)


def emit_qframes(frames, qframes):
    """
    Emit some number of stream frames, depending on cameras quantity
    Should do it through QFrame objects, defined above
    Don't forget to connect QFrame objects to PyQt slots in widgets!

    :param frames: dictionary of frames in format of {camera:frame}
    :param qframes: dictionary of qframes in the same format as frames
    """
    for camera in frames:
        # converting to RGB
        rgb_image = cv2.cvtColor(frames[camera], cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        # converting to QImage
        qpicture = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        # scaling QImage to resolution
        scaled_qpicture = qpicture.scaled(width, height, Qt.KeepAspectRatio)
        # emitting the picture
        qframes[camera].signal.emit(scaled_qpicture)
