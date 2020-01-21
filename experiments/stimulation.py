import time
import cv2
import numpy as np
from experiments.DAQ_output import DigitalModDevice


def show_visual_stim_img(type='background', name='vistim'):
    """
    Shows image in newly created or named window

    :param type: defines image through visual dictionary to be displayed
    :param name: name of window that is created or used by OpenCV to display image
    """
    # Show image when called
    visual = {'background': dict(path=r"./experiments/src/whiteback_1920_1080.png"),
              'Greenbar_whiteback': dict(path=r"./experiments/src/greenbar_whiteback_1920_1080.png"),
              'Bluebar_whiteback': dict(path=r"./experiments/src/bluebar_whiteback_1920_1080.png")}
    # load image unchanged (-1), greyscale (0) or color (1)
    img = cv2.imread(visual[type]['path'], -1)
    converted_image = np.uint8(img)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, converted_image)


def toggle_device():
    """Controls micro peristaltic pump via digital trigger signal."""
    device = DigitalModDevice('Dev1/PFI2')
    device.toggle()


if __name__ == '__main__':
    toggle_device()
    time.sleep(3.5)
    toggle_device()
