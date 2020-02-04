import time
import os
import cv2
import numpy as np
from experiments.DAQ_output import DigitalModDevice


def show_visual_stim_img(img_type='background', name='vistim', img_dict=None):
    """
    Shows image in newly created or named window

    :param img_type: defines image through visual dictionary to be displayed
    :param name: name of window that is created or used by OpenCV to display image
    :param img_dict: optional custom image paths dictionary
    """
    # Show image when called
    img_path = os.path.join(os.path.dirname(__file__), 'src')
    if img_dict is None:
        visual = {'background': r"whiteback_1920_1080.png",
                  'Greenbar_whiteback': r"greenbar_whiteback_1920_1080.png",
                  'Bluebar_whiteback': r"bluebar_whiteback_1920_1080.png"}
    else:
        visual = img_dict
    # load image unchanged (-1), greyscale (0) or color (1)
    img = cv2.imread(os.path.join(img_path, visual[img_type]), -1)
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
