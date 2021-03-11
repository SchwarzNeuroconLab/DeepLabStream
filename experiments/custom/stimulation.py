"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""

import time
import os
import cv2
import numpy as np
from experiments.utils.DAQ_output import DigitalModDevice


def show_visual_stim_img(img_type="background", name="vistim", img_dict=None):
    """
    Shows image in newly created or named window

    :param img_type: defines image through visual dictionary to be displayed
    :param name: name of window that is created or used by OpenCV to display image
    :param img_dict: optional custom image paths dictionary
    """
    # Show image when called
    img_path = os.path.join(os.path.dirname(__file__), "src")
    if img_dict is None:
        visual = {
            "background": r"whiteback_1920_1080.png",
            "Greenbar_whiteback": r"greenbar_whiteback_1920_1080.png",
            "Bluebar_whiteback": r"bluebar_whiteback_1920_1080.png",
        }
    else:
        visual = img_dict
    # load image unchanged (-1), greyscale (0) or color (1)
    img = cv2.imread(os.path.join(img_path, visual[img_type]), -1)
    converted_image = np.uint8(img)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, converted_image)


def toggle_device():
    """Controls micro peristaltic pump via digital trigger signal."""
    device = DigitalModDevice("Dev1/PFI2")
    device.toggle()


def show_visual_stim_img(type="background", name="vistim"):
    """
    Shows image in newly created or named window

    :param type: defines image through visual dictionary to be displayed
    :param name: name of window that is created or used by OpenCV to display image
    """
    # Show image when called
    visual = {
        "background": dict(path=r"./experiments/src/whiteback_1920_1080.png"),
        "Greenbar_whiteback": dict(
            path=r"./experiments/src/greenbar_whiteback_1920_1080.png"
        ),
        "Bluebar_whiteback": dict(
            path=r"./experiments/src/bluebar_whiteback_1920_1080.png"
        ),
        "DLStream_test": dict(path=r"./experiments/src/stuckinaloop.jpg"),
    }
    # load image unchanged (-1), greyscale (0) or color (1)
    img = cv2.imread(visual[type]["path"], -1)
    converted_image = np.uint8(img)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, converted_image)


def show_visual_stim_vid(type, name="vistim"):
    """
    Shows video in newly created or named window
    WARNING: LONG FILES WILL HOLD THE PROCESS NOTICEABLY
    :param type: defines video through visual dictionary to be displayed
    :param name: name of window that is created or used by OpenCV to display image
    """
    # Show image when called
    visual = {
        "Vid1": dict(path=r"./experiments/src/video1.mp4"),
        "Vid2": dict(path=r"./experiments/src/video2.mp4"),
    }
    cap = cv2.VideoCapture(visual[type]["path"])
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    while cap.isOpened():
        ret, frame = cap.read()

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if ret is True:
            cv2.imshow(name, frame)

        else:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()


def toggle_device():
    """Controls micro peristaltic pump via digital trigger signal."""
    device = DigitalModDevice("Dev1/PFI2")
    device.toggle()


"""The following is the original stimulation we used for our experiments! If you are interested in using this, 
you will need to adapt the stimulation to your system! Otherwise I recommend looking at them for ideas how to incorporate
your own experiment into DLStream!"""


def laser_toggle():
    """Toggle laser on or off
    Laser needs to be connected to DAQ_PORT and switched to "Digital modulation"
    If you use additional safety measurements to control the laser, make sure to undo them before starting the protocol!"""

    laser = DigitalModDevice(LSR_DAQ_PORT)
    laser.toggle()
    print("Laser was toggled")


def laser_switch(switch: bool = False):
    """Toggle laser on or off
    Laser needs to be connected to DAQ_PORT and switched to "Digital modulation"
    If you use additional safety measurements to control the laser, make sure to undo them before starting the protocol!"""

    laser = DigitalModDevice(LSR_DAQ_PORT)
    if switch:
        laser.turn_on()
        print("Laser is switched on")

    else:
        laser.turn_off()
        print("Laser is switched off")


def deliver_tone_shock():
    """
    Activates tone signal via digital trigger. Cycle is optional
    :param rep: Number of repetitions for signal [int]
    :param duration: Duration in seconds of signal for each rep [float]
    :param inter_time: Time in seconds between reps [float]
    """

    tone_gen = DigitalModDevice("Dev1/PFI5")
    tone_gen.toggle()


def deliver_airpuff(rep: int = 1, duration: float = 0.1, inter_time: float = 0.1):
    """Controls pressure micro-injector via digital trigger signal.
    All other parameters need to be manually changed on the Device, this only triggers it!"""

    pump = DigitalModDevice(AP_DAQ_PORT)
    if rep > 1:
        pump.cycle(rep, duration, inter_time)
    else:
        pump.trigger()


def deliver_liqreward():
    """Controls micro peristaltic pump via digital trigger signal."""
    pump_delivery = DigitalModDevice("Dev1/PFI2")
    pump_delivery.toggle()


def withdraw_liqreward():
    """activates micro peristaltic pump"""
    pump_withdraw = DigitalModDevice("Dev1/PFI6")
    pump_withdraw.timed_on(3.5)


if __name__ == "__main__":
    toggle_device()
    time.sleep(3.5)
    toggle_device()
