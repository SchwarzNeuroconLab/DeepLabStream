"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""
import time
import base64

import cv2
import numpy as np
import zmq

from utils.configloader import CAMERA_SOURCE, VIDEO_SOURCE, RESOLUTION, FRAMERATE, PORT, REPEAT_VIDEO


class GenericManager:
    """
    Camera manager class for generic (not specified) cameras
    """
    def __init__(self):
        """
        Generic camera manager from video source
        Uses pure opencv
        """
        source = CAMERA_SOURCE if CAMERA_SOURCE is not None else 0
        self._manager_name = "generic"
        self._enabled_devices = {}
        self._camera = cv2.VideoCapture(int(source))
        self._camera_name = "Camera {}".format(source)

    def get_connected_devices(self) -> list:
        """
        Getter for stored connected devices list
        """
        return [self._camera_name]

    def get_enabled_devices(self) -> dict:
        """
        Getter for enabled devices dictionary
        """
        return self._enabled_devices

    def enable_stream(self, resolution, framerate, *args):
        """
        Enable one stream with given parameters
        (hopefully)
        """
        width, height = resolution
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._camera.set(cv2.CAP_PROP_FPS, framerate)

    def enable_device(self, *args):
        """
        Redirects to enable_all_devices()
        """
        self.enable_all_devices()

    def enable_all_devices(self):
        """
        We don't need to enable anything with opencv
        """
        self._enabled_devices = {self._camera_name: self._camera}

    def get_frames(self) -> tuple:
        """
        Collect frames for camera and outputs it in 'color' dictionary
        ***depth and infrared are not used here***
        :return: tuple of three dictionaries: color, depth, infrared
        """
        color_frames = {}
        depth_maps = {}
        infra_frames = {}
        ret, image = self._camera.read()
        if ret:
            color_frames[self._camera_name] = image

        return color_frames, depth_maps, infra_frames

    def stop(self):
        """
        Stops camera
        """
        self._camera.release()
        self._enabled_devices = {}

    def get_name(self) -> str:
        return self._manager_name


class VideoManager(GenericManager):

    """
    Camera manager class for analyzing videos
    """
    def __init__(self):
        """
        Generic video manager from video files
        Uses pure opencv
        """
        super().__init__()
        self._camera = cv2.VideoCapture(VIDEO_SOURCE)
        self._camera_name = "Video"
        self.initial_wait = False
        self.last_frame_time = time.time()

    def get_frames(self) -> tuple:
        """
        Collect frames for camera and outputs it in 'color' dictionary
        ***depth and infrared are not used here***
        :return: tuple of three dictionaries: color, depth, infrared
        """

        color_frames = {}
        depth_maps = {}
        infra_frames = {}
        ret, image = self._camera.read()
        self.last_frame_time = time.time()
        if ret:
            if not self.initial_wait:
                cv2.waitKey(1000)
                self.initial_wait = True
            image = cv2.resize(image, RESOLUTION)
            color_frames[self._camera_name] = image
            running_time = time.time() - self.last_frame_time
            if running_time <= 1 / FRAMERATE:
                sleepy_time = int(np.ceil(1000/FRAMERATE - running_time / 1000))
                cv2.waitKey(sleepy_time)
        elif REPEAT_VIDEO:
            # cycle the video for testing purposes
            self._camera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return self.get_frames()

        return color_frames, depth_maps, infra_frames


class WebCamManager(GenericManager):

    def __init__(self):
        """
        Binds the computer to a ip address and starts listening for incoming streams.
        Adapted from StreamViewer.py https://github.com/CT83/SmoothStream
        """
        super().__init__()
        self._context = zmq.Context()
        self._footage_socket = self._context.socket(zmq.SUB)
        self._footage_socket.bind('tcp://*:' + PORT)
        self._footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))

        self._camera = None
        self._camera_name = "webcam"
        self.initial_wait = False
        self.last_frame_time = time.time()

    @ staticmethod
    def string_to_image(string):
        """
        Taken from https://github.com/CT83/SmoothStream
        """

        img = base64.b64decode(string)
        npimg = np.fromstring(img, dtype=np.uint8)
        return cv2.imdecode(npimg, 1)

    def get_frames(self) -> tuple:
        """
        Collect frames for camera and outputs it in 'color' dictionary
        ***depth and infrared are not used here***
        :return: tuple of three dictionaries: color, depth, infrared
        """

        color_frames = {}
        depth_maps = {}
        infra_frames = {}

        if self._footage_socket:
            ret = True
        else:
            ret = False
        self.last_frame_time = time.time()
        if ret:
            # if not self.initial_wait:
            #     cv2.waitKey(1000)
            #     self.initial_wait = True
            # receives frame from stream
            image = self._footage_socket.recv_string()
            # converts image from str to image format that cv can handle
            image = self.string_to_image(image)
            image = cv2.resize(image, RESOLUTION)
            color_frames[self._camera_name] = image
            running_time = time.time() - self.last_frame_time
            if running_time <= 1 / FRAMERATE:
                sleepy_time = int(np.ceil(1000/FRAMERATE - running_time / 1000))
                cv2.waitKey(sleepy_time)
        return color_frames, depth_maps, infra_frames

    def enable_stream(self, resolution, framerate, *args):
        """
        Not used for webcam streaming over network
        """
        pass
