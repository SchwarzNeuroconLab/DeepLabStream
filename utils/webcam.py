from utils.generic import GenericManager, MissingFrameError
import time
import base64

import cv2
import numpy as np
import zmq

from utils.configloader import RESOLUTION, FRAMERATE, PORT


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

        else:
            raise MissingFrameError('No frame was received from the webcam stream. Make sure that you started streaming on the host machine.')

        return color_frames, depth_maps, infra_frames

    def enable_stream(self, resolution, framerate, *args):
        """
        Not used for webcam streaming over network
        """
        pass