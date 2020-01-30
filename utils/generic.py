import cv2
from utils.configloader import VIDEO_SOURCE


class GenericManager:
    """
    Camera manager class for generic (not specified) cameras
    """
    def __init__(self):
        """
        Generic camera manager from video source
        Uses pure opencv
        """
        source = VIDEO_SOURCE if VIDEO_SOURCE is not None else 0
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
