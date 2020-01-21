from pypylon import pylon
import cv2


class PylonManager:
    """
    Basler cameras manager class
    """
    def __init__(self):
        self._manager_name = "Basler Pylon"
        self._factory = pylon.TlFactory.GetInstance()
        self._enabled_devices = {}
        self._resolution = None
        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    @property
    def _connected_devices(self) -> dict:
        """
        Create a dict with all connected devices from self._factory
        """
        return {device.GetSerialNumber(): device for device in self._factory.EnumerateDevices()}

    def get_connected_devices(self) -> list:
        """
        Getter for stored connected devices serials list
        """
        return list(self._connected_devices.keys())

    def enable_stream(self, resolution, *args):
        """
        Enable stream with given parameters
        Pretty meaningless for pylon manager, just sets the desired resolution
        """
        self._resolution = resolution

    def enable_device(self, device_serial: str, *args):
        """
        Camera starter
        """
        camera = pylon.InstantCamera(self._factory.CreateDevice(self._connected_devices[device_serial]))
        self._enabled_devices[camera.DeviceInfo.GetSerialNumber()] = camera
        # grabbing continuously (video) with minimal delay
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def enable_all_devices(self):
        """
        Starts the cameras with minimal delay
        """
        for device in self._connected_devices:
            self.enable_device(device)

    def get_enabled_devices(self) -> dict:
        """
        Getter for enabled devices dictionary
        """
        return self._enabled_devices

    def get_frames(self) -> tuple:
        """
        Collect frames for cameras and outputs it in 'color' dictionary
        ***depth and infrared are not used in pylon***
        :return: tuple of three dictionaries: color, depth, infrared
        """
        color_frames = {}
        depth_maps = {}
        infra_frames = {}
        for camera_name, camera in self._enabled_devices.items():
            grabbed_frame = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            # converting to opencv bgr format
            image = self._converter.Convert(grabbed_frame)
            img = image.GetArray()
            color_frames[camera_name] = cv2.resize(img, self._resolution)
            grabbed_frame.Release()
        return color_frames, depth_maps, infra_frames

    def stop(self):
        """
        Stops cameras
        """
        for camera_name, camera in self._enabled_devices.items():
            camera.StopGrabbing()

    def get_name(self) -> str:
        return self._manager_name
