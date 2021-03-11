"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""

import warnings
import numpy as np

warnings.filterwarnings(
    category=FutureWarning, action="ignore"
)  # filter unwanted warnings
import pyrealsense2 as prs2


class RealSenseManager:
    """
    RealSense cameras manager class
    """

    def __init__(self):
        """
        Everything needed to initialize we get from the environment
        List of connected devices is created once and is not updated later
        """
        self._manager_name = "Intel RealSense"
        self._config, self._context = self.realsense_environment()
        self._connected_devices = self.find_connected_devices()
        self._enabled_devices = {}
        # initializing colorizer, that is not really used afterwards
        self._colorizer = prs2.colorizer()
        # Create alignment primitive with color as its target stream:
        self._align = prs2.align(prs2.stream.color)

    @staticmethod
    def realsense_environment() -> tuple:
        """
        Getting config and context to find devices and enable them
        """
        config = prs2.config()
        context = prs2.context()
        return config, context

    def get_name(self):
        return self._manager_name

    def find_connected_devices(self) -> list:
        """
        Create a list with all connected devices from self._context
        """
        connected_devices = []
        for device in self._context.devices:
            connected_devices.append(device.get_info(prs2.camera_info.serial_number))
        return connected_devices

    def get_connected_devices(self) -> list:
        """
        Getter for stored connected devices list
        """
        return self._connected_devices

    def enable_stream(self, resolution: tuple, framerate: int, stream_type: str):
        """
        Enable one stream with given parameters
        :param resolution: resolution of the stream in format of (width, height)
        :param stream_type: type of stream to enable, supported ones: 'color', 'depth', 'infrared'
        :param framerate: maximum stream framerate
        """
        width, heigth = resolution
        if stream_type == "color":
            self._config.enable_stream(
                stream_type=prs2.stream.color,
                width=width,
                height=heigth,
                format=prs2.format.bgr8,
                framerate=framerate,
            )
        elif stream_type == "depth":
            self._config.enable_stream(
                stream_type=prs2.stream.depth,
                width=width,
                height=heigth,
                format=prs2.format.z16,
                framerate=framerate,
            )
        elif stream_type == "infrared":
            self._config.enable_stream(
                stream_type=prs2.stream.infrared,
                stream_index=1,  # we are using only the first camera here
                width=width,
                height=heigth,
                format=prs2.format.y8,
                framerate=framerate,
            )

    def enable_device(self, device_serial: str):
        """
        Enable one device with given serial
        """
        pipeline = prs2.pipeline(self._context)
        self._config.enable_device(device_serial)
        pipeline_profile = pipeline.start(self._config)
        # enabling the emitter
        sensor = pipeline_profile.get_device().first_depth_sensor()
        sensor.set_option(prs2.option.emitter_enabled, 0)
        # storing the enabled device in enabled devices dictionary
        self._enabled_devices[device_serial] = (pipeline, pipeline_profile)

    def enable_all_devices(self):
        """
        Cycles through all connected devices and enables them
        """
        for device_serial in self._connected_devices:
            self.enable_device(device_serial)

    def get_enabled_devices(self) -> dict:
        """
        Getter for enabled devices dictionary
        """
        return self._enabled_devices

    def get_frames(self) -> tuple:
        """
        Collect frames for each enabled stream from each enabled device and output them in the corresponding dictionary
        :return: tuple of three dictionaries: color, depth, infrared
        """
        color_frames = {}
        depth_maps = {}
        infra_frames = {}
        for serial, device in self._enabled_devices.items():
            device_pipeline, device_profile = device
            streams = device_profile.get_streams()
            frameset = device_pipeline.wait_for_frames()

            # Alignment for depth stream
            # currently not used
            # to enable it, uncomment following line
            # frameset = self._align.process(frameset)

            for stream in streams:
                if stream.stream_type() == prs2.stream.color:
                    color_frames[serial] = frameset.get_color_frame().get_data()
                elif stream.stream_type() == prs2.stream.depth:
                    depth_maps[serial] = frameset.get_depth_frame()
                elif stream.stream_type() == prs2.stream.infrared:
                    infra_frames[serial] = frameset.get_infrared_frame(1).get_data()

        return color_frames, depth_maps, infra_frames

    def colorize_depth_frame(self, depth_frame: prs2.depth_frame) -> prs2.depth_frame:
        """
        Colorizes the depth frame
        Not currently used due to very heavy CPU load
        OCV works with this better
        """
        colorized_frame = np.asanyarray(
            self._colorizer.colorize(depth_frame).get_data()
        )
        return colorized_frame

    def stop(self):
        """
        Stops every device and stream
        """
        self._config.disable_all_streams()
        for serial, device in self._enabled_devices.items():
            device_pipeline, device_profile = device
            device_pipeline.stop()
        self._enabled_devices = {}
