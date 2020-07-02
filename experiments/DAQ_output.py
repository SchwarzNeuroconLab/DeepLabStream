"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""

import nidaqmx
import time


class NotFoundException(nidaqmx._lib.DaqNotFoundError):
    pass


class Device:
    """
    Modulated devices connected to the DAQ Board
    """
    def __init__(self, port):
        """
        :param port: output port on the DAQ board connected to the Device
        """
        self.INPUT_PORT = port
        self._status = False

    def get_port(self):
        return self.INPUT_PORT

    def get_status(self):
        return self._status


class DigitalModDevice(Device):
    """
    Digital modulated devices
    """

    def __init__(self, digital_output_port):
        """
        :param digital_output_port: the digital output port on the DAQ board connected to the Device
        """
        super().__init__(digital_output_port)
        self._t_switch = False

    def trigger(self):
        """
        triggers devices via Digital output of NIDAQ board
        """
        TRIGGER = [True, False]
        try:
            with nidaqmx.Task() as task:
                task.do_channels.add_do_chan(self.INPUT_PORT)
                task.write(TRIGGER, auto_start=True)
        except NotFoundException:
            print("DAQ device not found")

    def turn_on(self):
        try:
            with nidaqmx.Task() as task:
                task.do_channels.add_do_chan(self.INPUT_PORT)
                task.write(True, auto_start=True)
        except NotFoundException:
            print("DAQ device not found")

    def turn_off(self):
        try:
            with nidaqmx.Task() as task:
                task.do_channels.add_do_chan(self.INPUT_PORT)
                task.write(False, auto_start=True)
        except NotFoundException:
            print("DAQ device not found")

    def toggle(self):
        """
        Digital modulation of Device via Digital output of NIDAQ board
        Toggles Device on if off and vice versa
        """
        self._t_switch = not self._t_switch
        try:
            with nidaqmx.Task() as task:
                task.do_channels.add_do_chan(self.INPUT_PORT)
                task.write(self._t_switch, auto_start=True)
        except NotFoundException:
            print("DAQ device not found")

    def timed_on(self, on_time):
        """
        Digital modulation of Device via Digital output of NIDAQ board
        :param on_time: the amount of time that the Device should stay turned ON in seconds
        """
        if on_time > 0:
            self.toggle()
            time.sleep(on_time)
            self.toggle()

    def cycle(self, repeats, on_time, off_time):
        """
        Digital modulation of Device via Digital output of NIDAQ board
        :param repeats: the number of repeats the ON/OFF cycle should run
        :param on_time: the amount of time that the Device should stay turned ON in seconds
        :param off_time: the amount of time that the Device should stay turned OFF between cycles in seconds
        """
        for i in range(repeats):
            self.timed_on(on_time)
            time.sleep(off_time)
