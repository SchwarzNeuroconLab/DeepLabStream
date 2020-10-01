from gpiozero import DigitalOutputDevice
from gpiozero.pins.pigpio import PiGPIOFactory

import serial



class DigitalPiDevice:
    """
    Digital modulated devices in combination with Raspberry Pi GPIO
    Setup: https://gpiozero.readthedocs.io/en/stable/remote_gpio.html
    """

    def __init__(self, PIN, BOARD_IP: str = None):

        """
        :param BOARD_IP:  IP adress of board connected to the Device
        """
        if BOARD_IP is not None:
            self._factory = PiGPIOFactory(host = BOARD_IP)
            self._device = DigitalOutputDevice(PIN, pin_factory = self._factory)
        else:
            self._factory = None
            self._device = DigitalOutputDevice(PIN)
        self._running = False

    def turn_on(self):
        self._device.on()
        self._running = True

    def turn_off(self):
        self._device.off()
        self._running = False

    def toggle(self):
        self._device.toggle()
        self._running = self._device.is_active


class DigitalArduinoDevice:
    """
    Digital modulated devices in combination with Arduino boards connected via USB
        setup: https://pythonforundergradengineers.com/python-arduino-LED.html

    """

    def __init__(self, PORT):
        """
        :param PORT:  USB PORT of the arduino board
        """
        self._device = serial.Serial(PORT, baudrate=9600)
        self._running = False

    def turn_on(self):
        self._device.write(b'H')
        self._running = True

    def turn_off(self):
        self._device.write(b'L')
        self._running = False

    def toggle(self):
        if self._running:
            self.turn_off()
        else:
            self.turn_on()
