from gpiozero import DigitalOutputDevice
from gpiozero.pins.pigpio import PiGPIOFactory

import serial
BOARD_IP = '192.168.1.2'



class DigitalPiBoardDevice:
    """
    Digital modulated devices in combination with Raspberry Pi GPIO
    """

    def __init__(self, PIN, BOARD_IP: str = None):

        """
        :param BOARD_IP:  IP adress of board connected to the Device
        """
        if BOARD_IP is not None:
            self._factory = PiGPIOFactory(host = BOARD_IP)
            self._device = DigitalOutputDevice(PIN= PIN, pin_factory = self._factory)
        else:
            self._factory = None
            self._device = DigitalOutputDevice(PIN = PIN)
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
    """

    def __init__(self, PORT):
        """
        :param PORT:  USB PORT of the arduino board
        """
        self._device = serial.Serial(PORT, baudrate=19200)
        self._running = False

    def turn_on(self):
        self._device.write(b'1')
        self._running = True

    def turn_off(self):
        self._device.write(b'0')
        self._running = False

    def toggle(self):
        if self._running:
            self.turn_off()
        else:
            self.turn_on()
