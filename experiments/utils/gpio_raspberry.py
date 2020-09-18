from gpiozero import DigitalOutputDevice
from gpiozero.pins.pigpio import PiGPIOFactory
BOARD_IP = '192.168.1.2'



class DigitialPiBoardDevice:
    """
    Digital modulated devices in combination with Raspberry Pi GPIO
    """

    def __init__(self, PIN, BOARD_IP: str = None,  remote: bool = False):
        """
        :param BOARD_IP:  IP adress of board connected to the Device
        """
        if remote:
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


