"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""

import time
import cv2
import numpy as np

from experiments.utils.exp_setup import get_stimulation_settings


class BaseStimulation:

    def __init__(self):
        self._name = 'BaseStimulation'
        self._parameter_dict = dict(TYPE='str',
                                    PORT='str',
                                    STIM_TIME='float')
        self._settings_dict = get_stimulation_settings(self._name, self._parameter_dict)
        self._running = False
        self._stim_device = self._setup_device(self._settings_dict['TYPE'], self._settings_dict['PORT'])

    @staticmethod
    def _setup_device(type, port):
        device = None
        if type == 'NI':
            from experiments.utils.DAQ_output import DigitalModDevice
            device = DigitalModDevice(port)

        return device

    def stimulate(self):
        """Run stimulation and stop after being done"""
        if self._settings_dict['STIM_TIME'] is not None:
            print('Stimulation: {} for {}.'.format(self._name, self._settings_dict['STIM_TIME']))
            self._stim_device.turn_on()
            self._running = True
            time.sleep(self._settings_dict['STIM_TIME'])
            self._stim_device.turn_off()
            self._running = False
        else:
            print('Stimulation: {} does not support stimulate().'.format(self._name))

    def remove(self):
        """remove stimulation (e.g. reward) and stop after being done"""
        print('Stimulation: {} does not support remove().'.format(self._name))

    def start(self):

        if not self._running:
            print('Stimulation: {} ON.'.format(self._name))
            self._stim_device.turn_on()
            self._running = True
        else:
            print('Stimulation was already ON.')

    def stop(self):
        if self._running:
            print('Stimulation: {} OFF.'.format(self._name))
            self._stim_device.turn_off()
            self._running = False
        else:
            print('Stimulation was already OFF.')


class RewardDispenser(BaseStimulation):

    def __init__(self):
        super().__init__()
        self._name = 'RewardDispenser'
        self._parameter_dict = dict(TYPE = 'str',
                                    STIM_PORT= 'str',
                                    REMOVAL_PORT = 'str',
                                    STIM_TIME = 'float',
                                    REMOVAL_TIME = 'float')
        self._settings_dict = get_stimulation_settings(self._name, self._parameter_dict)
        self._running = False
        self._stim_device = self._setup_device(self._settings_dict['TYPE'], self._settings_dict['STIM_PORT'])
        self._removal_device = self._setup_device(self._settings_dict['TYPE'], self._settings_dict['REMOVAL_PORT'])


    @staticmethod
    def _setup_device(type, port):
        device = None
        if type == 'NI':
            from experiments.utils.DAQ_output import DigitalModDevice
            device = DigitalModDevice(port)

        return device

    def stimulate(self):
        """Run stimulation and stop after being done"""
        print('Stimulation: {} for {}.'.format(self._name, self._settings_dict['STIM_TIME']))
        self._stim_device.turn_on()
        self._running = True
        time.sleep(self._settings_dict['STIM_TIME'])
        self._stim_device.turn_off()
        self._running = False

    def remove(self):
        """remove stimulation (e.g. reward) and stop after being done"""
        print('Stimulation: {} for {}.'.format(self._name, self._settings_dict['REMOVAL_TIME']))
        self._removal_device.turn_on()
        self._running = True
        time.sleep(self._settings_dict['REMOVAL_TIME'])
        self._removal_device.turn_off()
        self._running = False

    def start(self):
        print('Stimulation: {} does not support start(). Did you mean stimulate()?'.format(self._name))

    def stop(self):
        print('Stimulation: {} does not support stop(). Did you mean remove()?'.format(self._name))


class ScreenStimulation(BaseStimulation):

    def __init__(self):
        super().__init__()
        self._name = 'ScreenStimulation'
        self._parameter_dict = dict(TYPE='str',
                                    STIM_PATH='str',
                                    BACKGROUND_PATH='str')
        self._settings_dict = get_stimulation_settings(self._name, self._parameter_dict)
        self._running = False
        self._stim_device = None

        self._background = self._setup_stimulus(self._settings_dict['BACKGROUND_PATH'], type = 'image') \
            if self._settings_dict['BACKGROUND_PATH'] is not None else None
        self._stimulus = self._setup_stimulus(self._settings_dict['STIM_PATH'], type = self._settings_dict['TYPE'])
        self._window = None

    @staticmethod
    def _setup_stimulus(path, type = 'image'):
        if type == 'image':
            img = cv2.imread(path, -1)
            stimulus = np.uint8(img)
        elif type == 'video':
            stimulus = cv2.VideoCapture(path)

        return stimulus

    def _setup_window(self):
        cv2.namedWindow(self._name, cv2.WINDOW_NORMAL)

    def stimulate(self):
        """Run stimulation and stop after being done"""
        if self._window is None:
            self._setup_window()
        if self._settings_dict['TYPE'] == 'image':
            cv2.imshow(self._name, self._stimulus)

        elif self._settings_dict['TYPE'] == 'video':
            while self._stimulus.isOpened():
                self._running = True
                ret, frame = self._stimulus.read()
                if ret is True:
                    cv2.imshow(self._name, frame)
                else:
                    break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            self._running = False
            self._stimulus.release()

    def remove(self):
        """remove stimulation (e.g. reward) and stop after being done"""
        if self._window is None:
            self._setup_window()

        cv2.imshow(self._name, self._background)

    def start(self):
        print('Stimulation: {} does not support start(). Did you mean stimulate()?'.format(self._name))

    def stop(self):
        print('Stimulation: {} does not support stop(). Did you mean remove()?'.format(self._name))

