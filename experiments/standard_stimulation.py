"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""

import time
import os
import cv2
import numpy as np
from experiments.DAQ_output import DigitalModDevice


def get_stimulation_settings(stimulation_name, parameter_dict):
    import os
    import configparser as cfg
    exp_config = cfg.ConfigParser()
    exp_path = os.path.join(os.path.dirname(__file__), 'experiment_config.ini')
    with open(exp_path) as exp_file:
        exp_config.read_file(exp_file)

    experiment_config = {}
    for parameter in list(parameter_dict.keys()):
        if parameter_dict[parameter] == 'int':
            try:
                experiment_config[parameter] = exp_config[stimulation_name].getint(parameter)
            except:
                experiment_config[parameter] = None
                print('Did not find valid {} entry for {}. Setting to None.'.format(parameter, stimulation_name))
        if parameter_dict[parameter] == 'float':
            try:
                experiment_config[parameter] = exp_config[stimulation_name].getfloat(parameter)
            except:
                experiment_config[parameter] = None
                print('Did not find valid {} entry for {}. Setting to None.'.format(parameter, stimulation_name))

        elif parameter_dict[parameter] == 'tuple':
            try:
                experiment_config[parameter] = tuple(int(entry) for entry in exp_config[stimulation_name].get(parameter).split(','))
            except:
                experiment_config[parameter] = None
                print('Did not find valid {} entry for {}. Setting to None.'.format(parameter, stimulation_name))
        elif parameter_dict[parameter] == 'list':
            try:
                experiment_config[parameter] = list(
                    str(entry) for entry in exp_config[stimulation_name].get(parameter).split(','))
            except:
                experiment_config[parameter] = None
                print('Did not find valid {} entry for {}. Setting to None.'.format(parameter, stimulation_name))
        elif parameter_dict[parameter] == 'boolean':
            try:
                experiment_config[parameter] = exp_config[stimulation_name].getboolean(parameter)
            except:
                experiment_config[parameter] = None
                print('Did not find valid {} entry for {}. Setting to None.'.format(parameter, stimulation_name))

        elif parameter_dict[parameter] == 'str':
            try:
                experiment_config[parameter] = exp_config[stimulation_name].get(parameter)
            except:
                experiment_config[parameter] = None
                print('Did not find valid {} entry for {}. Setting to None.'.format(parameter, stimulation_name))

    return experiment_config


class StandardStimulation:

    def __init__(self):
        self._name = 'StandardStimulation'
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
            from experiments.DAQ_output import DigitalModDevice
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


class RewardDispenser(StandardStimulation):

    def __init__(self):

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
            from experiments.DAQ_output import DigitalModDevice
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


class ScreenStimulation(StandardStimulation):

    def __init__(self):
        self._name = 'ScreenStimulation'
        self._parameter_dict = dict(TYPE='str',
                                    STIM_PATH='str',
                                    BACKGROUND_PATH='str')
        self._settings_dict = get_stimulation_settings(self._name, self._parameter_dict)
        self._running = False
        self._stim_device = self._setup_device(self._settings_dict['TYPE'], self._settings_dict['PORT'])

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



def show_visual_stim_vid(type, name='vistim'):
    """
    Shows video in newly created or named window
    WARNING: LONG FILES WILL HOLD THE PROCESS NOTICEABLY
    :param type: defines video through visual dictionary to be displayed
    :param name: name of window that is created or used by OpenCV to display image
    """
    # Show image when called
    visual = {'Vid1': dict(path=r"./experiments/src/video1.mp4"),
              'Vid2': dict(path=r"./experiments/src/video2.mp4")}
    cap = cv2.VideoCapture(visual[type]['path'])
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    while cap.isOpened():
        ret, frame = cap.read()

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if ret is True:
            cv2.imshow(name, frame)

        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

