"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""

import time
import multiprocessing as mp
from experiments.utils.exp_setup import get_process_settings,setup_stimulation


class Timer:
    """
    Very simple timer
    """
    def __init__(self, seconds):
        """
        Setting the time the timer needs to run
        :param seconds: time in seconds
        """
        self._seconds = seconds
        self._start_time = None

    def start(self):
        """
        Starting the timer
        If already started does nothing
        """
        if not self._start_time:
            self._start_time = time.time()

    def check_timer(self):
        """
        Check if the time has run out or not
        Returns False if timer is not started
        Returns True if timer has run less then _seconds (still runs)
        """
        if self._start_time:
            current_time = time.time()
            return current_time - self._start_time <= self._seconds
        else:
            return False

    def return_time(self):

        if self._start_time:
            current_time = time.time()
            return current_time - self._start_time
        else:
            pass

    def reset(self):
        """
        Resets the timer
        """
        self._start_time = None

    def get_start_time(self):
        """
        Returns the start time of the timer
        """
        return self._start_time


def base_conditional_switch_protocol_run(condition_q: mp.Queue, stimulus_name):
    condition = False
    stimulation = setup_stimulation(stimulus_name)
    while True:
        if condition_q.full():
            condition = condition_q.get()
        if condition:
            stimulation.start()
        else:
            stimulation.stop()


def base_conditional_supply_protocol_run(condition_q: mp.Queue, stimulus_name):
    condition = False
    stimulation = setup_stimulation(stimulus_name)
    while True:
        if condition_q.full():
            condition = condition_q.get()
        if condition:
            stimulation.stimulate()
        else:
            stimulation.remove()


def base_trial_protocol_run(trial_q: mp.Queue, success_q: mp.Queue, trials: dict):
    """
    The function to use in ProtocolProcess class
    Designed to be run continuously alongside the main loop
    Three parameters are three mp.Queue classes, each passes corresponding values
    :param trial_q: the protocol name (inwards)
    :param success_q: the result of each protocol (outwards)
    :param trials: dict of possible trials
    :param stimulus_name: exact name of stimulus function in base_stimulus.py
    """
    current_trial = None
    # TODO: make this adaptive and working
    trial_dict = {}
    stimulus_name = 'BaseStimulation'
    stimulation = setup_stimulation(stimulus_name)
    # starting the main loop without any protocol running
    while True:
        if trial_q.empty() and current_trial is None:
            pass
        elif trial_q.full():
            current_trial = trial_q.get()
            print(current_trial)
            # this branch is for already running protocol
        elif current_trial is not None:
            success_q.put(True)
            stimulation.stimulate()
            current_trial = None


class BaseProtocolProcess:
    """
    Class to help work with protocol function in multiprocessing
    """
    def __init__(self, trials: dict = None):
        """
        Setting up the three queues and the process itself
        """
        self._name = 'BaseProtocolProcess'
        self._parameter_dict = dict(TYPE='str',
                                    STIMULATION ='str')
        self._settings_dict = get_process_settings(self._name, self._parameter_dict)

        if self._settings_dict['TYPE'] == 'trial' and trials is not None:
            self._trial_queue = mp.Queue(1)
            self._success_queue = mp.Queue(1)
            self._protocol_process = mp.Process(target=base_trial_protocol_run,
                                                args=(self._trial_queue, self._success_queue, trials))
        elif self._settings_dict['TYPE'] == 'switch':
            self._condition_queue = mp.Queue(1)
            self._protocol_process = mp.Process(target=base_conditional_switch_protocol_run,
                                                args=(self._condition_queue, self._settings_dict['STIMULATION']))

        elif self._settings_dict['TYPE'] == 'supply':
            self._condition_queue = mp.Queue(1)
            self._protocol_process = mp.Process(target=base_conditional_supply_protocol_run,
                                                args=(self._condition_queue, self._settings_dict['STIMULATION']))

        self._running = False
        self._current_trial = None

    def start(self):
        """
        Starting the process
        """
        self._protocol_process.start()

    def end(self):
        """
        Ending the process
        """
        if self._settings_dict['TYPE'] == 'condition':
            self._condition_queue.close()
        elif self._settings_dict['TYPE'] == 'trial':
            self._trial_queue.close()
            self._success_queue.close()

        self._protocol_process.terminate()

    def get_status(self):
        """
        Getting current status of the running protocol
        """
        return self._running, self._current_trial

    def put(self, input_p):
        """
        Passing the trial name to the process
        """
        if self._settings_dict['TYPE'] == 'trial':
            if self._trial_queue.empty() and self._success_queue.empty():
                self._trial_queue.put(input_p)
                self._running = True
                self._current_trial = input_p

        elif self._settings_dict['TYPE'] == 'condition':
            if self._condition_queue.empty():
                self._condition_queue.put(input_p)

    def get_result(self) -> bool:
        """
        Getting result from the process
        """
        if self._settings_dict['TYPE'] == 'trial':
            if self._success_queue.full():
                self._running = False
                return self._success_queue.get()
        else:
            return None
