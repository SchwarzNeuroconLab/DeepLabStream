"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""

import time
import cv2
import multiprocessing as mp
from experiments.custom.stimulation import show_visual_stim_img, deliver_liqreward, deliver_tone_shock, withdraw_liqreward, DigitalModDevice
import random


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


def example_protocol_run(condition_q: mp.Queue):
    current_trial = None
    #dmod_device = DigitalModDevice('Dev1/PFI0')
    while True:
        if condition_q.full():
            current_trial = condition_q.get()
        if current_trial is not None:
            show_visual_stim_img(type=current_trial, name='inside')
            #dmod_device.toggle()
        else:
            show_visual_stim_img(name='inside')
            #dmod_device.turn_off()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


class ProtocolProcess:
    """
    Class to help work with protocol function in multiprocessing
    """
    def __init__(self):
        """
        Setting up the three queues and the process itself
        """
        self._trial_queue = mp.Queue(1)
        self._success_queue = mp.Queue(1)
        self._condition_queue = mp.Queue(1)
        self._protocol_process = None
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
        self._trial_queue.close()
        self._success_queue.close()
        self._condition_queue.close()
        self._protocol_process.terminate()

    def get_status(self):
        """
        Getting current status of the running protocol
        """
        return self._running, self._current_trial

    def set_trial(self, trial: str):
        """
        Passing the trial name to the process
        """
        if self._trial_queue.empty() and self._success_queue.empty():
            self._trial_queue.put(trial)
            self._running = True
            self._current_trial = trial

    def pass_condition(self, condition: bool):
        """
        Passing the condition to the process
        """
        if self._condition_queue.empty():
            self._condition_queue.put(condition)

    def get_result(self) -> bool:
        """
        Getting result from the process
        """
        if self._success_queue.full():
            self._running = False
            return self._success_queue.get()


class ExampleProtocolProcess(ProtocolProcess):
    """
    Class to help work with protocol function in multiprocessing with simple stimulation
    """
    def __init__(self):
        """
        Setting up the three queues and the process itself
        """
        super().__init__()
        self._protocol_process = mp.Process(target=example_protocol_run, args=(self._trial_queue,))



"""The following is the original protocols we used for our experiments! If you are interested in using this, 
you will need to adapt the stimulation to your system! Otherwise I recommend looking at them for ideas how to incorporate
your own experiment into DLStream!"""


def start_unconditional(protocol):
    print('Running some stuff, water or sound for {} protocol'.format(protocol))


def classic_protocol_run_old(trial_q: mp.Queue, condition_q: mp.Queue, success_q: mp.Queue, trials: dict):
    """
    The function to use in ProtocolProcess class
    Designed to be run continuously alongside the main loop
    Three parameters are three mp.Queue classes, each passes corresponding values
    :param trial_q: the protocol name (inwards)
    :param condition_q: the condition (inwards)
    :param success_q: the result of each protocol (outwards)
    :param trials: dict of possible trials
    """
    # setting up different trials
    current_trial = None
    # starting the main loop without any protocol running
    while True:
        # if no protocol is selected, running default picture (background)
        if trial_q.empty() and current_trial is None:
            # print('No protocol running')
            show_visual_stim_img(name='inside')
        # if some protocol is passed, set up protocol timers and variables
        elif trial_q.full():
            current_trial = trial_q.get()
            finished_trial = False
            # starting timers
            stimulus_timer = trials[current_trial]['stimulus_timer']
            success_timer = trials[current_trial]['success_timer']
            print('Starting protocol {}'.format(current_trial))
            stimulus_timer.start()
            success_timer.start()
            condition_list = []
        # this branch is for already running protocol
        elif current_trial is not None:
            # checking for stimulus timer and outputting correct image
            if stimulus_timer.check_timer():
                # if stimulus timer is running, show stimulus
                show_visual_stim_img(current_trial, name='inside')
            else:
                # if the timer runs out, finish protocol and reset timer
                trials[current_trial]['stimulus_timer'].reset()
                current_trial = None

            # checking if any condition was passed
            if condition_q.full():
                stimulus_condition = condition_q.get()
                # checking if timer for condition is running and condition=True
                if success_timer.check_timer():
                    # print('That was a success!')
                    condition_list.append(stimulus_condition)
                # elif success_timer.check_timer() and not stimulus_condition:
                #     # print('That was not a success')
                #     condition_list.append(False)

            # checking if the timer for condition has run out
            if not success_timer.check_timer() and not finished_trial:
                if CTRL:
                    #start a random time interval
                    #TODO: working ctrl timer that does not set new time each frame...
                    ctrl_time = random.randint(0, INTERTRIAL_TIME+1)
                    ctrl_timer = Timer(ctrl_time)
                    ctrl_timer.start()
                    print('Waiting for extra' + str(ctrl_time) + ' sec')
                    if not ctrl_timer.check_timer():
                        # in ctrl just randomly decide between the two
                        print('Random choice between both stimuli')
                        if random.random() >= 0.5:
                            # very fast random choice between TRUE and FALSE
                            deliver_liqreward()
                            print('Delivered Reward')

                        else:
                            deliver_tone_shock()
                            print('Delivered Aversive')

                        ctrl_timer.reset()
                        finished_trial = True
                        # outputting the result, whatever it is
                        success = trials[current_trial]['result_func'](condition_list)
                        success_q.put(success)
                        trials[current_trial]['success_timer'].reset()

                else:
                    if current_trial == 'Bluebar_whiteback':
                        deliver_tone_shock()
                        print('Delivered Aversive')
                    elif current_trial == 'Greenbar_whiteback':
                        if trials[current_trial]['random_reward']:
                            if random.random() >= 0.5:
                                #very fast random choice between TRUE and FALSE
                                deliver_liqreward()
                                print('Delivered Reward')
                            else:
                                print('No Reward')
                        else:
                            deliver_liqreward()
                    # resetting the timer
                    print('Timer for condition run out')
                    finished_trial = True
                    # outputting the result, whatever it is
                    success = trials[current_trial]['result_func'](condition_list)
                    success_q.put(success)
                    trials[current_trial]['success_timer'].reset()

        # don't delete that
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def classic_protocol_run(trial_q: mp.Queue, condition_q: mp.Queue, success_q: mp.Queue, trials: dict):
    """
    The function to use in ProtocolProcess class
    Designed to be run continuously alongside the main loop
    Three parameters are three mp.Queue classes, each passes corresponding values
    :param trial_q: the protocol name (inwards)
    :param condition_q: the condition (inwards)
    :param success_q: the result of each protocol (outwards)
    :param trials: dict of possible trials
    """
    # setting up different trials
    current_trial = None
    # starting the main loop without any protocol running
    while True:
        # if no protocol is selected, running default picture (background)
        if trial_q.empty() and current_trial is None:
            # print('No protocol running')
            show_visual_stim_img(name='inside')
        # if some protocol is passed, set up protocol timers and variables
        elif trial_q.full():
            current_trial = trial_q.get()
            finished_trial = False
            delivery = False
            reward_del = False
            # starting timers
            stimulus_timer = trials[current_trial]['stimulus_timer']
            collection_timer = trials[current_trial]['collection_timer']
            success_timer = trials[current_trial]['success_timer']
            delivery_timer = Timer(3.5)
            shock_timer = Timer(3.5)
            # withdraw_timer = Timer(3.5)
            print('Starting protocol {}'.format(current_trial))
            stimulus_timer.start()
            success_timer.start()
            condition_list = []
            collection_list = []
        # this branch is for already running protocol
        elif current_trial is not None:
            # checking for stimulus timer and outputting correct image
            if stimulus_timer.check_timer():
                # if stimulus timer is running, show stimulus
                show_visual_stim_img(current_trial, name='inside')
            else:
                # if the timer runs out, finish protocol and reset timer
                trials[current_trial]['stimulus_timer'].reset()
                show_visual_stim_img(name='inside')
            # checking if any condition was passed
            if condition_q.full():
                stimulus_condition = condition_q.get()
                # checking if timer for condition is running and condition=True
                if success_timer.check_timer():
                    condition_list.append(stimulus_condition)
                elif not success_timer.check_timer() and collection_timer.check_timer():
                    collection_list.append(stimulus_condition)

            # checking if the timer for condition has run out
            if not success_timer.check_timer() and not finished_trial:

                if not delivery:
                    if current_trial is not None:
                        print('Timer for condition ran out')
                        print_check = True
                        #check wether animal collected within success timer
                        success = trials[current_trial]['result_func'](condition_list)
                        trials[current_trial]['success_timer'].reset()

                        print('Stimulation.')

                        if current_trial == 'Bluebar_whiteback':
                            deliver_tone_shock()
                            print('Aversive')
                            shock_timer.start()
                        elif current_trial == 'Greenbar_whiteback':
                            deliver_liqreward()
                            delivery_timer.start()
                            reward_del = True
                            print('Reward')
                        delivery = True
                        collection_timer.start()
                elif delivery:
                    # resetting the timer
                    if not collection_timer.check_timer():
                        finished_trial = True
                        # check whether animal collected at all
                        collect = any(collection_list)
                        if not collect and reward_del:
                            # if the animal didnt go to collect reward, withdraw reward again.
                            withdraw_liqreward()
                            # withdraw_timer.start()
                        trials[current_trial]['collection_timer'].reset()
                        current_trial = None
                        # put success in queue and finish trial
                        success_q.put(success)

            if not delivery_timer.check_timer() and delivery_timer.get_start_time() is not None:
                deliver_liqreward()
                delivery_timer.reset()
            if not shock_timer.check_timer() and shock_timer.get_start_time()is not None:
                deliver_tone_shock()
                shock_timer.reset()


            # if not withdraw_timer.check_timer() and withdraw_timer.get_start_time() is not None:
            #     withdraw_liqreward(False)
            #     withdraw_timer.reset()
            #     delivery = False

        # don't delete that
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def simple_protocol_run(trial_q: mp.Queue, success_q: mp.Queue, trials: dict):
    """
    The function to use in ProtocolProcess class
    Designed to be run continuously alongside the main loop
    Three parameters are three mp.Queue classes, each passes corresponding values
    :param trial_q: the protocol name (inwards)
    :param success_q: the result of each protocol (outwards)
    :param trials: dict of possible trials
    """
    current_trial = None
    # starting the main loop without any protocol running
    while True:
        if trial_q.empty() and current_trial is None:
            pass
        elif trial_q.full():
            current_trial = trial_q.get()
            print(current_trial)
            # this branch is for already running protocol
        elif current_trial is not None:
            print('Stimulating...')
            current_trial = None
            success_q.put(True)
            deliver_liqreward()
            time.sleep(3.5)
            deliver_liqreward()


class ClassicProtocolProcess:
    """
    Class to help work with protocol function in multiprocessing
    """
    def __init__(self, trials):
        """
        Setting up the three queues and the process itself
        """
        self._trial_queue = mp.Queue(1)
        self._success_queue = mp.Queue(1)
        self._condition_queue = mp.Queue(1)
        self._protocol_process = mp.Process(target=classic_protocol_run, args=(self._trial_queue,
                                                                               self._condition_queue,
                                                                               self._success_queue,
                                                                               trials))
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
        self._trial_queue.close()
        self._success_queue.close()
        self._condition_queue.close()
        self._protocol_process.terminate()

    def get_status(self):
        """
        Getting current status of the running protocol
        """
        return self._running, self._current_trial

    def set_trial(self, trial: str):
        """
        Passing the trial name to the process
        """
        if self._trial_queue.empty() and self._success_queue.empty():
            self._trial_queue.put(trial)
            self._running = True
            self._current_trial = trial

    def pass_condition(self, condition: bool):
        """
        Passing the condition to the process
        """
        if self._condition_queue.empty():
            self._condition_queue.put(condition)

    def get_result(self) -> bool:
        """
        Getting result from the process
        """
        if self._success_queue.full():
            self._running = False
            return self._success_queue.get()


class SimpleProtocolProcess(ClassicProtocolProcess):
    """
    Class to help work with protocol function in multiprocessing with simple stimulation
    """
    def __init__(self, trials):
        """
        Setting up the three queues and the process itself
        """
        super().__init__(trials)
        self._protocol_process = mp.Process(target=simple_protocol_run, args=(self._trial_queue,
                                                                              self._success_queue,
                                                                              trials))
