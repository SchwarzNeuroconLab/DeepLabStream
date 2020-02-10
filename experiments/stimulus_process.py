import time
import cv2
import multiprocessing as mp
from experiments.stimulation import show_visual_stim_img, DigitalModDevice


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
    # dmod_device = DigitalModDevice('Dev1/PFI0')
    while True:
        if condition_q.full():
            current_trial = condition_q.get()
        if current_trial is not None:
            show_visual_stim_img(img_type=current_trial, name='inside')
            # dmod_device.toggle()
        else:
            show_visual_stim_img(name='inside')
            # dmod_device.turn_off()

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
