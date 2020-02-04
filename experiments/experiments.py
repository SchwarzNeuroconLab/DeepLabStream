from experiments.stimulus_process import ExampleProtocolProcess, Timer
from experiments.triggers import RegionTrigger
from utils.plotter import plot_triggers_response
from experiments.stimulation import show_visual_stim_img


class ExampleExperiment:
    """
    Simple class to contain all of the experiment properties
    Uses multiprocess to ensure the best possible performance and
        to showcase that it is possible to work with any type of equipment, even timer-dependant
    """
    def __init__(self):
        self.experiment_finished = False
        self._process = ExampleProtocolProcess()
        self._green_point = (550, 163)
        self._blue_point = (372, 163)
        self._radius = 20
        self._event = None
        self._current_trial = None
        self._trial_count = {trial: 0 for trial in self._trials}
        self._trial_timers = {trial: Timer(10) for trial in self._trials}
        self._exp_timer = Timer(600)

    def check_skeleton(self, frame, skeleton):
        """
        Checking each passed animal skeleton for a pre-defined set of conditions
        Outputting the visual representation, if exist
        Advancing trials according to inherent logic of an experiment
        :param frame: frame, on which animal skeleton was found
        :param skeleton: skeleton, consisting of multiple joints of an animal
        """
        self.check_exp_timer()  # checking if experiment is still on
        for trial in self._trial_count:
            # checking if any trial hit a predefined cap
            if self._trial_count[trial] >= 10:
                self.stop_experiment()

        if not self.experiment_finished:
            for trial in self._trials:
                # check for all trials if condition is met
                result, response = self._trials[trial]['trigger'](skeleton=skeleton)
                plot_triggers_response(frame, response)
                if result:
                    if self._current_trial is None:
                        if not self._trial_timers[trial].check_timer():
                            self._current_trial = trial
                            self._trial_timers[trial].reset()
                            self._trial_count[trial] += 1
                            print(trial, self._trial_count[trial])
                else:
                    if self._current_trial == trial:
                        self._current_trial = None
                        self._trial_timers[trial].start()
                return result, response

            self._process.set_trial(self._current_trial)

    @property
    def _trials(self):
        """
        Defining the trials
        """
        green_roi = RegionTrigger('circle', self._green_point, self._radius * 2 + 7.5, 'neck')
        blue_roi = RegionTrigger('circle', self._blue_point, self._radius * 2 + 7.5, 'neck')
        trials = {'Greenbar_whiteback': dict(trigger=green_roi.check_skeleton,
                                             count=0),
                  'Bluebar_whiteback': dict(trigger=blue_roi.check_skeleton,
                                            count=0)}
        return trials

    def check_exp_timer(self):
        """
        Checking the experiment timer
        """
        if not self._exp_timer.check_timer():
            print("Experiment is finished")
            print("Time ran out.")
            self.stop_experiment()

    def start_experiment(self):
        """
        Start the experiment
        """
        self._process.start()
        if not self.experiment_finished:
            self._exp_timer.start()

    def stop_experiment(self):
        """
        Stop the experiment and reset the timer
        """
        self.experiment_finished = True
        print('Experiment completed!')
        self._exp_timer.reset()
        # don't forget to end the process!
        self._process.end()

    def get_trial(self):
        """
        Check which trial is going on right now
        """
        return self._event
