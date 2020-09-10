"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""

import time

from experiments.base.stimulation import BaseStimulation
from experiments.base.stimulus_process import BaseProtocolProcess, Timer
from experiments.utils.exp_setup import get_experiment_settings, setup_trigger, setup_process
from utils.plotter import plot_triggers_response


class BaseExperiment:
    """
        Base class for standard experiments"""

    def __init__(self):
        self._name = 'BaseExperiment'
        self._settings_dict = {}
        self.experiment_finished = False
        self._process = None
        self._event = None
        self._current_trial = None
        self._exp_timer = Timer(10)

    def check_skeleton(self, frame, skeleton):
        """
        Checking each passed animal skeleton for a pre-defined set of conditions
        Outputting the visual representation, if exist
        Advancing trials according to inherent logic of an experiment
        :param frame: frame, on which animal skeleton was found
        :param skeleton: skeleton, consisting of multiple joints of an animal
        """
        pass


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
        if self._process is not None:
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
        if self._process is not None:
            self._process.end()

    def get_settings(self):

        return self._settings_dict

    def get_name(self):

        return self._name






class BaseExampleExperiment(BaseExperiment):
    """
    Simple class to contain all of the experiment properties
    Uses multiprocess to ensure the best possible performance and
        to showcase that it is possible to work with any type of equipment, even timer-dependent
    """
    def __init__(self):
        self._name = 'BaseExampleExperiment'
        self._parameter_dict = dict(TRIGGER = 'str',
                                    PROCESS = 'str',
                                    INTERTRIAL_TIME = 'int',
                                    TRIAL_TIME = 'int',
                                    EXP_LENGTH = 'int',
                                    EXP_COMPLETION = 'int',
                                    EXP_TIME = 'int')
        self._settings_dict = get_experiment_settings(self._name, self._parameter_dict)
        self.experiment_finished = False
        self._process = setup_process(self._settings_dict['PROCESS'])
        self._event = None
        self._current_trial = None
        self._trial_count = {trial: 0 for trial in self._trials}
        self._trial_timers = {trial: Timer(self._settings_dict['TRIAL_TIME']) for trial in self._trials}
        self._exp_timer = Timer(self._settings_dict['EXP_TIME'])

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
            result, response = False, None
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

            self._process.put(self._current_trial)
            return result, response

    @property
    def _trials(self):
        """
        Defining the trials
        """
        trigger = setup_trigger(self._settings_dict['TRIGGER'])
        trials = {'Greenbar_whiteback': dict(trigger=trigger.check_skeleton,
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
        return self._current_trial



class BaseConditionalExperiment(BaseExperiment):
    """
    Simple class to contain all of the experiment properties
    Uses multiprocess to ensure the best possible performance and
        to showcase that it is possible to work with any type of equipment, even timer-dependent
    """
    def __init__(self):
        self._name = 'BaseExampleExperiment'
        self._parameter_dict = dict(TRIGGER = 'str',
                                    STIMULATION = 'str',
                                    INTERTRIAL_TIME = 'int',
                                    EXP_LENGTH = 'int',
                                    EXP_COMPLETION = 'int',
                                    EXP_TIME = 'int')
        self._settings_dict = get_experiment_settings(self._name, self._parameter_dict)
        self.experiment_finished = False
        self._process = BaseProtocolProcess(process_type='switch',
                                            stimulus_name= self._settings_dict['STIMULATION'])
        self._event = None
        self._event_count = 0
        self._current_trial = None

        self._exp_timer = Timer(self._settings_dict['EXP_TIME'])

        self._trigger = setup_trigger(self._settings_dict['TRIGGER'])

    def check_skeleton(self, frame, skeleton):
        """
        Checking each passed animal skeleton for a pre-defined set of conditions
        Outputting the visual representation, if exist
        Advancing trials according to inherent logic of an experiment
        :param frame: frame, on which animal skeleton was found
        :param skeleton: skeleton, consisting of multiple joints of an animal
        """
        self.check_exp_timer()  # checking if experiment is still on

        if self._event_count >= self._settings_dict['EXP_COMPLETION']:
            self.stop_experiment()

        elif not self.experiment_finished:

            # check if condition is met
            result, response = self._trigger.check_skeleton(skeleton=skeleton)
            plot_triggers_response(frame, response)
            if result:
                self._event_count += 1
                print('Stimulation #{self._event_count}'.format())

            self._process.put(result)
            return result, response

    def check_exp_timer(self):
        """
        Checking the experiment timer
        """
        if not self._exp_timer.check_timer():
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
        return self._current_trial






"""Standardexperiments that can be setup by using the experiment config"""

class BaseOptogeneticExperiment(BaseExperiment):
    """Standard implementation of an optogenetic experiment"""

    def __init__(self):
        self.experiment_finished = False
        self._name = 'BaseOptogeneticExperiment'

        #loading settings
        self._exp_parameter_dict = dict(TRIGGER ='str',
                                        INTERSTIM_TIME = 'int',
                                        MAX_STIM_TIME = 'int',
                                        MIN_STIM_TIME='int',
                                        MAX_TOTAL_STIM_TIME = 'int',
                                        EXP_LENGTH = 'int',
                                        EXP_COMPLETION = 'int',
                                        EXP_TIME = 'int')
        self._settings_dict = get_experiment_settings(self._name, self._exp_parameter_dict)

        self._intertrial_timer = Timer(self._settings_dict['INTERTRIAL_TIME'])
        self._exp_timer = Timer(self._settings_dict['EXP_TIME'])
        self._event = False
        self._event_start = None

        #setting limits
        self._max_trial_time = self._settings_dict['MAX_STIM_TIME']
        self._min_trial_time = self._settings_dict['MIN_STIM_TIME']
        self._max_total_time = self._settings_dict['MAX_TOTAL_STIM_TIME'] if self._settings_dict['MAX_TOTAL_STIM_TIME']\
                                                                     is not None else self._settings_dict['EXP_TIME'] + 1

        #keeping count
        self._results = []
        self._total_time = 0
        self._trial_time = 0
        #trigger
        self._trigger = setup_trigger(self._settings_dict['TRIGGER'])

        #Stimulation
        self._stimulus = BaseStimulation()


    def check_skeleton(self, frame, skeleton):

        if self._exp_timer.check_timer():
            if self._total_time >= self._max_total_time:
                # check if total time to stimulate per experiment is reached
                print("Ending experiment, total event time ran out")
                self.stop_experiment()
            else:
                # if not continue
                if not self._intertrial_timer.check_timer():
                    # check if there is an intertrial time running right now, if not continue
                    # check if the trigger is true
                    trigger_result, _ = self._trigger.check_skeleton(skeleton)
                    if trigger_result:
                        if not self._event:
                            # if a stimulation event wasn't started already, start one
                            print("Starting Stimulation")
                            self._event = True
                            # and activate the laser, start the timer and reset the intertrial timer
                            self._stimulus.start()
                            self._event_start = time.time()
                            self._intertrial_timer.reset()
                        else:
                            if time.time() - self._event_start <= self._max_trial_time:
                                # if the total event time has not reached the maximum time per event
                                pass
                            else:
                                # if the maximum event time was reached, reset the event,
                                # turn off the laser and start inter-trial time
                                print("Ending Stimulation, Stimulation time ran out")
                                self._event = False
                                self._stimulus.stop()
                                trial_time = time.time() - self._event_start
                                self._total_time += trial_time
                                self._results.append(trial_time)
                                print("Stimulation duration", trial_time)
                                self._intertrial_timer.start()
                    else:
                        # if the trigger is false
                        if self._event:
                            # but the stimulation is still going
                            if time.time() - self._event_start < self._min_trial_time:
                                # check if the minimum event time was not reached, then pass
                                pass
                            else:
                                # if minumum event time has been reached, reset the event,
                                # turn of the laser and start intertrial time
                                print("Ending Stimulation, Trigger is False")
                                self._event = False
                                self._stimulus.stop()
                                trial_time = time.time() - self._event_start
                                self._total_time += trial_time
                                self._results.append(trial_time)
                                print("Stimulation duration", trial_time)
                                self._intertrial_timer.start()

        else:
            # if maximum experiment time was reached, stop experiment
            print("Ending experiment, timer ran out")
            self.stop_experiment()

        return self._event

    def start_experiment(self):
        self._exp_timer.start()

    def stop_experiment(self):
        self.experiment_finished = True
        print('Experiment completed!')
        print("Total event duration", sum(self._results))
        print(self._results)

    def get_trial(self):
        return self._event





