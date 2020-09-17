"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""

import time

from experiments.base.stimulation import BaseStimulation
from experiments.base.stimulus_process import Timer
from experiments.utils.exp_setup import get_experiment_settings, setup_trigger, setup_process
from utils.plotter import plot_triggers_response
import random


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


class BaseTrialExperiment(BaseExperiment):
    def __init__(self):
        super().__init__()
        self._name = 'BaseTrialExperiment'
        self.experiment_finished = False
        self._event = None
        self._print_check = False
        self._current_trial = None
        self._result_list = []
        self._success_count = 0

        self._parameter_dict = dict(TRIGGER = 'str',
                                    PROCESS = 'str',
                                    INTERTRIAL_TIME = 'int',
                                    TRIAL_NAME = 'str',
                                    TRIAL_TRIGGER = 'str',
                                    TRIAL_TIME = 'int',
                                    STIMULUS_TIME = 'int',
                                    RESULT_FUNC = 'str',
                                    EXP_LENGTH = 'int',
                                    EXP_COMPLETION = 'int',
                                    EXP_TIME = 'int')

        self._settings_dict = get_experiment_settings(self._name, self._parameter_dict)
        self._process = setup_process(self._settings_dict['PROCESS'])
        self._init_trigger = setup_trigger(self._settings_dict['TRIGGER'])
        self._trials_list = self.generate_trials_list(self._trials, self._settings_dict['EXP_LENGTH'])
        self._trial_timer = Timer(self._settings_dict['TRIAL_TIME'])
        self._exp_timer = Timer(self._settings_dict['EXP_TIME'])
        self._intertrial_timer = Timer(self._settings_dict['INTERTRIAL_TIME'])

    def check_skeleton(self, frame, skeleton):
        status, trial = self._process.get_status()
        if status:
            current_trial = self._trials[trial]
            condition, response = current_trial['trigger'].check_skeleton(skeleton)
            self._process.put(condition)
            result = self._process.get_result()
            if result is not None:
                self.process_result(result, trial)
                self._current_trial = None
                # check if all trials were successful until completion
                if self._success_count >= self._settings_dict['EXP_COMPLETION']:
                    print("Experiment is finished")
                    print("Trial reached required amount of successes")
                    self.stop_experiment()

                # if not continue
                print(' Going into Intertrial time.')
                self._intertrial_timer.reset()
                self._intertrial_timer.start()
            result = None
            plot_triggers_response(frame, response)

        elif not self._intertrial_timer.check_timer():
            if self._current_trial is None:
                self._current_trial = next(self._trials_list,False)
            elif not self._current_trial:
                print("Experiment is finished due to max. trial repetition.")
                print(self._result_list)
                self.stop_experiment()
            else:
                init_result, response_body = self._init_trigger.check_skeleton(skeleton)
                if init_result:
                    # check trial start triggers
                    self._process.put_trial(self._trials[self._current_trial], self._current_trial)
                    self._print_check = False
                elif not self._print_check:
                    print('Next trial: #' + str(len(self._result_list) + 1) + ' ' + self._current_trial)
                    print('Animal is not meeting trial start criteria, the start of trial is delayed.')
                    self._print_check = True
        # if experimental time ran out, finish experiments
        super().check_exp_timer()

    def process_result(self, result, trial):
        """
        Will add result if TRUE or reset comp_counter if FALSE
        :param result: bool if trial was successful
        :param trial: str name of the trial
        :return:
        """
        self._result_list.append((trial, result))
        if result is True:
            self._success_count +=1
            print('Trial successful!')
        else:
            print('Trial failed.')
            #

    @staticmethod
    def generate_trials_list(trials: dict, length: int):
        trials_list = []
        for trial in range(length):
            trials_list.append(random.choice(list(trials.keys())))
        return iter(trials_list)

    @property
    def _trials(self):

        trigger = setup_trigger(self._settings_dict['TRIAL_TRIGGER'])
        if self._settings_dict['RESULT_FUNC'] == 'all':
            result_func = all
        elif self._settings_dict['RESULT_FUNC'] == 'any':
            result_func = any
        else:
            raise ValueError(f'Result function can only be "all" or "any", not {self._settings_dict["RESULT_FUNC"]}.')
        trials = {self._settings_dict['TRIAL_NAME']: dict(stimulus_timer=Timer(self._settings_dict['STIMULUS_TIME']),
                               success_timer=Timer(self._settings_dict['TRIAL_TIME']),
                               trigger=trigger,
                               result_func=result_func)}

        return trials


class BaseConditionalExperiment(BaseExperiment):
    """
    Simple class to contain all of the experiment properties
    Uses multiprocess to ensure the best possible performance and
        to showcase that it is possible to work with any type of equipment, even timer-dependent
    """
    def __init__(self):
        super().__init__()
        self._name = 'BaseConditionalExperiment'
        self._parameter_dict = dict(TRIGGER = 'str',
                                    PROCESS = 'str',
                                    INTERTRIAL_TIME = 'int',
                                    EXP_LENGTH = 'int',
                                    EXP_TIME = 'int')
        self._settings_dict = get_experiment_settings(self._name, self._parameter_dict)
        self.experiment_finished = False
        self._process = setup_process(self._settings_dict['PROCESS'])
        self._event = None
        self._event_count = 0
        self._current_trial = None

        self._exp_timer = Timer(self._settings_dict['EXP_TIME'])
        self._intertrial_timer = Timer(self._settings_dict['INTERTRIAL_TIME'])

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

        if self._event_count >= self._settings_dict['EXP_LENGTH']:
            self.stop_experiment()

        elif not self.experiment_finished:
            if not self._intertrial_timer.check_timer():
                # check if condition is met
                result, response = self._trigger.check_skeleton(skeleton=skeleton)
                if result:
                    self._event_count += 1
                    print('Stimulation #{self._event_count}'.format())
                    self._intertrial_timer.reset()
                    self._intertrial_timer.start()

                plot_triggers_response(frame, response)
                self._process.put(result)

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
        super().__init__()
        self.experiment_finished = False
        self._name = 'BaseOptogeneticExperiment'

        #loading settings
        self._exp_parameter_dict = dict(TRIGGER ='str',
                                        INTERSTIM_TIME = 'int',
                                        MAX_STIM_TIME = 'int',
                                        MIN_STIM_TIME='int',
                                        MAX_TOTAL_STIM_TIME = 'int',
                                        EXP_TIME = 'int',
                                        PROCESS = 'str')
        self._settings_dict = get_experiment_settings(self._name, self._exp_parameter_dict)
        self._process = setup_process(self._settings_dict['PROCESS'])
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
                    result, _ = self._trigger.check_skeleton(skeleton)
                    if result:
                        if not self._event:
                            # if a stimulation event wasn't started already, start one
                            print("Starting Stimulation")
                            self._event = True
                            # and activate the laser, start the timer and reset the intertrial timer
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
                                trial_time = time.time() - self._event_start
                                self._total_time += trial_time
                                self._results.append(trial_time)
                                print("Stimulation duration", trial_time)
                                self._intertrial_timer.start()
            self._process.put(self._event)


        else:
            # if maximum experiment time was reached, stop experiment
            print("Ending experiment, timer ran out")
            self.stop_experiment()

    def start_experiment(self):
        self._exp_timer.start()

    def stop_experiment(self):
        self.experiment_finished = True
        print('Experiment completed!')
        print("Total event duration", sum(self._results))
        print(self._results)

    def get_trial(self):
        return self._event





