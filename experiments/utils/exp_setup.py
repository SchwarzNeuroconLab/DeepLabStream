"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""


import os
import configparser as cfg
from datetime import date
from utils.configloader import EXP_NAME



def get_config_settings(name, parameter_dict, config_file_name):

    config = cfg.ConfigParser()
    path = os.path.join(os.path.dirname(__file__), '..', 'configs', config_file_name)
    with open(path) as file:
        config.read_file(file)

    config_dict = {}
    for parameter in list(parameter_dict.keys()):
        if parameter_dict[parameter] == 'int':
            try:
                config_dict[parameter] = config[name].getint(parameter)
            except:
                config_dict[parameter] = None
                print('Did not find valid {} entry for {} in {}. Setting to None.'.format(parameter, name, config_file_name))
        if parameter_dict[parameter] == 'float':
            try:
                config_dict[parameter] = config[name].getfloat(parameter)
            except:
                config_dict[parameter] = None
                print('Did not find valid {} entry for {} in {}. Setting to None.'.format(parameter, name, config_file_name))

        elif parameter_dict[parameter] == 'tuple':
            try:
                config_dict[parameter] = tuple(int(entry) for entry in config[name].get(parameter).split(','))
            except:
                config_dict[parameter] = None
                print('Did not find valid {} entry for {} in {}. Setting to None.'.format(parameter, name, config_file_name))

        elif parameter_dict[parameter] == 'list':
            try:
                config_dict[parameter] = list(
                    str(entry) for entry in config[name].get(parameter).split(','))
            except:
                config_dict[parameter] = None
                print('Did not find valid {} entry for {} in {}. Setting to None.'.format(parameter, name, config_file_name))

        elif parameter_dict[parameter] == 'boolean':
            try:
                config_dict[parameter] = config[name].getboolean(parameter)
            except:
                config_dict[parameter] = None
                print('Did not find valid {} entry for {} in {}. Setting to None.'.format(parameter, name, config_file_name))


        elif parameter_dict[parameter] == 'str':
            try:
                config_dict[parameter] = config[name].get(parameter)
            except:
                config_dict[parameter] = None
                print('Did not find valid {} entry for {} in {}. Setting to None.'.format(parameter, name, config_file_name))


    return config_dict



def get_experiment_settings(experiment_name, parameter_dict):

    experiment_config = get_config_settings(experiment_name, parameter_dict, f'{EXP_NAME}.ini')

    return experiment_config


def get_stimulation_settings(stimulation_name, parameter_dict):
    experiment_config = get_config_settings(stimulation_name, parameter_dict, f'{EXP_NAME}.ini')

    return experiment_config

def get_trigger_settings(trigger_name, parameter_dict):
    experiment_config = get_config_settings(trigger_name, parameter_dict, f'{EXP_NAME}.ini')

    return experiment_config

def get_process_settings(process_name, parameter_dict):
    experiment_config = get_config_settings(process_name, parameter_dict, f'{EXP_NAME}.ini')

    return experiment_config


def setup_experiment():
    config = cfg.ConfigParser()
    path = os.path.join(os.path.dirname(__file__), '..', 'configs', f'{EXP_NAME}.ini')
    with open(path) as file:
        config.read_file(file)

    experiment_name = config['EXPERIMENT']['BASE']
    import importlib
    mod = importlib.import_module('experiments.base.experiments')
    try:
        experiment_class = getattr(mod, experiment_name)
        experiment = experiment_class()
    except Exception:
        raise ValueError(f'Experiment: {experiment_name} not in base.experiments.py.')

    return experiment


def setup_trigger(trigger_name):
    import importlib
    mod = importlib.import_module('experiments.base.triggers')
    try:
        trigger_class = getattr(mod, trigger_name)
        trigger = trigger_class()
    except Exception:
        raise ValueError(f'Trigger: {trigger_name} not in base.triggers.py.')

    return trigger


def setup_process(process_name):
    import importlib
    mod = importlib.import_module('experiments.base.stimulus_process')
    try:
        process_class = getattr(mod, process_name)
        process = process_class()
    except Exception:
        raise ValueError(f'Process: {process_name} not in base.stimulus_process.py.')

    return process


def setup_stimulation(stimulus_name):
    import importlib
    mod = importlib.import_module('experiments.base.stimulation')
    try:
        stimulation_class = getattr(mod, stimulus_name)
        stimulation = stimulation_class()
    except Exception:
        raise ValueError(f'Stimulus: {stimulus_name} not in stimulation.py.')

    return stimulation


class DlStreamConfigWriter:

    def __init__(self):
        self._config = self._init_configparser()
        self._default_config = self._init_configparser()
        self._init_configparser()
        self._filename = None
        self._default_path = os.path.join(os.path.dirname(__file__),'..', 'configs')
        self._dlstream_dict = dict(EXPERIMENT = dict(BASE='DEFAULT',
                                                 EXPERIMENTOR = 'DEFAULT'))
        self._date = date.today().strftime("%d%m%Y")

    @staticmethod
    def _init_configparser():
        config = cfg.ConfigParser()
        config.optionxform=str
        return config

    def set_experimentor(self, name):
        self._dlstream_dict['EXPERIMENT']['EXPERIMENTOR'] = name

    def set_experiment(self, experiment_name):
        self._dlstream_dict['EXPERIMENT']['BASE'] = experiment_name

    def set_default(self, experiment_name, trigger_name = None, process_name = None, stimulation_name = None):

        try:
            self._default_config.read(os.path.join(self._default_path, 'default_config.ini'))
        except FileNotFoundError:
            raise FileNotFoundError('The default_config.ini was not found. Make sure it exists.')

        self.set_experiment(experiment_name)

        try:
            self._dlstream_dict[experiment_name] =  self._default_config[experiment_name]
        except Exception:
            raise ValueError(f'Unknown Experiment: {experiment_name}.')

        if trigger_name is not None:
            self._dlstream_dict[trigger_name] =  self._default_config[trigger_name]
            self._dlstream_dict[experiment_name]['TRIGGER'] = trigger_name
        else:
            trigger_name = self._dlstream_dict[experiment_name]['TRIGGER']
            if trigger_name is not None:
                self._dlstream_dict[trigger_name] =  self._default_config[trigger_name]

        if process_name is not None:
            self._dlstream_dict[process_name] =  self._default_config[process_name]
            self._dlstream_dict[experiment_name]['PROCESS'] = process_name
        else:
            process_name = self._dlstream_dict[experiment_name]['PROCESS']
            if process_name is not None:
                self._dlstream_dict[process_name] =  self._default_config[process_name]

        if stimulation_name is not None:
            self._dlstream_dict[stimulation_name] = self._default_config[stimulation_name]
            self._dlstream_dict[process_name]['STIMULATION'] = stimulation_name
        else:
            stimulation_name = self._dlstream_dict[process_name]['STIMULATION']
            if stimulation_name is not None:
                self._dlstream_dict[stimulation_name] = self._default_config[stimulation_name]

    def set_custom(self, config_path):
            try:
                self._config.read(config_path)
            except FileNotFoundError:
                raise FileNotFoundError('Config file does not exist at this location.')

            self._dlstream_dict = self._config._sections

    def write_ini(self):
        self._init_configparser()
        if self._filename is None:
            experiment_name = self._dlstream_dict['EXPERIMENT']['BASE']
            self._filename = f'{experiment_name}_{self._date}.ini'

        file = open(os.path.join(self._default_path, self._filename), 'w')
        for key in self._dlstream_dict.keys():
            self._config.add_section(key)
            for parameter, value in self._dlstream_dict[key].items():
                self._config.set(key, parameter, str(value))
        self._config.write(file)
        file.close()
        print(f'Created {self._filename}.')

    def set_filename(self, filename):
        self._filename = filename + '.ini'

    def set_parameter(self):
        pass
    def get_current_config(self):
        return self._dlstream_dict


if __name__ == '__main__':

    exp = setup_experiment()
    print(exp)


