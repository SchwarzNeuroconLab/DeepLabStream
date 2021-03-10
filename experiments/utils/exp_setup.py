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
from utils.configloader import EXP_NAME, EXP_ORIGIN


def get_config_settings(name, parameter_dict, config_file_name):

    config = cfg.ConfigParser()
    path = os.path.join(os.path.dirname(__file__), "..", "configs", config_file_name)
    with open(path) as file:
        config.read_file(file)

    config_dict = {}
    for parameter in list(parameter_dict.keys()):
        if parameter_dict[parameter] == "int":
            try:
                config_dict[parameter] = config[name].getint(parameter)
            except:
                config_dict[parameter] = None
                print(
                    "Did not find valid {} entry for {} in {}. Setting to None.".format(
                        parameter, name, config_file_name
                    )
                )
        if parameter_dict[parameter] == "float":
            try:
                config_dict[parameter] = config[name].getfloat(parameter)
            except:
                config_dict[parameter] = None
                print(
                    "Did not find valid {} entry for {} in {}. Setting to None.".format(
                        parameter, name, config_file_name
                    )
                )

        elif parameter_dict[parameter] == "tuple":
            try:
                config_dict[parameter] = tuple(
                    int(entry) for entry in config[name].get(parameter).split(",")
                )
            except:
                config_dict[parameter] = None
                print(
                    "Did not find valid {} entry for {} in {}. Setting to None.".format(
                        parameter, name, config_file_name
                    )
                )

        elif parameter_dict[parameter] == "list":
            try:
                config_dict[parameter] = list(
                    str(entry) for entry in config[name].get(parameter).split(",")
                )
            except:
                config_dict[parameter] = None
                print(
                    "Did not find valid {} entry for {} in {}. Setting to None.".format(
                        parameter, name, config_file_name
                    )
                )

        elif parameter_dict[parameter] == "boolean":
            try:
                config_dict[parameter] = config[name].getboolean(parameter)
            except:
                config_dict[parameter] = None
                print(
                    "Did not find valid {} entry for {} in {}. Setting to None.".format(
                        parameter, name, config_file_name
                    )
                )

        elif parameter_dict[parameter] == "str":
            try:
                config_dict[parameter] = config[name].get(parameter)
            except:
                config_dict[parameter] = None
                print(
                    "Did not find valid {} entry for {} in {}. Setting to None.".format(
                        parameter, name, config_file_name
                    )
                )

    return config_dict


def get_experiment_settings(experiment_name, parameter_dict):

    experiment_config = get_config_settings(
        experiment_name, parameter_dict, f"{EXP_NAME}.ini"
    )

    return experiment_config


def get_stimulation_settings(stimulation_name, parameter_dict):
    experiment_config = get_config_settings(
        stimulation_name, parameter_dict, f"{EXP_NAME}.ini"
    )

    return experiment_config


def get_trigger_settings(trigger_name, parameter_dict):
    experiment_config = get_config_settings(
        trigger_name, parameter_dict, f"{EXP_NAME}.ini"
    )

    return experiment_config


def get_process_settings(process_name, parameter_dict):
    experiment_config = get_config_settings(
        process_name, parameter_dict, f"{EXP_NAME}.ini"
    )

    return experiment_config


def setup_experiment():
    if EXP_ORIGIN.upper() == "BASE":
        config = cfg.ConfigParser()
        path = os.path.join(
            os.path.dirname(__file__), "..", "configs", f"{EXP_NAME}.ini"
        )
        try:
            with open(path) as file:
                config.read_file(file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{EXP_NAME}.ini was not found. Make sure it exists."
            )

        experiment_name = config["EXPERIMENT"]["BASE"]
        import importlib

        mod = importlib.import_module("experiments.base.experiments")
        try:
            experiment_class = getattr(mod, experiment_name)
            experiment = experiment_class()
        except Exception:
            raise ValueError(
                f"Experiment: {experiment_name} not in base.experiments.py."
            )

    elif EXP_ORIGIN.upper() == "CUSTOM":

        experiment_name = EXP_NAME
        import importlib

        mod = importlib.import_module("experiments.custom.experiments")
        try:
            experiment_class = getattr(mod, experiment_name)
            experiment = experiment_class()
        except Exception:
            raise ValueError(
                f"Experiment: {experiment_name} not in custom.experiments.py."
            )

    else:
        raise ValueError(
            f'Experiment Origin "{EXP_ORIGIN}" not valid. Pick CUSTOM or BASE.'
        )

    return experiment


def setup_trigger(trigger_name):
    import importlib

    mod = importlib.import_module("experiments.base.triggers")
    try:
        trigger_class = getattr(mod, trigger_name)
        trigger = trigger_class()
    except Exception:
        raise ValueError(f"Trigger: {trigger_name} not in base.triggers.py.")

    return trigger


def setup_process(process_name):
    import importlib

    mod = importlib.import_module("experiments.base.stimulus_process")
    try:
        process_class = getattr(mod, process_name)
        process = process_class()
    except Exception:
        raise ValueError(f"Process: {process_name} not in base.stimulus_process.py.")

    return process


def setup_stimulation(stimulus_name):
    import importlib

    mod = importlib.import_module("experiments.base.stimulation")
    try:
        stimulation_class = getattr(mod, stimulus_name)
        stimulation = stimulation_class()
    except Exception:
        raise ValueError(f"Stimulus: {stimulus_name} not in stimulation.py.")

    return stimulation


class DlStreamConfigWriter:
    def __init__(self):
        self._config = self._init_configparser()
        self._default_config = self._init_configparser()
        self._init_configparser()
        self._filename = None
        self._default_path = os.path.join(os.path.dirname(__file__), "..", "configs")
        self._dlstream_dict = dict(
            EXPERIMENT=dict(BASE="DEFAULT", EXPERIMENTER="DEFAULT")
        )
        self._date = date.today().strftime("%d%m%Y")
        # TODO: Make this adaptive!
        self._available_modules = dict(
            EXPERIMENT=[
                "BaseConditionalExperiment",
                "BaseOptogeneticExperiment",
                "BaseTrialExperiment",
            ],
            TRIGGER=[
                "BaseRegionTrigger",
                "BaseOutsideRegionTrigger",
                "BaseHeaddirectionTrigger",
                "BaseEgoHeaddirectionTrigger",
                "BaseScreenTrigger",
                "BaseSpeedTrigger",
                "BaseFreezeTrigger",
                "BaseHeaddirectionROITrigger",
            ],
            PROCESS=["BaseProtocolProcess"],
            STIMULATION=["BaseStimulation", "RewardDispenser", "ScreenStimulation"],
        )

    @staticmethod
    def _init_configparser():
        config = cfg.ConfigParser()
        config.optionxform = str
        return config

    def _read_default_config(self):
        try:
            self._default_config.read(
                os.path.join(self._default_path, "default_config.ini")
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "The default_config.ini was not found. Make sure it exists."
            )

    def _read_config(self, config_path):

        try:
            self._config.read(config_path)
        except FileNotFoundError:
            raise FileNotFoundError("Config file does not exist at this location.")

    def set_experimenter(self, name):
        self._dlstream_dict["EXPERIMENT"]["EXPERIMENTER"] = name

    def set_experiment(self, experiment_name):
        self._dlstream_dict["EXPERIMENT"]["BASE"] = experiment_name

    def import_default(
        self,
        experiment_name,
        trigger_name=None,
        process_name=None,
        stimulation_name=None,
        trial_trigger_name=None,
    ):

        self._read_default_config()

        self.set_experiment(experiment_name)

        try:
            self._dlstream_dict[experiment_name] = self._default_config[experiment_name]
        except Exception:
            raise ValueError(f"Unknown Experiment: {experiment_name}.")

        if trigger_name is not None:
            self._dlstream_dict[trigger_name] = self._default_config[trigger_name]
            self._dlstream_dict[experiment_name]["TRIGGER"] = trigger_name
        else:
            trigger_name = self._dlstream_dict[experiment_name]["TRIGGER"]
            if trigger_name is not None:
                self._dlstream_dict[trigger_name] = self._default_config[trigger_name]

        if process_name is not None:
            self._dlstream_dict[process_name] = self._default_config[process_name]
            self._dlstream_dict[experiment_name]["PROCESS"] = process_name
        else:
            process_name = self._dlstream_dict[experiment_name]["PROCESS"]
            if process_name is None:
                self._dlstream_dict[process_name] = self._default_config[process_name]

        if stimulation_name is not None:
            self._dlstream_dict[stimulation_name] = self._default_config[
                stimulation_name
            ]
            self._dlstream_dict[process_name]["STIMULATION"] = stimulation_name
        else:
            stimulation_name = self._dlstream_dict[process_name]["STIMULATION"]
            if stimulation_name is not None:
                self._dlstream_dict[stimulation_name] = self._default_config[
                    stimulation_name
                ]

        # TODO: Make this adaptive
        if experiment_name == "BaseTrialExperiment" and trial_trigger_name is not None:
            # TODO: ADD option to use the same trigger as trigger_name
            if not trigger_name == trial_trigger_name:
                self._dlstream_dict[trial_trigger_name] = self._default_config[
                    trial_trigger_name
                ]
            else:
                raise ValueError(
                    f"Trial Trigger can currently not be the same as Trigger."
                )

    def import_custom(self, config_path):

        self._read_config(config_path)
        self._dlstream_dict = self._config._sections

    def _set_path(self):
        if self._filename is None:
            experiment_name = self._dlstream_dict["EXPERIMENT"]["BASE"]
            self._filename = f"{experiment_name}_{self._date}.ini"

        file = open(os.path.join(self._default_path, self._filename), "w")
        return file

    def _change_module(self, module_type: str, module_name: str):
        """Changes module in a given settings.ini (resets parameters on that module)
        :param module_type str: Module type (TRIGGER, PROCESS, STIMULATION, EXPERIMENT etc.)
        :param module_name str: Exact name of new module (with Camelcase)
        :param config_path: path to config that needs changing"""

        module_type = module_type.upper()
        self._read_default_config()
        # self.import_custom(config_path)

        for key in self._dlstream_dict.keys():
            if module_type in self._dlstream_dict[key].keys():
                old_module = self._dlstream_dict[key][module_type]
                self._dlstream_dict[key][module_type] = module_name
        self._dlstream_dict.pop(old_module, None)
        self._dlstream_dict[module_name] = self._default_config[module_name]
        print(f"Changed {old_module} to {module_name}.")

    def _change_parameter(
        self, module_name: str, parameter_name: str, parameter_value: str
    ):

        parameter_name = parameter_name.upper()
        # self.import_custom(config_path)
        if module_name in self._dlstream_dict.keys():
            if parameter_name in self._dlstream_dict[module_name].keys():
                old_value = self._dlstream_dict[module_name][parameter_name]
                if not isinstance(parameter_value, str):
                    parameter_value = str(parameter_value)
                self._dlstream_dict[module_name][parameter_name] = parameter_value
                print(
                    f"Changed {parameter_name} in {module_name} from {old_value} to {parameter_value}."
                )
            else:
                raise ValueError(
                    f"Parameter {parameter_name} does not exist in given config."
                )

        else:
            raise ValueError(f"Module {module_name} does not exist in given config.")

    def check_if_default_exists(self, module_name, module_type):
        # TODO: adjust to make adaptive
        # self._read_default_config()
        if module_name in self._available_modules[module_type]:
            return True
        else:
            return False

    def get_available_module_names(self, module_type):
        return self._available_modules[module_type]

    def change_modules(self, config_path, module_dict: dict, overwrite: bool = False):
        """Changes multiple modules at once,
        :param module_dict: dictionary in style:(module_type = module_name)"""

        self.import_custom(config_path)

        for key, value in module_dict.items():
            self._change_module(module_type=str(key).upper(), module_name=value)

        if overwrite:
            self.write_ini(path=config_path)

    def change_parameters(
        self, config_path, parameter_dict: dict, overwrite: bool = False
    ):
        """Changes multiple modules at once,
        :param paramter_dict: nested dictionary in style:{module_name: dict(parameter_name = value)}"""

        self.import_custom(config_path)
        for key in parameter_dict.keys():
            for inner_key, value in parameter_dict[key].items():
                self._change_parameter(
                    module_name=str(key).upper(),
                    parameter_name=inner_key,
                    parameter_value=value,
                )

        if overwrite:
            self.write_ini(path=config_path)

    def change_config(self, config_path, config_dict: dict, overwrite: bool = False):
        """Changes multiple modules and parameters at once,
        :param config_dict: nested dictionary in style:dict(module_type ={module_name: dict(parameter_name = value)})"""

        self.import_custom(config_path)

        for key, inner_dict in config_dict.items():
            if isinstance(inner_dict, dict):
                for inner_key, most_inner_dict in inner_dict.items():
                    self._change_module(module_type=key, module_name=inner_key)
                    if isinstance(most_inner_dict, dict):
                        for most_inner_key in most_inner_dict.keys():

                            self._change_parameter(
                                module_name=inner_key,
                                parameter_name=str(most_inner_key).upper(),
                                parameter_value=most_inner_dict[most_inner_key],
                            )
            else:
                self._change_module(module_type=key, module_name=inner_dict)

        if overwrite:
            self.write_ini(path=config_path)

    def write_ini(self, path: str = None):
        if path is None:
            file = self._set_path()
        else:
            self._filename = os.path.basename(path)
            file = open(path, "w")
        self._config = self._init_configparser()
        for key in self._dlstream_dict.keys():
            self._config.add_section(key)
            for parameter, value in self._dlstream_dict[key].items():
                self._config.set(key, parameter, str(value))
        self._config.write(file)
        file.close()
        print(f"Created {self._filename}.")

    def set_filename(self, filename):
        self._filename = filename + ".ini"

    def get_current_config(self):
        return self._dlstream_dict

    def get_parameters(self, module_name):
        if module_name in self._dlstream_dict.keys():
            print(self._dlstream_dict.keys())
            return self._dlstream_dict[module_name]
        else:
            raise ValueError(f"{module_name} is not valid.")


if __name__ == "__main__":

    exp = setup_experiment()
    print(exp)
