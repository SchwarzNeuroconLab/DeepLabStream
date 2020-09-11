import os
from datetime import date
import configparser as cfg



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
    config = DlStreamConfigWriter()

    settings_dict = dict(bodyparts='neck', stummbel=1, fumbel=(1, 2, 2), rumbel=False)
    experiment_dict = dict(experiment=settings_dict, trigger=settings_dict)

    config.set_default('BaseConditionalExperiment', trigger_name= 'BaseHeaddirectionTrigger', stimulation_name= 'ScreenStimulation')
    config.write_ini()
