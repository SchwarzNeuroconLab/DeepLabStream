import os
from datetime import date
import configparser as cfg



class DlStreamConfigWriter:

    def __init__(self):
        self._config = cfg.RawConfigParser()
        self._filename = None
        self._default_path = os.path.join(os.path.dirname(__file__),'..', 'configs')
        self._dlstream_dict = {}
        self._date = date.today().strftime("%d%m%Y")

    def set_default(self, experiment_name):
        default_config = cfg.RawConfigParser()
        default_config.read(os.path.join(self._default_path, 'default_config.ini'))

        self._dlstream_dict['EXPERIMENT'] = dict(BASE=experiment_name,
                                                 EXPERIMENTOR = 'DEFAULT')
        try:
            self._dlstream_dict[experiment_name] = default_config[experiment_name]
        except Exception:
            raise ValueError(f'Unknown Experiment: {experiment_name}.')

        keys = ['TRIGGER', 'PROCESS']
        for key in keys:
            new_section = self._dlstream_dict[experiment_name][key]
            if new_section is not None:
                self._dlstream_dict[new_section] = default_config[new_section]

    def set_custom(self, config_path):
        self._config.read(config_path)
        self._dlstream_dict = self._config._sections

    def write_ini(self):
        if self._filename is None:
            print(self._dlstream_dict)
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

    def get_current_config(self):
        return self._dlstream_dict


if __name__ == '__main__':
    config = DlStreamConfigWriter()

    settings_dict = dict(bodyparts='neck', stummbel=1, fumbel=(1, 2, 2), rumbel=False)
    experiment_dict = dict(experiment=settings_dict, trigger=settings_dict)

    config.set_default('BaseConditionalExperiment')
    config.write_ini()
