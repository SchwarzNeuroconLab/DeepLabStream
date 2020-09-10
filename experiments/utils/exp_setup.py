import os
import configparser as cfg


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
                print('Did not find valid {} entry for {}. Setting to None.'.format(parameter, name))
        if parameter_dict[parameter] == 'float':
            try:
                config_dict[parameter] = config[name].getfloat(parameter)
            except:
                config_dict[parameter] = None
                print('Did not find valid {} entry for {}. Setting to None.'.format(parameter, name))

        elif parameter_dict[parameter] == 'tuple':
            try:
                config_dict[parameter] = tuple(int(entry) for entry in config[name].get(parameter).split(','))
            except:
                config_dict[parameter] = None
                print('Did not find valid {} entry for {}. Setting to None.'.format(parameter, name))

        elif parameter_dict[parameter] == 'list':
            try:
                config_dict[parameter] = list(
                    str(entry) for entry in config[name].get(parameter).split(','))
            except:
                config_dict[parameter] = None
                print('Did not find valid {} entry for {}. Setting to None.'.format(parameter, name))

        elif parameter_dict[parameter] == 'boolean':
            try:
                config_dict[parameter] = config[name].getboolean(parameter)
            except:
                config_dict[parameter] = None
                print('Did not find valid {} entry for {}. Setting to None.'.format(parameter, name))


        elif parameter_dict[parameter] == 'str':
            try:
                config_dict[parameter] = config[name].get(parameter)
            except:
                config_dict[parameter] = None
                print('Did not find valid {} entry for {}. Setting to None.'.format(parameter, name))


    return config_dict



def get_experiment_settings(experiment_name, parameter_dict):
    experiment_config = get_config_settings(experiment_name, parameter_dict, 'base_config.ini')

    return experiment_config


def get_stimulation_settings(stimulation_name, parameter_dict):
    experiment_config = get_config_settings(stimulation_name, parameter_dict, 'base_config.ini')

    return experiment_config

def get_trigger_settings(trigger_name, parameter_dict):
    experiment_config = get_config_settings(trigger_name, parameter_dict, 'base_config.ini')

    return experiment_config
