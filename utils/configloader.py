import time
import os
import sys
import configparser as cfg

# loading DeepLabStream configuration
dsc_config = cfg.ConfigParser()


def get_script_path():
    return os.path.dirname(os.path.realpath(sys.argv[0]))


cfg_path = os.path.join(get_script_path(), 'settings.ini')
with open(cfg_path) as cfg_file:
    dsc_config.read_file(cfg_file)

# DeepLabCut
deeplabcut_config = dict(dsc_config.items('DeepLabCut'))

# Streaming items
try:
    RESOLUTION = tuple(int(part) for part in dsc_config['Streaming'].get('RESOLUTION').split(','))
except ValueError:
    print('Incorrect resolution in config!\n'
          'Using default value "RESOLUTION = 848, 480"')
    RESOLUTION = (848, 480)
FRAMERATE = dsc_config['Streaming'].getint('FRAMERATE')
OUT_DIR = dsc_config['Streaming'].get('OUTPUT_DIRECTORY')
STREAM = dsc_config['Streaming'].getboolean('STREAM')
MULTI_CAM = dsc_config['Streaming'].getboolean('MULTIPLE_DEVICES')
STACK_FRAMES = dsc_config['Streaming'].getboolean('STACK_FRAMES') if dsc_config['Streaming'].getboolean(
    'STACK_FRAMES') is not None else False
ANIMALS_NUMBER = dsc_config['Streaming'].getint('ANIMALS_NUMBER') if dsc_config['Streaming'].getint(
    'ANIMALS_NUMBER') is not None else 1
STREAMS = [str(part).strip() for part in dsc_config['Streaming'].get('STREAMS').split(',')]
VIDEO_SOURCE = dsc_config['Streaming'].get('VIDEO_SOURCE')

# experiment
EXP_NUMBER = dsc_config['Experiment'].getint('EXP_NUMBER')
RECORD_EXP = dsc_config['Experiment'].getboolean('RECORD_EXP')

# images
# BACKGRND_PATH = dsc_config['Visual'].get('BACKGRND_PATH')
# IMG1_INFO = dsc_config['Image'].get('IMG1_INFO')
# IMG1_PATH = dsc_config['Image'].get('IMG1_PATH')
# IMG2_INFO = dsc_config['Image'].get('IMG2_INFO')
# IMG2_PATH = dsc_config['Image'].get('IMG2_PATH')

START_TIME = time.time()
