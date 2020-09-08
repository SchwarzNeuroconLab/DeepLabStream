"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""

import time
import os
import configparser as cfg

# loading DeepLabStream configuration
# remember when it was called DSC?
dsc_config = cfg.ConfigParser()


def get_script_path():
    return os.path.dirname(os.path.join(os.path.dirname(__file__), '..'))


cfg_path = os.path.join(os.path.dirname(__file__), '..', 'settings.ini')
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
MODEL = dsc_config['Streaming'].get('MODEL')
FRAMERATE = dsc_config['Streaming'].getint('FRAMERATE')
OUT_DIR = dsc_config['Streaming'].get('OUTPUT_DIRECTORY')
STREAM = dsc_config['Streaming'].getboolean('STREAM')
MULTI_CAM = dsc_config['Streaming'].getboolean('MULTIPLE_DEVICES')
STACK_FRAMES = dsc_config['Streaming'].getboolean('STACK_FRAMES') if dsc_config['Streaming'].getboolean(
    'STACK_FRAMES') is not None else False
ANIMALS_NUMBER = dsc_config['Streaming'].getint('ANIMALS_NUMBER') if dsc_config['Streaming'].getint(
    'ANIMALS_NUMBER') is not None else 1
STREAMS = [str(part).strip() for part in dsc_config['Streaming'].get('STREAMS').split(',')]
CAMERA_SOURCE = dsc_config['Streaming'].get('CAMERA_SOURCE')

# Video
VIDEO_SOURCE = dsc_config['Video'].get('VIDEO_SOURCE')

# experiment
EXP_NUMBER = dsc_config['Experiment'].getint('EXP_NUMBER')
RECORD_EXP = dsc_config['Experiment'].getboolean('RECORD_EXP')

START_TIME = time.time()
EGG = "".join(format(ord(x), 'b') for x in "Hello there!")
