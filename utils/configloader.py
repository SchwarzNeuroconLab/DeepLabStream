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
adv_dsc_config = cfg.ConfigParser()


def get_script_path():
    return os.path.dirname(os.path.join(os.path.dirname(__file__), '..'))


cfg_path = os.path.join(os.path.dirname(__file__), '..', 'settings.ini')
with open(cfg_path) as cfg_file:
    dsc_config.read_file(cfg_file)

adv_cfg_path = os.path.join(os.path.dirname(__file__), 'advanced_settings.ini')
with open(adv_cfg_path) as adv_cfg_file:
    adv_dsc_config.read_file(adv_cfg_file)

#poseestimation
MODEL_ORIGIN = dsc_config['Pose Estimation'].get('MODEL_ORIGIN')
MODEL_PATH = dsc_config['Pose Estimation'].get('MODEL_PATH')
MODEL_NAME = dsc_config['Pose Estimation'].get('MODEL_NAME')
ALL_BODYPARTS = tuple(part for part in dsc_config['Pose Estimation'].get('ALL_BODYPARTS').split(','))

# Streaming items

try:
    RESOLUTION = tuple(int(part) for part in dsc_config['Streaming'].get('RESOLUTION').split(','))
except ValueError:
    print('Incorrect resolution in config!\n'
          'Using default value "RESOLUTION = 848, 480"')
    RESOLUTION = (848, 480)

FRAMERATE = dsc_config['Streaming'].getint('FRAMERATE')
OUT_DIR = dsc_config['Streaming'].get('OUTPUT_DIRECTORY')
CAMERA_SOURCE = dsc_config['Streaming'].get('CAMERA_SOURCE')
STREAMING_SOURCE = dsc_config['Streaming'].get('STREAMING_SOURCE')
# Video
VIDEO_SOURCE = dsc_config['Video'].get('VIDEO_SOURCE')

#IPWEBCAM
PORT = dsc_config['IPWEBCAM'].get('PORT')


# experiment
EXP_ORIGIN = dsc_config['Experiment'].get('EXP_ORIGIN')
EXP_NAME = dsc_config['Experiment'].get('EXP_NAME')
RECORD_EXP = dsc_config['Experiment'].getboolean('RECORD_EXP')

START_TIME = time.time()

"""advanced settings"""
STREAMS = [str(part).strip() for part in adv_dsc_config['Streaming'].get('STREAMS').split(',')]
MULTI_CAM = adv_dsc_config['Streaming'].getboolean('MULTIPLE_DEVICES')
STACK_FRAMES = adv_dsc_config['Streaming'].getboolean('STACK_FRAMES') if adv_dsc_config['Streaming'].getboolean(
    'STACK_FRAMES') is not None else False
ANIMALS_NUMBER = adv_dsc_config['Streaming'].getint('ANIMALS_NUMBER') if adv_dsc_config['Streaming'].getint(
    'ANIMALS_NUMBER') is not None else 1

REPEAT_VIDEO = adv_dsc_config['Video'].getboolean('REPEAT_VIDEO')
CROP = adv_dsc_config['Streaming'].getboolean('CROP')
CROP_X = [int(str(part).strip()) for part in adv_dsc_config['Streaming'].get('CROP_X').split(',')]
CROP_Y = [int(str(part).strip()) for part in adv_dsc_config['Streaming'].get('CROP_Y').split(',')]

FLATTEN_MA = adv_dsc_config['Pose Estimation'].getboolean('FLATTEN_MA')


