"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""

import cv2
import time
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from DeepLabStream import create_row
from utils.poser import load_deeplabcut, get_pose, find_local_peaks_new, calculate_skeletons
from utils.plotter import plot_bodyparts, plot_metadata_frame, plot_triggers_response
from utils.configloader import VIDEO_SOURCE, OUT_DIR, ANIMALS_NUMBER
from experiments.experiments import ExampleExperiment
from deepposekit.models import load_model
import numpy as np





def create_dataframes(data_output):
    """
    Outputting dataframe to csv
    """
    df = pd.DataFrame(data_output)
    df.index.name = 'Frame'
    print("Saving database for {}".format(video_name))
    df.to_csv(OUT_DIR + '/DataOutput{}'.format(video_name) + '-' + time.strftime('%d%m%Y-%H%M%S') + '.csv', sep=';')


print("Loading the video")
video = cv2.VideoCapture(r"D:\DeepPoseKit\Videos\video_fly.avi")
name = os.path.split(VIDEO_SOURCE)[-1]
video_name = "{}".format(name.split('.')[0])
resolution = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
animals_list = ["Animal{}".format(num) for num in range(1, ANIMALS_NUMBER + 1)]

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
output_file = os.path.join(OUT_DIR, "Analysed_" + video_name + time.strftime('%d%m%Y-%H%M%S') + ".avi")
video_file = cv2.VideoWriter(output_file, fourcc, 30, resolution)

def plot_dlc_bodyparts(image, bodyparts):
    """
    Plots dlc bodyparts on given image
    adapted from plotter
    """

    for bp in bodyparts:
        center = tuple(bp.astype(int))
        cv2.circle(image, center=center, radius=3, color=(255, 0, 0), thickness=2)
    return image

def start_videoanalyser():
    print("Starting DeepPoseKit")
    path_model = r"D:\DeepPoseKit\models\Fly\best_model_densenet.h5"
    model = load_model(path_model)


    experiment_enabled = False
    video_output = True

    if experiment_enabled:
        print("Initializing experiment")
        experiment = ExampleExperiment()
        experiment.start_experiment()

    # some variables initialization
    all_rows = []
    index = 0

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frame = frame = frame[..., 1][..., None]
            predictions = model.predict(frame, verbose = 1)
            # scmap, locref, pose = get_pose(frame, config, sess, inputs, outputs)
            # peaks = find_local_peaks_new(scmap, locref, ANIMALS_NUMBER, config)
            # skeletons = calculate_skeletons(peaks, ANIMALS_NUMBER)
            # if skeletons:
            #     for skeleton in skeletons:
            #         if experiment_enabled:
            #             result, response = experiment.check_skeleton(frame, skeleton)
            #             plot_triggers_response(frame, response)
            #     out_frame = plot_bodyparts(frame, skeletons)
            # else:
            x,y,confidence = np.split(predictions,3,-1)
            plot_dlc_bodyparts(predictions[:,:3])
            out_frame = frame
            cv2.imshow('stream', out_frame)
            if video_output:
                video_file.write(out_frame)
            if experiment_enabled:
                all_rows.append(create_row(index, skeletons, experiment_enabled, experiment.get_trial()))
            else:
                all_rows.append(create_row(index, skeletons, experiment_enabled, None))
            index += 1
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if experiment_enabled:
        experiment.stop_experiment()
    if video_output:
        print('Saving analyzed video for {}'.format(video_name))
        video_file.release()
    video.release()
    create_dataframes(all_rows)


if __name__ == "__main__":
    start_videoanalyser()
