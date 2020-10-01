"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""

import cv2
from deepposekit.models import load_model
from deepposekit.io import VideoReader
import numpy as np
import tensorflow as tf



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
    video = cv2.VideoCapture(r"D:\DeepPoseKit-Data-master\datasets\fly\video.avi")
    resolution = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    path_model = r"D:\DeepPoseKit-Data-master\datasets\fly\best_model_densenet.h5"
    model = load_model(path_model)


    predict_model = model.predict_model
    # predict_model.layers.pop(0)  # remove current input layer
    #
    # inputs = tf.keras.layers.Input((resolution[0], resolution[1], 3))
    # outputs = predict_model(inputs)
    # predict_model = tf.keras.Model(inputs, outputs)


    experiment_enabled = False

    index = 0
    while video.isOpened():
        ret, frame = video.read()
        if ret is not None:
            org_frame = frame
            frame = frame[..., 1][..., None]
            st_frame = np.stack([frame])
            prediction = predict_model.predict(st_frame, batch_size= 1,  verbose=True)
            x, y, confidence = np.split(prediction, 3, -1)

            print(prediction.shape)
            predi = prediction[0,:,:2]
            pre = predi[:, :2]
            print(pre)
            out_frame = plot_dlc_bodyparts(org_frame, predi)
            # out_frame = org_frame
            cv2.imshow('stream', out_frame)
            index += 1
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()


if __name__ == "__main__":
    start_videoanalyser()