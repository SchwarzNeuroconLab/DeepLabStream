"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""

import cv2
import numpy as np


def plot_dots(image, coordinates, color, cond=False):
    """
    Takes the image and positional arguments from pose to plot corresponding dot
    Returns the resulting image
    """
    cv2.circle(image, coordinates, 3, color, -1)
    if cond:
        cv2.circle(image, (10, 10), 10, (0, 255, 0), -1)
    return image


def plot_bodyparts(image, skeletons):
    """
    Takes the image and skeletons list to plot them
    :return: resulting image
    """
    res_image = image.copy()
    # predefined colors list
    colors_list = [(0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255), (255, 255, 0), (255, 255, 128),
                   (0, 0, 128), (0, 128, 0), (0, 128, 128), (0, 128, 255), (0, 255, 128), (128, 0, 0), (128, 0, 128),
                    (128, 0, 255), (128, 128, 0), (128, 128, 128), (128, 128, 255), (128, 255, 0), (128, 255, 128),
                    (128, 255, 255), (255, 0, 128), (255, 128, 0), (255, 128, 128), (255, 128, 255)]
    #color = (255, 0, 0)

    for num, animal in enumerate(skeletons):
        bodyparts = animal.keys()
        bp_count = len(bodyparts)
        #colors = dict(zip(bodyparts, colors_list[:bp_count]))
        for part in animal:
            #check for NaNs and skip
            if not any(np.isnan(animal[part])):
                plot_dots(res_image, tuple(map(int, animal[part])), colors_list[num])
                #plot_dots(res_image, tuple(animal[part]), colors[part])
    return res_image


def plot_metadata_frame(image, frame_width, frame_height, current_fps, current_elapsed_time):
    """
    Takes the image and plots metadata
    :return: resulting image
    """
    res_image = image.copy()
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(res_image, 'Time: ' + str(round(current_elapsed_time, 2)),
                (int(frame_width * 0.8), int(frame_height * 0.9)), font, 1, (255, 255, 0))
    cv2.putText(res_image, 'FPS: ' + str(round(current_fps, 1)),
                (int(frame_width * 0.8), int(frame_height * 0.95)), font, 1, (255, 255, 0))
    return res_image

def plot_dlc_bodyparts(image, bodyparts):
    """
    Plots dlc bodyparts on given image
    adapted from plotter
    """

    for bp in bodyparts:
        center = tuple(bp.astype(int))
        cv2.circle(image, center=center, radius=3, color=(255, 0, 0), thickness=2)
    return image

def plot_triggers_response(image, response):
    """
    Plots trigger response on given image
    """
    if 'plot' in response:
        plot = response['plot']
        if 'line' in plot:
            cv2.line(image, **plot['line'], thickness=4)
        if 'text' in plot:
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(image, **plot['text'], fontFace=font, fontScale=1)
        if 'circle' in plot:
            # if result2:
            cv2.circle(image, **plot['circle'], thickness=2)
        if 'square' in plot:
            cv2.rectangle(image, **plot['square'], thickness=2)



