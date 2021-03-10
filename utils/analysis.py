# -*- coding: utf-8 -*-
"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""


import os
from typing import Union, List, Tuple
import numpy as np
import pandas as pd
import math


class ROI:
    """
    Creating a ROI with given parameters
    center - a tuple with (x,y) coordinates
    h - radius by Y-axis
    k - radius by X-axis
    name (optional) - name for ROI
    """

    def __init__(self, center: tuple, h: int, k: int, name: str = "ROI"):
        self._name = name
        self._x_center, self._y_center = center
        self._x_radius = k
        self._y_radius = h

        # creating a coordinates box
        self._box = [
            self._x_center - self._x_radius,
            self._y_center - self._y_radius,
            self._x_center + self._x_radius,
            self._y_center + self._y_radius,
        ]

    def get_box(self):
        """
        Returns box coordinates as [x1, y1, x2, y2]
        """
        return self._box

    def get_x_radius(self):
        """
        Returns x_radius
        """
        return self._x_radius

    def get_y_radius(self):
        """
        Returns y_radius
        """
        return self._y_radius

    def get_center(self):
        """
        Returns center coordinates as x, y
        """
        return self._x_center, self._y_center

    def get_name(self):
        """
        Returns ROI name
        """
        return self._name

    def set_name(self, name: str):
        """
        Returns ROI name
        """
        self._name = name


class RectangleROI(ROI):
    """
    Creating a rectangle ROI with given parameters
    center - a tuple with (x,y) coordinates
    h - radius by Y-axis
    k - radius by X-axis
    name (optional) - name for ROI
    """

    def __init__(self, center: tuple, h: int, k: int, name: str = "RectangleROI"):
        super().__init__(center, h, k, name)

    def check_point(self, x: int, y: int):
        """
        Checking if point with given coordinates x,y is inside ROI
        Returns True or False
        """
        check = (-self._x_radius <= x - self._x_center <= self._x_radius) and (
            -self._y_radius <= y - self._y_center <= self._y_radius
        )
        return check


class EllipseROI(ROI):
    """
    Creating a ROI ellipse with given parameters
    center - a tuple with (x,y) coordinates
    h - radius by Y-axis
    k - radius by X-axis
    name (optional) - name for ROI
    """

    def __init__(self, center: tuple, h: int, k: int, name: str = "EllipseROI"):
        super().__init__(center, h, k, name)

    def check_point(self, x: int, y: int):
        """
        Checking if point with given coordinates x,y is inside ROI
        Returns True or False
        """
        check = ((x - self._x_center) ** 2 / self._x_radius ** 2) + (
            (y - self._y_center) ** 2 / self._y_radius ** 2
        )
        return check <= 1


def calculate_distance(point1: tuple, point2: tuple) -> float:
    """
    Calculates distance between two points (x1,y1) and (x2,y2)
    """
    if np.isnan(point1).any() or np.isnan(point2).any():
        return np.nan
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return distance


def calculate_distance_for_bodyparts(
    dataframe: pd.DataFrame, body_parts: Union[List[str], str]
) -> List[pd.Series]:
    """
    Calculating distances traveled for each frame for desired body parts
    :param dataframe DataFrame: dataframe to calculate distances on
    Should have columns with X and Y coordinates of desired body_parts
    :param body_parts str or list of str: part or parts to calculate distances for
    Can be either string or list of strings
    :return list: returns list of pd.Series with distances for each bodypart
    """
    df = dataframe
    # creating temporary dataframe for calculations
    temp_df = pd.DataFrame()
    # creating empty list for results
    results = []

    def distance_func(row, bodypart):
        """
        Function to actually calculate distance
        """
        return math.sqrt(
            row["{}_travel_X".format(bodypart)] ** 2
            + row["{}_travel_Y".format(bodypart)] ** 2
        )

    def calc_distance(bodypart):
        """
        Function to create temporary dataframe columns and do calculations on them
        Then append the resulting series to results list
        """
        temp_df["{}_travel_X".format(bodypart)] = (
            df["{}_X".format(bodypart)].diff().astype(float)
        )
        temp_df["{}_travel_Y".format(bodypart)] = (
            df["{}_Y".format(bodypart)].diff().astype(float)
        )
        results.append(temp_df.apply(distance_func, axis=1, args=(bodypart,)))

    # checking if provided body_parts is list or not
    if isinstance(body_parts, list):
        # if list, calculate for every body part in it
        for part in body_parts:
            calc_distance(part)
    else:
        # if not, calculate for one provided part
        calc_distance(body_parts)
    return results


def calculate_speed_for_bodyparts(
    dataframe: pd.DataFrame, body_parts: Union[List[str], str]
) -> List[pd.Series]:
    """
    Calculating speed in pixels per seconds for each frame for desired body parts
    :param dataframe DataFrame: dataframe to calculate speeds on
    Should have columns distances travelled for each desired body part
    :param body_parts str or list of str: part or parts to calculate distances for
    Can be either string or list of strings
    :return list: returns list of pd.Series with speeds for each bodypart
    """
    df = dataframe
    # creating temporary dataframe for calculations
    temp_df = pd.DataFrame()
    # calculating time differences between each frame
    temp_df["Time_diff"] = df["Time"].diff().astype(float)
    # creating empty list for results
    results = []

    def speed_func(row, bodypart):
        """
        Function to actually calculate speed
        """
        if row["Time_diff"] != 0:
            return row["distance_{}".format(bodypart)] / row["Time_diff"]
        else:
            return np.nan

    def check_for_distance(bodypart):
        """
        Check if column with distance for desired body part exists in provided dataframe
        If true, copy it to temp_df
        Otherwise, raise ValueError exception
        """
        if "distance_{}".format(bodypart) in df.columns:
            temp_df["distance_{}".format(bodypart)] = df["distance_{}".format(bodypart)]
        else:
            raise ValueError(
                "Distances travelled should be calculated beforehand for each bodypart"
            )

    # checking if provided body_parts is list or not
    if isinstance(body_parts, list):
        # if list, calculate for every body part in it
        for part in body_parts:
            check_for_distance(part)
            results.append(temp_df.apply(speed_func, axis=1, args=(part,)))
    else:
        # if not, calculate for one provided part
        check_for_distance(body_parts)
        results.append(temp_df.apply(speed_func, axis=1, args=(body_parts,)))
    return results


def angle_between_vectors(
    xa: int, ya: int, xb: int, yb: int, xc: int, yc: int
) -> Tuple[str, float]:
    """
    Calculating angle between vectors, defined by coordinates
    Returns angle and direction (left, right, forward or backward)
    *ISSUE* - if y axis is reversed, directions would also be reversed
    """
    # using atan2() formula for both vectors
    dir_ab = math.atan2(ya - yb, xa - xb)
    dir_bc = math.atan2(yb - yc, xb - xc)

    # angle between vectors in radians
    rad_angle = dir_ab - dir_bc
    pi = math.pi

    # converting to degrees
    angle = rad_angle
    if pi < angle:
        angle -= 2 * pi
    elif -pi > angle:
        angle += 2 * pi
    angle = math.degrees(angle)

    # defining the direction
    if 180 > angle > 0:
        direction = "left"
    elif -180 < angle < 0:
        direction = "right"
    elif abs(angle) == 180:
        direction = "backwards"
    else:
        direction = "forward"

    return direction, angle


## miscellaneous ##
def cls():
    os.system("cls" if os.name == "nt" else "clear")
