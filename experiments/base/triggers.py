"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""
from experiments.utils.exp_setup import get_trigger_settings
from utils.analysis import (
    angle_between_vectors,
    calculate_distance,
    EllipseROI,
    RectangleROI,
)

"""BaseTrigger class"""


class BaseTrigger:
    """Base class for standard triggers"""

    def __init__(self):
        """
        Initialising trigger with following parameters:
        """

        self._name = "BaseTrigger"
        self._settings_dict = {}
        self._debug = True

    def check_skeleton(self, skeleton: dict):
        """
        Checking skeleton for trigger
        :param skeleton: a skeleton dictionary, returned by calculate_skeletons() from poser file
        :return: response, a tuple of result (bool) and response body
        Response body is used for plotting and outputting results to trials dataframes
        """

        result = True

        color = (0, 255, 0) if result else (0, 0, 255)
        if self._debug:
            center = skeleton[self._bodyparts[1]]

            response_body = {
                "plot": {
                    "text": dict(
                        text="BaseTrigger",
                        org=skeleton[self._bodyparts[1]],
                        color=color,
                    ),
                }
            }
        else:
            response_body = None

        response = (result, response_body)
        return response

    def get_name(self):
        return self._name

    def get_settings(self):
        return self._settings_dict


"""Single posture triggers"""


class BaseHeaddirectionTrigger(BaseTrigger):
    """Trigger to check if animal is turning head in a specific angle to a reference point"""

    def __init__(self):
        """
        Initialising trigger with following parameters:
        :param int angle: angle to meet for condition
        :param tuple point: point used as reference to measure headdirection angle

        """
        super().__init__()
        self._name = "BaseHeaddirectionTrigger"

        # loading settings
        self._trigger_parameter_dict = dict(
            POINT="tuple", ANGLE="int", BODYPARTS="list", DEBUG="boolean"
        )
        self._settings_dict = get_trigger_settings(self._name, self._trigger_parameter_dict)

        self._point = self._settings_dict["POINT"]
        self._angle = self._settings_dict["ANGLE"]
        self._bodyparts = self._settings_dict["BODYPARTS"]
        self._debug = (
            self._settings_dict["DEBUG"]
            if self._settings_dict["DEBUG"] is not None
            else False
        )

    def check_skeleton(self, skeleton: dict):
        """
        Checking skeleton for trigger
        :param skeleton: a skeleton dictionary, returned by calculate_skeletons() from poser file
        :return: response, a tuple of result (bool) and response body
        Response body is used for plotting and outputting results to trials dataframes
         [point , 'neck', 'nose'] you need to pass this to angle between vectors to get headdirection
        """
        ret_head_dir, angle = angle_between_vectors(
            *skeleton[self._bodyparts[0]], *skeleton[self._bodyparts[1]], *self._point
        )
        true_angle = abs(angle)

        if true_angle <= self._angle:
            result = True
        else:
            result = False

        color = (0, 255, 0) if result else (0, 0, 255)
        if self._debug:
            center = skeleton[self._bodyparts[1]]

            response_body = {
                "plot": {
                    "text": dict(
                        text=str(true_angle),
                        org=skeleton[self._bodyparts[1]],
                        color=(255, 255, 255),
                    ),
                    "circle": dict(center=center, radius=5, color=color),
                }
            }
        else:
            response_body = {"angle": true_angle}

        response = (result, response_body)
        return response


class BaseEgoHeaddirectionTrigger(BaseTrigger):
    """Trigger to check if animal is turning head in a specific angle and egocentric direction"""

    def __init__(self):
        """
        Initialising trigger with following parameters:
        :param int angle: angle to meet for condition
        :param str head_dir: head direction from egocentric position of the animal (left, right or both)

        """
        super().__init__()
        self._name = "BaseEgoHeaddirectionTrigger"

        # loading settings
        self._trigger_parameter_dict = dict(
            ANGLE="int", HEADDIRECTION="str", BODYPARTS="list", DEBUG="boolean"
        )
        self._settings_dict = get_trigger_settings(self._name, self._trigger_parameter_dict)

        self._point = self._settings_dict["POINT"]
        self._angle = self._settings_dict["ANGLE"]
        self._bodyparts = self._settings_dict["BODYPARTS"]
        self._debug = (
            self._settings_dict["DEBUG"]
            if self._settings_dict["DEBUG"] is not None
            else False
        )

    def check_skeleton(self, skeleton: dict):
        """
        Checking skeleton for trigger
        :param skeleton: a skeleton dictionary, returned by calculate_skeletons() from poser file
        :return: response, a tuple of result (bool) and response body
        Response body is used for plotting and outputting results to trials dataframes
         ['neck', 'nose', 'tailroot'] you need to pass this to angle between vectors to get headdirection
        """
        ret_head_dir, angle = angle_between_vectors(
            *skeleton[self._bodyparts[0]],
            *skeleton[self._bodyparts[1]],
            *skeleton[self._bodyparts[2]]
        )
        true_angle = abs(angle)

        if true_angle <= self._angle:
            if self._head_dir == ret_head_dir:
                result = True
            elif self._head_dir == "both":
                result = True
        else:
            result = False

        color = (0, 255, 0) if result else (0, 0, 255)
        if self._debug:
            center = skeleton[self._bodyparts[1]]

            response_body = {
                "plot": {
                    "text": dict(
                        text=str(true_angle),
                        org=skeleton[self._bodyparts[1]],
                        color=(255, 255, 255),
                    ),
                    "circle": dict(center=center, radius=5, color=color),
                }
            }
        else:
            response_body = {"angle": true_angle}

        response = (result, response_body)
        return response


class BaseScreenTrigger(BaseTrigger):
    """
    Trigger to check if animal is looking at the screen
    """

    def __init__(self):
        """
        Initialising trigger with following parameters:
        :param direction: a direction where the screen is located in the stream or video.
        All possible directions: 'North' (or top of the frame), 'East' (right), 'South' (bottom), 'West' (left)
        Note that directions are not tied to real-world cardinal directions
        :param angle: angle, at which animal is considered looking at the screen
        :param bodyparts: a pair of joints of animal (tuple or list) that represent 'looking vector' like (start, end)
        For example,
         ('neck', 'nose') pair would mean that direction in which animal is looking defined by vector from neck to nose
        """
        from utils.configloader import RESOLUTION

        super().__init__()
        self._name = "BaseScreenTrigger"
        max_x, max_y = RESOLUTION
        direction_dict = {
            "North": (int(max_x / 2), 0),
            "South": (int(max_x / 2), max_y),
            "West": (0, int(max_y / 2)),
            "East": (max_x, int(max_y / 2)),
        }

        # loading settings
        self._trigger_parameter_dict = dict(
            ANGLE="int", DIRECTION="str", BODYPARTS="list", DEBUG="boolean"
        )
        self._settings_dict = get_trigger_settings(self._name, self._trigger_parameter_dict)

        self._direction = self._settings_dict["DIRECTION"]
        self._point = direction_dict[self._direction]
        self._angle = self._settings_dict["ANGLE"]
        self._bodyparts = self._settings_dict["BODYPARTS"]
        self._debug = (
            self._settings_dict["DEBUG"]
            if self._settings_dict["DEBUG"] is not None
            else False
        )

    def check_skeleton(self, skeleton: dict):
        """
        Checking skeleton for trigger
        :param skeleton: a skeleton dictionary, returned by calculate_skeletons() from poser file
        :return: response, a tuple of result (bool) and response body
        Response body is used for plotting and outputting results to trials dataframes
        """
        direction_x, direction_y = self._point
        head_dir, angle = angle_between_vectors(
            direction_x,
            direction_y,
            *skeleton[self._bodyparts[0]],
            *skeleton[self._bodyparts[1]]
        )
        true_angle = 180 - abs(angle)

        result = true_angle <= self._angle

        color = (0, 255, 0) if result else (0, 0, 255)
        if self._debug:
            response_body = {
                "plot": {
                    "line": dict(
                        pt1=skeleton[self._bodyparts[0]], pt2=self._point, color=color
                    ),
                    "text": dict(
                        text=str(true_angle),
                        org=skeleton[self._bodyparts[0]],
                        color=(255, 255, 255),
                    ),
                }
            }
        else:
            response_body = {"angle": true_angle}

        response = (result, response_body)
        return response


class BaseRegionTrigger(BaseTrigger):
    """
    Trigger to check if animal is in Region Of Interest (ROI)
    """

    def __init__(self):
        """
        Initialising trigger with following parameters:
        :param region_type: type of a ROI
        Currently available ROIs are 'square' and 'circle'
        Both of them use the same scheme to define coordinates
        :param center: center of a ROI
        :param radius: radius of a ROI
        For circle - literally radius of a circle
        For square - half of a side
        :param bodyparts: joint or a list of joints for which we are checking the ROI
        """

        super().__init__()
        self._name = "BaseRegionTrigger"

        # loading settings
        self._trigger_parameter_dict = dict(
            CENTER="tuple", TYPE="str", RADIUS="int", BODYPARTS="list", DEBUG="boolean"
        )
        self._settings_dict = get_trigger_settings(
            self._name, self._trigger_parameter_dict
        )

        region_types = {"circle": EllipseROI, "square": RectangleROI}
        self._region_of_interest = region_types[self._settings_dict["TYPE"].lower()](
            self._settings_dict["CENTER"],
            self._settings_dict["RADIUS"],
            self._settings_dict["RADIUS"],
        )

        self._bodyparts = (
            self._settings_dict["BODYPARTS"]
            if len(self._settings_dict["BODYPARTS"]) > 1
            else self._settings_dict["BODYPARTS"][0]
        )

        self._debug = (
            self._settings_dict["DEBUG"]
            if self._settings_dict["DEBUG"] is not None
            else False
        )

    def check_skeleton(self, skeleton: dict):
        """
        Checking skeleton for trigger
        :param skeleton: a skeleton dictionary, returned by calculate_skeletons() from poser file
        :return: response, a tuple of result (bool) and response body
        Response body is used for plotting and outputting results to trials dataframes
        """
        if isinstance(self._bodyparts, list):
            results = []
            for part in self._bodyparts:
                bp_x, bp_y = skeleton[part]
                results.append(self._region_of_interest.check_point(bp_x, bp_y))
            result = any(results)
        else:
            bp_x, bp_y = skeleton[self._bodyparts]
            result = self._region_of_interest.check_point(bp_x, bp_y)

        color = (0, 255, 0) if result else (0, 0, 255)

        if self._roi_type == "circle":
            response_body = {
                "plot": {
                    "circle": dict(
                        center=self._region_of_interest.get_center(),
                        radius=int(self._region_of_interest.get_x_radius()),
                        color=color,
                    )
                }
            }
        elif self._roi_type == "square":
            box = self._region_of_interest.get_box()
            x1, y1, x2, y2 = box
            pt1 = (x1, y2)
            pt2 = (x2, y1)
            response_body = {"plot": {"square": dict(pt1=pt1, pt2=pt2, color=color)}}

        response = (result, response_body)
        return response


class BaseOutsideRegionTrigger(BaseRegionTrigger):
    """
    Trigger to check if animal is out of the Region Of Interest (ROI)
    """

    def __init__(self):
        """
        Initialising trigger with following parameters:
        :param region_type: type of a ROI
        Currently available ROIs are 'square' and 'circle'
        Both of them use the same scheme to define coordinates
        :param center: center of a ROI
        :param radius: radius of a ROI
        For circle - literally radius of a circle
        For square - half of a side
        :param bodyparts: joint or a list of joints for which we are checking the ROI
        """
        super().__init__()

    def check_skeleton(self, skeleton: dict):
        """
        Checking skeleton for trigger
        :param skeleton: a skeleton dictionary, returned by calculate_skeletons() from poser file
        :return: response, a tuple of result (bool) and response body
        Response body is used for plotting and outputting results to trials dataframes
        """
        result, response_body = super().check_skeleton(skeleton)
        response = (not result, response_body)  # flips result bool
        return response


"""Combination triggers"""


class BaseHeaddirectionROITrigger(BaseTrigger):
    """Trigger to check if animal is turning its head in a specific angle to a reference point (center of the region of interest)
    and if the animal is not in the ROI"""

    def __init__(
        self, center: tuple, radius: int, angle: float = 45, debug: bool = False
    ):
        """
        Initialising trigger with following parameters:
        :param int angle: angle to meet for condition
        :param tuple center: point used as reference to measure headdirection angle and center of ROI
        :param int radius: radius of the ROI
        :param debug: Not used in this trigger
        """

        super().__init__()
        self._name = "BaseHeaddirectionROITrigger"

        # loading settings
        self._trigger_parameter_dict = dict(
            CENTER="tuple",
            TYPE="str",
            RADIUS="int",
            POINT="tuple",
            ANGLE="int",
            ROI_BODYPARTS="list",
            ANGLE_BODYPARTS="list",
            DEBUG="boolean",
        )
        self._settings_dict = get_trigger_settings(
            self._name, self._trigger_parameter_dict
        )

        region_types = {"circle": EllipseROI, "square": RectangleROI}
        self._region_of_interest = region_types[self._settings_dict["TYPE"].lower()](
            self._settings_dict["CENTER"],
            self._settings_dict["RADIUS"],
            self._settings_dict["RADIUS"],
        )

        self._roi_bodyparts = (
            self._settings_dict["BODYPARTS"]
            if len(self._settings_dict["BODYPARTS"]) > 1
            else self._settings_dict["BODYPARTS"][0]
        )

        self._point = self._settings_dict["POINT"]
        self._angle = self._settings_dict["ANGLE"]
        self._angle_bodyparts = self._settings_dict["ANGLE_BODYPARTS"]

        self._debug = (
            self._settings_dict["DEBUG"]
            if self._settings_dict["DEBUG"] is not None
            else False
        )

    def check_skeleton(self, skeleton: dict):
        """
        Checking skeleton for trigger
        :param skeleton: a skeleton dictionary, returned by calculate_skeletons() from poser file
        :return: response, a tuple of result (bool) and response body
        Response body is used for plotting and outputting results to trials dataframes
        """
        if isinstance(self._roi_bodyparts, list):
            roi_result = []
            for part in self._roi_bodyparts:
                bp_x, bp_y = skeleton[part]
                roi_result.append(self._region_of_interest.check_point(bp_x, bp_y))
            roi_result = any(roi_result)
        else:
            bp_x, bp_y = skeleton[self._roi_bodyparts]
            roi_result = self._region_of_interest.check_point(bp_x, bp_y)

        ret_head_dir, angle = angle_between_vectors(
            *skeleton[self._angle_bodyparts[0]],
            *skeleton[self._angle_bodyparts[1]],
            *self._point
        )
        true_angle = abs(angle)

        if true_angle <= self._angle:
            angle_result = True
        else:
            angle_result = False

        if angle_result and roi_result:
            result = True
        else:
            result = False

        color = (0, 255, 0) if result else (0, 0, 255)

        if self._roi_type == "circle":
            response_body = {
                "plot": {
                    "circle": dict(
                        center=self._region_of_interest.get_center(),
                        radius=int(self._region_of_interest.get_x_radius()),
                        color=color,
                    ),
                    "text": dict(
                        text=str(true_angle),
                        org=skeleton[self._angle_bodyparts[1]],
                        color=(255, 255, 255),
                    ),
                }
            }
        elif self._roi_type == "square":
            box = self._region_of_interest.get_box()
            x1, y1, x2, y2 = box
            pt1 = (x1, y2)
            pt2 = (x2, y1)
            response_body = {
                "plot": {
                    "square": dict(pt1=pt1, pt2=pt2, color=color),
                    "text": dict(
                        text=str(true_angle),
                        org=skeleton[self._angle_bodyparts[1]],
                        color=(255, 255, 255),
                    ),
                }
            }

        response = (result, response_body)
        return response


"""Posture sequence triggers"""


class BaseSpeedTrigger(BaseTrigger):
    """
    Trigger to check if animal is moving above a certain speed
    """

    def __init__(self):
        """
        Initializing trigger with given threshold
        :param threshold: int in pixel how much of a movement does not count
        :param bodypart: str or list of str, bodypart or list of bodyparts in skeleton to use for trigger,
         if "any" will check if any bodypart reaches treshold; default "any"
        For example threshold of 5 would mean that all movements less then 5 pixels would be ignored
        """
        super().__init__()
        self._name = "BaseRegionTrigger"

        # loading settings
        self._trigger_parameter_dict = dict(
            THRESHOLD="float", BODYPARTS="list", DEBUG="boolean"
        )
        self._settings_dict = get_trigger_settings(self._name, self._trigger_parameter_dict)

        region_types = {"circle": EllipseROI, "square": RectangleROI}
        self._region_of_interest = region_types[self._settings_dict["TYPE"].lower()](
            self._settings_dict["CENTER"],
            self._settings_dict["RADIUS"],
            self._settings_dict["RADIUS"],
        )

        self._threshold = self._settings_dict["THRESHOLD"]

        self._bodyparts = (
            self._settings_dict["BODYPARTS"]
            if len(self._settings_dict["BODYPARTS"]) > 1
            else self._settings_dict["BODYPARTS"][0]
        )

        self._debug = (
            self._settings_dict["DEBUG"]
            if self._settings_dict["DEBUG"] is not None
            else False
        )

        self._skeleton = None

    def check_skeleton(self, skeleton: dict):
        """
        Checking skeleton for trigger
        :param skeleton: a skeleton dictionary, returned by calculate_skeletons() from poser file
        :return: response, a tuple of result (bool) and response body
        Response body is used for plotting and outputting results to trials dataframes
        """
        # choosing a point to draw near the skeleton
        org_point = skeleton[list(skeleton.keys())[0]]
        joint_moved = []
        if self._skeleton is None:
            result = False
            text = "First frame"
            self._skeleton = skeleton
        else:
            if self._bodypart is "any":
                for joint in skeleton:
                    joint_travel = calculate_distance(
                        skeleton[joint], self._skeleton[joint]
                    )
                    joint_moved.append(abs(joint_travel) >= self._threshold)

            elif isinstance(self._bodypart, list):
                for joint in self._bodypart:
                    joint_travel = calculate_distance(
                        skeleton[joint], self._skeleton[joint]
                    )
                    joint_moved.append(abs(joint_travel) >= self._threshold)
            else:
                joint_travel = calculate_distance(
                    skeleton[self._bodypart], self._skeleton[self._bodypart]
                )
                joint_moved.append(abs(joint_travel) >= self._threshold)

            if all(joint_moved):
                result = True
                text = "Running"
            else:
                result = False
                text = ""
            self._skeleton = skeleton

        color = (0, 255, 0) if result else (0, 0, 255)
        response_body = {"plot": {"text": dict(text=text, org=org_point, color=color)}}
        response = (result, response_body)
        return response


class BaseFreezeTrigger(BaseSpeedTrigger):
    """
    Trigger to check if animal is moving below a certain speed
    """

    def __init__(self):
        """
        Initializing trigger with given threshold
        :param threshold: int in pixel how much of a movement does not count
        :param bodypart: str or list of str, bodypart or list of bodyparts in skeleton to use for trigger,
         if "any" will check if any bodypart reaches treshold; default "any"
        For example threshold of 5 would mean that all movements higher then 5 pixels would be ignored
        """
        super().__init__()

    def check_skeleton(self, skeleton: dict):
        """
        Checking skeleton for trigger
        :param skeleton: a skeleton dictionary, returned by calculate_skeletons() from poser file
        :return: response, a tuple of result (bool) and response body
        Response body is used for plotting and outputting results to trials dataframes
        """
        result, response_body = super().check_skeleton(skeleton)
        result = not result
        color = (0, 255, 0) if result else (0, 0, 255)
        if result:
            text = "Not Moving"
        else:
            text = ""
        response_body = {
            "plot": {
                "text": dict(text=text, org=skeleton[super()._bodypart[0]], color=color)
            }
        }

        response = (result, response_body)  # flips result bool
        return response
