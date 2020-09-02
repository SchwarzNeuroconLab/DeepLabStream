"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""


from utils.analysis import angle_between_vectors, calculate_distance, EllipseROI, RectangleROI
from utils.configloader import RESOLUTION

"""Single posture triggers"""

class HeaddirectionROITrigger:
    """Trigger to check if animal is turning its head in a specific angle to a reference point (center of the region of interest)
    and if the animal is not in the ROI"""
    def __init__(self, center: tuple, radius: int, angle: float = 45, debug: bool = False):
        """
        Initialising trigger with following parameters:
        :param int angle: angle to meet for condition
        :param tuple center: point used as reference to measure headdirection angle and center of ROI
        :param int radius: radius of the ROI
        :param debug: Not used in this trigger
         """

        self._headdirection_trigger = HeaddirectionTrigger(angle, center)
        self._region_trigger = RegionTrigger(region_type= 'circle', center= center, radius = radius, bodyparts= 'nose')
        self._center = center
        self._angle = angle
        self._radius = radius
        self._debug = debug

    def check_skeleton(self, skeleton: dict):
        """
        Checking skeleton for trigger
        :param skeleton: a skeleton dictionary, returned by calculate_skeletons() from poser file
        :return: response, a tuple of result (bool) and response body
        Response body is used for plotting and outputting results to trials dataframes
         [point , 'neck', 'nose'] is used for headdirection
        """
        _ , angle = angle_between_vectors(*skeleton['neck'], *skeleton['nose'], *self._center)
        true_angle = abs(angle)

        result_head, _ = self._headdirection_trigger.check_skeleton(skeleton)
        result_roi, _ = self._region_trigger.check_skeleton(skeleton)

        if result_head is True and result_roi is True:
            result = True
        else:
            result = False

        color = (0, 255, 0) if result else (0, 0, 255)

        if self._debug:
            point = skeleton['nose']

            response_body = {'plot': {'text': dict(text=str(true_angle),
                                                   org=point,
                                                   color=(255, 255, 255)),
                                      'circle': dict(center= self._center,
                                                     radius= self._radius,
                                                     color=color)
                             }}
        else:
            response_body = {'angle': true_angle}

        response = (result, response_body)
        return response


class HeaddirectionTrigger:
    """Trigger to check if animal is turning head in a specific angle to a reference point"""
    def __init__(self, angle: int, point: tuple = (0,0), debug: bool = False):
        """
        Initialising trigger with following parameters:
        :param int angle: angle to meet for condition
        :param tuple point: point used as reference to measure headdirection angle

         """
        self._point = point
        self._angle = angle
        self._debug = debug

    def check_skeleton(self, skeleton: dict):
        """
        Checking skeleton for trigger
        :param skeleton: a skeleton dictionary, returned by calculate_skeletons() from poser file
        :return: response, a tuple of result (bool) and response body
        Response body is used for plotting and outputting results to trials dataframes
         [point , 'neck', 'nose'] you need to pass this to angle between vectors to get headdirection
        """
        ret_head_dir, angle = angle_between_vectors(*skeleton['neck'], *skeleton['nose'], *self._point)
        true_angle = abs(angle)

        if true_angle <= self._angle:
            result = True
        else:
            result = False


        color = (0, 255, 0) if result else (0, 0, 255)
        if self._debug:
            center = skeleton['nose']

            response_body = {'plot': {'text': dict(text=str(true_angle),
                                                   org=skeleton[self._end_point],
                                                   color=(255, 255, 255)),
                                      'circle': dict(center= center,
                                                     radius= 5,
                                                     color=color)
                             }}
        else:
            response_body = {'angle': true_angle}

        response = (result, response_body)
        return response

class EgoHeaddirectionTrigger:
    """Trigger to check if animal is turning head in a specific angle and egocentric direction"""
    def __init__(self, angle: int, head_dir: str = 'both', debug: bool = False):
        """
        Initialising trigger with following parameters:
        :param int angle: angle to meet for condition
        :param str head_dir: head direction from egocentric position of the animal (left, right or both)

         """
        self._head_dir = head_dir
        self._angle = angle
        self._debug = debug

    def check_skeleton(self, skeleton: dict):
        """
        Checking skeleton for trigger
        :param skeleton: a skeleton dictionary, returned by calculate_skeletons() from poser file
        :return: response, a tuple of result (bool) and response body
        Response body is used for plotting and outputting results to trials dataframes
         ['tailroot', 'neck', 'nose'] you need to pass this to angle between vectors to get headdirection
        """
        tailroot_x, tailroot_y = skeleton['tailroot']
        neck_x, neck_y = skeleton['neck']
        nose_x, nose_y = skeleton['nose']
        ret_head_dir, angle = angle_between_vectors(tailroot_x, tailroot_y, neck_x, neck_y , nose_x, nose_y)
        true_angle = 180 - abs(angle)

        if true_angle <= self._angle:
            if self._head_dir == ret_head_dir:
                result = True
            elif self._head_dir == 'both':
                result = True
        else:
            result = False


        color = (0, 255, 0) if result else (0, 0, 255)
        if self._debug:
            center = (nose_x, nose_y)

            response_body = {'plot': {'text': dict(text=str(true_angle),
                                                   org=skeleton[self._end_point],
                                                   color=(255, 255, 255)),
                                      'circle': dict(center= center,
                                                     radius= 5,
                                                     color=color)
                             }}
        else:
            response_body = {'angle': true_angle}

        response = (result, response_body)
        return response



class DirectionTrigger:
    """
    Trigger to check if animal is looking in direction of some point
    """
    def __init__(self, point: tuple, angle: int, bodyparts: iter, debug: bool = False):
        """
        Initialising trigger with following parameters:
        :param tuple point: a point of interest in (x,y) format.
        :param int angle: angle, at which animal is considered looking at the screen
        :param iter bodyparts: a pair of joints of animal (tuple or list) that represent 'looking vector' like (start, end)
        For example,
         ('neck', 'nose') pair would mean that direction in which animal is looking defined by vector from neck to nose
        """
        self._angle = angle
        self._debug = debug
        self._point = point
        self._start_point, self._end_point = bodyparts

    def check_skeleton(self, skeleton: dict):
        """
        Checking skeleton for trigger
        :param skeleton: a skeleton dictionary, returned by calculate_skeletons() from poser file
        :return: response, a tuple of result (bool) and response body
        Response body is used for plotting and outputting results to trials dataframes
        """
        start_x, start_y = skeleton[self._start_point]
        end_x, end_y = skeleton[self._end_point]
        direction_x, direction_y = self._point
        head_dir, angle = angle_between_vectors(direction_x, direction_y, start_x, start_y, end_x, end_y)
        true_angle = 180 - abs(angle)

        result = true_angle <= self._angle

        color = (0, 255, 0) if result else (0, 0, 255)
        if self._debug:
            response_body = {'plot': {'line': dict(pt1=skeleton[self._end_point],
                                                   pt2=self._point,
                                                   color=color),
                                      'text': dict(text=str(true_angle),
                                                   org=skeleton[self._end_point],
                                                   color=(255, 255, 255))}}
        else:
            response_body = {'angle': true_angle}

        response = (result, response_body)
        return response


class ScreenTrigger(DirectionTrigger):
    """
    Trigger to check if animal is looking at the screen
    """
    def __init__(self, direction: str, angle: int, bodyparts: iter, debug: bool = False):
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
        self._direction = direction
        max_x, max_y = RESOLUTION
        direction_dict = {'North': (int(max_x / 2), 0), 'South': (int(max_x / 2), max_y),
                          'West': (0, int(max_y / 2)), 'East': (max_x, int(max_y / 2))}
        super().__init__(direction_dict[self._direction], angle, bodyparts, debug)


class RegionTrigger:
    """
    Trigger to check if animal is in Region Of Interest (ROI)
    """
    def __init__(self, region_type: str, center: tuple, radius: float, bodyparts, debug: bool = False):
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
        self._roi_type = region_type.lower()
        region_types = {'circle': EllipseROI, 'square': RectangleROI}
        self._region_of_interest = region_types[self._roi_type](center, radius, radius)
        self._bodyparts = bodyparts
        self._debug = debug  # not used in this trigger

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

        if self._roi_type == 'circle':
            response_body = {'plot': {'circle': dict(center=self._region_of_interest.get_center(),
                                                     radius=int(self._region_of_interest.get_x_radius()),
                                                     color=color)}}
        elif self._roi_type == 'square':
            box = self._region_of_interest.get_box()
            x1, y1, x2, y2 = box
            pt1 = (x1, y2)
            pt2 = (x2, y1)
            response_body = {'plot': {'square': dict(pt1=pt1,
                                                     pt2=pt2,
                                                     color=color)}}

        response = (result, response_body)
        return response


class OutsideTrigger(RegionTrigger):
    """
    Trigger to check if animal is out of the Region Of Interest (ROI)
    """
    def __init__(self, region_type: str, center: tuple, radius: float, bodyparts, debug: bool = False):
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
        super().__init__(region_type, center, radius, bodyparts, debug)

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


"""Posture sequence triggers"""

class FreezeTrigger:
    """
    Trigger to check if animal is in freezing state
    """
    def __init__(self, threshold: int, debug: bool = False):
        """
        Initializing trigger with given threshold
        :param threshold: int in pixel how much of a movement does not count
        For example threshold of 5 would mean that all movements less then 5 pixels would be ignored
        """
        self._threshold = threshold
        self._skeleton = None
        self._debug = debug  # not used in this trigger

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
            text = 'Not freezing'
            self._skeleton = skeleton
        else:
            for joint in skeleton:
                joint_travel = calculate_distance(skeleton[joint], self._skeleton[joint])
                joint_moved.append(abs(joint_travel) <= self._threshold)
            if all(joint_moved):
                result = True
                text = 'Freezing'
            else:
                result = False
                text = 'Not freezing'
            self._skeleton = skeleton

        color = (0, 255, 0) if result else (0, 0, 255)
        response_body = {'plot': {'text': dict(text=text,
                                               org=org_point,
                                               color=color)}}
        response = (result, response_body)
        return response


class SpeedTrigger:
    """
    Trigger to check if animal is moving above a certain speed
    """
    def __init__(self, threshold: int, bodypart: str = 'any', debug: bool = False):
        """
        Initializing trigger with given threshold
        :param threshold: int in pixel how much of a movement does not count
        :param bodypart: str or list of str, bodypart or list of bodyparts in skeleton to use for trigger,
         if "any" will check if any bodypart reaches treshold; default "any"
        For example threshold of 5 would mean that all movements less then 5 pixels would be ignored
        """
        self._bodypart = bodypart
        self._threshold = threshold
        self._skeleton = None
        self._debug = debug  # not used in this trigger

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
            text = 'First frame'
            self._skeleton = skeleton
        else:
            if self._bodypart is "any":
                for joint in skeleton:
                    joint_travel = calculate_distance(skeleton[joint], self._skeleton[joint])
                    joint_moved.append(abs(joint_travel) >= self._threshold)

            elif isinstance(self._bodypart, list):
                for joint in self._bodypart:
                    joint_travel = calculate_distance(skeleton[joint], self._skeleton[joint])
                    joint_moved.append(abs(joint_travel) >= self._threshold)
            else:
                joint_travel = calculate_distance(skeleton[self._bodypart], self._skeleton[self._bodypart])
                joint_moved.append(abs(joint_travel) >= self._threshold)

            if all(joint_moved):
                result = True
                text = 'Running'
            else:
                result = False
                text = ''
            self._skeleton = skeleton

        color = (0, 255, 0) if result else (0, 0, 255)
        response_body = {'plot': {'text': dict(text=text,
                                               org=org_point,
                                               color=color)}}
        response = (result, response_body)
        return response