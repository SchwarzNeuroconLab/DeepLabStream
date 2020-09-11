"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""
from experiments.utils.exp_setup import get_trigger_settings
from utils.analysis import angle_between_vectors, calculate_distance, EllipseROI, RectangleROI

"""BaseTrigger class"""

class BaseTrigger:
    """Base class for standard triggers"""
    def __init__(self):
        """
        Initialising trigger with following parameters:
         """

        self._name = 'BaseTrigger'
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

            response_body = {'plot': {'text': dict(text='BaseTrigger',
                                                   org=skeleton[self._bodyparts[1]],
                                                   color=color),
                                     }}
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
        self._name = 'BaseHeaddirectionTrigger'

        #loading settings
        self._trigger_parameter_dict = dict(POINT = 'tuple',
                                            ANGLE = 'int',
                                            BODYPARTS = 'list',
                                            DEBUG = 'boolean')
        self._settings_dict = get_trigger_settings(self._name, self._exp_parameter_dict)

        self._point = self._settings_dict['POINT']
        self._angle = self._settings_dict['ANGLE']
        self._bodyparts = self._settings_dict['BODYPARTS']
        self._debug = self._settings_dict['DEBUG'] if self._settings_dict['DEBUG'] is not None else False

    def check_skeleton(self, skeleton: dict):
        """
        Checking skeleton for trigger
        :param skeleton: a skeleton dictionary, returned by calculate_skeletons() from poser file
        :return: response, a tuple of result (bool) and response body
        Response body is used for plotting and outputting results to trials dataframes
         [point , 'neck', 'nose'] you need to pass this to angle between vectors to get headdirection
        """
        ret_head_dir, angle = angle_between_vectors(*skeleton[self._bodyparts[0]], *skeleton[self._bodyparts[1]], *self._point)
        true_angle = abs(angle)

        if true_angle <= self._angle:
            result = True
        else:
            result = False


        color = (0, 255, 0) if result else (0, 0, 255)
        if self._debug:
            center = skeleton[self._bodyparts[1]]

            response_body = {'plot': {'text': dict(text=str(true_angle),
                                                   org=skeleton[self._bodyparts[1]],
                                                   color=(255, 255, 255)),
                                      'circle': dict(center= center,
                                                     radius= 5,
                                                     color=color)
                             }}
        else:
            response_body = {'angle': true_angle}

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
        self._name = 'BaseRegionTrigger'

        #loading settings
        self._trigger_parameter_dict = dict(CENTER = 'tuple',
                                            TYPE = 'int',
                                            RADIUS = 'int',
                                            BODYPARTS = 'list',
                                            DEBUG = 'boolean')
        self._settings_dict = get_trigger_settings(self._name, self._exp_parameter_dict)

        region_types = {'circle': EllipseROI, 'square': RectangleROI}
        self._region_of_interest = region_types[self._settings_dict['TYPE'].lower()](self._settings_dict['CENTER'],
                                                                self._settings_dict['RADIUS'],
                                                                self._settings_dict['RADIUS'])

        self._bodyparts = self._settings_dict['BODYPARTS'] if len(self._settings_dict['BODYPARTS']) > 1 \
                                                           else self._settings_dict['BODYPARTS'][0]

        self._debug = self._settings_dict['DEBUG'] if self._settings_dict['DEBUG'] is not None else False



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
        self._name = 'BaseRegionTrigger'

        #loading settings
        self._trigger_parameter_dict = dict(THRESHOLD = 'float',
                                            BODYPARTS = 'list',
                                            DEBUG = 'boolean')
        self._settings_dict = get_trigger_settings(self._name, self._exp_parameter_dict)

        region_types = {'circle': EllipseROI, 'square': RectangleROI}
        self._region_of_interest = region_types[self._settings_dict['TYPE'].lower()](self._settings_dict['CENTER'],
                                                                self._settings_dict['RADIUS'],
                                                                self._settings_dict['RADIUS'])

        self._threshold = self._settings_dict['THRESHOLD']

        self._bodyparts = self._settings_dict['BODYPARTS'] if len(self._settings_dict['BODYPARTS']) > 1 \
                                                           else self._settings_dict['BODYPARTS'][0]

        self._debug = self._settings_dict['DEBUG'] if self._settings_dict['DEBUG'] is not None else False

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