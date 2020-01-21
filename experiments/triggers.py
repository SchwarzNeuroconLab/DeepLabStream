from utils.analysis import EllipseROI, RectangleROI


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
