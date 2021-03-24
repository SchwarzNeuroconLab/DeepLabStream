"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0

Simba Feature extraction functions were provided by Simon Nilsson from Golden Lab
Main developer of SiMBA https://github.com/sgoldenlab/simba
and integrated into the SimbaFeatureExtractor

Bsoid Feature extraction functions were adpated from code orginally by Alexander Hsu from Yttri Lab
Main developer of B-Soid https://github.com/YttriLab/B-SOID
and integrated into the BsoidFeatureExtractor
"""

from utils.configloader import PIXPERMM, FRAMERATE
import numpy as np
import pandas as pd
import math
from numba import jit
import itertools


### EUCLIDIAN DISTANCES IN SINGLE FRAMES
@jit(nopython=True, cache=True)
def EuclidianDistCalc(inArr):
    """provided by Simon Nilsson from Golden Lab; Main developer of SiMBA https://github.com/sgoldenlab/simba"""
    Mouse_1_nose_to_tail = int(
        np.sqrt(
            ((inArr[-1][4] - inArr[-1][12]) ** 2 + (inArr[-1][5] - inArr[-1][13]) ** 2)
        )
    )
    Mouse_2_nose_to_tail = int(
        np.sqrt(
            (
                (inArr[-1][18] - inArr[-1][26]) ** 2
                + (inArr[-1][19] - inArr[-1][27]) ** 2
            )
        )
    )
    Mouse_1_Ear_distance = int(
        np.sqrt(
            ((inArr[-1][0] - inArr[-1][2]) ** 2 + (inArr[-1][1] - inArr[-1][3]) ** 2)
        )
    )
    Centroid_distance = int(
        np.sqrt(
            ((inArr[-1][6] - inArr[-1][20]) ** 2 + (inArr[-1][7] - inArr[-1][21]) ** 2)
        )
    )
    Nose_to_nose_distance = int(
        np.sqrt(
            ((inArr[-1][4] - inArr[-1][18]) ** 2 + (inArr[-1][5] - inArr[-1][19]) ** 2)
        )
    )
    M1_Nose_to_M2_lat_left = int(
        np.sqrt(
            ((inArr[-1][4] - inArr[-1][22]) ** 2 + (inArr[-1][5] - inArr[-1][23]) ** 2)
        )
    )
    M1_Nose_to_M2_lat_right = int(
        np.sqrt(
            ((inArr[-1][4] - inArr[-1][24]) ** 2 + (inArr[-1][5] - inArr[-1][25]) ** 2)
        )
    )
    M2_Nose_to_M1_lat_left = int(
        np.sqrt(
            ((inArr[-1][18] - inArr[-1][8]) ** 2 + (inArr[-1][19] - inArr[-1][9]) ** 2)
        )
    )
    M2_Nose_to_M1_lat_right = int(
        np.sqrt(
            (
                (inArr[-1][18] - inArr[-1][10]) ** 2
                + (inArr[-1][19] - inArr[-1][11]) ** 2
            )
        )
    )
    M1_Nose_to_M2_tail_base = int(
        np.sqrt(
            ((inArr[-1][4] - inArr[-1][26]) ** 2 + (inArr[-1][5] - inArr[-1][27]) ** 2)
        )
    )
    M2_Nose_to_M1_tail_base = int(
        np.sqrt(
            (
                (inArr[-1][18] - inArr[-1][12]) ** 2
                + (inArr[-1][19] - inArr[-1][13]) ** 2
            )
        )
    )

    return np.array(
        [
            Mouse_1_nose_to_tail,
            Mouse_2_nose_to_tail,
            Mouse_1_Ear_distance,
            Centroid_distance,
            Nose_to_nose_distance,
            M1_Nose_to_M2_lat_left,
            M1_Nose_to_M2_lat_right,
            M2_Nose_to_M1_lat_left,
            M2_Nose_to_M1_lat_right,
            M1_Nose_to_M2_tail_base,
            M2_Nose_to_M1_tail_base,
        ]
    )


### EUCLIDEAN DISTANCES BETWEEN CENTROIDS IN ROLLING WINDOWS
@jit(nopython=True, cache=True)
def distancesBetweenBps(bpArray, bp):
    #TODO: adapt to flexible window
    """provided by Simon Nilsson from Golden Lab; Main developer of SiMBA https://github.com/sgoldenlab/simba"""

    frames2Process = bpArray.shape[0]
    bps2process = int(bpArray.shape[1] / 2)
    outputArray = np.zeros((frames2Process, bps2process - 1))
    for frame in range(frames2Process):
        outputArray[frame] = np.sqrt(
            (bpArray[frame][0] - bpArray[frame][2]) ** 2
            + (bpArray[frame][1] - bpArray[frame][3]) ** 2
        )
    Distance_median_2, Distance_mean_2, Distance_sum_2 = (
        int(np.median(outputArray)),
        int(np.mean(outputArray)),
        int(np.sum(outputArray)),
    )
    if bp == "centroid":
        #currently hardcoded to 15 frames
        msArr200, msArr166, msArr133, msArr66 = (
            outputArray[8:15],
            outputArray[10:15],
            outputArray[11:15],
            outputArray[13:15],
        )
        Distance_median_5, Distance_mean_5, Distance_sum_5 = (
            int(np.median(msArr200)),
            int(np.mean(msArr200)),
            int(np.sum(msArr200)),
        )
        Distance_median_6, Distance_mean_6, Distance_sum_6 = (
            int(np.median(msArr166)),
            int(np.mean(msArr166)),
            int(np.sum(msArr166)),
        )
        Distance_median_7, Distance_mean_7, Distance_sum_7 = (
            int(np.median(msArr133)),
            int(np.mean(msArr133)),
            int(np.sum(msArr133)),
        )
        Distance_median_15, Distance_mean_15, Distance_sum_15 = (
            int(np.median(msArr66)),
            int(np.mean(msArr66)),
            int(np.sum(msArr66)),
        )

        return np.array(
            [
                Distance_median_2,
                Distance_mean_2,
                Distance_sum_2,
                Distance_median_5,
                Distance_mean_5,
                Distance_sum_5,
                Distance_median_6,
                Distance_mean_6,
                Distance_sum_6,
                Distance_median_7,
                Distance_mean_7,
                Distance_sum_7,
                Distance_median_15,
                Distance_mean_15,
                Distance_sum_15,
            ]
        )
    if bp == "width":
        return np.array([Distance_median_2, Distance_mean_2, Distance_sum_2])


### EUCLIDEAN DISTANCES BETWEEN BODY-PARTS WITHIN EACH ANIMALS HULL
@jit(nopython=True, cache=True)
def bpDistancesInHull(animal1arraySplit):
    """provided by Simon Nilsson from Golden Lab; Main developer of SiMBA https://github.com/sgoldenlab/simba"""

    frames2process = animal1arraySplit.shape[0]
    bps2process = animal1arraySplit[1].shape[0]
    outputArray = np.zeros((frames2process, bps2process))
    for frame in range(frames2process):
        for bodypart in range(bps2process):
            for otherBps in [x for x in range(bps2process) if x != bodypart]:
                outputArray[frame][bodypart] = np.sqrt(
                    (
                        animal1arraySplit[frame][bodypart][0]
                        - animal1arraySplit[frame][otherBps][0]
                    )
                    ** 2
                    + (
                        animal1arraySplit[frame][bodypart][1]
                        - animal1arraySplit[frame][otherBps][1]
                    )
                    ** 2
                )
    flattenedOutput = outputArray.flatten()
    Mean_euclid_distances_median_2 = np.median(flattenedOutput)
    Mean_euclid_distances_mean_2 = np.mean(flattenedOutput)
    return np.array([Mean_euclid_distances_median_2, Mean_euclid_distances_mean_2])


### TOTAL MOVEMENT OF ALL ANIMALS IN ROLLING WINDOWS
@jit(nopython=True, cache=True)
def TotalMovementBodyparts(arrayConcat_Animal1, arrayConcat_Animal2):
    """provided by Simon Nilsson from Golden Lab; Main developer of SiMBA https://github.com/sgoldenlab/simba"""

    frames2process = arrayConcat_Animal2.shape[0]
    bps2process = int(arrayConcat_Animal2.shape[1] / 2)
    outputArray_animal_1, outputArray_animal_2 = (
        np.zeros((frames2process, bps2process)),
        np.zeros((frames2process, bps2process)),
    )
    for frame in range(frames2process):
        for bp_current, bp_shifted in zip(range(0, 7), range(7, 14)):
            outputArray_animal_1[frame][bp_current] = np.sqrt(
                (
                    arrayConcat_Animal1[frame][bp_current][0]
                    - arrayConcat_Animal1[frame][bp_shifted][0]
                )
                ** 2
                + (
                    arrayConcat_Animal1[frame][bp_current][1]
                    - arrayConcat_Animal1[frame][bp_shifted][1]
                )
                ** 2
            )
            outputArray_animal_2[frame][bp_current] = np.sqrt(
                (
                    arrayConcat_Animal2[frame][bp_current][0]
                    - arrayConcat_Animal2[frame][bp_shifted][0]
                )
                ** 2
                + (
                    arrayConcat_Animal2[frame][bp_current][1]
                    - arrayConcat_Animal2[frame][bp_shifted][1]
                )
                ** 2
            )
        sumAnimal1, sumAnimal2 = np.sum(outputArray_animal_1, axis=1), np.sum(
            outputArray_animal_2, axis=1
        )
        sumConcat = np.concatenate((sumAnimal1, sumAnimal2))
        Total_movement_all_bodyparts_both_mice_median_2 = int(np.median(sumConcat))
        Total_movement_all_bodyparts_both_mice_mean_2 = int(np.mean(sumConcat))
        Total_movement_all_bodyparts_both_mice_sum_2 = int(np.sum(sumConcat))
    last200msArrayAnimal1, last200msArrayAnimal2 = sumAnimal1[9:13], sumAnimal2[9:13]
    sumConcat200ms = np.concatenate((last200msArrayAnimal1, last200msArrayAnimal2))
    Total_movement_all_bodyparts_both_mice_mean_5 = int(np.mean(sumConcat200ms))
    Total_movement_all_bodyparts_both_mice_sum_5 = int(np.sum(sumConcat200ms))

    return (
        np.array(
            [
                Total_movement_all_bodyparts_both_mice_median_2,
                Total_movement_all_bodyparts_both_mice_mean_2,
                Total_movement_all_bodyparts_both_mice_sum_2,
                Total_movement_all_bodyparts_both_mice_mean_5,
                Total_movement_all_bodyparts_both_mice_sum_5,
            ]
        ),
        outputArray_animal_1,
        outputArray_animal_2,
    )


### MOVEMENTS OF INDIVIDUAL BODY-PARTS
@jit(nopython=True, cache=True)
def singleAnimalBpMovements(tail_1, tail_2, center_1, center_2, nose_1, nose_2):
    """provided by Simon Nilsson from Golden Lab; Main developer of SiMBA https://github.com/sgoldenlab/simba"""

    Tail_base_movement_M1_median_2, Tail_base_movement_M2_median_2 = (
        int(np.median(tail_1)),
        int(np.median(tail_2)),
    )
    Tail_base_movement_M2_mean_2, Tail_base_movement_M2_sum_2 = (
        int(np.mean(tail_2)),
        int(np.sum(tail_2)),
    )
    Centroid_movement_M1_mean_2, Centroid_movement_M1_sum_2 = (
        int(np.mean(center_1)),
        int(np.sum(center_1)),
    )
    Centroid_movement_M2_mean_2, Centroid_movement_M2_sum_2 = (
        int(np.mean(center_2)),
        int(np.sum(center_2)),
    )
    Nose_movement_M1_median_2, Nose_movement_M1_mean_2 = int(np.median(nose_1)), int(
        np.mean(nose_1)
    )
    Nose_movement_M1_sum_2, Nose_movement_M2_mean_2, Nose_movement_M2_sum_2 = (
        int(np.sum(nose_1)),
        int(np.mean(nose_2)),
        int(np.sum(nose_2)),
    )
    return np.array(
        [
            Tail_base_movement_M1_median_2,
            Tail_base_movement_M2_median_2,
            Tail_base_movement_M2_mean_2,
            Tail_base_movement_M2_sum_2,
            Centroid_movement_M1_mean_2,
            Centroid_movement_M1_sum_2,
            Centroid_movement_M2_mean_2,
            Centroid_movement_M2_sum_2,
            Nose_movement_M1_median_2,
            Nose_movement_M1_mean_2,
            Nose_movement_M1_sum_2,
            Nose_movement_M2_mean_2,
            Nose_movement_M2_sum_2,
        ]
    )


class SimbaFeatureExtractor:
    """Feature extraction module for integration in behavior trigger, takes list of postures as input
    and calculates features to pass to classifier. Features and classifier have to match!
    Designed to work with Simba https://github.com/sgoldenlab/simba"""

    def __init__(self, input_array_length):
        self.currPixPerMM = PIXPERMM
        self.input_array_length = input_array_length
        #TODO: Collect bodypart position in input array from skeleton automatically (this will make this much easier!)

    def get_currPixPerMM(self):
        return self.currPixPerMM

    def get_input_array_length(self):
        return self.input_array_length

    def set_input_array_length(self, new_input_array_length):
        self.input_array_length = new_input_array_length

    def extract_features(self, input_array):
        """Takes bp coordinates of length input_list_length and extract features.
        :return extracted feature list for input to classifier"
        Adapted from code provided by Simon Nilsson from Golden Lab; Main developer of SiMBA https://github.com/sgoldenlab/simba"""

        def append2featureList(featureList, measures2add):
            featureList.extend(measures2add)
            return featureList

        featureList = []
        if len(input_array) == self.input_array_length:
            input_array = np.array(input_array).astype(int)
            """Start extracting features"""
            ### EUCLIDIAN DISTANCES IN SINGLE FRAMES
            distanceMeasures = EuclidianDistCalc(input_array)
            featureList = append2featureList(featureList, list(distanceMeasures))

            ### EUCLIDEAN DISTANCES BETWEEN CENTROIDS/width IN ROLLING WINDOWS
            centroidM1x, centroidM1y, centroidM2x, centroidM2y = (
                input_array[:, [6]],
                input_array[:, [7]],
                input_array[:, [20]],
                input_array[:, [21]],
            )
            bpArray = np.concatenate(
                (centroidM1x, centroidM1y, centroidM2x, centroidM2y), 1
            )
            distancefeatures = distancesBetweenBps(bpArray, "centroid")
            featureList = append2featureList(featureList, list(distancefeatures))

            latLeftM1x, latLeftM1y, latRightM1x, latRightM1y = (
                input_array[:, [8]],
                input_array[:, [9]],
                input_array[:, [10]],
                input_array[:, [11]],
            )
            bpArray = np.concatenate(
                (latLeftM1x, latLeftM1y, latRightM1x, latRightM1y), 1
            )
            animal_1_Widths = distancesBetweenBps(bpArray, "width")
            featureList = append2featureList(featureList, list(animal_1_Widths))

            latLeftM2x, latLeftM2y, latRightM2x, latRightM2y = (
                input_array[:, [22]],
                input_array[:, [23]],
                input_array[:, [24]],
                input_array[:, [25]],
            )
            bpArray = np.concatenate(
                (latLeftM2x, latLeftM2y, latRightM2x, latRightM2y), 1
            )
            animal_2_Widths = distancesBetweenBps(bpArray, "width")
            featureList = append2featureList(featureList, list(animal_2_Widths))

            ### EUCLIDEAN DISTANCES BETWEEN BODY-PARTS WITHIN EACH ANIMALS HULL
            animal1array, animal2array = input_array[:, 0:14], input_array[:, 14:28]
            animal1arraySplit, animal2arraySplit = (
                animal1array.reshape(-1, 7, 2),
                animal2array.reshape(-1, 7, 2),
            )
            hullVars1 = bpDistancesInHull(animal1arraySplit)
            hullVars2 = bpDistancesInHull(animal2arraySplit)
            featureList = append2featureList(featureList, list(hullVars1))
            featureList = append2featureList(featureList, list(hullVars2))

            ### TOTAL MOVEMENT OF ALL ANIMALS IN ROLLING WINDOWS
            animalSplitArray = input_array.reshape(-1, 7, 2)
            animal_1_SplitArray = animalSplitArray[0::2]
            animal_2_SplitArray = animalSplitArray[1::2]
            shiftedSplitArray_animal_1 = np.roll(animal_1_SplitArray, -1, axis=0)
            shiftedSplitArray_animal_2 = np.roll(animal_2_SplitArray, -1, axis=0)
            shiftedSplitArray_animal_1, shiftedSplitArray_animal_2 = (
                shiftedSplitArray_animal_1[:-1].copy(),
                shiftedSplitArray_animal_2[:-1].copy(),
            )
            animal_1_SplitArray, animal_2_SplitArray = (
                animal_1_SplitArray[:-1].copy(),
                animal_2_SplitArray[:-1].copy(),
            )
            arrayConcat_Animal1 = np.concatenate(
                (animal_1_SplitArray, shiftedSplitArray_animal_1), 1
            )
            arrayConcat_Animal2 = np.concatenate(
                (animal_2_SplitArray, shiftedSplitArray_animal_2), 1
            )
            (
                totalMovementVars,
                outputArray_animal_1,
                outputArray_animal_2,
            ) = TotalMovementBodyparts(arrayConcat_Animal1, arrayConcat_Animal2)
            featureList = append2featureList(featureList, list(totalMovementVars))

            tail1, tail2 = outputArray_animal_1[:, [6]], outputArray_animal_2[:, [6]]
            center1, center2 = (
                outputArray_animal_1[:, [3]],
                outputArray_animal_2[:, [3]],
            )
            nose1, nose2 = outputArray_animal_1[:, [0]], outputArray_animal_2[:, [0]]
            indBpMovs = singleAnimalBpMovements(
                tail1, tail2, center1, center2, nose1, nose2
            )
            featureList = append2featureList(featureList, list(indBpMovs))

            featureList = [x / self.currPixPerMM for x in featureList]
            featureList = np.array(featureList)
            featureList = featureList.reshape(1, -1)

            return featureList

        else:
            return None


class BsoidFeatureExtractor:
    """Feature extraction module for integration in behavior trigger, takes list of postures as input
    and calculates features to pass to classifier. Features and classifier have to match!
    Designed to work with BSOID; https://github.com/YttriLab/B-SOID"""

    def __init__(self, input_array_length):
        self.input_array_length = input_array_length
        self._fps = FRAMERATE

    def get_currPixPerMM(self):
        return None

    def get_input_array_length(self):
        return self.input_array_length

    def set_input_array_length(self, new_input_array_length):
        self.input_array_length = new_input_array_length

    def extract_features(self, input_array):
        """
        Extracts features based on (x,y) positions
        :param input_array: list of poses
        :return f_10fps: 2D array, extracted features
           Adapted from BSOID; https://github.com/YttriLab/B-SOID
        """

        def boxcar_center(a, n):
            a1 = pd.Series(a)
            moving_avg = np.array(
                a1.rolling(window=n, min_periods=1, center=True).mean()
            )

            return moving_avg

        def adp_filt_pose(pose_estimation):
            """Adapted from adp_filt function in BSOID"""
            currdf = np.array(pose_estimation)
            datax = currdf[:, :, 0]
            datay = currdf[:, :, 1]
            # TODO: Adapt filter to work without workaround and skeleton
            # data_lh = currdf[:, :, 2]
            data_lh = np.full_like(datax, 0.9)
            currdf_filt = np.zeros((datax.shape[0], (datax.shape[1]) * 2))
            perc_rect = []
            for i in range(data_lh.shape[1]):
                perc_rect.append(0)
            for x in range(data_lh.shape[1]):
                a, b = np.histogram(data_lh[1:, x].astype(np.float))
                rise_a = np.where(np.diff(a) >= 0)
                if rise_a[0][0] > 1:
                    llh = b[rise_a[0][0]]
                else:
                    llh = b[rise_a[0][1]]
                data_lh_float = data_lh[:, x].astype(np.float)
                perc_rect[x] = np.sum(data_lh_float < llh) / data_lh.shape[0]
                currdf_filt[0, (2 * x) : (2 * x + 2)] = np.hstack(
                    [datax[0, x], datay[0, x]]
                )
                for i in range(1, data_lh.shape[0]):
                    if data_lh_float[i] < llh:
                        currdf_filt[i, (2 * x) : (2 * x + 2)] = currdf_filt[
                            i - 1, (2 * x) : (2 * x + 2)
                        ]
                    else:
                        currdf_filt[i, (2 * x) : (2 * x + 2)] = np.hstack(
                            [datax[i, x], datay[i, x]]
                        )
            currdf_filt = np.array(currdf_filt)
            currdf_filt = currdf_filt.astype(np.float)
            return currdf_filt, perc_rect

        if len(input_array) == self.input_array_length:
            data, p_sub_threshold = adp_filt_pose(input_array)
            data = np.array([data])
            win_len = np.int(np.round(0.05 / (1 / self._fps)) * 2 - 1)
            feats = []
            for m in range(len(data)):  # 1
                dataRange = len(data[m])  # 5
                dxy_r = []
                dis_r = []
                for r in range(dataRange):  # 0-4
                    if r < dataRange - 1:
                        dis = []
                        for c in range(0, data[m].shape[1], 2):  # 0-17, 2
                            dis.append(
                                np.linalg.norm(
                                    data[m][r + 1, c : c + 2] - data[m][r, c : c + 2]
                                )
                            )
                        dis_r.append(dis)
                    dxy = []
                    for i, j in itertools.combinations(
                        range(0, data[m].shape[1], 2), 2
                    ):  # 0-17, 2
                        dxy.append(data[m][r, i : i + 2] - data[m][r, j : j + 2])
                    dxy_r.append(dxy)
                dis_r = np.array(dis_r)
                dxy_r = np.array(dxy_r)
                dis_smth = []
                dxy_eu = np.zeros([dataRange, dxy_r.shape[1]])  # 5,36
                ang = np.zeros([dataRange - 1, dxy_r.shape[1]])
                dxy_smth = []
                ang_smth = []
                for l in range(dis_r.shape[1]):  # 0-8
                    dis_smth.append(boxcar_center(dis_r[:, l], win_len))
                for k in range(dxy_r.shape[1]):  # 0-35
                    for kk in range(dataRange):  # 0
                        dxy_eu[kk, k] = np.linalg.norm(dxy_r[kk, k, :])
                        if kk < dataRange - 1:
                            b_3d = np.hstack([dxy_r[kk + 1, k, :], 0])
                            a_3d = np.hstack([dxy_r[kk, k, :], 0])
                            c = np.cross(b_3d, a_3d)
                            ang[kk, k] = np.dot(
                                np.dot(np.sign(c[2]), 180) / np.pi,
                                math.atan2(
                                    np.linalg.norm(c),
                                    np.dot(dxy_r[kk, k, :], dxy_r[kk + 1, k, :]),
                                ),
                            )
                    dxy_smth.append(boxcar_center(dxy_eu[:, k], win_len))
                    ang_smth.append(boxcar_center(ang[:, k], win_len))
                dis_smth = np.array(dis_smth)
                dxy_smth = np.array(dxy_smth)
                ang_smth = np.array(ang_smth)
                feats.append(np.vstack((dxy_smth[:, 1:], ang_smth, dis_smth)))

            f_10fps = []
            for n in range(0, len(feats)):
                feats1 = np.zeros(len(data[n]))
                for s in range(math.floor(self._fps / 10)):
                    for k in range(
                        round(self._fps / 10) + s,
                        len(feats[n][0]),
                        round(self._fps / 10),
                    ):
                        if k > round(self._fps / 10) + s:
                            feats1 = np.concatenate(
                                (
                                    feats1.reshape(feats1.shape[0], feats1.shape[1]),
                                    np.hstack(
                                        (
                                            np.mean(
                                                (
                                                    feats[n][
                                                        0 : dxy_smth.shape[0],
                                                        range(
                                                            k - round(self._fps / 10), k
                                                        ),
                                                    ]
                                                ),
                                                axis=1,
                                            ),
                                            np.sum(
                                                (
                                                    feats[n][
                                                        dxy_smth.shape[0] : feats[
                                                            n
                                                        ].shape[0],
                                                        range(
                                                            k - round(self._fps / 10), k
                                                        ),
                                                    ]
                                                ),
                                                axis=1,
                                            ),
                                        )
                                    ).reshape(len(feats[0]), 1),
                                ),
                                axis=1,
                            )
                        else:
                            feats1 = np.hstack(
                                (
                                    np.mean(
                                        (
                                            feats[n][
                                                0 : dxy_smth.shape[0],
                                                range(k - round(self._fps / 10), k),
                                            ]
                                        ),
                                        axis=1,
                                    ),
                                    np.sum(
                                        (
                                            feats[n][
                                                dxy_smth.shape[0] : feats[n].shape[0],
                                                range(k - round(self._fps / 10), k),
                                            ]
                                        ),
                                        axis=1,
                                    ),
                                )
                            ).reshape(len(feats[0]), 1)
                    f_10fps.append(feats1)
            return f_10fps

        else:
            return None
