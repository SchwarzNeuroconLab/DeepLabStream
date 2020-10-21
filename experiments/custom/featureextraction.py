"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0

Feature extraction functions were provided by Simon Nillson from Golden Lab
Main developer of SiMBA https://github.com/sgoldenlab/simba
and integrated into the FeatureExtractor
"""

from utils.configloader import PIXPERMM
import numpy as np
from numba import jit


### EUCLIDIAN DISTANCES IN SINGLE FRAMES
@jit(nopython=True, cache=True)
def EuclidianDistCalc(inArr):
    Mouse_1_nose_to_tail = int(np.sqrt(((inArr[-1][4] - inArr[-1][12]) ** 2 + (inArr[-1][5] - inArr[-1][13]) ** 2)))
    Mouse_2_nose_to_tail = int(np.sqrt(((inArr[-1][18] - inArr[-1][26]) ** 2 + (inArr[-1][19] - inArr[-1][27]) ** 2)))
    Mouse_1_Ear_distance = int(np.sqrt(((inArr[-1][0] - inArr[-1][2]) ** 2 + (inArr[-1][1] - inArr[-1][3]) ** 2)))
    Centroid_distance = int(np.sqrt(((inArr[-1][6] - inArr[-1][20]) ** 2 + (inArr[-1][7] - inArr[-1][21]) ** 2)))
    Nose_to_nose_distance = int(np.sqrt(((inArr[-1][4] - inArr[-1][18]) ** 2 + (inArr[-1][5] - inArr[-1][19]) ** 2)))
    M1_Nose_to_M2_lat_left = int(np.sqrt(((inArr[-1][4] - inArr[-1][22]) ** 2 + (inArr[-1][5] - inArr[-1][23]) ** 2)))
    M1_Nose_to_M2_lat_right = int(np.sqrt(((inArr[-1][4] - inArr[-1][24]) ** 2 + (inArr[-1][5] - inArr[-1][25]) ** 2)))
    M2_Nose_to_M1_lat_left = int(np.sqrt(((inArr[-1][18] - inArr[-1][8]) ** 2 + (inArr[-1][19] - inArr[-1][9]) ** 2)))
    M2_Nose_to_M1_lat_right = int(np.sqrt(((inArr[-1][18] - inArr[-1][10]) ** 2 + (inArr[-1][19] - inArr[-1][11]) ** 2)))
    M1_Nose_to_M2_tail_base = int(np.sqrt(((inArr[-1][4] - inArr[-1][26]) ** 2 + (inArr[-1][5] - inArr[-1][27]) ** 2)))
    M2_Nose_to_M1_tail_base = int(np.sqrt(((inArr[-1][18] - inArr[-1][12]) ** 2 + (inArr[-1][19] - inArr[-1][13]) ** 2)))

    return np.array([Mouse_1_nose_to_tail, Mouse_2_nose_to_tail, Mouse_1_Ear_distance,
                            Centroid_distance, Nose_to_nose_distance, M1_Nose_to_M2_lat_left, M1_Nose_to_M2_lat_right,
                            M2_Nose_to_M1_lat_left, M2_Nose_to_M1_lat_right, M1_Nose_to_M2_tail_base,
                            M2_Nose_to_M1_tail_base])

### EUCLIDEAN DISTANCES BETWEEN CENTROIDS IN ROLLING WINDOWS
@jit(nopython=True, cache=True)
def distancesBetweenBps(bpArray, bp):
    frames2Process = bpArray.shape[0]
    bps2process = int(bpArray.shape[1] / 2)
    outputArray = np.zeros((frames2Process, bps2process - 1))
    for frame in range(frames2Process):
        outputArray[frame] = np.sqrt((bpArray[frame][0] - bpArray[frame][2]) ** 2 + (bpArray[frame][1] - bpArray[frame][3]) ** 2)
    Distance_median_2, Distance_mean_2, Distance_sum_2 = int(np.median(outputArray)), int(np.mean(outputArray)), int(np.sum(outputArray))
    if bp == 'centroid':
        msArr200,msArr166,msArr133,msArr66 = outputArray[8:15],outputArray[10:15],outputArray[11:15],outputArray[13:15]
        Distance_median_5,Distance_mean_5,Distance_sum_5 = int(np.median(msArr200)),int(np.mean(msArr200)),int(
            np.sum(msArr200))
        Distance_median_6,Distance_mean_6,Distance_sum_6 = int(np.median(msArr166)),int(np.mean(msArr166)),int(
            np.sum(msArr166))
        Distance_median_7,Distance_mean_7,Distance_sum_7 = int(np.median(msArr133)),int(np.mean(msArr133)),int(
            np.sum(msArr133))
        Distance_median_15,Distance_mean_15,Distance_sum_15 = int(np.median(msArr66)),int(np.mean(msArr66)),int(
            np.sum(msArr66))

        return np.array([Distance_median_2, Distance_mean_2, Distance_sum_2,
                         Distance_median_5, Distance_mean_5, Distance_sum_5, Distance_median_6, Distance_mean_6,
                         Distance_sum_6, Distance_median_7, Distance_mean_7,
                         Distance_sum_7, Distance_median_15, Distance_mean_15, Distance_sum_15])
    if bp == 'width':
        return np.array([Distance_median_2, Distance_mean_2, Distance_sum_2])


### EUCLIDEAN DISTANCES BETWEEN BODY-PARTS WITHIN EACH ANIMALS HULL
@jit(nopython=True, cache=True)
def bpDistancesInHull(animal1arraySplit):
    frames2process = animal1arraySplit.shape[0]
    bps2process = animal1arraySplit[1].shape[0]
    outputArray = np.zeros((frames2process, bps2process))
    for frame in range(frames2process):
        for bodypart in range(bps2process):
            for otherBps in [x for x in range(bps2process) if x != bodypart]:
                outputArray[frame][bodypart] = np.sqrt((animal1arraySplit[frame][bodypart][0] - animal1arraySplit[frame][otherBps][0]) ** 2 + (animal1arraySplit[frame][bodypart][1] - animal1arraySplit[frame][otherBps][1]) ** 2)
    flattenedOutput = outputArray.flatten()
    Mean_euclid_distances_median_2 = np.median(flattenedOutput)
    Mean_euclid_distances_mean_2 = np.mean(flattenedOutput)
    return np.array([Mean_euclid_distances_median_2, Mean_euclid_distances_mean_2])



### TOTAL MOVEMENT OF ALL ANIMALS IN ROLLING WINDOWS
@jit(nopython=True, cache=True)
def TotalMovementBodyparts(arrayConcat_Animal1, arrayConcat_Animal2):
    frames2process = arrayConcat_Animal2.shape[0]
    bps2process = int(arrayConcat_Animal2.shape[1] / 2)
    outputArray_animal_1, outputArray_animal_2 = np.zeros((frames2process, bps2process)), np.zeros(
        (frames2process, bps2process))
    for frame in range(frames2process):
        for bp_current, bp_shifted in zip(range(0, 7), range(7, 14)):
            outputArray_animal_1[frame][bp_current] = np.sqrt(
                (arrayConcat_Animal1[frame][bp_current][0] - arrayConcat_Animal1[frame][bp_shifted][0]) ** 2 + (
                            arrayConcat_Animal1[frame][bp_current][1] - arrayConcat_Animal1[frame][bp_shifted][1]) ** 2)
            outputArray_animal_2[frame][bp_current] = np.sqrt(
                (arrayConcat_Animal2[frame][bp_current][0] - arrayConcat_Animal2[frame][bp_shifted][0]) ** 2 + (
                            arrayConcat_Animal2[frame][bp_current][1] - arrayConcat_Animal2[frame][bp_shifted][1]) ** 2)
        sumAnimal1, sumAnimal2 = np.sum(outputArray_animal_1, axis=1), np.sum(outputArray_animal_2, axis=1)
        sumConcat = np.concatenate((sumAnimal1, sumAnimal2))
        Total_movement_all_bodyparts_both_mice_median_2 = int(np.median(sumConcat))
        Total_movement_all_bodyparts_both_mice_mean_2 = int(np.mean(sumConcat))
        Total_movement_all_bodyparts_both_mice_sum_2 = int(np.sum(sumConcat))
    last200msArrayAnimal1, last200msArrayAnimal2 = sumAnimal1[9:13], sumAnimal2[9:13]
    sumConcat200ms = np.concatenate((last200msArrayAnimal1, last200msArrayAnimal2))
    Total_movement_all_bodyparts_both_mice_mean_5 = int(np.mean(sumConcat200ms))
    Total_movement_all_bodyparts_both_mice_sum_5 = int(np.sum(sumConcat200ms))

    return np.array([Total_movement_all_bodyparts_both_mice_median_2, Total_movement_all_bodyparts_both_mice_mean_2,
                     Total_movement_all_bodyparts_both_mice_sum_2, Total_movement_all_bodyparts_both_mice_mean_5,
                     Total_movement_all_bodyparts_both_mice_sum_5]), outputArray_animal_1, outputArray_animal_2

### MOVEMENTS OF INDIVIDUAL BODY-PARTS
@jit(nopython=True, cache=True)
def singleAnimalBpMovements(tail_1, tail_2, center_1, center_2, nose_1, nose_2):
    Tail_base_movement_M1_median_2, Tail_base_movement_M2_median_2 = int(np.median(tail_1)), int(np.median(tail_2))
    Tail_base_movement_M2_mean_2, Tail_base_movement_M2_sum_2 = int(np.mean(tail_2)), int(np.sum(tail_2))
    Centroid_movement_M1_mean_2, Centroid_movement_M1_sum_2 = int(np.mean(center_1)), int(np.sum(center_1))
    Centroid_movement_M2_mean_2, Centroid_movement_M2_sum_2 = int(np.mean(center_2)), int(np.sum(center_2))
    Nose_movement_M1_median_2, Nose_movement_M1_mean_2 = int(np.median(nose_1)), int(np.mean(nose_1))
    Nose_movement_M1_sum_2, Nose_movement_M2_mean_2, Nose_movement_M2_sum_2 = int(np.sum(nose_1)), int(np.mean(nose_2)), int(np.sum(nose_2))
    return np.array([Tail_base_movement_M1_median_2, Tail_base_movement_M2_median_2, Tail_base_movement_M2_mean_2,
                     Tail_base_movement_M2_sum_2, Centroid_movement_M1_mean_2, Centroid_movement_M1_sum_2,
                     Centroid_movement_M2_mean_2, Centroid_movement_M2_sum_2, Nose_movement_M1_median_2, Nose_movement_M1_mean_2,
                     Nose_movement_M1_sum_2, Nose_movement_M2_mean_2, Nose_movement_M2_sum_2])



class FeatureExtractor:
    """Feature extraction module for integration in behavior trigger, takes list of postures as input
     and calculates features to pass to classifier. Features and classifier have to match!"""

    def __init__(self, input_array_length):
        self.currPixPerMM = PIXPERMM
        self.input_array_length = input_array_length

    def get_currPixPerMM(self):
        return self.currPixPerMM

    def get_input_array_length(self):
        return self.input_array_length

    def set_input_array_length(self,new_input_array_length):
        self.input_array_length = new_input_array_length

    def extract_features(self, input_array):
        """Takes bp coordinates of length input_list_length and extract features.
        :return extracted feature list for input to classifier"""

        def append2featureList(featureList,measures2add):
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
            centroidM1x, centroidM1y, centroidM2x, centroidM2y = input_array[:,[6]],input_array[:,[7]],input_array[:,[20]],input_array[:,[21]]
            bpArray = np.concatenate((centroidM1x, centroidM1y, centroidM2x, centroidM2y), 1)
            distancefeatures = distancesBetweenBps(bpArray, 'centroid')
            featureList = append2featureList(featureList, list(distancefeatures))

            latLeftM1x, latLeftM1y, latRightM1x, latRightM1y = input_array[:,[8]],input_array[:,[9]],input_array[:,[10]],input_array[:,[11]]
            bpArray = np.concatenate((latLeftM1x, latLeftM1y, latRightM1x, latRightM1y), 1)
            animal_1_Widths = distancesBetweenBps(bpArray, 'width')
            featureList = append2featureList(featureList, list(animal_1_Widths))

            latLeftM2x, latLeftM2y, latRightM2x, latRightM2y = input_array[:,[22]],input_array[:,[23]],input_array[:,[24]],input_array[:,[25]]
            bpArray = np.concatenate((latLeftM2x, latLeftM2y, latRightM2x, latRightM2y), 1)
            animal_2_Widths = distancesBetweenBps(bpArray, 'width')
            featureList = append2featureList(featureList, list(animal_2_Widths))

            ### EUCLIDEAN DISTANCES BETWEEN BODY-PARTS WITHIN EACH ANIMALS HULL
            animal1array, animal2array = input_array[:,0:14],input_array[:,14:28]
            animal1arraySplit, animal2arraySplit = animal1array.reshape(-1, 7, 2), animal2array.reshape(-1, 7, 2)
            hullVars1 = bpDistancesInHull(animal1arraySplit)
            hullVars2 = bpDistancesInHull(animal2arraySplit)
            featureList = append2featureList(featureList, list(hullVars1))
            featureList = append2featureList(featureList, list(hullVars2))

            ### TOTAL MOVEMENT OF ALL ANIMALS IN ROLLING WINDOWS
            animalSplitArray = (input_array.reshape(-1,7,2))
            animal_1_SplitArray = animalSplitArray[0::2]
            animal_2_SplitArray = animalSplitArray[1::2]
            shiftedSplitArray_animal_1 = np.roll(animal_1_SplitArray, -1, axis=0)
            shiftedSplitArray_animal_2 = np.roll(animal_2_SplitArray, -1, axis=0)
            shiftedSplitArray_animal_1, shiftedSplitArray_animal_2 = shiftedSplitArray_animal_1[:-1].copy(), shiftedSplitArray_animal_2[:-1].copy()
            animal_1_SplitArray, animal_2_SplitArray = animal_1_SplitArray[:-1].copy(), animal_2_SplitArray[:-1].copy()
            arrayConcat_Animal1 = np.concatenate((animal_1_SplitArray, shiftedSplitArray_animal_1), 1)
            arrayConcat_Animal2 = np.concatenate((animal_2_SplitArray, shiftedSplitArray_animal_2), 1)
            totalMovementVars, outputArray_animal_1, outputArray_animal_2 = TotalMovementBodyparts(arrayConcat_Animal1, arrayConcat_Animal2)
            featureList = append2featureList(featureList, list(totalMovementVars))

            tail1, tail2 = outputArray_animal_1[:, [6]], outputArray_animal_2[:, [6]]
            center1, center2 = outputArray_animal_1[:, [3]], outputArray_animal_2[:, [3]]
            nose1, nose2 = outputArray_animal_1[:, [0]], outputArray_animal_2[:, [0]]
            indBpMovs = singleAnimalBpMovements(tail1, tail2, center1, center2, nose1, nose2)
            featureList = append2featureList(featureList, list(indBpMovs))

            featureList = [x / self.currPixPerMM for x in featureList]
            featureList = np.array(featureList)
            featureList = featureList.reshape(1,-1)

            return featureList

        else:
            return None
