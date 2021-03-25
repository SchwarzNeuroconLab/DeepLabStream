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
import scipy
from scipy.spatial import ConvexHull
from utils.configloader import PIXPERMM, FRAMERATE
import numpy as np
import pandas as pd
import math
import time
from numba import jit
import itertools

"""For standard SIMBA feature extraction"""

def count_values_in_range(series,values_in_range_min,values_in_range_max):
    return series.between(left=values_in_range_min,right=values_in_range_max).sum()

@jit(nopython=True, cache=True)
def euclidean_distance(x1, x2, y1, y2):
    result = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return result

@jit(nopython=True, cache=True)
def angle3pt(ax,ay,bx,by,cx,cy):
    ang = math.degrees(
        math.atan2(cy - by,cx - bx) - math.atan2(ay - by,ax - bx))
    return ang + 360 if ang < 0 else ang

def as_strided(*args):
    return np.lib.stride_tricks.as_strided(*args)



class SimbaFeatureExtractorStandard14bp:
    """Feature extraction module for integration in behavior trigger, takes list of postures as input
    and calculates features to pass to classifier. Features and classifier have to match!
    Designed to work with Simba https://github.com/sgoldenlab/simba"""

    def __init__(self, input_array_length, debug = False):
        self._debug = debug
        self.input_array_length = input_array_length
        #TODO: Collect bodypart position in input array from skeleton automatically (this will make this much easier!)
        self._currPixPerMM = PIXPERMM
        self._fps = FRAMERATE
        self._num_bodyparts = 14
        self._roll_window_values = [2,5,6,7.5,15]
        self._roll_window_values_len = len(self._roll_window_values)
        self._roll_windows =  [int(self._fps / num) for num in self._roll_window_values]
        self._columnheaders = ["Nose_1_x","Nose_1_y"
            ,"Ear_left_1_x","Ear_left_1_y"
            ,"Ear_right_1_x","Ear_right_1_y"
            ,"Center_1_x","Center_1_y"
            ,"Lat_left_1_x","Lat_left_1_y"
            ,"Lat_right_1_x","Lat_right_1_y"
            ,"Tail_base_1_x","Tail_base_1_y"
            ,"Nose_2_x","Nose_2_y"
            ,"Ear_left_2_x","Ear_left_2_y"
            ,"Ear_right_2_x","Ear_right_2_y"
            ,"Center_2_x","Center_2_y"
            ,"Lat_left_2_x","Lat_left_2_y"
            ,"Lat_right_2_x","Lat_right_2_y"
            ,"Tail_base_2_x","Tail_base_2_y"
                         ]
        self._pheaders = ["Nose_1_p","Ear_left_1_p","Ear_right_1_p","Center_1_p","Lat_left_1_p","Lat_right_1_p",
                     "Tail_base_1_p","Nose_2_p","Ear_left_2_p","Ear_right_2_p","Center_2_p","Lat_left_2_p",
                     "Lat_right_2_p","Tail_base_2_p"]

    def convert_pandas(self, input_array):
        """This is a hardcoded version for the 14 bp SiMBA classification"""
        input_array_np = np.array(input_array)

        input_df = pd.DataFrame(input_array_np,columns=self._columnheaders)
        # add fake prediction column
        for p_clm in self._pheaders:
            input_df[p_clm] = 0.99

        input_df = input_df.fillna(0)
        input_df = input_df.drop(input_df.index[[0]])
        input_df = input_df.apply(pd.to_numeric)
        input_df = input_df.reset_index()
        input_df = input_df.reset_index(drop=True)
        return input_df

    def extract_features_simba14bp(self, input_df):
        # adapted from SIMBA: https://github.com/sgoldenlab/simba/blob/master/simba/features_scripts/extract_features_14bp.py
        ########### CREATE PD FOR RAW DATA AND PD FOR MOVEMENT BETWEEN FRAMES ###########
        M1_hull_large_euclidean_list = []
        M1_hull_small_euclidean_list = []
        M1_hull_mean_euclidean_list = []
        M1_hull_sum_euclidean_list = []
        M2_hull_large_euclidean_list = []
        M2_hull_small_euclidean_list = []
        M2_hull_mean_euclidean_list = []
        M2_hull_sum_euclidean_list = []

        start_time = time.time()
        ########### MOUSE AREAS ###########################################
        input_df['Mouse_1_poly_area'] = input_df.apply(lambda x: ConvexHull(np.array(
            [[x['Ear_left_1_x'],x["Ear_left_1_y"]],
             [x['Ear_right_1_x'],x["Ear_right_1_y"]],
             [x['Nose_1_x'],x["Nose_1_y"]],
             [x['Lat_left_1_x'],x["Lat_left_1_y"]], \
             [x['Lat_right_1_x'],x["Lat_right_1_y"]],
             [x['Tail_base_1_x'],x["Tail_base_1_y"]],
             [x['Center_1_x'],x["Center_1_y"]]])).area,axis=1)
        input_df['Mouse_1_poly_area'] = input_df['Mouse_1_poly_area'] / self._currPixPerMM


        input_df['Mouse_2_poly_area'] = input_df.apply(lambda x: ConvexHull(np.array(
            [[x['Ear_left_2_x'],x["Ear_left_2_y"]],
             [x['Ear_right_2_x'],x["Ear_right_2_y"]],
             [x['Nose_2_x'],x["Nose_2_y"]],
             [x['Lat_left_2_x'],x["Lat_left_2_y"]], \
             [x['Lat_right_2_x'],x["Lat_right_2_y"]],
             [x['Tail_base_2_x'],x["Tail_base_2_y"]],
             [x['Center_2_x'],x["Center_2_y"]]])).area,axis=1)
        input_df['Mouse_2_poly_area'] = input_df['Mouse_2_poly_area'] / self._currPixPerMM

        if self._debug:
            print('Evaluating convex hulls...')
            print((time.time() -start_time) *1000, 'ms')

        ########### CREATE SHIFTED DATAFRAME FOR DISTANCE CALCULATIONS ###########################################
        start_time = time.time()
        input_df_shifted = input_df.shift(periods=1)
        input_df_shifted = input_df_shifted.rename(
            columns={'Ear_left_1_x': 'Ear_left_1_x_shifted','Ear_left_1_y': 'Ear_left_1_y_shifted',
                     'Ear_left_1_p': 'Ear_left_1_p_shifted','Ear_right_1_x': 'Ear_right_1_x_shifted', \
                     'Ear_right_1_y': 'Ear_right_1_y_shifted','Ear_right_1_p': 'Ear_right_1_p_shifted',
                     'Nose_1_x': 'Nose_1_x_shifted','Nose_1_y': 'Nose_1_y_shifted', \
                     'Nose_1_p': 'Nose_1_p_shifted','Center_1_x': 'Center_1_x_shifted',
                     'Center_1_y': 'Center_1_y_shifted','Center_1_p': 'Center_1_p_shifted','Lat_left_1_x': \
                         'Lat_left_1_x_shifted','Lat_left_1_y': 'Lat_left_1_y_shifted',
                     'Lat_left_1_p': 'Lat_left_1_p_shifted','Lat_right_1_x': 'Lat_right_1_x_shifted',
                     'Lat_right_1_y': 'Lat_right_1_y_shifted', \
                     'Lat_right_1_p': 'Lat_right_1_p_shifted','Tail_base_1_x': 'Tail_base_1_x_shifted',
                     'Tail_base_1_y': 'Tail_base_1_y_shifted', \
                     'Tail_base_1_p': 'Tail_base_1_p_shifted',
                     'Ear_left_2_x': 'Ear_left_2_x_shifted','Ear_left_2_y': 'Ear_left_2_y_shifted',
                     'Ear_left_2_p': 'Ear_left_2_p_shifted','Ear_right_2_x': 'Ear_right_2_x_shifted', \
                     'Ear_right_2_y': 'Ear_right_2_y_shifted','Ear_right_2_p': 'Ear_right_2_p_shifted',
                     'Nose_2_x': 'Nose_2_x_shifted','Nose_2_y': 'Nose_2_y_shifted', \
                     'Nose_2_p': 'Nose_2_p_shifted','Center_2_x': 'Center_2_x_shifted',
                     'Center_2_y': 'Center_2_y_shifted','Center_2_p': 'Center_2_p_shifted','Lat_left_2_x': \
                         'Lat_left_2_x_shifted','Lat_left_2_y': 'Lat_left_2_y_shifted',
                     'Lat_left_2_p': 'Lat_left_2_p_shifted','Lat_right_2_x': 'Lat_right_2_x_shifted',
                     'Lat_right_2_y': 'Lat_right_2_y_shifted', \
                     'Lat_right_2_p': 'Lat_right_2_p_shifted','Tail_base_2_x': 'Tail_base_2_x_shifted',
                     'Tail_base_2_y': 'Tail_base_2_y_shifted', \
                     'Tail_base_2_p': 'Tail_base_2_p_shifted',
                     'Mouse_1_poly_area': 'Mouse_1_poly_area_shifted',
                     'Mouse_2_poly_area': 'Mouse_2_poly_area_shifted'})
        input_df_combined = pd.concat([input_df,input_df_shifted],axis=1,join='inner')
        input_df_combined = input_df_combined.fillna(0)
        input_df_combined = input_df_combined.reset_index(drop=True)

        if self._debug:
            print('Creating shifted dataframes for distance calculations')
            print((time.time() -start_time) *1000, 'ms')

        ########### EUCLIDEAN DISTANCES ###########################################
        start_time = time.time()

        #within mice

        eucl_distance_dict_wm = dict(
            nose_to_tail = ('Nose', 'Tail_base')
            ,width = ('Lat_left', 'Lat_right')
            ,Ear_distance = ('Ear_right','Ear_left')
            ,Nose_to_centroid = ('Nose', 'Center')
            ,Nose_to_lateral_left= ('Nose', 'Lat_left')
            ,Nose_to_lateral_right=('Nose','Lat_right')
            ,Centroid_to_lateral_left = ('Center', 'Lat_left')
            ,Centroid_to_lateral_right=('Center','Lat_right')
        )

        mice = [1, 2]
        for mouse in mice:
            for distance_measurement, bodyparts in eucl_distance_dict_wm.items():
                x1 = input_df[f'{bodyparts[0]}_{mouse}_x'].to_numpy()
                y1 = input_df[f'{bodyparts[0]}_{mouse}_y'].to_numpy()
                x2 = input_df[f'{bodyparts[1]}_{mouse}_x'].to_numpy()
                y2 = input_df[f'{bodyparts[1]}_{mouse}_y'].to_numpy()
                input_df[f'Mouse_{mouse}_{distance_measurement}'] = euclidean_distance(x1,x2,y1,y2) / self._currPixPerMM

        #between mice

        eucl_distance_dict_bm = dict(
            Centroid_distance = ('Center_1','Center_2')
            ,Nose_to_nose_distance = ('Nose_1', 'Nose_2')
            ,M1_Nose_to_M2_lat_left = ('Nose_1','Lat_left_2')
            ,M1_Nose_to_M2_lat_right = ('Nose_1', 'Lat_right_2')
            ,M2_Nose_to_M1_lat_left = ('Nose_2','Lat_left_1')
            ,M2_Nose_to_M1_lat_right = ('Nose_2', 'Lat_right_1')
            ,M1_Nose_to_M2_tail_base = ('Nose_1', 'Tail_base_2')
            ,M2_Nose_to_M1_tail_base=('Nose_2', 'Tail_base_1')
        )

        for distance_measurement,bodyparts in eucl_distance_dict_bm.items():
            x1 = input_df[f'{bodyparts[0]}_x'].to_numpy()
            y1 = input_df[f'{bodyparts[0]}_y'].to_numpy()
            x2 = input_df[f'{bodyparts[1]}_x'].to_numpy()
            y2 = input_df[f'{bodyparts[1]}_y'].to_numpy()
            input_df[f'{distance_measurement}'] = euclidean_distance(x1,x2,y1,y2) / self._currPixPerMM

        #Movement

        bp_list = ('Center', 'Nose', 'Lat_left','Lat_right', 'Tail_base', 'Ear_left', 'Ear_right')

        mice = [1, 2]
        for mouse in mice:
            for bp in bp_list:
                x1 = input_df_combined[f'{bp}_{mouse}_x_shifted'].to_numpy()
                y1 = input_df_combined[f'{bp}_{mouse}_y_shifted'].to_numpy()
                x2 = input_df_combined[f'{bp}_{mouse}_x'].to_numpy()
                y2 = input_df_combined[f'{bp}_{mouse}_y'].to_numpy()
                'Movement_mouse_1_centroid'
                if bp == 'Center':
                    input_df[f'Movement_mouse_{mouse}_centroid'] = euclidean_distance(x1,x2,y1,y2) / self._currPixPerMM
                elif bp == 'Ear_left':
                    input_df[f'Movement_mouse_{mouse}_left_ear'] = euclidean_distance(x1,x2,y1,y2) / self._currPixPerMM
                elif bp == 'Ear_right':
                    input_df[f'Movement_mouse_{mouse}_right_ear'] = euclidean_distance(x1,x2,y1,y2) / self._currPixPerMM
                elif bp == 'Lat_left':
                    input_df[f'Movement_mouse_{mouse}_lateral_left'] = euclidean_distance(x1,x2,y1,y2) / self._currPixPerMM
                elif bp == 'Lat_right':
                    input_df[f'Movement_mouse_{mouse}_lateral_right'] = euclidean_distance(x1,x2,y1,y2) / self._currPixPerMM
                else:
                    input_df[f'Movement_mouse_{mouse}_{bp.lower()}'] = euclidean_distance(x1,x2,y1,y2) / self._currPixPerMM


        input_df['Mouse_1_polygon_size_change'] = (
                input_df_combined['Mouse_1_poly_area_shifted'] - input_df_combined['Mouse_1_poly_area'])
        input_df['Mouse_2_polygon_size_change'] = (
                input_df_combined['Mouse_2_poly_area_shifted'] - input_df_combined['Mouse_2_poly_area'])

        if self._debug:
            print('Calculating euclidean distances...')
            print((time.time() -start_time) *1000, 'ms')

        ########### HULL - EUCLIDEAN DISTANCES ###########################################
        start_time = time.time()

        for index,row in input_df.iterrows():
            M1_np_array = np.array(
                [[row['Ear_left_1_x'],row["Ear_left_1_y"]],[row['Ear_right_1_x'],row["Ear_right_1_y"]],
                 [row['Nose_1_x'],row["Nose_1_y"]],[row['Center_1_x'],row["Center_1_y"]],
                 [row['Lat_left_1_x'],row["Lat_left_1_y"]],[row['Lat_right_1_x'],row["Lat_right_1_y"]],
                 [row['Tail_base_1_x'],row["Tail_base_1_y"]]]).astype(int)
            M2_np_array = np.array(
                [[row['Ear_left_2_x'],row["Ear_left_2_y"]],[row['Ear_right_2_x'],row["Ear_right_2_y"]],
                 [row['Nose_2_x'],row["Nose_2_y"]],[row['Center_2_x'],row["Center_2_y"]],
                 [row['Lat_left_2_x'],row["Lat_left_2_y"]],[row['Lat_right_2_x'],row["Lat_right_2_y"]],
                 [row['Tail_base_2_x'],row["Tail_base_2_y"]]]).astype(int)
            M1_dist_euclidean = scipy.spatial.distance.cdist(M1_np_array,M1_np_array,metric='euclidean')
            M1_dist_euclidean = M1_dist_euclidean[M1_dist_euclidean != 0]
            M1_hull_large_euclidean = np.amax(M1_dist_euclidean)
            M1_hull_small_euclidean = np.min(M1_dist_euclidean)
            M1_hull_mean_euclidean = np.mean(M1_dist_euclidean)
            M1_hull_sum_euclidean = np.sum(M1_dist_euclidean)
            M1_hull_large_euclidean_list.append(M1_hull_large_euclidean)
            M1_hull_small_euclidean_list.append(M1_hull_small_euclidean)
            M1_hull_mean_euclidean_list.append(M1_hull_mean_euclidean)
            M1_hull_sum_euclidean_list.append(M1_hull_sum_euclidean)
            M2_dist_euclidean = scipy.spatial.distance.cdist(M2_np_array,M2_np_array,metric='euclidean')
            M2_dist_euclidean = M2_dist_euclidean[M2_dist_euclidean != 0]
            M2_hull_large_euclidean = np.amax(M2_dist_euclidean)
            M2_hull_small_euclidean = np.min(M2_dist_euclidean)
            M2_hull_mean_euclidean = np.mean(M2_dist_euclidean)
            M2_hull_sum_euclidean = np.sum(M2_dist_euclidean)
            M2_hull_large_euclidean_list.append(M2_hull_large_euclidean)
            M2_hull_small_euclidean_list.append(M2_hull_small_euclidean)
            M2_hull_mean_euclidean_list.append(M2_hull_mean_euclidean)
            M2_hull_sum_euclidean_list.append(M2_hull_sum_euclidean)
        input_df['M1_largest_euclidean_distance_hull'] = list(
            map(lambda x: x / self._currPixPerMM,M1_hull_large_euclidean_list))
        input_df['M1_smallest_euclidean_distance_hull'] = list(
            map(lambda x: x / self._currPixPerMM,M1_hull_small_euclidean_list))
        input_df['M1_mean_euclidean_distance_hull'] = list(map(lambda x: x / self._currPixPerMM,M1_hull_mean_euclidean_list))
        input_df['M1_sum_euclidean_distance_hull'] = list(map(lambda x: x / self._currPixPerMM,M1_hull_sum_euclidean_list))
        input_df['M2_largest_euclidean_distance_hull'] = list(
            map(lambda x: x / self._currPixPerMM,M2_hull_large_euclidean_list))
        input_df['M2_smallest_euclidean_distance_hull'] = list(
            map(lambda x: x / self._currPixPerMM,M2_hull_small_euclidean_list))
        input_df['M2_mean_euclidean_distance_hull'] = list(map(lambda x: x / self._currPixPerMM,M2_hull_mean_euclidean_list))
        input_df['M2_sum_euclidean_distance_hull'] = list(map(lambda x: x / self._currPixPerMM,M2_hull_sum_euclidean_list))
        input_df['Sum_euclidean_distance_hull_M1_M2'] = (
                input_df['M1_sum_euclidean_distance_hull'] + input_df['M2_sum_euclidean_distance_hull'])

        if self._debug:
            print('Calculating hull variables...')
            print((time.time() -start_time) *1000, 'ms')

        ########### COLLAPSED MEASURES ###########################################

        start_time = time.time()
        input_df['Total_movement_centroids'] = input_df['Movement_mouse_1_centroid'] + input_df['Movement_mouse_2_centroid']
        input_df['Total_movement_all_bodyparts_M1'] = input_df['Movement_mouse_1_centroid'] + input_df[
            'Movement_mouse_1_nose'] + input_df['Movement_mouse_1_tail_base'] + \
                                                      input_df['Movement_mouse_1_left_ear'] + input_df[
                                                        'Movement_mouse_1_right_ear'] + input_df[
                                                        'Movement_mouse_1_lateral_left'] + input_df[
                                                        'Movement_mouse_1_lateral_right']
        input_df['Total_movement_all_bodyparts_M2'] = input_df['Movement_mouse_2_centroid'] + input_df[
            'Movement_mouse_2_nose'] + input_df['Movement_mouse_2_tail_base'] + \
                                                      input_df['Movement_mouse_2_left_ear'] + input_df[
                                                        'Movement_mouse_2_right_ear'] + input_df[
                                                        'Movement_mouse_2_lateral_left'] + input_df[
                                                        'Movement_mouse_2_lateral_right']
        input_df['Total_movement_all_bodyparts_both_mice'] = input_df['Total_movement_all_bodyparts_M1'] + input_df[
            'Total_movement_all_bodyparts_M2']

        if self._debug:
            print('Collapsed measures')
            print((time.time() -start_time) *1000, 'ms')

        ########### CALC ROLLING WINDOWS MEDIANS AND MEANS ###########################################
        # step  simplification: remove pd.DataFrame.rolling() for fixed selection. In a limited time window a rolling approach is not neccessary
        start_time = time.time()

        for roll_value in self._roll_windows:

            parameter_dict = dict(
                Mouse1_width = 'Mouse_1_width'
                ,Mouse2_width = 'Mouse_2_width'
                ,Distance = 'Centroid_distance'
                ,Movement = 'Total_movement_centroids'
                ,Sum_euclid_distances_hull = 'Sum_euclidean_distance_hull_M1_M2'
            )
            for key, clm_name in parameter_dict.items():

                clm_array = input_df[clm_name][:roll_value].to_numpy()

                currentcolname = f'{key}_mean_' + str(roll_value)
                input_df[currentcolname] = np.mean(clm_array)

                currentcolname = f'{key}_median_' + str(roll_value)
                input_df[currentcolname] = np.median(clm_array)

                currentcolname = f'{key}_sum_' + str(roll_value)
                input_df[currentcolname] = np.sum(clm_array)


            clm_name = 'euclidean_distance_hull'
            clm_name2 = 'euclid_distances'
            for mouse in mice:
                for value in ('largest', 'smallest', 'mean'):

                    clm_array = input_df[f'M{mouse}_{value}_{clm_name}'][:roll_value].to_numpy()

                    currentcolname = f'Mouse{mouse}_{value}_{clm_name2}_mean_' + str(roll_value)
                    input_df[currentcolname] = np.mean(clm_array)

                    currentcolname = f'Mouse{mouse}_{value}_{clm_name2}_median_' + str(roll_value)
                    input_df[currentcolname] = np.median(clm_array)

                    currentcolname = f'Mouse{mouse}_{value}_{clm_name2}_sum_' + str(roll_value)
                    input_df[currentcolname] = np.sum(clm_array)

            clm_list = [
                'Total_movement_all_bodyparts_both_mice'
                ,'Total_movement_centroids'
            ]

            for clm_name in clm_list:
                clm_array = input_df[clm_name][:roll_value].to_numpy()

                currentcolname = clm_name + '_mean_' + str(roll_value)
                input_df[currentcolname] = np.mean(clm_array)

                currentcolname = clm_name + '_median_' + str(roll_value)
                input_df[currentcolname] = np.median(clm_array)

                currentcolname = clm_name + '_sum_' + str(roll_value)
                input_df[currentcolname] = np.sum(clm_array)

            parameter_dict = dict(
                Nose_movement = 'nose'
                ,Centroid_movement = 'centroid'
                ,Tail_base_movement = 'tail_base'
            )
            for mouse in mice:
                for key, bp in parameter_dict.items():
                    clm_array = input_df[f'Movement_mouse_{mouse}_{bp.lower()}'][:roll_value].to_numpy()

                    currentcolname = f'{key}_M{mouse}_mean_' + str(roll_value)
                    input_df[currentcolname] = np.mean(clm_array)

                    currentcolname = f'{key}_M{mouse}_median_' + str(roll_value)
                    input_df[currentcolname] = np.median(clm_array)

                    currentcolname = f'{key}_M{mouse}_sum_' + str(roll_value)
                    input_df[currentcolname] = np.sum(clm_array)

        if self._debug:
            print('Calculating rolling windows: medians, medians, and sums...')
            print((time.time() -start_time) *1000, 'ms')

        ########### BODY PARTS RELATIVE TO EACH OTHER ##################

        ################# EMPETY #########################################

        ########### ANGLES ###########################################
        start_time = time.time()
        input_df['Mouse_1_angle'] = input_df.apply(
            lambda x: angle3pt(x['Nose_1_x'],x['Nose_1_y'],x['Center_1_x'],x['Center_1_y'],x['Tail_base_1_x'],
                               x['Tail_base_1_y']),axis=1)
        input_df['Mouse_2_angle'] = input_df.apply(
            lambda x: angle3pt(x['Nose_2_x'],x['Nose_2_y'],x['Center_2_x'],x['Center_2_y'],x['Tail_base_2_x'],
                               x['Tail_base_2_y']),axis=1)
        input_df['Total_angle_both_mice'] = input_df['Mouse_1_angle'] + input_df['Mouse_2_angle']
        for roll_value in self._roll_windows:
            currentcolname = 'Total_angle_both_mice_' + str(roll_value)
            input_df[currentcolname] = np.sum(input_df['Total_angle_both_mice'][:roll_value].to_numpy())

        if self._debug:
            print('Calculating angles...')
            print((time.time() -start_time) *1000, 'ms')

        ########### DEVIATIONS ###########################################
        start_time = time.time()
        parameter_dict = dict(
            Total_movement_all_bodyparts_both_mice_deviation = 'Total_movement_all_bodyparts_both_mice'
            ,Sum_euclid_distances_hull_deviation = 'Sum_euclidean_distance_hull_M1_M2'
            ,M1_smallest_euclid_distances_hull_deviation = 'M1_smallest_euclidean_distance_hull'
            ,M1_largest_euclid_distances_hull_deviation = 'M1_largest_euclidean_distance_hull'
            ,M1_mean_euclid_distances_hull_deviation = 'M1_mean_euclidean_distance_hull'
            ,Centroid_distance_deviation = 'Centroid_distance'
            ,Total_angle_both_mice_deviation = 'Total_angle_both_mice'
            ,Movement_mouse_1_deviation_centroid = 'Movement_mouse_1_centroid'
            ,Movement_mouse_2_deviation_centroid = 'Movement_mouse_2_centroid'
            ,Mouse_1_polygon_deviation = 'Mouse_1_poly_area'
            ,Mouse_2_polygon_deviation = 'Mouse_2_poly_area'

        )
        for key, value in parameter_dict.items():
            currentClm = input_df[value].to_numpy()
            input_df[key] = (np.mean(currentClm) - currentClm)


        for roll_value in self._roll_windows:

            clm_names = (
                'Total_movement_all_bodyparts_both_mice_mean_'
                ,'Sum_euclid_distances_hull_mean_'
                ,'Mouse1_smallest_euclid_distances_mean_'
                ,'Mouse1_largest_euclid_distances_mean_'
                ,'Movement_mean_'
                ,'Distance_mean_'
                ,'Total_angle_both_mice_'
            )

            for part in clm_names:
                currentcolname = part + str(roll_value)
                currentClm = input_df[currentcolname].to_numpy()
                output_array = (np.mean(currentClm) - currentClm)
                input_df[f'{currentcolname}_deviation'] = output_array
                # same values will be applied to another clm ... weird but what should i do?
                #Yes I checked. Twice, actually 3 times.
                input_df[f'{currentcolname}_percentile_rank'] = output_array

        if self._debug:
            print('Calculating deviations...')
            print((time.time() -start_time) *1000, 'ms')

        ########### PERCENTILE RANK ###########################################

        start_time = time.time()
        input_df['Movement_percentile_rank'] = input_df['Total_movement_centroids'].rank(pct=True)
        input_df['Distance_percentile_rank'] = input_df['Centroid_distance'].rank(pct=True)
        input_df['Movement_mouse_1_percentile_rank'] = input_df['Movement_mouse_1_centroid'].rank(pct=True)
        input_df['Movement_mouse_2_percentile_rank'] = input_df['Movement_mouse_1_centroid'].rank(pct=True)
        input_df['Movement_mouse_1_deviation_percentile_rank'] = input_df['Movement_mouse_1_deviation_centroid'].rank(
            pct=True)
        input_df['Movement_mouse_2_deviation_percentile_rank'] = input_df['Movement_mouse_2_deviation_centroid'].rank(
            pct=True)
        input_df['Centroid_distance_percentile_rank'] = input_df['Centroid_distance'].rank(pct=True)
        input_df['Centroid_distance_deviation_percentile_rank'] = input_df['Centroid_distance_deviation'].rank(pct=True)

        if self._debug:
            print('Calculating percentile ranks...')
            print((time.time() -start_time) *1000, 'ms')

        ########### CALCULATE STRAIGHTNESS OF POLYLINE PATH: tortuosity  ###########################################
        #as_strided = np.lib.stride_tricks.as_strided
        start_time = time.time()

        win_size = 3
        centroidList_Mouse1_x = as_strided(input_df.Center_1_x,(len(input_df) - (win_size - 1),win_size),
                                           (input_df.Center_1_x.values.strides * 2))
        centroidList_Mouse1_y = as_strided(input_df.Center_1_y,(len(input_df) - (win_size - 1),win_size),
                                           (input_df.Center_1_y.values.strides * 2))
        centroidList_Mouse2_x = as_strided(input_df.Center_2_x,(len(input_df) - (win_size - 1),win_size),
                                           (input_df.Center_2_x.values.strides * 2))
        centroidList_Mouse2_y = as_strided(input_df.Center_2_y,(len(input_df) - (win_size - 1),win_size),
                                           (input_df.Center_2_y.values.strides * 2))

        for k in range(self._roll_window_values_len):
            start = 0
            end = start + int(self._roll_window_values[k])
            tortuosity_M1 = []
            tortuosity_M2 = []
            for y in range(len(input_df)):
                tortuosity_List_M1 = []
                tortuosity_List_M2 = []
                CurrCentroidList_Mouse1_x = centroidList_Mouse1_x[start:end]
                CurrCentroidList_Mouse1_y = centroidList_Mouse1_y[start:end]
                CurrCentroidList_Mouse2_x = centroidList_Mouse2_x[start:end]
                CurrCentroidList_Mouse2_y = centroidList_Mouse2_y[start:end]
                for i in range(len(CurrCentroidList_Mouse1_x)):
                    currMovementAngle_mouse1 = (
                        angle3pt(CurrCentroidList_Mouse1_x[i][0],CurrCentroidList_Mouse1_y[i][0],
                                 CurrCentroidList_Mouse1_x[i][1],CurrCentroidList_Mouse1_y[i][1],
                                 CurrCentroidList_Mouse1_x[i][2],CurrCentroidList_Mouse1_y[i][2]))
                    currMovementAngle_mouse2 = (
                        angle3pt(CurrCentroidList_Mouse2_x[i][0],CurrCentroidList_Mouse2_y[i][0],
                                 CurrCentroidList_Mouse2_x[i][1],CurrCentroidList_Mouse2_y[i][1],
                                 CurrCentroidList_Mouse2_x[i][2],CurrCentroidList_Mouse2_y[i][2]))
                    tortuosity_List_M1.append(currMovementAngle_mouse1)
                    tortuosity_List_M2.append(currMovementAngle_mouse2)
                tortuosity_M1.append(sum(tortuosity_List_M1) / (2 * math.pi))
                tortuosity_M2.append(sum(tortuosity_List_M2) / (2 * math.pi))
                start += 1
                end += 1
            currentcolname1 = str('Tortuosity_Mouse1_') + str(self._roll_window_values[k])
            # currentcolname2 = str('Tortuosity_Mouse2_') + str(self._roll_window_values[k])
            input_df[currentcolname1] = tortuosity_M1
            # input_df[currentcolname2] = tortuosity_M2
        if self._debug:
            print('Calculating path tortuosities...')
            print((time.time() -start_time) *1000, 'ms')

        ########### CALC THE NUMBER OF LOW PROBABILITY DETECTIONS & TOTAL PROBABILITY VALUE FOR ROW###########################################
        start_time = time.time()

        #SKIPPING BECAUSE DLSTREAM DOES NOT USE PROBABILITY
        input_df['Sum_probabilities'] = 0.99 * self._num_bodyparts
        input_df['Sum_probabilities_deviation'] = 0
        #output of ranking with all 0
        input_df['Sum_probabilities_deviation_percentile_rank'] = 0.53571
        input_df['Sum_probabilities_percentile_rank'] = 0.53571

        input_df["Low_prob_detections_0.1"] = 0.0
        input_df["Low_prob_detections_0.5"] = 0.0
        input_df["Low_prob_detections_0.75"] = 0.0
        if self._debug:
            print('Calculating pose probability scores...')
            print((time.time() -start_time) *1000, 'ms')

        ########### DROP COORDINATE COLUMNS ###########################################

        input_df = input_df.reset_index(drop=True)
        input_df = input_df.fillna(0)
        input_df = input_df.drop(columns=['index'])
        #drop coordinates and likelyhood
        #input_df = input_df.drop(columns = self._columnheaders + self._pheaders)

        return input_df

    def extract_features(self, input_array):
        """Takes bp coordinates of length input_list_length and extract features.
        :return extracted feature list for input to classifier"
        Adapted from code provided by Simon Nilsson from Golden Lab; Main developer of SiMBA https://github.com/sgoldenlab/simba"""

        if len(input_array) == self.input_array_length:
            start_time = time.time()
            input_df = self.convert_pandas(input_array)
            features = self.extract_features_simba14bp(input_df)

            if self._debug:
                print('Full feature extraction time: ', (time.time() -start_time) *1000)
            return features.to_numpy()

        else:
            return None

    def get_currPixPerMM(self):
        return self.currPixPerMM

    def get_input_array_length(self):
        return self.input_array_length

    def set_input_array_length(self, new_input_array_length):
        self.input_array_length = new_input_array_length


"""For fast simba extraction"""


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
