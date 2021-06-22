"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""

import sys
import os
import importlib.util
from itertools import product, combinations

import numpy as np
from skimage.feature import peak_local_max
from scipy.ndimage.measurements import label, maximum_position
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter

from utils.analysis import calculate_distance
from utils.configloader import (
    MODEL_ORIGIN,
    MODEL_NAME,
    MODEL_PATH,
    ALL_BODYPARTS,
    FLATTEN_MA,
    SPLIT_MA,
    HANDLE_MISSING,
    ANIMALS_NUMBER,
    FILTER_LIKELIHOOD,
    LIKELIHOOD_THRESHOLD,
    USE_DLSTREAM_POSTURE_DETECTION,
)

# suppressing unnecessary warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

# trying importing functions using deeplabcut module, if DLC 2 is installed correctly
if MODEL_ORIGIN in ("DLC", "MADLC"):
    try:
        # checking for DLC-core
        if importlib.util.find_spec("deeplabcutcore") is not None:
            import deeplabcutcore.pose_estimation_tensorflow.nnet.predict as predict
            from deeplabcutcore.pose_estimation_tensorflow.config import load_config
        # trying to import "classic" DLC2
        else:
            import deeplabcut.pose_estimation_tensorflow.nnet.predict as predict
            from deeplabcut.pose_estimation_tensorflow.config import load_config

            if MODEL_ORIGIN == "MADLC":
                from deeplabcut.pose_estimation_tensorflow.nnet import (
                    predict_multianimal,
                )

        models_folder = "pose_estimation_tensorflow/models/"
    # if not DLC 2 is not installed, try import from DLC 1 the old way
    except ImportError:
        # adding DLC posing path and loading modules from it
        sys.path.insert(0, MODEL_PATH + "/pose-tensorflow")
        from config import load_config
        from nnet import predict

        models_folder = "pose-tensorflow/models/"

elif MODEL_ORIGIN == "DEEPPOSEKIT":
    from deepposekit.models import load_model

elif MODEL_ORIGIN == "DLC-LIVE":
    from dlclive import DLCLive
    from utils.configloader import MODEL_PATH

elif MODEL_ORIGIN == "SLEAP":
    from sleap import load_model
    from utils.configloader import MODEL_PATH


class SkeletonError(Exception):
    """Custom expection to be raised when issues with the skeleton is not received"""


def load_deeplabcut():
    """
    Loads TensorFlow with predefined in config DeepLabCut model

    :return: tuple of DeepLabCut config, TensorFlow session, inputs and outputs
    """
    model = os.path.join(MODEL_PATH, models_folder, MODEL_NAME)
    cfg = load_config(os.path.join(model, "test/pose_cfg.yaml"))
    snapshots = sorted(
        [sn.split(".")[0] for sn in os.listdir(model + "/train/") if "index" in sn]
    )
    cfg["init_weights"] = model + "/train/" + snapshots[-1]

    sess, inputs, outputs = predict.setup_pose_prediction(cfg)
    return cfg, sess, inputs, outputs


# pure DLC
def get_pose(image, config, session, inputs, outputs):
    """
    Gets scoremap, local reference and pose from DeepLabCut using given image
    Pose is most probable points for each joint, and not really used later
    Scoremap and local reference is essential to extract skeletons
    :param image: frame which would be analyzed
    :param config, session, inputs, outputs: DeepLabCut configuration and TensorFlow variables from load_deeplabcut()

    :return: tuple of scoremap, local reference and pose
    """
    scmap, locref, pose = predict.getpose(image, config, session, inputs, outputs, True)
    return scmap, locref, pose


def find_local_peaks_new(
    scoremap: np.ndarray, local_reference: np.ndarray, animal_number: int, config: dict
) -> dict:
    """
    Function for finding local peaks for each joint on provided scoremap
    :param scoremap: scmap from get_pose function
    :param local_reference: locref from get_pose function
    :param animal_number: number of animals for which we need to find peaks, also used for critical joints
        Critical joint are used to define skeleton of an animal
        There can not be more than animal_number point for each critical joint
    :param config: DeepLabCut config from load_deeplabcut()

    :returns all_joints dictionary with coordinates as list of tuples for each joint
    """

    # loading animal joints from config
    all_joints_names = config["all_joints_names"]
    # critical_joints = ['neck', 'tailroot']
    all_peaks = {}
    # loading stride from config
    stride = config["stride"]
    # filtering scoremap
    scoremap[scoremap < 0.1] = 0
    for joint_num, joint in enumerate(all_joints_names):
        all_peaks[joint] = []
        # selecting the joint in scoremap and locref
        lr_joint = local_reference[:, :, joint_num]
        sm_joint = scoremap[:, :, joint_num]

        # applying maximum filter with footprint
        neighborhood = generate_binary_structure(2, 1)
        sm_max_filter = maximum_filter(sm_joint, footprint=neighborhood)
        # eroding filtered scoremap
        erosion_structure = generate_binary_structure(2, 3)
        sm_max_filter_eroded = binary_erosion(
            sm_max_filter, structure=erosion_structure
        ).astype(sm_max_filter.dtype)
        # labeling eroded filtered scoremap
        labeled_sm_eroded, num = label(sm_max_filter_eroded)
        # if joint is 'critical' and we have too few labels then we try a workaround to ensure maximum found peaks
        # for all other joints - normal procedure with cutoff point at animal_number
        peaks = maximum_position(
            sm_joint, labels=labeled_sm_eroded, index=range(1, num + 1)
        )
        if num != animal_number:
            peaks = [
                tuple(peak)
                for peak in peak_local_max(
                    sm_joint, min_distance=4, num_peaks=animal_number
                )
            ]

        if len(peaks) > animal_number:
            peaks = peaks[: animal_number + 1]

        # using scoremap peaks to get the coordinates on original image
        for peak in peaks:
            offset = lr_joint[peak]
            prob = sm_joint[peak]  # not used
            # some weird DLC magic with stride and offsets
            coordinates = np.floor(
                np.array(peak)[::-1] * stride + 0.5 * stride + offset
            )
            all_peaks[joint].append([tuple(coordinates.astype(int)), joint])
    return all_peaks


def filter_pose_by_likelihood(pose, threshold: float = 0.1):
    """
       Filters pose estimation by likelihood threshold. Estimates below threshold are set to NaN and handled downstream
    of this function in calculate skeletons.
    :param pose: pose estimation (e.g., from DLC)
    :param threshold: likelihood threshold to filter by
    :return filtered_pose: pose estimation filtered by likelihood (may contain NaN) in shape of pose,
                            likelihood will be set to 2 in case of filtered bodyparts
    """
    filtered_pose = pose.copy()
    if MODEL_ORIGIN == 'DLC':
        """ DLC pose output is an np.array with [bp*[X,Y, Likelihood]]"""
        for num, bp in enumerate(filtered_pose):
            if bp[2] < threshold:
                # set new threshold to "2" (number outside of normal range to signify filter)
                # if just past to "calculate_skeleton" it will be ignored
                filtered_pose[num] = np.array([np.NaN, np.NaN, 2])

    return filtered_pose


def calculate_dlstream_skeletons(peaks: dict, animals_number: int) -> list:
    """
    Creating skeletons from given peaks
    There could be no more skeletons than animals_number
    Only unique skeletons output
    """
    # creating a cartesian product out of all joints
    # this product contains all possible variations (dots clusters) of all joints groups
    cartesian_p = product(*peaks.values(), repeat=1)

    def calculate_closest_distances(dots_cluster: list) -> float:
        """
        Calculating a sum of all distances between all dots in a cluster
        """
        # extracting dots coordinates from given list
        dots_coordinates = (dot[0] for dot in dots_cluster)
        # calculating sum of each dots cluster
        product_sum = sum(
            calculate_distance(*c) for c in combinations(dots_coordinates, 2)
        )
        return product_sum

    # sorting groups by their sum
    sorted_product = sorted(
        cartesian_p, key=lambda c: calculate_closest_distances(c), reverse=False
    )

    # creating skeletons from top dots cluster
    def compare_clusters(unique_clusters: list, new_cluster: tuple) -> bool:
        """
        Compare some new cluster against every existing unique cluster to find if it is unique
        :param unique_clusters: list of existing unique cluster
        :param new_cluster: cluster with same dots
        :return: if new cluster is unique
        """
        # compare each element of tuple for uniqueness
        # finding unique combinations of joints within all possible combinations
        compare = lambda cl1, cl2: not any([(s1 == s2) for s1, s2 in zip(cl1, cl2)])
        # create a uniqueness check list
        # so if a list consists at least one False then new_cluster is not unique
        comparison = [compare(u_cluster, new_cluster) for u_cluster in unique_clusters]
        return all(comparison)

    def create_animal_skeleton(dots_cluster: tuple) -> dict:
        """
        Creating a easy to read skeleton from dots cluster
        Format for each joint:
        {'joint_name': (x,y)}
        """
        skeleton = {}
        for dot in dots_cluster:
            skeleton[dot[-1]] = dot[0]
        return skeleton

    top_unique_clusters = []
    animal_skeletons = []

    if sorted_product:
        # add first cluster in our sorted list
        top_unique_clusters.append(sorted_product[0])
        for cluster in sorted_product[1:]:
            # check if cluster is unique and we have a room for it
            if (
                compare_clusters(top_unique_clusters, cluster)
                and len(top_unique_clusters) < animals_number
            ):
                top_unique_clusters.append(cluster)
            # there couldn't be more clusters then animal_number limit
            elif len(top_unique_clusters) == animals_number:
                break

    # creating a skeleton out of each cluster in our top clusters list
    for unique_cluster in top_unique_clusters:
        animal_skeletons.append(create_animal_skeleton(unique_cluster))

    return animal_skeletons


# maDLC
def get_ma_pose(image, config, session, inputs, outputs):
    """
    Gets scoremap, local reference and pose from DeepLabCut using given image
    Pose is most probable points for each joint, and not really used later
    Scoremap and local reference is essential to extract skeletons
    :param image: frame which would be analyzed
    :param config, session, inputs, outputs: DeepLabCut configuration and TensorFlow variables from load_deeplabcut()

    :return: tuple of scoremap, local reference and pose
    """
    scmap, locref, paf, pose = predict_multianimal.get_detectionswithcosts(
        image,
        config,
        session,
        inputs,
        outputs,
        outall=True,
        nms_radius=5.0,
        det_min_score=0.1,
        c_engine=False,
    )

    return pose


def calculate_ma_skeletons(
    pose: dict, animals_number: int, threshold: float = 0.1
) -> list:
    """
    Creating skeletons from given pose in maDLC
    There could be no more skeletons than animals_number
    Only unique skeletons output
    """

    def filter_mapredictions(pose):
        detection = []
        conf = np.array(pose["confidence"])
        coords = np.array(pose["coordinates"])
        for num, bp in enumerate(pose["coordinates"][0]):
            if len(bp) > 0:
                conf_bp = conf[num].flatten()
                fltred_bp = bp[conf_bp >= threshold, :]
                # todo: add function to only take top k-highest poses with k = animal number
                detection.append(fltred_bp)
            else:
                detection.append(np.array([]))
        return detection

    def extract_to_animal_skeleton(coords):
        """
        Creating a easy to read skeleton from dots cluster
        Format for each joint:
        {'joint_name': (x,y)}
        """
        bodyparts = np.array(coords)
        skeletons = {}
        for bp in range(len(bodyparts)):
            for animal_num in range(animals_number):
                if "Mouse" + str(animal_num + 1) not in skeletons.keys():
                    skeletons["Mouse" + str(animal_num + 1)] = {}
                if len(bodyparts[bp]) >= animals_number:
                    skeletons["Mouse" + str(animal_num + 1)][
                        "bp" + str(bp + 1)
                    ] = bodyparts[bp][animal_num].astype(float)
                else:
                    if animal_num < len(bodyparts[bp]):
                        skeletons["Mouse" + str(animal_num + 1)][
                            "bp" + str(bp + 1)
                        ] = bodyparts[bp][animal_num].astype(float)
                    else:
                        skeletons["Mouse" + str(animal_num + 1)][
                            "bp" + str(bp + 1)
                        ] = np.array([np.NaN, np.NaN])

        return skeletons

    detections = filter_mapredictions(pose)
    animal_skeletons = extract_to_animal_skeleton(detections)
    animal_skeletons = list(animal_skeletons.values())

    return animal_skeletons


# DLC LIVE & DeepPoseKit
def load_dpk():
    model = load_model(MODEL_PATH)
    return model.predict_model


def load_dlc_live():
    return DLCLive(MODEL_PATH)


def load_sleap():
    model = load_model(MODEL_PATH, batch_size=1)
    model.inference_model
    return model.inference_model


def flatten_maDLC_skeletons(skeletons):
    """Flattens maDLC multi skeletons into one skeleton to simulate dlc output
    where animals are not identical e.g. for animals with different fur colors (SIMBA)"""
    flat_skeletons = dict()
    for num, skeleton in enumerate(skeletons):
        for bp, value in skeleton.items():
            flat_skeletons[f"{num}_{bp}"] = value

    return [flat_skeletons]


def split_flat_skeleton(skeletons):
    """Splits flat multi skeletons (e.g. from flatten_maDLCskeleton) into seperate skeleton to simulate output
    where animals are identity tracked (e.g. SLEAP)"""
    flat_skeletons = skeletons[0]
    split_skeletons = []
    bp_per_animal, remainder = divmod(len(flat_skeletons), ANIMALS_NUMBER)
    if remainder > 0:
        raise SkeletonError(
            f"The number of body parts ({len(flat_skeletons)}) cannot be split equally into {ANIMALS_NUMBER} animals."
        )
    else:
        for animal in range(ANIMALS_NUMBER):
            single_skeleton = list(flat_skeletons.keys())[
                bp_per_animal * animal : bp_per_animal * animal + bp_per_animal
            ]
            split_skeletons.append(
                {x: flat_skeletons[x] for x in flat_skeletons if x in single_skeleton}
            )

    return split_skeletons


def transform_2skeleton(pose):
    """
    Transforms pose estimation into DLStream style "skeleton" posture.
    If ALL_BODYPARTS is not sufficient, it will autoname the bodyparts in style bp0, bp1 ...
    """
    try:
        skeleton = dict()
        counter = 0
        for bp in pose:
            skeleton[ALL_BODYPARTS[counter]] = tuple(np.array(bp[0:2], dtype=float))
            counter += 1
    except IndexError:
        skeleton = dict()
        counter = 0
        for bp in pose:
            skeleton[f"bp{counter}"] = tuple(np.array(bp[0:2], dtype=float))
            counter += 1

    return skeleton


def transform_2pose(skeleton):
    pose = np.array([*skeleton.values()])
    return pose


def handle_missing_bp(animal_skeletons: list):
    """handles missing bodyparts (NaN values) by selected method in advanced_settings.ini
    If HANDLE_MISSING is skip: the complete skeleton is removed (default);
    If HANDLE_MISSING is null: the missing coordinate is set to 0.0, not recommended for experiments
     where continuous monitoring of parameters is necessary.
    If HANDLE_MISSING is pass: the missing coordinate is left NaN. This is useful to keep identities, but can yield
        unexpected results down the line if NaN values are not caught.
    If HANDLE_MISSING is reset: the whole skeleton is set to NaN. This is useful to keep identities, but can yield
        unexpected results down the line if NaN values are not caught.

    Missing skeletons will not be passed to the trigger, while resetting coordinates might lead to false results returned
    by triggers.


    :param: animal_skeletons: list of skeletons returned by calculate skeleton
    :return animal_skeleton with handled missing values"""

    for skeleton in animal_skeletons:
        for bodypart, coordinates in skeleton.items():
            np_coords = np.array((coordinates))
            if any(np.isnan(np_coords)):
                if HANDLE_MISSING == "pass":
                    # do nothing
                    pass
                elif HANDLE_MISSING == "skip":
                    # remove the whole skeleton
                    animal_skeletons.remove(skeleton)
                    break
                elif HANDLE_MISSING == "null":
                    # remove replace coordinates with 0,0
                    new_coordinates = np.nan_to_num(np_coords, copy=True)
                    skeleton[bodypart] = tuple(new_coordinates)
                elif HANDLE_MISSING == "reset":
                    # reset complete skeleton to NaN, NaN
                    reset_skeleton = {bp: (np.NaN, np.NaN) for bp in skeleton}
                    animal_skeletons = [
                        reset_skeleton if i == skeleton else i for i in animal_skeletons
                    ]
                    break
                else:
                    animal_skeletons.remove(skeleton)
                    break

    return animal_skeletons


def arrange_flatskeleton(skeleton, n_animals, n_bp_animal, switch_dict):
    """changes sequence of bodypart sets (skeletons) in multi animal tracking with flat skeleton output (multiple animals in single skeleton) by switching position of pairs.
    E.g. in pose estimation with different fur colors. Note: When switching muliple animals the new position of the previous switches will be used.
    :param skeleton: flat skeleton of pose estimation in style {bp1: (x,y), bp2: (x2,y2) ...}
    :param n_animals: number of animals in total represented by skeleton
    :param n_bp_animal: number of bodyparts per animal in skeleton
    :param switch_dict: dictionary containing position of bodypart set (animal) in flat skeleton as key and bp set to exchange with as value.
            e.g. switch_dict = dict(1 = 2, 3 = 4)
    :return: skeleton with new order
    """
    flat_pose = transform_2pose(skeleton)
    ra_dict = {}
    # slicing the animals out
    for num_animal in range(n_animals):
        ra_dict[num_animal] = flat_pose[
            num_animal * n_bp_animal : num_animal * n_bp_animal + n_bp_animal
        ]
    # switching positions
    for orig_pos, switch_pos in switch_dict.items():
        # extract old
        orig = ra_dict[orig_pos]
        switch = ra_dict[switch_pos]
        # set to new position
        ra_dict[orig_pos] = switch
        ra_dict[switch_pos] = orig
    # extracting pose
    arranged_pose = np.array([*ra_dict.values()]).reshape(flat_pose.shape)
    # transforming it to skeleton
    flat_skeleton = transform_2skeleton(arranged_pose)
    return flat_skeleton


def calculate_skeletons_dlc_live(pose) -> list:
    """
    Creating skeletons from given pose
    There could be no more skeletons than animals_number
    Only unique skeletons output
    """
    skeletons = [transform_2skeleton(pose)]
    return skeletons


def calculate_sleap_skeletons(pose) -> list:
    """
    Creating skeleton from sleap output
    """
    skeletons = []
    for animal in range(pose.shape[0]):
        skeleton = transform_2skeleton(pose[animal])
        skeletons.append(skeleton)
    return skeletons


def calculate_skeletons(peaks: dict, animals_number: int) -> list:
    """
    Creating skeletons from given peaks
    There could be no more skeletons than animals_number
    Only unique skeletons output
    adaptive to chosen model origin
    """
    if MODEL_ORIGIN == "DLC":
        if USE_DLSTREAM_POSTURE_DETECTION:
            animal_skeletons = calculate_dlstream_skeletons(peaks, animals_number)
        else:
            if FILTER_LIKELIHOOD:
                peaks = filter_pose_by_likelihood(peaks, LIKELIHOOD_THRESHOLD)
            animal_skeletons = calculate_skeletons_dlc_live(peaks)

        if animals_number != 1 and SPLIT_MA:
            animal_skeletons = split_flat_skeleton(animal_skeletons)
        else:
            pass

    elif MODEL_ORIGIN == "MADLC":
        animal_skeletons = calculate_ma_skeletons(peaks, animals_number)
        if FLATTEN_MA:
            animal_skeletons = flatten_maDLC_skeletons(animal_skeletons)
        else:
            pass

    elif MODEL_ORIGIN == "DLC-LIVE" or MODEL_ORIGIN == "DEEPPOSEKIT":
        animal_skeletons = calculate_skeletons_dlc_live(peaks)
        if animals_number != 1 and not SPLIT_MA:
            raise SkeletonError(
                "Multiple animals are currently not supported by DLC-LIVE."
                " If you are using differently colored animals, please refer to the bodyparts directly (as a flattened skeleton) or use SPLIT_MA in the advanced settings."
            )
        elif SPLIT_MA:
            animal_skeletons = split_flat_skeleton(animal_skeletons)
        else:
            pass

    elif MODEL_ORIGIN == "SLEAP":
        animal_skeletons = calculate_sleap_skeletons(peaks)
        if FLATTEN_MA:
            animal_skeletons = flatten_maDLC_skeletons(animal_skeletons)
        elif animals_number != 1 and SPLIT_MA:
            animal_skeletons = split_flat_skeleton(animal_skeletons)
        else:
            pass

    animal_skeletons = handle_missing_bp(animal_skeletons)

    return animal_skeletons
