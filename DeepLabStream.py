#!/usr/bin/env python
"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""
import multiprocessing as mp
import os
import sys
import time
from importlib.util import find_spec

import click
import cv2
import numpy as np
import pandas as pd

from utils.generic import VideoManager, WebCamManager, GenericManager
from utils.configloader import RESOLUTION, FRAMERATE, OUT_DIR, MODEL_NAME, MULTI_CAM, STACK_FRAMES, \
    ANIMALS_NUMBER, STREAMS, STREAMING_SOURCE, MODEL_ORIGIN, CROP, CROP_X, CROP_Y
from utils.plotter import plot_bodyparts,plot_metadata_frame
from utils.poser import load_deeplabcut,load_dpk,load_dlc_live,get_pose,calculate_skeletons, \
    find_local_peaks_new,get_ma_pose,load_sleap


def create_video_files(directory, devices, resolution, framerate, codec):
    """
    Function to create videofiles for every available camera at once
    :param directory: directory to output files to
    :param devices: list of camera ids/names
    :param resolution: resolution in which to save the video
    :param framerate: framerate in which to save the video
    :param codec: codec in which to save the video
    :return: dictionary with files
    """
    files = {}
    for device in devices:
        file_name = 'VideoOutput' + device + '-' + time.strftime('%d%m%Y-%H%M%S') + '.avi'
        output_file = os.path.join(directory, file_name)
        out = cv2.VideoWriter(output_file, codec, framerate, resolution)
        files[device] = out
    return files


def create_row(index, animal_skeletons, experiment_status, experiment_trial, start_time=None):
    """
    Create a pd.Series for each frame from each camera with joints position
    :param experiment_trial: current trial name
    :param experiment_status: current experiment status
    :param index: frame index
    :param animal_skeletons: skeletons for that frame
    :param start_time: (optional) starting time point for Time column
    """
    row_dict = {}
    # creating joints columns
    for num, animal in enumerate(animal_skeletons):
        for joint, value in animal.items():
            row_dict[("Animal{}".format(num + 1),
                      joint, 'x')], row_dict[("Animal{}".format(num + 1), joint, 'y')] = value
    # optional time column
    if start_time is not None:
        row_dict[('Time', '', '')] = round(time.time() - start_time, 3)
    # experiment columns
    row_dict[('Experiment', 'Status', '')] = experiment_status
    if experiment_trial is None and experiment_status:
        row_dict[('Experiment', 'Trial', '')] = 'InterTrial'
    else:
        row_dict[('Experiment', 'Trial', '')] = experiment_trial
    row = pd.Series(row_dict, name=index)
    return row


def show_stream(frames):
    """
    Shows some number of stream frames, depending on cameras quantity
    :param frames: dictionary of frames in format of {camera:frame}
    """
    if STACK_FRAMES:
        if len(frames.values()) >= 2:
            stack = np.hstack(frames.values())
            cv2.imshow('stacked stream', stack.copy())
    else:
        for camera in frames:
            cv2.imshow(str(camera) + ' stream', frames[camera])


def cls():
    """
    Clear the screen
    """
    os.system('cls' if os.name == 'nt' else 'clear')


class DeepLabStream:
    """
    Class for managing everything stream-related
    """

    def __init__(self):
        """
        Initializing the DeepLabStream class with some predefined variables
        (more information for each below)
        """
        self._camera_manager = self.set_camera_manager()  # camera manager, used to get frames from the camera
        self._video_codec = cv2.VideoWriter_fourcc(*'DIVX')  # codec in which we output the videofiles
        self._start_time = None  # time of initialization
        self._data_row = {camera: {} for camera in self.cameras}  # dictionary for creating row of data for each frame
        self._data_output = {}  # dictionary for storing rows for dataframes
        self._stored_frames = {}  # dictionary for storing frames
        self._dlc_running = False  # has DeepLabCut started?
        self._experiment_running = False  # has experiment started?
        self._recording_running = False  # has recording started?
        self._video_files = None
        self._multiprocessing = None  # variable for multiprocessing tools
        self._experiment = self.set_up_experiment()
        self.frame_index = 0
        self._fps_counter = []
        self._fps = 0
        self.greetings()

    @staticmethod
    def set_camera_manager():
        """
        Trying to load each present camera manager, if installed
        Then checking for connected cameras and choosing the one with at least some cameras connected
        ! Camera managers cannot be mixed
        :return: the chosen camera manager
        """

        def select_camera_manager():
            """
            Function to select from all available camera managers
            """
            manager_list = []
            # loading realsense manager, if installed
            if find_spec("pyrealsense2") is not None:
                from utils.realsense import RealSenseManager
                realsense_manager = RealSenseManager()
                manager_list.append(realsense_manager)

            # loading basler manager, if installed
            if find_spec("pypylon") is not None:
                from utils.pylon import PylonManager
                pylon_manager = PylonManager()
                manager_list.append(pylon_manager)

            def check_for_cameras(camera_manager):
                """
                Helper method to get cameras, connected to that camera manager
                """
                cameras = camera_manager.get_connected_devices()
                if cameras:
                    print("Found {} {} camera(s)!".format(len(cameras), camera_manager.get_name()))
                    return True
                else:
                    return False

            # checking for connected cameras for all installed managers
            for manager in manager_list:
                if check_for_cameras(manager):
                    return manager
            else:
                # if no camera is found, try generic openCV manager
                generic_manager = GenericManager()
                return generic_manager

        MANAGER_SOURCE = {
            'video': VideoManager,
            'ipwebcam': WebCamManager,
            'camera': select_camera_manager
        }

        # initialize selected manager
        camera_manager = MANAGER_SOURCE.get(STREAMING_SOURCE)()
        if camera_manager is not None:
            return camera_manager
        else:
            raise ValueError(f'Streaming source {STREAMING_SOURCE} is not a valid option. \n'
                             f'Please choose from "video", "camera" or "ipwebcam".')

    @property
    def cameras(self):
        """
        Used to dynamically get every connected camera from camera manager
        """
        return self._camera_manager.get_connected_devices()

    @property
    def enabled_cameras(self):
        """
        Used to dynamically get every enabled camera from camera manager
        """
        return self._camera_manager.get_enabled_devices()

    ##########################
    # start cameras and stream
    ##########################
    def enable_streams(self, streams: list):
        """
        Enabling streams if they are recognized
        :param streams: list of stream names
        """
        available_streams = ['color', 'depth', 'infrared']
        for stream in streams:
            if stream in available_streams:
                self._camera_manager.enable_stream(RESOLUTION, FRAMERATE, stream)
            else:
                print('Stream type {} not found!\n'
                      'Allowed stream types: {}'.format(stream, available_streams))

    def start_cameras(self, streams: list, multi_cam: bool = True):
        """
        Enabling streams and starting cameras with enabled streams
        If multicam then use all available cameras, else prompt for each
        :param streams: list
        :param multi_cam: bool
        """
        # enabling streams
        self.enable_streams(streams)
        # starting cameras
        # self._start_time = time.time()
        if multi_cam:
            print("Starting all available cameras")
            self._camera_manager.enable_all_devices()
        else:
            if len(self.cameras) != 0:
                print("Starting first available camera {}".format(self.cameras[0]))
                self._camera_manager.enable_device(self.cameras[0])
            else:
                print("No cameras found.")
                print("Exiting script...")
                sys.exit()
        # creating output for dataframe
        # self.create_output()

    def stop_cameras(self):
        print("Stopping all cameras")
        self._camera_manager.stop()

    #######################
    # video outputting part
    #######################
    def start_recording(self):
        self.create_videofiles()
        self._recording_running = True

    def create_videofiles(self):
        """
        Create video files dictionary by cameras names
        """
        self._video_files = create_video_files(OUT_DIR, self.enabled_cameras, RESOLUTION, FRAMERATE, self._video_codec)

    def write_video(self, frames: dict, index: int):
        """
        Make actual videos from frames
        :param frames: dict with frames, sorted by camera
        :param index: int frame number
        """
        for camera in frames:
            font = cv2.FONT_HERSHEY_SIMPLEX
            # puts the frame index in the top-left corner
            cv2.putText(frames[camera], str(index), (1, 15), font, 0.5, (0, 0, 255))
            self._video_files[camera].write(frames[camera])

    ######################
    # setting up DLC usage
    ######################
    @staticmethod
    def get_pose_mp(input_q, output_q):
        """
        Process to be used for each camera/DLC stream of analysis
        Designed to be run in an infinite loop
        :param input_q: index and corresponding frame
        :param output_q: index and corresponding analysis
        """

        if MODEL_ORIGIN in ('DLC', 'MADLC'):
            config, sess, inputs, outputs = load_deeplabcut()
            while True:
                if input_q.full():
                    index, frame = input_q.get()
                    if MODEL_ORIGIN == 'DLC':
                        scmap, locref, pose = get_pose(frame, config, sess, inputs, outputs)
                        # TODO: Remove alterations to original
                        #peaks = find_local_peaks_new(scmap, locref, ANIMALS_NUMBER, config)
                        peaks = pose
                    if MODEL_ORIGIN == 'MADLC':
                        peaks = get_ma_pose(frame, config, sess, inputs, outputs)

                    output_q.put((index, peaks))

        elif MODEL_ORIGIN == 'DLC-LIVE':
            dlc_live = load_dlc_live()
            while True:
                if input_q.full():
                    index, frame = input_q.get()
                    if not dlc_live.is_initialized:
                        peaks = dlc_live.init_inference(frame)
                    else:
                        peaks = dlc_live.get_pose(frame)

                    output_q.put((index, peaks))

        elif MODEL_ORIGIN == 'DEEPPOSEKIT':
            predict_model = load_dpk()
            while True:
                if input_q.full():
                    index, frame = input_q.get()
                    frame = frame[..., 1][..., None]
                    st_frame = np.stack([frame])
                    prediction = predict_model.predict(st_frame, batch_size=1, verbose=True)
                    peaks = prediction[0, :, :2]
                    output_q.put((index, peaks))

        elif MODEL_ORIGIN == 'SLEAP':
            sleap_model = load_sleap()
            while True:
                if input_q.full():
                    index, frame = input_q.get()
                    input_frame = frame[:, :, ::-1]
                    #this is weird, but without it, it does not seem to work...
                    frames = np.array([input_frame])
                    prediction = sleap_model.predict(frames[[0]], batch_size=1)
                    #check if this is multiple animal instances or single animal model
                    if  sleap_model.name == 'single_instance_inference_model':
                        #get predictions (wrap it again, so the behavior is the same for both model types)
                        peaks = np.array([prediction['peaks'][0, :]])
                    else:
                        peaks = prediction['instance_peaks'][0, :]
                    output_q.put((index, peaks))
        else:
            raise ValueError(f'Model origin {MODEL_ORIGIN} not available.')

    @staticmethod
    def create_mp_tools(devices):
        """
        Creating easy to use dictionaries for our multiprocessing needs
        :param devices: list of cameras for each we should create a separate process
        :return: dictionary with process and queues for each device
        """
        device_mps = {}
        for device in devices:
            # creating queues
            device_mps[device] = {'input': mp.Queue(1), 'output': mp.Queue(1)}

            # creating process
            process = mp.Process(target=DeepLabStream.get_pose_mp, args=(device_mps[device]['input'],
                                                                         device_mps[device]['output']),
                                 name=device)
            device_mps[device]['process'] = process
        return device_mps

    def set_up_multiprocessing(self):
        """
        Creating multiprocessing tools
        """
        self._multiprocessing = self.create_mp_tools(self.enabled_cameras)

    def start_dlc(self):
        """
        Starting DLC in stand-alone process for each available camera
        """
        for camera in self.enabled_cameras:
            self._multiprocessing[camera]['process'].start()
        self._dlc_running = True

    #####################
    # working with frames
    #####################
    def get_frames(self) -> tuple:
        """
        Get new frameset from each camera and make color frames and infrared frames useful
        :return: color_frames, depth_maps, infrared_frames
        """
        c_frames, d_maps, i_frames = self._camera_manager.get_frames()
        for camera in c_frames:
            c_frames[camera] = np.asanyarray(c_frames[camera])
            if CROP:
                c_frames[camera] = c_frames[camera][CROP_Y[0]:CROP_Y[1],CROP_X[0]:CROP_X[1]].copy()

        for camera in i_frames:
            i_frames[camera] = np.asanyarray(i_frames[camera])

        return c_frames, d_maps, i_frames

    def input_frames_for_analysis(self, frames: tuple, index: int):
        """
        Passing the frame and its index to DLC process
        :param frames: frameset from camera manager
        :param index: index of a frameset
        """
        if self._dlc_running:
            c_frames, d_maps, i_frames = frames
            for camera in self._multiprocessing:
                if self._multiprocessing[camera]['input'].empty():
                    # passes color frame to analysis
                    frame = c_frames[camera]
                    frame_time = time.time()
                    self._multiprocessing[camera]['input'].put((index, frame))
                    if d_maps:
                        self.store_frames(camera, frame, d_maps[camera], frame_time, index)
                    else:
                        self.store_frames(camera, frame, None, frame_time, index)

    def get_analysed_frames(self) -> tuple:
        """
        The main magic is happening here
        Getting the data from DLC processes
        Plotting the data
        Checking data in experiments
        Gathering data to a series
        """
        if self._dlc_running:
            analysed_frames = {}
            analysis_time = None
            frame_width, frame_height = RESOLUTION
            for camera in self._multiprocessing:
                if self._multiprocessing[camera]['output'].full():
                    if self._start_time is None:
                        self._start_time = time.time()  # getting the first frame here

                    # Getting the analysed data
                    analysed_index, peaks = self._multiprocessing[camera]['output'].get()
                    skeletons = calculate_skeletons(peaks, ANIMALS_NUMBER)
                    print('', end='\r', flush=True)  # this is the line you should not remove
                    analysed_frame , depth_map, input_time = self.get_stored_frames(camera, analysed_index)
                    analysis_time = time.time() - input_time
                    # Calculating FPS and plotting the data on frame
                    self.calculate_fps(analysis_time if analysis_time != 0 else 0.01)
                    frame_time = time.time() - self._start_time
                    analysed_image = plot_metadata_frame(
                        plot_bodyparts(analysed_frame, skeletons),
                        frame_width, frame_height, self._fps, frame_time)

                    # Experiments
                    if self._experiment.experiment_finished and self._experiment_running:
                        self._experiment_running = False

                    if self._experiment_running and not self._experiment.experiment_finished:
                        # #TODO: Update to work for multiple animal and single animal experiments
                        # #Shift responsibility to experiments
                        # if ANIMALS_NUMBER > 1:
                        #     self._experiment.check_skeleton(analysed_image,skeletons)
                        # else:
                        for skeleton in skeletons:
                            self._experiment.check_skeleton(analysed_image, skeleton)

                    # Gathering data as pd.Series for output
                    if self._data_output:
                        self.append_row(camera, analysed_index, skeletons,
                                        self._experiment_running, self._experiment.get_trial(), self._start_time)

                    analysed_frames[camera] = analysed_image
            return analysed_frames, analysis_time

    def store_frames(self, camera: str, c_frame, d_map, frame_time: float, index: int):
        """
        Store frames currently sent for analysis in index based dictionary
        :param camera: camera name
        :param c_frame: color frame
        :param d_map: depth map
        :param frame_time: inputting time of frameset
        :param index: index of frame that is currently analysed
        """
        if camera in self._stored_frames.keys():
            self._stored_frames[camera][index] = c_frame, d_map, frame_time

        else:
            self._stored_frames[camera] = {}
            self._stored_frames[camera][index] = c_frame, d_map, frame_time

    def get_stored_frames(self, camera: str, index: int):
        """
        Retrieve frames currently sent for analysis, retrieved frames will be removed (popped) from the dictionary
        :param camera: camera name
        :param index: index of analysed frame
        :return:
        """
        c_frame, d_map, frame_time = self._stored_frames[camera].pop(index, None)
        return c_frame, d_map, frame_time

    def convert_depth_map_to_image(self, d_map):
        """
        Colorize depth map using build in camera manager colorizer
        :param d_map: depth map
        :return: colorized image of a depth map
        """
        colorized_depth_frame = self._camera_manager.colorize_depth_frame(d_map)
        return colorized_depth_frame

    def calculate_fps(self, current_analysis_time):
        """
        Calculates average FPS for 10 frames
        :param current_analysis_time: instant analysis time for one frame
        """
        current_fps = 1 / current_analysis_time
        self._fps_counter.append(current_fps)
        if len(self._fps_counter) == 10:
            self._fps = np.average(self._fps_counter)
            self._fps_counter.clear()

    ##################
    # experiments part
    ##################
    @staticmethod
    def set_up_experiment():
        from experiments.utils.exp_setup import setup_experiment
        experiment = setup_experiment()
        return experiment

    def start_experiment(self):
        if not self._experiment_running:
            self._experiment.start_experiment()
            self._experiment_running = True

    #########################
    # after streaming is done
    #########################
    def stop_recording(self):
        # finishing with the videos
        if self._recording_running:
            print("Saving the video")
            for file in self._video_files.values():
                file.release()
            self._recording_running = False
            self._video_files = None
            print("Video saved")

    def stop_dlc(self):
        # cleaning up the dlc processes
        if self._dlc_running:
            for camera in self._multiprocessing:
                # finishing the process
                self._multiprocessing[camera]['process'].terminate()
                # closing all the Queues
                self._multiprocessing[camera]['input'].close()
                self._multiprocessing[camera]['output'].close()
            self._dlc_running = False
            self._multiprocessing = None
            self._start_time = None
            # writing database
            if self._data_output:
                self.create_dataframes()

    def stop_experiment(self):
        # stopping the experiment
        if self._experiment_running:
            self._experiment.stop_experiment()
            self._experiment_running = False
            self._experiment = self.set_up_experiment()

    def finish_streaming(self):
        """
        Clean up after ourselves
        """
        self.stop_experiment()
        self.stop_dlc()
        self.stop_recording()
        cv2.destroyAllWindows()

    #####################
    # working with pandas
    #####################
    def create_output(self):
        """
        Create lists to contain serialized skeletons
        """
        for camera in self.enabled_cameras:
            self._data_output[camera] = []

    def append_row(self, camera, index, animal_skeletons, experiment_status, experiment_trial, start_time=None):
        """
        Create a pd.Series for each frame from each camera with joints position and store it
        :param experiment_trial: current trial name
        :param experiment_status: current experiment status
        :param camera: camera name
        :param index: frame index
        :param animal_skeletons: skeletons for that frame
        :param start_time: (optional) starting time point for Time column
        """
        row = create_row(index, animal_skeletons, experiment_status, experiment_trial, start_time)
        self._data_output[camera].append(row)

    def create_dataframes(self):
        """
        Outputting dataframes to csv
        """
        for num, camera in enumerate(self._data_output):
            print("Saving database for device {}".format(camera))
            df = pd.DataFrame(self._data_output[camera])
            df.index.name = 'Frame'
            df.to_csv(OUT_DIR + '/DataOutput{}'.format(camera) + '-' + time.strftime('%d%m%Y-%H%M%S') + '.csv', sep=';')
            print("Database saved")

    ######
    # meta
    ######
    @staticmethod
    def greetings():
        from utils.configloader import EXP_NAME, EXP_ORIGIN
        print("This is DeepLabStream")
        print("Developed by: Jens Schweihoff and Matvey Loshakov")
        print(f'Initializing {EXP_ORIGIN.lower()} experiment: {EXP_NAME}...')

    def get_camera_manager(self):
        return self._camera_manager

    def get_video_files(self):
        return self._video_files

    def get_multiprocessing_tools(self):
        return self._multiprocessing

    def get_enabled_cameras(self):
        return self.enabled_cameras

    def get_connected_cameras(self):
        return self._camera_manager.get_connected_devices()

    def dlc_status(self):
        return self._dlc_running

    def experiment_status(self):
        return self._experiment_running

    def recording_status(self):
        return self._recording_running

    def get_fps(self):
        return self._fps

    def get_start_time(self):
        return self._start_time


# testing part
@click.command()
@click.option('--dlc-enabled', 'dlc_enabled', is_flag=True)
@click.option('--benchmark-enabled', 'benchmark_enabled', is_flag=True)
@click.option('--recording-enabled', 'recording_enabled', is_flag=True)
@click.option('--data-output-enabled', 'data_output_enabled', is_flag=True)
def start_deeplabstream(dlc_enabled, benchmark_enabled, recording_enabled, data_output_enabled):
    if not dlc_enabled and benchmark_enabled:
        print("Cannot benchmark with DLC turned off")
        print("Please enable DLC with --dlc-enabled flag")

    if not dlc_enabled and data_output_enabled:
        print("Cannot output data with DLC turned off")
        print("Please enable DLC with --dlc-enabled flag")

    # initializing DeepLabStream
    print("Your current config:")
    print("Resolution : {}".format(RESOLUTION))
    print("Framerate : {}".format(FRAMERATE))
    print("Output directory : {}".format(OUT_DIR))
    print("Use multiple cameras : {}".format(MULTI_CAM))
    print("DLC enabled: {}".format(dlc_enabled))
    print("Benchmarking enabled: {}".format(benchmark_enabled))
    print("Data output enabled: {}".format(data_output_enabled))
    print("Start with current config? (y/n)")
    if input().lower() != 'y':
        print("Please edit config and restart the script.")
        print("Exiting script...")
        sys.exit()
    print("Starting DeepLabStream manager")
    stream_manager = DeepLabStream()

    # starting streams
    print("Starting cameras with defined above streams")
    stream_manager.start_cameras(STREAMS, MULTI_CAM)

    if dlc_enabled:
        # starting DeepLabCut
        print("Starting DeepLabCut")
        stream_manager.set_up_multiprocessing()
        stream_manager.start_dlc()

    if benchmark_enabled:
        # benchmarking tools
        fps_data = []
        analysis_time_data = []
        whole_loop_time_data = []
        tracking_accuracy_counter = 0

        def describe_dataset(dataset, name):
            """
            Function to describe dataset and print out the results
            """
            average = np.average(dataset)
            print("Average {0} time {1:6.5f}".format(name, average))
            maximum = np.max(dataset)
            print("Maximum {0} time {1:6.5f}".format(name, maximum))
            minimum = np.min(dataset)
            print("Minimum {0} time {1:6.5f}".format(name, minimum))
            standart_deviation = np.std(dataset)
            print("Standard deviation {0} time {1:6.5f}".format(name, standart_deviation))

        def show_benchmark_statistics():
            """
            Outputting all stream benchmark statistics
            """
            analysis_full_time = time.time() - start_time
            print("Full analysis time {0:4.2f}".format(analysis_full_time))
            ##################################################################
            avg_fps = np.average(fps_data)
            print("Average FPS {0:4.2f}".format(avg_fps))
            frame_count = len(fps_data)
            print("Got total {0:.0f}".format(frame_count))
            tracking_accuracy = tracking_accuracy_counter / frame_count * 100
            print("Tracking accuracy {0:.3f}%".format(tracking_accuracy))
            ##################################################################
            describe_dataset(analysis_time_data, "analysis")
            describe_dataset(whole_loop_time_data, "whole_loop")

    if recording_enabled:
        stream_manager.start_recording()

    if data_output_enabled:
        stream_manager.create_output()

    got_first_analysed_frame = False

    while True:
        loop_time = time.time()  # start of the loop
        all_frames = stream_manager.get_frames()
        color_frames, depth_maps, infrared_frames = all_frames
        if recording_enabled:
            stream_manager.write_video(color_frames, stream_manager.frame_index)
        if dlc_enabled:
            ###########################################################
            # Analysis part
            # outputting the frames
            res_frames, res_time = stream_manager.get_analysed_frames()
            # inputting the frames
            stream_manager.input_frames_for_analysis(all_frames, stream_manager.frame_index)

            ###########################################################
            # Benchmarking part
            if res_time is not None:
                if benchmark_enabled and got_first_analysed_frame:
                    analysis_time_data.append(res_time)
            else:
                if benchmark_enabled and got_first_analysed_frame:
                    tracking_accuracy_counter += 1

            ###########################################################
            # streaming the stream
            if res_frames:
                if not got_first_analysed_frame and benchmark_enabled:
                    got_first_analysed_frame = True
                show_stream(res_frames)
        else:
            show_stream(color_frames)

        ###########################################################
        # finishing the loop
        stream_manager.frame_index += 1

        whole_loop = time.time() - loop_time
        current_fps = stream_manager.get_fps()
        if benchmark_enabled:
            if got_first_analysed_frame:
                fps_data.append(current_fps)
                whole_loop_time_data.append(whole_loop)

        # exit clauses
        if cv2.waitKey(1) & 0xFF == ord('q'):
            start_time = stream_manager.get_start_time()
            stream_manager.finish_streaming()
            if benchmark_enabled:
                print("Benchmark statistics:")
                show_benchmark_statistics()
            if recording_enabled:
                stream_manager.stop_recording()
            break

        if benchmark_enabled:
            if len(analysis_time_data) > 3000:
                start_time = stream_manager.get_start_time()
                stream_manager.finish_streaming()
                print("Benchmark statistics:")
                show_benchmark_statistics()
                if recording_enabled:
                    stream_manager.stop_recording()
                break
            elif got_first_analysed_frame:
                print("[{0}/3000] Benchmarking in progress".format(len(analysis_time_data)))

    if benchmark_enabled:
        model_parts = MODEL_NAME.split('_')
        if len(model_parts) == 3:
            short_model = model_parts[0] + '_' + model_parts[2]
        else:
            short_model = MODEL_NAME
        # the best way to save files
        np.savetxt(f'{OUT_DIR}/{short_model}_framerate_{FRAMERATE}_resolution_{RESOLUTION[0]}_{RESOLUTION[1]}.txt',
                   np.transpose([fps_data, whole_loop_time_data]))


if __name__ == '__main__':
    mp.freeze_support()
    cls()
    start_deeplabstream()
