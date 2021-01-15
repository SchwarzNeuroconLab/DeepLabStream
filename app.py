"""
DeepLabStream
© J.Schweihoff, M. Loshakov
University Bonn Medical Faculty, Germany
https://github.com/SchwarzNeuroconLab/DeepLabStream
Licensed under GNU General Public License v3.0
"""

import sys
import os
import cv2

from DeepLabStream import DeepLabStream, show_stream
from utils.generic import MissingFrameError
from utils.configloader import MULTI_CAM, STREAMS, RECORD_EXP
from utils.gui_image import QFrame, ImageWindow, emit_qframes

from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QPushButton, QApplication, QWidget, QGridLayout
from PyQt5.QtGui import QIcon


# creating a complete thread process to work in the background
class AThread(QThread):
    """
    QThread is just one of the many PyQt ways to do multitasking
    This is, for most intents and purposes, identical to Python multithreading
    """
    def start(self, **kwargs):
        """
        Setting thread to active, creating a QFrames dictionary
        Then just starting parent method
        """
        self.threadactive = True
        self.qframes = {}
        # changePixmap = pyqtSignal(QImage)
        for camera in stream_manager.enabled_cameras:
            self.qframes[camera] = QFrame()
        super().start(**kwargs)

    def run(self):
        """
        Infinite loop with all the streaming, analysis and recording logic
        """
        while self.threadactive:
            try:
                all_frames = stream_manager.get_frames()
            except MissingFrameError as e:
                """catch missing frame, stop Thread and save what can be saved"""
                print(*e.args, '\nShutting down DLStream and saving data...')
                stream_manager.finish_streaming()
                stream_manager.stop_cameras()
                self.stop()
                break

            color_frames, depth_maps, infra_frames = all_frames

            # writing the video
            if stream_manager.recording_status():
                stream_manager.write_video(color_frames, stream_manager.frame_index)

            if stream_manager.dlc_status():
                # outputting the frames
                res_frames, res_time = stream_manager.get_analysed_frames()
                # inputting the frames
                stream_manager.input_frames_for_analysis(all_frames, stream_manager.frame_index)
                # streaming the stream
                if res_frames:
                    self._stream_frames(res_frames)
            else:
                self._stream_frames(color_frames)

            stream_manager.frame_index += 1

    def stop(self):
        """
        Setting thread to active, thus stopping the infinite loop
        """
        self.threadactive = False

    def _stream_frames(self, frames):
        """
        Shows some number of stream frames, depending on cameras quantity
        Method of streaming depends on platform
        Windows -> through openCV with their window objects
        Unix -> thought PyQt with some widget window
        :param frames: dictionary of frames in format of {camera:frame}
        """
        if os.name == 'nt':
            show_stream(frames)
            # very important line for openCV to work correctly
            # actually does nothing, but do NOT delete
            cv2.waitKey(1)
        else:
            emit_qframes(frames, self.qframes)


class ButtonWindow(QWidget):
    def __init__(self):
        super().__init__()
        # setting the icon for window
        self.setWindowIcon(QIcon('misc/DLStream_Logo_small.png'))
        self.setWindowTitle('DeepLabStream')
        self.title = 'ButtonWindow'
        # next is the complete buttons dictionary with buttons, icons, functions and layouts
        self._buttons_dict = {'Start_Stream': {"Button": QPushButton('Start Stream'),
                                               "Icon": QIcon('misc/StartStream2.png'),
                                               "Function": self.start_stream,
                                               "Layout": (0, 0, 2, 2),
                                               "State": True},

                              'Start_Analysis': {"Button": QPushButton('Start Analysis'),
                                                 "Icon": QIcon('misc/StartAnalysis2.png'),
                                                 "Function": self.start_analysis,
                                                 "Layout": (2, 0, 2, 1),
                                                 "State": False},

                              'Start_Experiment': {"Button": QPushButton('Start Experiment'),
                                                   "Icon": QIcon('misc/StartExperiment2.png'),
                                                   "Function": self.start_experiment,
                                                   "Layout": (4, 0, 2, 1),
                                                   "State": False},

                              'Start_Recording': {"Button": QPushButton('Start Recording'),
                                                  "Icon": QIcon('misc/StartRecording2.png'),
                                                  "Function": self.start_recording,
                                                  "Layout": (6, 0, 2, 1),
                                                  "State": False},

                              'Stop_Stream': {"Button": QPushButton('Stop Stream'),
                                              "Icon": QIcon('misc/StopStream2.png'),
                                              "Function": self.stop_stream,
                                              "Layout": (8, 0, 2, 2),
                                              "State": False},

                              'Stop_Analysis': {"Button": QPushButton('Stop Analysis'),
                                                "Icon": QIcon('misc/StopAnalysis2.png'),
                                                "Function": self.stop_analysis,
                                                "Layout": (2, 1, 2, 1),
                                                "State": False},

                              'Stop_Experiment': {"Button": QPushButton('Stop Experiment'),
                                                  "Icon": QIcon('misc/StopExperiment2.png'),
                                                  "Function": self.stop_experiment,
                                                  "Layout": (4, 1, 2, 1),
                                                  "State": False},

                              'Stop_Recording': {"Button": QPushButton('Stop Recording'),
                                                 "Icon": QIcon('misc/StopRecording2.png'),
                                                 "Function": self.stop_recording,
                                                 "Layout": (6, 1, 2, 1),
                                                 "State": False}}

        # creating button layout with icons and functionality
        self.initialize_buttons()
        self._thread = None
        self.image_windows = {}

    def start(self):
        self._thread.start()

    def stop(self):
        self._thread.stop()

    def initialize_buttons(self):
        """
        Function to make button window great again
        Sets all buttons with an icon, function and position
        """
        layout = QGridLayout()
        for func in self._buttons_dict:
            # setting icon
            self._buttons_dict[func]["Button"].setIcon(self._buttons_dict[func]["Icon"])
            # setting function
            self._buttons_dict[func]["Button"].clicked.connect(self._buttons_dict[func]["Function"])
            # setting position
            layout.addWidget(self._buttons_dict[func]["Button"], *self._buttons_dict[func]["Layout"])
            # setting default state
            self._buttons_dict[func]["Button"].setEnabled(self._buttons_dict[func]["State"])
            # setting button size
            self._buttons_dict[func]["Button"].setMinimumHeight(100)
        # setting window layout for all buttons
        self.setLayout(layout)

    def buttons_toggle(self, *buttons):
        for button in buttons:
            self._buttons_dict[button]["Button"].setEnabled(not self._buttons_dict[button]["Button"].isEnabled())

    """ Button functions"""

    def start_stream(self):
        # initializing the stream manager cameras
        stream_manager.start_cameras(STREAMS, MULTI_CAM)

        # initializing background thread
        self._thread = AThread(self)
        self._thread.start()
        print("Streaming started")

        # flipping the state of the buttons
        self.buttons_toggle("Start_Analysis",
                            "Start_Recording",
                            "Start_Stream",
                            "Stop_Stream")

        # initializing image windows for Unix systems via PyQt
        if os.name != 'nt':
            for camera in stream_manager.enabled_cameras:
                self.image_windows[camera] = ImageWindow(camera)
                self._thread.qframes[camera].signal.connect(self.image_windows[camera].set_image)
                self.image_windows[camera].show()
        else:
            # for Windows it is taken care by openCV
            pass

    def stop_stream(self):
        # stopping background thread
        self._thread.stop()

        # flipping the state of the buttons
        for func in self._buttons_dict:
            self._buttons_dict[func]["Button"].setEnabled(self._buttons_dict[func]["State"])

        if os.name != 'nt':
            for camera in self.image_windows:
                self.image_windows[camera].hide()
        else:
            pass

        print("Streaming stopped")
        stream_manager.finish_streaming()
        stream_manager.stop_cameras()

    def start_analysis(self):
        print("Analysis starting")
        self.buttons_toggle("Stop_Analysis", "Start_Analysis", "Start_Experiment")
        stream_manager.set_up_multiprocessing()
        stream_manager.start_dlc()
        stream_manager.create_output()

    def stop_analysis(self):
        print("Analysis stopped")
        self.buttons_toggle("Stop_Analysis", "Start_Analysis", "Start_Experiment")
        stream_manager.stop_dlc()

    def start_experiment(self):
        print("Experiment started")
        self.buttons_toggle("Stop_Experiment",
                            "Start_Experiment")
        stream_manager.set_up_experiment()
        stream_manager.start_experiment()
        if RECORD_EXP:
            self.start_recording()

    def stop_experiment(self):
        print("Experiment stopped")
        self.buttons_toggle("Stop_Experiment",
                            "Start_Experiment")
        stream_manager.stop_experiment()
        if RECORD_EXP:
            self.stop_recording()

    def start_recording(self):
        print("Recording started")
        self.buttons_toggle("Stop_Recording",
                            "Start_Recording")
        stream_manager.start_recording()

    def stop_recording(self):
        print("Recording stopped")
        self.buttons_toggle("Stop_Recording",
                            "Start_Recording")
        stream_manager.stop_recording()


if __name__ == "__main__":
    stream_manager = DeepLabStream()
    app = QApplication([])
    bt = ButtonWindow()
    bt.show()
    sys.exit(app.exec_())
