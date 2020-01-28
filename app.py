import sys
import os
import cv2

from DeepLabStream import DeepLabStream, show_stream
from utils.configloader import MULTI_CAM, STREAMS, RESOLUTION, RECORD_EXP

from PyQt5.QtCore import QObject, QThread, pyqtSignal, Qt, pyqtSlot

from PyQt5.QtWidgets import QPushButton, QApplication, QWidget, QLabel, QGridLayout

from PyQt5.QtGui import QImage, QPixmap, QIcon

width, height = RESOLUTION


class ImageWindow(QWidget):
    """
    Image window is used to show an image within PyQt GUI
    This example is hardcoded for width and height of currently defined resolution
    """
    def __init__(self, name):
        super().__init__()
        self.title = name
        self.left = 0
        self.top = 0
        self.width, self.height = width, height
        self.label = QLabel(self)
        self.init_ui()

    @pyqtSlot(QImage)
    def set_image(self, image):
        """
        This is a slot for QImage widget
        We use this to "catch" an emitted image
        :param image: QImage
        """
        # using label to output the image
        self.label.setPixmap(QPixmap.fromImage(image))

    def init_ui(self):
        """
        Creating a UI itself
        """
        # title
        self.setWindowTitle(self.title)
        # geometric parameters
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.resize(self.width, self.height)
        # create a label
        self.label.resize(self.width, self.height)

    def closeEvent(self, event):
        # keeps window from closing on accident
        event.ignore()


class QFrame(QObject):
    """
    This is a dummy class to workaround PyQt5 restrictions on declaring signals
    Basically, it does not allow to be declared in an instance of a class, or it will become bounded
    You need to define it as a class variable, but that does not work every time, especially for dynamic tasks
    In short -
    Okay:
        class A:
            smth = pyqtSignal()
    Not okay:
        class A:
            def __init__()
            self.smth = pyqtSignal()

    But both of the examples provide the same functionality, A.smth is a pyqtSignal
    So we workaround it, making a dummy class and going with the "okay" route
    This is essential for multicam support

    ! Probably have a tax on the performance
    """
    signal = pyqtSignal(QImage)


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
            all_frames = stream_manager.get_frames()
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
                    self.stream_frames(res_frames)
            else:
                self.stream_frames(color_frames)

            stream_manager.frame_index += 1

    def stop(self):
        """
        Setting thread to active, thus stopping the infinite loop
        """
        self.threadactive = False

    def stream_frames(self, frames):
        """
        Shows some number of stream frames, depending on cameras quantity
        :param frames: dictionary of frames in format of {camera:frame}
        """
        if os.name == 'nt':
            show_stream(frames)
        else:
            for camera in frames:
                # converting to RGB
                rgb_image = cv2.cvtColor(frames[camera], cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                # converting to QImage
                qpicture = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
                # scaling QImage to resolution
                scaled_qpicture = qpicture.scaled(width, height, Qt.KeepAspectRatio)
                # emitting the picture
                self.qframes[camera].signal.emit(scaled_qpicture)


class ButtonWindow(QWidget):
    def __init__(self):
        super().__init__()
        # setting the icon for window
        self.setWindowIcon(QIcon('misc/DLStream_Logo_small.png'))
        self.setWindowTitle('DeepLabStream')
        self.title = 'ButtonWindow'
        # next is the complete buttons dictionary with buttons, icons, functions and layouts
        self.buttons_dict = {'Start_Stream': {"Button": QPushButton('Start Stream'),
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
        self.buttons_pairing = {

        }
        # creating button layout with icons and functionality
        self.initialize_buttons()
        self.thread = None
        self.image_windows = {}

    def start(self):
        self.thread.start()

    def stop(self):
        self.thread.stop()

    def initialize_buttons(self):
        """
        Function to make button window great again
        Sets all buttons with an icon, function and position
        """
        layout = QGridLayout()
        for func in self.buttons_dict:
            # setting icon
            self.buttons_dict[func]["Button"].setIcon(self.buttons_dict[func]["Icon"])
            # setting function
            self.buttons_dict[func]["Button"].clicked.connect(self.buttons_dict[func]["Function"])
            # setting position
            layout.addWidget(self.buttons_dict[func]["Button"], *self.buttons_dict[func]["Layout"])
            # setting default state
            self.buttons_dict[func]["Button"].setEnabled(self.buttons_dict[func]["State"])
            # setting button size
            self.buttons_dict[func]["Button"].setMinimumHeight(100)
        # setting window layout for all buttons
        self.setLayout(layout)

    def buttons_toggle(self, *buttons):
        for button in buttons:
            self.buttons_dict[button]["Button"].setEnabled(not self.buttons_dict[button]["Button"].isEnabled())

    """ Button functions"""

    def start_stream(self):
        # initializing the stream manager cameras
        stream_manager.start_cameras(STREAMS, MULTI_CAM)

        # initializing background thread
        self.thread = AThread(self)
        self.thread.start()

        # flipping the state of the buttons
        # for func in self.buttons_dict:
        #     self.buttons_dict[func]["Button"].setEnabled(not self.buttons_dict[func]["State"])
        self.buttons_toggle("Start_Analysis",
                            "Start_Recording",
                            "Start_Stream",
                            "Stop_Stream")

        if os.name != 'nt':
            for camera in stream_manager.enabled_cameras:
                self.image_windows[camera] = ImageWindow(camera)
                self.thread.qframes[camera].signal.connect(self.image_windows[camera].set_image)
                self.image_windows[camera].show()
        else:
            pass

    def stop_stream(self):
        # stopping background thread
        self.thread.stop()

        # flipping the state of the buttons
        for func in self.buttons_dict:
            self.buttons_dict[func]["Button"].setEnabled(self.buttons_dict[func]["State"])

        if os.name != 'nt':
            for camera in self.image_windows:
                self.image_windows[camera].hide()
        else:
            pass

        print("Cameras stopping")
        stream_manager.finish_streaming()
        stream_manager._camera_manager.stop()

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
