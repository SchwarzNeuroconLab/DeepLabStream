# DeepLabStream
![GraphAbstract](docs/GraphAbstract.png)
DeepLabStream is a python based multi-purpose tool that enables the realtime tracking of animals and manipulation of experiments.
Our toolbox is adapted from the previously published [DeepLabCut](https://github.com/AlexEMG/DeepLabCut) ([Mathis et al., 2018](https://www.nature.com/articles/s41593-018-0209-y)] and expands on its core capabilities.
DeepLabStreams core feature is the real-time analysis using any type of camera-based video stream (incl. multiple streams).  Building onto that, we designed a full experimental closed-loop toolkit. It enables running experimental protocols that are dependent on a constant stream of bodypart positions and feedback activation of several input/output devices. It's capabilities range from simple region of interest (ROI) based triggers to headdirection or behavior dependent stimulation.

## Installation

### Hardware requirements

Your hardware, obviously, should be in compliance with DeepLabCut requirements to be able to run DeepLabStream. 
But DeepLabStream also requires more processing power than DeepLabCut to be easy and convenient to use.
We are really **not** recommending using this software without GPU, even though DeepLabCut supports this.

In general, you need:
- CUDA-compatible Nvidia videocard (Nvidia GTX 1080 or better is recommended);
- CPU with at least 4 cores to properly utilize parallelization;
- Decent amount of RAM, at least 16 Gb;

We tested DeepLabStream with different setups and would say that a minimum reasonable configuration would be:
```
CPU: Intel Core i7-7700K CPU @ 4.20GHz
RAM: 32GB DDR4
GPU: Nvidia GeForce GTX 1050 (3GB)
```

However, our recommended setup, with which we did achieve constant 30 FPS with a two camera setup at 848x480 resolution:
```
CPU: Intel Core i7-9700K @ 3.60GHz
RAM: 64 GB DDR4
GPU: Nvidia GeForce RTX 2080 (12GB) 
```

### Preparations

In short, you need to be able to run DeepLabCut on your system before installing and running DeepLabStream.

[Here](https://github.com/AlexEMG/DeepLabCut/blob/master/docs/installation.md) is a full instruction by DeepLabCut,
but we will provide a short version/checklist below.

1. Make sure that you have the proper Nvidia drivers installed;
2. Install [CUDA](https://developer.nvidia.com/cuda-downloads) by Nvidia.
Please refer to this [table](https://stackoverflow.com/questions/30820513/what-is-the-correct-version-of-cuda-for-my-nvidia-driver/30820690#30820690) to ensure you have the correct driver/CUDA combination;
3. Verify your CUDA installation on [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#verify-installation)
or [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#verify-installation);
4. Create an environment. We strongly recommend using [environments provided by DeepLabCut](https://github.com/AlexEMG/DeepLabCut/blob/master/conda-environments/README.md);
5. If you are not using DeepLabCut-provided environments for step 4, install [cuDNN](https://developer.nvidia.com/cudnn).
Otherwise, skip this step;
6. Make sure that [Tensorflow](https://www.tensorflow.org/install/) is installed in your environment. Manual installation goes as followed:
    ```bash
    pip install tensorflow-gpu==1.12
    ``` 
    But a lot of different problems could arise, depending on your software and hardware setup;
7. Verify that your TensorFlow is working correctly by using [this](https://askubuntu.com/questions/872098/how-to-check-if-i-installed-tensorflow-with-gpu-support-correctly/875992) (Linux)
or [this](https://towardsdatascience.com/installing-tensorflow-with-cuda-cudnn-and-gpu-support-on-windows-10-60693e46e781) (Windows) manual. 
The latter also provides a great overview of the whole process with the previous six steps.

DeepLabStream was originally designed with [DeepLabCut v1.11](https://github.com/AlexEMG/DeepLabCut/blob/1.11/docs/installation.md) in mind, but for the ease of installation and future-proofing we recommend current [DeepLabCut 2.x](https://github.com/AlexEMG/DeepLabCut) ([Nath, Mathis et al, 2019](https://www.nature.com/articles/s41596-019-0176-0)) . Both versions and networks trained with them worked fine within our tests.

### DeepLabStream installation

The easiest way of installing DeepLabStream would be the following:

*(Make sure that you are working in the same environment that you installed DeepLabCut in!)*

```bash
git clone https://github.com/SchwarzNeuroconLab/DeepLabStream.git
cd DeepLabStream
pip install -r requirements.txt
```

#### Config editing

You need to modify the DeepLabStream config in `settings.ini`  after installation to specify with which model and DeepLabCut folder it will work.

1. Change `DLC_PATH` variable to wherever your DeepLabCut installation is.
 
If you installed it like a package with DeepLabCut's provided environment files, it would be approximately here in your Anaconda environment:
```../anaconda3/envs/dlc-ubuntu-GPU/lib/python3.6/site-packages/deeplabcut```. Of course, the specific folder may vary.

2. Change the `MODEL` variable to the name of your model, found in `../deeplabcut/pose_estimation_tensorflow/models` (`../deeplabcut/pose_estimation/models` for DLC v1.11) folder.
if you are using DeepLabCut 2.0+ you first have to copy the model folder from the corresponding DLC project directory into the aforementioned pose estimation models folder.
3. Change variables in the `[Streaming]` portion of the config to the most
   suitable for you:

    - RESOLUTION - choose the resolution, supported by your camera
    - FRAMERATE - choose the framerate, supported by your camera
    - STREAMS - (color, depth, infrared) choose this only for Intel RealSense cameras, otherwise leave this empty
    - OUTPUT_DIRECTORY - folder for data and video output
    - MULTIPLE_DEVICES - Set `True` if using multicamera setup, otherwise `False`
    - VIDEO_SOURCE - if you are not using RealSense or Basler cameras, you need to choose the correct source for your camera manually. It should be recognized by openCV.

#### Multicam support

To correctly enable multiple camera support, you need not only to set the variable `MULTIPLE_DEVICES` to `True` in the config, but also edit one of the DeepLabCut files.

Locate the file `predict.py` in your DeepLabCut folder (for DLC v2.x it would be in `../deeplabcut/pose_estimation_tensorflow/nnet` folder), and change the line in function `setup_pose_prediction`
```python 
sess = TF.Session()
```
to the following lines, maintaining the correct indentation 
```python
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
```

#### Intel RealSense support

DeepLabStream was written with [Intel RealSense cameras](https://www.intelrealsense.com/) support in mind, to be able to get depth data and use infrared cameras for experiments in low-lighting conditions.

To enable these features, you need to install an additional Python library: **PyRealSense2**
```bash
pip install pyrealsense2
```
In an ideal scenario, that would install it fully, but in some specific cases, for example, if you are using Python 3.5 on a Windows machine, the corresponding wheel file can not be available.
If that is the case, you need to manually build it from source from this official [GitHub repository](https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python).

#### Basler Pylon support

DeepLabStream also supports the usage of [Basler cameras](https://www.baslerweb.com/en/) through their Python wrapper [pypylon](https://github.com/basler/pypylon) .

To enable this, you need to install an additional Python library: **PyPylon**
```bash
pip install pypylon
```

or use the official provided instruction

```bash
git clone https://github.com/basler/pypylon.git
cd pypylon
pip install .
```

#### Generic camera support

If you wish to not use either Intel RealSense or Basler cameras, DeepLabStream can work with any camera supported by **opencv**.

By default, DeepLabStream would try to open a camera from source 0 (like ```cv2.VideoCapture(0)```), but you can modify this and use a camera from any source.
You just need to specify your desired video source in `settings.ini`. For this, add a new parameter `VIDEO_SOURCE = N` in the `[Streaming]` section, where N would be your desired video source, supported by **opencv**.
Your resolution and framerate, described in the config would also apply, but beware that **opencv** does not always support every native camera resolution and/or framerate. Some experimenting might be required. 

Very important note: **with this generic camera mode you will not be able to use multiple cameras!**

## Testing

To properly test your DeepLabStream installation, we included a testing script that you can run in three different modes. ```DeepLabStream.py``` allows you to test your cameras, your DeepLabcut installation,
and to benchmark your DeepLabStream performance.

1. Run the following command to test your cameras:
```bash
python DeepLabStream.py
```

2. Next, you can test how your DeepLabCut installation behaves and if you did correctly set the DeepLabCut path in the config:
```bash
python DeepLabStream.py --dlc-enabled
```

3. And finally you can benchmark your system automatically:
```bash
python DeepLabStream.py --dlc-enabled --benchmark-enabled
```
The stream would run until it gets 3000 analyzed frames (you can always stop it manually at any point, just press 'Q' while the stream window is in focus). 
Then it will show you a detailed statistic of the overall performance timings, analysis timings, percentage of frames where it did lose tracking and your average FPS.
 
#### Recording testing

Additionally, you can test and see the results of the build-in video recorder. Run the following command to test it:
```bash
python DeepLabStream.py --recording-enabled
```

This will record the videofeed from the camera to your `OUTPUT_DIRECTORY`. You can also add this flag to any of the previously mentioned tests to check performance with recording enabled.

Important note: **recording will always save only "raw" video, without analysis, with framerate as close to specified as possible**


## Usage

### How to use DeepLabStream

Just run 
```
cd DeepLabStream
python app.py
``` 

You will see the main control panel of a GUI app.

![Main](docs/screen_gui.png)

To start working with DeepLabStream, press the `Start Stream` button. It will activate the camera manager and show you the current view from the connected cameras.

![Stream](docs/screen_stream.png)

After that you can `Start Analysis` to start DeepLabCut and receive a pose estimations for each frame, or, additionally, you can `Start Recording` to record a
video of the current feed (visible in the stream window). You will see your current video timestamp (counted in frames) and FPS after you pressed the `Start Analysis` button.

![Analisys](docs/screen_analysis.png)

As you can see, we track three points that represent three body parts of the mouse - nose, neck and tail root.
Every single frame where the animal was tracked is outputted to the dataframe, which would create a .csv file after the analysis is finished.

After you finish with tracking and/or recording the video, you can stop either function by specifically pressing on corresponding "stop" button
(so, `Stop Analysis` or `Stop Recording`) or you can stop the app and refresh all the timing at once, by pressing `Stop Streaming` button.

#### Experiments

DeepLabStream was build specifically for closed-loop experiments, so with a properly implemented experiment protocol, running experiments on this system is as easy as 
pressing the `Start Experiment` button. Depending on your protocol and experimental goals, experiments could run and finish without any further engagement from the user.

![Start](docs/screen_exp_start.png)

In the provided `ExampleExperiment` two regions of interest (ROIs) are created inside an arena. The experiment is designed to count the number of times the mouse enters a ROI and trigger a corresponding visual stimulus on a screen.
The high contrast stimuli (image files) are located within the `experiments/src` folder and specified within the `experiments.py` `ExampleExperiments` Class.

![Experiment](docs/screen_exp.png)

As a visual representation of this event, the border of the ROI will turn green.

All experimental output will be stored to a .csv file for easy postprocessing.

### How does this work

DeepLabStream uses the camera's video stream to simultaneously record a raw (read as unmodified) video of the ongoing experiment,
send frames one-by-one to the neuronal network for analysis, and use returned analysed data to plot and show a video stream for the experimenter to observe and control the experiment.
Analysed data will also be utilized to enable closed-loop experiments without any human interference, using triggers to operate equipment on predefined conditions
and to end, prolong or modify parts of experimental protocol.

![Flowchart](docs/flowchart2.png)

### Testing experiments offline

Within  `/utils`  we implemented an offline testing script `VideoAnalyzer.py` that enables experiment tests using prerecorded videos. 


As an example: You can run the `ExampleExperiment` on any video by simply inserting the path of the video into `settings.ini`:
```
[Video]
VIDEO_SOURCE = FullVideoPath.avi
``` 
Then run `VideoAnalyzer.py` the same as you would run `app.py` (Note: `VideoAnalyzer.py` does not offer a GUI).

To test your own experiments, you have to import your custom `ExperimentClass` like this:
```
# add it to the import line
    from experiments.experiments import ExampleExperiment, YourExperiment
``` 
and change the following line to create an instance of that experiment:
```
# old line:
    experiment = ExampleExperiment()
# new line:
    experiment = YourExperiment()
``` 

If you want to run the DeepLabStream posture detection on a prerecorded video without running an experiment just set the `experiment_enabled` to `False`.
A video of your offline test will be saved if `video_output` is set `True` (default = `True`). As usual all experimental data will be exported in a .csv file.

Note: `VideoAnalyzer.py` is not build to quickly analyze videos, but is specifically build to show the result immediately as a "live" stream.

### Known issues

#### Error when stopping the analysis:
```
BrokenPipeError: [Errno 32] Broken pipe
```

This is an `multiprocessing.Queue` closing error, when some items are still in the Queue.
This should not affect the app in any meaningful way.



## License
