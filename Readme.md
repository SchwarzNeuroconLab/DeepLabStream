

# DeepLabStream

![GraphAbstract](docs/GraphAbstract.png)

[![GitHub stars](https://img.shields.io/github/stars/SchwarzNeuroconLab/DeepLabStream.svg?style=social&label=Star)](https://github.com/SchwarzNeuroconLab/DeepLabStream)
[![GitHub forks](https://img.shields.io/github/forks/SchwarzNeuroconLab/DeepLabStream.svg?style=social&label=Fork)](https://github.com/SchwarzNeuroconLab/DeepLabStream)
![GitHub All Releases](https://img.shields.io/github/downloads/SchwarzNeuroconLab/DeepLabStream/total?style=social)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Twitter Follow](https://img.shields.io/twitter/follow/SNeuroconnect.svg?label=SNeuroconnect&style=social)](https://twitter.com/SNeuroconnect)

DeepLabStream is a python based multi-purpose tool that enables the realtime tracking of animals and manipulation of experiments.
Our toolbox is adapted from the previously published [DeepLabCut](https://github.com/AlexEMG/DeepLabCut) ([Mathis et al., 2018](https://www.nature.com/articles/s41593-018-0209-y)) and expands on its core capabilities.
DeepLabStreams core feature is the real-time analysis using any type of camera-based video stream (incl. multiple streams).  Building onto that, we designed a full experimental closed-loop toolkit. It enables running experimental protocols that are dependent on a constant stream of bodypart positions and feedback activation of several input/output devices. It's capabilities range from simple region of interest (ROI) based triggers to headdirection or behavior dependent stimulation.

![DLS_Stim](docs/DLSSTim_example.gif)

### Quick Reference:

 ### Read the preprint: [Schweihoff et al, 2019](https://doi.org/10.1101/2019.12.20.884478).
 
 ### 1. [Installation & Testing](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Installation-&-Testing)
 
 ### 2. [Introduction to experiments](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Introduction)
 
 ### 3. [Design your first experiment](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/My-first-experiment)
 
 ### 4. [Adapting an existing experiment to your own needs](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Adapting-an-existing-experiment-to-your-own-needs)
 
 #### Check out our [Out-of-the-Box](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Out-Of-The-Box:-What-Triggers-are-available%3F) section to get a good idea, what DLStream has in stock for your own experiments!
 
### How does this work

DeepLabStream uses the camera's video stream to simultaneously record a raw (read as unmodified) video of the ongoing experiment,
send frames one-by-one to the neuronal network for analysis, and use returned analysed data to plot and show a video stream for the experimenter to observe and control the experiment.
Analysed data will also be utilized to enable closed-loop experiments without any human interference, using triggers to operate equipment on predefined conditions
and to end, prolong or modify parts of experimental protocol.

![Flowchart](docs/flowchart2.png)

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

Look at the [Introduction to experiments](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Introduction) to get an idea how to design your own experiment in DeepLabStream or learn how to adapt one of the already published experiments at [Adapting an existing experiment](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Adapting-an-existing-experiment-to-your-own-needs).

### Known issues

If you encounter any issues or errors, you can check out the wiki article ([Help there is an error!](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Help-there-is-an-error!)). If your issue is not listed yet, please refer to the issues and either submit a new issue or find a reported issue (which might be already solved) there. Thank you!

## References:

If you use this code or data please cite [Schweihoff et al, 2019](https://doi.org/10.1101/2019.12.20.884478).

## License
This project is licensed under the GNU General Public License v3.0. Note that the software is provided "as is", without warranty of any kind, expressed or implied. 
## Authors
Lead Researcher: Jens Schweihoff, jens.schweihoff@ukbonn.de

Lead Developer: Matvey Loshakov, matveyloshakov@gmail.com

Corresponding Author: Martin Schwarz, Martin.Schwarz@ukbonn.de
