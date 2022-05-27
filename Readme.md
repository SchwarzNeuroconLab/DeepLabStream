

# DeepLabStream

![GraphAbstract](docs/GraphAbstract.png)

[![GitHub stars](https://img.shields.io/github/stars/SchwarzNeuroconLab/DeepLabStream.svg?style=social&label=Star)](https://github.com/SchwarzNeuroconLab/DeepLabStream)
[![GitHub forks](https://img.shields.io/github/forks/SchwarzNeuroconLab/DeepLabStream.svg?style=social&label=Fork)](https://github.com/SchwarzNeuroconLab/DeepLabStream)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Twitter Follow](https://img.shields.io/twitter/follow/SNeuroconnect.svg?label=SNeuroconnect&style=social)](https://twitter.com/SNeuroconnect) 

[DeepLabStream](https://www.nature.com/articles/s42003-021-01654-9) is a python based multi-purpose tool that enables the realtime tracking and manipulation of animals during ongoing experiments.
Our toolbox was orginally adapted from the previously published [DeepLabCut](https://github.com/AlexEMG/DeepLabCut) ([Mathis et al., 2018](https://www.nature.com/articles/s41593-018-0209-y)) and expanded on its core capabilities, but is now able to utilize a variety of different network architectures for online pose estimation
 ([SLEAP](https://github.com/murthylab/sleap), [DLC-Live](https://github.com/DeepLabCut/DeepLabCut-live), [DeepPosekit's](https://github.com/jgraving/DeepPoseKit) StackedDenseNet, StackedHourGlass and [LEAP](https://github.com/murthylab/sleap)).

DeepLabStreams core feature is the utilization of real-time tracking to orchestrate closed-loop experiments. This can be achieved using any type of camera-based video stream (incl. multiple streams).  It enables running experimental protocols that are dependent on a constant stream of bodypart positions and feedback activation of several input/output devices. It's capabilities range from simple region of interest (ROI) based triggers to headdirection or behavior dependent stimulation, including online classification ([SiMBA](https://www.biorxiv.org/content/10.1101/2020.04.19.049452v2), [B-SOID](https://www.biorxiv.org/content/10.1101/770271v2)).

![DLS_Stim](docs/DLSSTim_example.gif)

## Read the news:

- Watch an introductory talk about DLStream on [Youtube](https://www.youtube.com/watch?v=ZspLDZb_kMI)
- Real-time behavioral analysis using artificial intelligence @ [Federal Ministry of Education and Research](http://www.research-in-germany.org/news/2021/2/2021-02-19_Real-time_behavioural_analysis_using_artificial_intelligence.html), [Uni Bonn](https://www.uni-bonn.de/news/real-time-behavioral-analysis-using-artificial-intelligence) and [Phys.org](https://phys.org/news/2021-02-real-time-behavioral-analysis-artificial-intelligence.html)
- We are featured on [OpenBehavior](https://edspace.american.edu/openbehavior/project/dlstream/)
- We are also featured on [open-neuroscience.com](https://open-neuroscience.com/post/deeplabstream/)!

## New features:

#### 03/2021: Online Behavior Classification using SiMBA and B-SOID:

- full integration of online classification of user-defined behavior using [SiMBA](https://github.com/sgoldenlab/simba) and [B-SOID](https://github.com/YttriLab/B-SOID).
- SOCIAL CLASSIFICATION with SiMBA 14bp two animal classification (more to come!)
- Unsupervised Classification with B-SOID
- New wiki guide and example experiment to get started with online classification: [Advanced Behavior Classification](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Advanced-Behavior-Classification)
- this version has new requirements (numba, pure, scikit-learn), so be sure to install them (e.g. `pip install -r requirements.txt`).

#### 02/2021: Multiple Animal Experiments (Pre-release): Full [SLEAP](https://github.com/murthylab/sleap) integration (Full release coming soon!)

- Updated [Installation](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Installation-&-Testing) (for SLEAP support)
- Single Instance and Multiple Instance models (TopDown & BottomUp) integration
- [New Multiple Animal Experiment Tutorial](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Multiple-Animal-Experiments)

#### 01/2021: DLStream was published in [Communications Biology](https://www.nature.com/articles/s42003-021-01654-9)

#### 12/2021: New pose estimation model integration
 - ([DLC-Live](https://github.com/DeepLabCut/DeepLabCut-live)) and pre-release of further integration ([DeepPosekit's](https://github.com/jgraving/DeepPoseKit) StackedDenseNet, StackedHourGlass and [LEAP](https://github.com/murthylab/sleap))

## Quick Reference:

 #### Check out or wiki: [DLStream Wiki](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki)

 #### Read the paper: [Schweihoff, et al. 2021](https://www.nature.com/articles/s42003-021-01654-9)
 
 #### Contributing

If you have feature requests or questions regarding the design of experiments join our [slack group](https://join.slack.com/t/dlstream/shared_invite/zt-pdh6twf5-xHbpLvJtrTx32t12w8RDVQ).

We are constantly working to update and increase the capabilities of DLStream. 
We welcome all feedback and input from your side.

 
 ### 1. [Updated Installation & Testing](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Installation-&-Testing)
 
 ### 2. [How to use DLStream GUI](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/How-to-use-DLStream)
 
 ### 3. Check out our [Out-of-the-Box](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Out-Of-The-Box:-Overview) 
 
 ### 4. [Design an Out-of-the-Box Experiment](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Out-Of-The-Box:-Design-Experiments)
 
### What's underneath?:
 
 ### 5. [Introduction to experiments](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Introduction)
 
 ### For advanced users:
 
 ### 6. [Design your first experiment](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/My-first-experiment)
 
 ### 7. [Adapting an existing experiment to your own needs](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Adapting-an-existing-experiment-to-your-own-needs)
 


### How to use DeepLabStream

Just run 
```
cd DeepLabStream
python app.py
``` 

You will see the main control panel of a GUI app.

![Main](https://user-images.githubusercontent.com/44863941/91172971-59faf000-e6dd-11ea-8b68-3c36db0ff22f.png)

To start working with DeepLabStream, press the `Start Stream` button. It will activate the camera manager and show you the current view from the connected cameras.

![Stream](https://user-images.githubusercontent.com/44863941/91173024-7008b080-e6dd-11ea-84b0-b05ac408d9a2.png)

After that you can `Start Analysis` to start DeepLabCut and receive a pose estimations for each frame, or, additionally, you can `Start Recording` to record a
video of the current feed (visible in the stream window). You will see your current video timestamp (counted in frames) and FPS after you pressed the `Start Analysis` button.

![Analysis](https://user-images.githubusercontent.com/44863941/91173049-7ac34580-e6dd-11ea-80b6-ad56cb9cf22c.png)

As you can see, we track three points that represent three body parts of the mouse - nose, neck and tail root.
Every single frame where the animal was tracked is outputted to the dataframe, which would create a .csv file after the analysis is finished.

After you finish with tracking and/or recording the video, you can stop either function by specifically pressing on corresponding "stop" button
(so, `Stop Analysis` or `Stop Recording`) or you can stop the app and refresh all the timing at once, by pressing `Stop Streaming` button.

#### Experiments

DeepLabStream was build specifically for closed-loop experiments, so with a properly implemented experiment protocol, running experiments on this system is as easy as 
pressing the `Start Experiment` button. Depending on your protocol and experimental goals, experiments could run and finish without any further engagement from the user.

![Start](https://user-images.githubusercontent.com/44863941/91173075-857dda80-e6dd-11ea-90a4-1e768cab41ad.png)

In the provided `ExampleExperiment` two regions of interest (ROIs) are created inside an arena. The experiment is designed to count the number of times the mouse enters a ROI and trigger a corresponding visual stimulus on a screen.
The high contrast stimuli (image files) are located within the `experiments/src` folder and specified within the `experiments.py` `ExampleExperiments` Class.

![Experiment](https://user-images.githubusercontent.com/44863941/91173098-90d10600-e6dd-11ea-94be-63e99f88df0a.png)

As a visual representation of this event, the border of the ROI will turn green.

All experimental output will be stored to a .csv file for easy postprocessing. Check out [Working with DLStream output](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Working-with-DLStream-output) for further details.

Look at the [Introduction to experiments](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Introduction) to get an idea how to design your own experiment in DeepLabStream or learn how to adapt one of the already published experiments at [Adapting an existing experiment](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Adapting-an-existing-experiment-to-your-own-needs).

## How does this work

DeepLabStream uses the camera's video stream to simultaneously record a raw (read as unmodified) video of the ongoing experiment,
send frames one-by-one to the neuronal network for analysis, and use returned analysed data to plot and show a video stream for the experimenter to observe and control the experiment.
Analysed data will also be utilized to enable closed-loop experiments without any human interference, using triggers to operate equipment on predefined conditions
and to end, prolong or modify parts of experimental protocol.

![Flowchart](docs/flowchart2.png)

### Known issues

If you encounter any issues or errors, you can check out the wiki article ([Help there is an error!](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Help-there-is-an-error!)). If your issue is not listed yet, please refer to the issues and either submit a new issue or find a reported issue (which might be already solved) there. Thank you!

## References:

If you use this code or data please cite:

Schweihoff, J.F., Loshakov, M., Pavlova, I. et al. DeepLabStream enables closed-loop behavioral experiments using deep learning-based markerless, real-time posture detection. 

Commun Biol 4, 130 (2021). https://doi.org/10.1038/s42003-021-01654-9

## License
This project is licensed under the GNU General Public License v3.0. Note that the software is provided "as is", without warranty of any kind, expressed or implied. 

## Authors

Developed by:
- Jens Schweihoff, jens.schweihoff@ukbonn.de

- Matvey Loshakov, matveyloshakov@gmail.com

Corresponding Author: Martin Schwarz, Martin.Schwarz@ukbonn.de

## Other References

If you are using any of the following open-source code please cite them accordingly:

>  Simple Behavioral Analysis (SimBA) – an open source toolkit for computer classification of complex social behaviors in experimental animals;
 Simon RO Nilsson, Nastacia L. Goodwin, Jia Jie Choong, Sophia Hwang, Hayden R Wright, Zane C Norville, Xiaoyu Tong, Dayu Lin, Brandon S. Bentzley, Neir Eshel, Ryan J McLaughlin, Sam A. Golden
 bioRxiv 2020.04.19.049452; doi: https://doi.org/10.1101/2020.04.19.049452 

>  B-SOiD: An Open Source Unsupervised Algorithm for Discovery of Spontaneous Behaviors;
 Alexander I. Hsu, Eric A. Yttri
 bioRxiv 770271; doi: https://doi.org/10.1101/770271 

> SLEAP: Multi-animal pose tracking;
Talmo D. Pereira, Nathaniel Tabris, Junyu Li, Shruthi Ravindranath, Eleni S. Papadoyannis, Z. Yan Wang, David M. Turner, Grace McKenzie-Smith, Sarah D. Kocher, Annegret L. Falkner, Joshua W. Shaevitz, Mala Murthy
bioRxiv 2020.08.31.276246; doi: https://doi.org/10.1101/2020.08.31.276246 

>Real-time, low-latency closed-loop feedback using markerless posture tracking;
    Gary A Kane, Gonçalo Lopes, Jonny L Saunders, Alexander Mathis, Mackenzie W Mathis;
 eLife 2020;9:e61909 doi: 10.7554/eLife.61909
