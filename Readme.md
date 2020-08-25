

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

## Quick Reference:

 ### Check out or wiki: [DLStream Wiki](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki)

 ### Read the preprint: [Schweihoff et al, 2019](https://doi.org/10.1101/2019.12.20.884478).
 
 ### 1. [Installation & Testing](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Installation-&-Testing)
 
 ### 2. [How to use DLStream GUI](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/How-to-use-DLStream)
 
 ### 3. [Introduction to experiments](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Introduction)
 
 ### 4. [Design your first experiment](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/My-first-experiment)
 
 ### 5. [Adapting an existing experiment to your own needs](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Adapting-an-existing-experiment-to-your-own-needs)
 
 #### Check out our [Out-of-the-Box](https://github.com/SchwarzNeuroconLab/DeepLabStream/wiki/Out-Of-The-Box:-What-Triggers-are-available%3F) section to get a good idea, what DLStream has in stock for your own experiments!
 
## How does this work

DeepLabStream uses the camera's video stream to simultaneously record a raw (read as unmodified) video of the ongoing experiment,
send frames one-by-one to the neuronal network for analysis, and use returned analysed data to plot and show a video stream for the experimenter to observe and control the experiment.
Analysed data will also be utilized to enable closed-loop experiments without any human interference, using triggers to operate equipment on predefined conditions
and to end, prolong or modify parts of experimental protocol.

![Flowchart](docs/flowchart2.png)

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
