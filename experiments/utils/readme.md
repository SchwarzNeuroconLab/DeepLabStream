###How to use the predesigned experiment features

With the new version of DLStream, creating your own experiments has become much more accessible to researchers that do not want to dive into the DLStream code.

This guide will discuss the new feature as well as our `base` modules that can be used to generate combinations of prebuild `EXPERIMENT`, `TRIGGER`, `STIMULATION` modules and their `parameters` without any need to access the code.

First the function: `design_experiment()`

In your console of choice or IDE, run `design_experiment.py`

You will be asked to enter the base modules that can be used to design a basic experiment in DLStream.


![Design_experiment](experiments/utils/misc/design_experiment_gif.gif)


#### Selecting a designed experiment in DLStream
Once a experiment config file has been created, the experiment can be used in DLStream by entering the filename to `settings.ini`.
````
[Experiment]
EXP_NAME = CONFIG_NAME
````

Starting `app.py` and `Start Experiment` will load the configuration into DLStream and run the experiment based on the entered parameters.

####General Build-up and custom parameters
Second, let's discuss the general build-up.

Experiments can now be combined, configured and saved in `.ini` files, very similiar to `settings.ini`:
Each module has a `[Section]` and corresponding `PARAMETERS`, that are used to configure each experiment design.
````
[EXPERIMENT]
BASE = BaseConditionalExperiment
EXPERIMENTOR = DEFAULT

[BaseConditionalExperiment]
TRIGGER = BaseHeaddirectionTrigger
PROCESS = BaseProtocolProcess
INTERTRIAL_TIME = 40
EXP_LENGTH = 40
EXP_TIME = 3600
EXP_COMPLETION = 10

[BaseHeaddirectionTrigger]

POINT = 550, 63
ANLGE = 30
BODYPARTS = neck
DEBUG = False

[BaseProtocolProcess]

TYPE = switch
STIMULATION = ScreenStimulation

[ScreenStimulation]

TYPE = image
STIM_PATH = PATH_TO_IMAGE
BACKGROUND_PATH = PATH_TO_BACKGROUND
````

#### Changing parameters in an already created config

If you want to change any parameter, you can do so directly in the experiment config file (and save them).

````
[BaseConditionalExperiment]

INTERTRIAL_TIME = 40
````
Change the number and save the file.
````
[BaseConditionalExperiment]

INTERTRIAL_TIME = 20
````

When you load the experiment again, it will use the new parameter. Note, that in case of non-sense parameter changes (e.g. `INTERSTIMULUS_TIME = banana`).
DLStream will throw an error and the experiment won't run (as expected).

Several parameters only except preselected options, so make sure that you check out the available options in the corresponding wiki article.

#### Changing modules in an already created config

Modules are found as `PARAMETERS` in their corresponding parent module and as `[Sections]` in the `.ini` file.
````
[EXPERIMENT]
BASE = BaseConditionalExperiment
...

[BaseConditionalExperiment]
TRIGGER = BaseHeaddirectionTrigger
...
[BaseHeaddirectionTrigger]
...
````

Above you see that in the `[Experiment]` section the base experiment `BaseConditionalExperiment` is named, while underneath a `[Section]` with the same name appears with all relevant `PARAMETERS`.

If you do not want to manually change the modules, you can use our new functions to do so.
The easiest way is to create a new experiment, with your selection of modules and then copy the parameters into the new config file.

If you want to change a module manually (e.g.  `TRIGGER`), you will need to change the entry under `TRIGGER`, remove the old section `[BaseHeaddirectionTrigger]` and add the new section from `experiment/configs/default_config.ini`.
Afterwards check if all parameters are conclusive and save the file.

````
[EXPERIMENT]
BASE = BaseConditionalExperiment
...

[BaseConditionalExperiment]

# change to BaseRegionTrigger:
TRIGGER = BaseHeaddirectionTrigger 
...
# delete this section completely:
[BaseHeaddirectionTrigger]
POINT = 550, 63
ANGLE = 30
BODYPARTS = neck
DEBUG = False

# add the new section from the default_config.ini:
[BaseRegionTrigger]
TYPE = circle
CENTER= 550, 63
RADIUS = 30
BODYPARTS = neck
DEBUG = False
````

