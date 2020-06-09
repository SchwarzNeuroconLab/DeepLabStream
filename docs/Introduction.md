# Introduction

This short introductory tutorial is targeted at experimenters with intermediate or advanced Python skills and will not
go into full details of class based programming or explain the underlying processing functions.

To design and successfully implement an experiment in DeepLabStream you need:

1. A clear idea of the design and necessary steps of the planned experiment
2. Good understanding of the relationship between detected behavior and the desired closed loop event
3. A network and system that can detect the behavior of choice and react within a time frame that your experiment demands

Let's do this step by step. We will take the example experiment included in DeepLabStream as a basis.

## The general structure

An experiment in DeepLabStream is made up by several interacting parts. If you are already familiar with the overall design of DeepLabStream you can skip this part.

### 1. Triggers:

A Trigger is an object that is specifically created to check whether a certain predefined condition is true in the current frame.
It will be checked repetitively and returns either `True` or `False` depending if the condition was met. For this the position of a body part or the posture of an animal is compared each frame.

Let's take the RegionTrigger as an example: 
```
class RegionTrigger:

    def __init__(self, region_type: str, center: tuple, radius: float, bodyparts, debug: bool = False):

        self._roi_type = region_type.lower()
        region_types = {'circle': EllipseROI, 'square': RectangleROI}
        self._region_of_interest = region_types[self._roi_type](center, radius, radius)
        self._bodyparts = bodyparts
```
When creating an experiment, we are creating an instance of the `RegionTrigger` class with the parameters `region_type`, `center`, `radius` and `bodyparts`.
In this case we are creating one of two different ROI types depending on the `region_type` with a `center` and a `radius` (or width/length) in pixels.

As we will see later, each time DeepLabStream is analysing a frame and passes it to the `ExampleExperiment`, the `Trigger` is called by it's main function `check_skeleton`.
Let's have a look at a simplified version of this function:
```
    def check_skeleton(self, skeleton: dict):
    
        bp_x, bp_y = skeleton[self._bodyparts]
        result = self._region_of_interest.check_point(bp_x, bp_y)

        color = (0, 255, 0) if result else (0, 0, 255)

        if self._roi_type == 'circle':
            response_body = {'plot': {'circle': dict(center=self._region_of_interest.get_center(),
                                                     radius=int(self._region_of_interest.get_x_radius()),
                                                     color=color)}}
                                                     
        response = (result, response_body)
        return response
```
Whenever `check_skeleton()` is called, it returns whether the bodypart within the skeleton dictionary that was defined by the `bodyparts` parameter is inside the ROI. It will return `True` or `False` and an additional component that is visualized on the stream.
To simplify this even further let's assume a different trigger where I want to check whether my animal has crossed the middle of my arena `x_center` (from left to right) and is in the right side of my arena.
It can be easily done with a simple `if` statement.
```
    def check_skeleton(self, skeleton: dict):
    
        bp_x, bp_y = skeleton[self._bodyparts]
        if bp_x > x_center:
            result = True
        else:
            result = False
        
        return result
```
You can go as simple or as complex as you want, taking multiple body parts or even other objects and their relation into account when designing a trigger. Try it yourself!

### 2. Stimulation:

Stimulations are a bit trickier to explain, because they heavily depend on your setup and your experiment. Let's go through some basics.

A stimulation is triggered and reacts to a given condition. That's easy, we know a simple way of doing that! `Trigger`s!

It runs in parallel with the experiment and does not stop or slow down the procedure. That's harder, depending on our stimulation we might be engaged for a longer time and this would halt the whole process. 

Luckily DeepLabStream was designed to account for such things by using multiprocessing.
We suggest that you leave the general design of experiments in place and adapt your stimulations accordingly as we designed them to run in parallel to the experiment even if the actual stimulation is a multistep process itself.
We have two important parts here:
 
`stimulation.py` contains the actual stimulation. `show_visual_stim_img` for example creates a window and displays an image in it. In the `ExampleExperiment` this function is used to switch between background and stimulation images on a screen that is visible to the animal from inside the arena. 
`toggle_device` is a function that controls a device connected via a NI DAQ-board and sends a digital trigger (TTL) signal. It can be used to toggle lasers or any other device that can be connected and modulated through such a board. Most boards that are equipped with USB can be controlled through an API to interact with via Python. 
The rule of thumb here is: If you can control it with Python, DeepLabStream can control it.
 
`stimulation_process.py` is the protocol that orchestrates the stimulation in another process. It also contains `Timer` which can be very useful in many cases as we will see later. 
Let's have a look at the main function inside of the process `example_protocol_run`:
  ```
 def example_protocol_run(condition_q: mp.Queue):
    current_trial = None
    dmod_device = DigitalModDevice('Dev1/PFI0')
    while True:
        if condition_q.full():
            current_trial = condition_q.get()
        if current_trial is not None:
            show_visual_stim_img(img_type=current_trial, name='inside')
            dmod_device.toggle()
        else:
            show_visual_stim_img(name='inside')
            dmod_device.turn_off()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

To simplify the multiprocessing part assume that we have a connection (queue) between the DeepLabStream app (that analyses and displays the stream) and the experimental protocol (that controls the stimulation). This connection is very simple and we are passing a single argument. Whenever we tell the experimental protocol that a stimulation should be started (or trial) it passes this through the connection.
```
    while True:
        if condition_q.full():
            current_trial = condition_q.get()
```
This literally says: Check if there is something waiting for you in the connection (queue) and take it.
```
        if current_trial is not None:
            show_visual_stim_img(img_type=current_trial, name='inside')
            dmod_device.toggle()
        else:
            show_visual_stim_img(name='inside')
            dmod_device.turn_off()
```
Here we are checking whether the parameter taken out of the connection actually means something for our stimulation. In this simple example we are just checking if `current_trial` is something at all and then pass it to the before mentioned stimulation function and simultaneously activates a device via a digital trigger. This way the image will be decided in the stimulation function.
But we also can decide directly inside this function.

For example:
```
        if current_trial == 'Trial_1':
            show_visual_stim_img(img_type= 'Trial_1', name='inside')
        elif current_trial == 'Trial_2':
            dmod_device.toggle()
        else:
            show_visual_stim_img(name='inside')
            dmod_device.turn_off()
```        
Now we are specifically looking for a string in `current_trial`. It will now either show an image 'Trial_1' or activate the device 'Trial_2'. If no argument was passed, it will just show a background image and deactivate the device. What we pass to the experimental protocol will be decided in the next part.

### 3. The experiment:

Now we are coming to the scaffold that holds everything together and makes sense out of it. 

Let's go through this step by step again:
```
class ExampleExperiment:

    def __init__(self):
        self.experiment_finished = False
        self._process = ExampleProtocolProcess()
        self._green_point = (550, 163)
        self._blue_point = (372, 163)
        self._radius = 40
        self._event = None
        self._current_trial = None
        self._trial_count = {trial: 0 for trial in self._trials}
        self._trial_timers = {trial: Timer(10) for trial in self._trials}
        self._exp_timer = Timer(600)

    @property
    def _trials(self):
        """
        Defining the trials
        """
        green_roi = RegionTrigger('circle', self._green_point, self._radius * 2 + 7.5, 'neck')
        blue_roi = RegionTrigger('circle', self._blue_point, self._radius * 2 + 7.5, 'neck')
        trials = {'Greenbar_whiteback': dict(trigger=green_roi.check_skeleton,
                                             count=0),
                  'Bluebar_whiteback': dict(trigger=blue_roi.check_skeleton,
                                            count=0)}
        return trials
        
```
The class `ExampleExperiment` is initiated with several parameters, including the actual process that orchestrates stimulation `ExampleProtocolProcess`. Here you will set most experimental defined parameters.
To directly build on the last part, we will first have a look at `_trials`. As you can see we are creating two things here. First we are initiating two different `RegionTrigger` (have a look at 1. Trigger if this is not telling you anything), 
second we are creating a dictionary which includes the trigger, a count and a key that refers to each trial. To jump a little bit a head: The key or name of the trial is actually passed to the `ExampleProtocolProcess` as we have seen in 3.

Okay, we now have successfully connected trial, stimulation and trigger, but an experiment is more than that. Now we come to the actual scaffold i was talking about:

The function `check_skeleton` (remember the trigger function with the same name!) is where the magic happens. This function will get every frame analyzed by DeepLabStream and the corresponding posture or "skeleton" of the animal.

Here is a simplified version:
```
    def check_skeleton(self, frame, skeleton):

        if not self.experiment_finished:
            result, response = False, None
            for trial in self._trials:
                # check for all trials if condition is met
                result, response = self._trials[trial]['trigger'](skeleton=skeleton)
                if result:
                    if self._current_trial is None:
                        self._current_trial = trial
                        self._trial_count[trial] += 1
                        print(trial, self._trial_count[trial])
                else:
                    if self._current_trial == trial:
                        self._current_trial = None

            self._process.set_trial(self._current_trial)
            
```
When the experiment is not finished, check for both trials (in this case the term trial is confusing, as it is actually just connected to a condition in this experiment) if the condition/trigger is met. 
If it was met, increase the trial counter by 1 and pass the trial name to the stimulation.

This simplified example would show images and trigger a device indefinitely as long as the animal is entering the defined ROIs. We are missing a crucial component that is part of any experiment: `Timer`!

### 4. Timer:

You probably spotted them already. We actually mentioned them earlier. But let's first look at the basic function and then their implementation.

A timer should be able to track time independent of the actual processing speed that the rest of the software is limited by. 
So it's quite simple, when we create a instance of the `Timer` class, we specify the time in seconds it should keep track of. Every time we check `Timer` it will tell us whether that time has run out or not. The rest of the functionality of this class is mainly utility wise.
For example we can reset a `Timer` to start it again, without the need to create it anew.

Let's talk about implementations. At this points it's not very useful to talk about the code inside the experiment as they all follow the same easy principle. If you have a look into the actual `ExampleExperiment`, you will see several cases.

a. Total experimental time aka `exp_timer`. If this timer runs out, your total experimental time was reached and the experiment should stop itself. In almost all experiments this is a must, this timer will keep track of time for you and ends the experiment in a coordinated fashion automatically. This does not get the animal out of the arena though... sorry.

b. Inter trial time aka inter stimulation time. Here is where the reset comes in handy. When we want to have a time
 between each triggered stimulation or trial, just add one of these and reset them after each trial/event again. Don't forget to start them again though!

c. Stimulation time. Assuming that you want to stimulate your animal not only when the condition/trigger was met, but also for some time after. This timer is useful to turn off your stimulus again after that time passed. Depending on the experiment it might be used in the `Experiment` class but most likely you will implement it in `stimulus_process.py`.

There are several other possibilities to use the `Timer` but this covers the basics that are most likely in any experiment you will design. As they say: `Timer` is of the essence!


## Design your own experiment

With the basics in place, you should be ready to start adapting the `ExampleExperiment` to fit your experimental needs. Rather then rewriting the entire code, we recommend taking the example and modify the predefined structure. If you want to design your own triggers, we recommend looking at your previous post hoc data analysis.
If you worked with DeepLabCut in the past, you most likely have an idea how to find out if certain conditions were met by the animal during the offline experiment. Can you reduce it to an frame by frame `True` or `False` output?

Yes? Congratulations, you got your first `Trigger` candidate!

Look at the [MyFirstExperiment tutorial](MyFirstExperiment.md) to get an idea how to design your own experiment in DeepLabStream.


### Testing experiments offline

To test your design, we recommend using `VideoAnalyzer.py`. It will give you an idea of the feasibility of your design.
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

## Concluding remarks

We did not cover all functions within the classes and parts we discussed, but most of them are commented extensively. Have a look at the script. Now that you know the basic principle, it should be much easier to understand.
