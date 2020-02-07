### Design your own experiment

Okay, now that we understand the basics, let's try to create an experiment from scratch.

First we take the experiment class and create a copy of the `ExampleExperiment` and remove all parts that are not essential.
Let's call this `MyFirstExperiment`


```

class MyFirstExperiment:
    """
        Experiment Class containing MyFirstExperiment designed in DeepLabStream
    """
```
When we initiate the class, we want all relevant parameters to be set. So at this point we should consider the experimental structure.
Let's assume the following experiment (based on a place avoidance task):

An animal is in an o-maze arena that it can freely explore. Whenever it goes to the top short arm and reaches its center, we want to deliver an aversive stimulus (e.g. an airpuff)
through a device than can be triggered by a TTL signal. After the animal experienced this for 10 times, we want the stimulus to stop at the top but start at the bottom. Between each stimulus should be a 5 sec period where no new event is triggered.
We decide that a session should take 10 min max and that after 10 additional trials on the bottom the experiment should stop, whatever comes first decides.

The parameters we need to set are: `_exp_timer` , `_trial_max_count`, `_trials` and  `_intertrial_timer`

```
    def __init__(self):
        self.experiment_finished = False
        self._process = MyFirstProtocolProcess() # we have not created the ProtocolProcess yet, but we know how it will be called
        self._event = None
        self._current_trial = None
        self._trial_count = {trial: 0 for trial in self._trials}
        
        # here are the parameters we change or add:
        self._trial_1_roi_center = (x1, y1) # change this if you want to use this for the real coordinates
        self._trial_2_roi_center = (x2, y2)
        self._trial_max_count = 10
        self._exp_timer = Timer(600) # 600 sec are 10 min
        self._intertrial_timer = Timer(10)

```
Let's review what we need: We want to check if the animal is in a certain ROI inside the arena. The `RegionTrigger` that was already used in the example is a what we can easily use for this. We already defined the center of the ROIs, so we are left with the size (e.g. radius), the body part or parts that we are interested in and the shape of the ROI (here: circle or square).
```
    @property
    def _trials(self):
        """
        Defining the trials
        """
        radius = radius_int # make sure to add a real number before trying this
        trial_1_roi = RegionTrigger('circle', self._trial_1_roi_center, radius, 'nose')
        trial_2_roi = RegionTrigger('circle', self._trial_2_roi_center, radius, 'nose')
        trials = {'Trial_1': dict(trigger=trial_1_roi.check_skeleton,
                                             count=0),
                  'Trial_2': dict(trigger=trial_2_roi.check_skeleton,
                                            count=0)}
        return trials
```
Now that we set the overall structure, we are building our main function `check_skeleton`. The rest of the functions we can leave unchanged from the `ExampleExperiment`.

As we described above, we want to differentiate between several things here. 

We want our experiment to end after both trials reach 10 repetitions or after 10 min. So the first thing we will check is whether one of those conditions is `True`.
```

    def check_skeleton(self, frame, skeleton):
        """
        Checking each passed animal skeleton for a pre-defined set of conditions
        Outputting the visual representation, if exist
        Advancing trials according to inherent logic of an experiment
        :param frame: frame, on which animal skeleton was found
        :param skeleton: skeleton, consisting of multiple joints of an animal
        """
        self.check_exp_timer()  # checking if experiment is still on and stopping the experiment if time ran out
        if self._trial_count['Trial_1'] == self._trial_max_count and self._trial_count['Trial_2'] == self._trial_max_count:
            # checking if both trials hit a predefined cap
            self.stop_experiment()
```
As long as we count the trials, the experiment will end automatically now after 20 trials (10 each). Next step is what actually decides what happens:
```
        if not self.experiment_finished:
            result, response = False, None
            
            if not self._intertrial_timer.check_timer():
            # check whether the intertrial timer runs
```
Now that we have the basic structure let's go into the trials and their role:
```
                if self._trial_count['Trial_1'] != self._trial_max_count:
                    trial = 'Trial_1'
                    # check for Trial_1 if condition is met
                    result, response = self._trials[trial]['trigger'](skeleton=skeleton)
                    plot_triggers_response(frame, response)
                    
                    if result: # if the animal is in the ROI
                        if self._current_trial is None: # if currently no trial is set
                            self._current_trial = trial
                            self._trial_count[trial] += 1
                            print(trial, self._trial_count[trial])
                            self._intertrial_timer.reset()

                    else: # stop trial when animal is outside of ROI and start the intertrial timer
                        if self._current_trial == trial:
                            self._current_trial = None
                            self._intertrial_timer.start()
```   
For Trial_2 it's almost the same, although we need to add the condition that Trial_1 needs to be finished. To make this more pythonic you can also wrap it in a function itself, but for readability we will keep it like this.                        
```                        
                elif self._trial_count['Trial_2'] != self._trial_max_count and self._trial_count['Trial_1'] == self._trial_max_count:
                #when Trial_1 reached criteria switch to Trial_2
                    trial = 'Trial_2'
                    # check for Trial_2 if condition is met
                    result, response = self._trials[trial]['trigger'](skeleton=skeleton)
                    plot_triggers_response(frame, response)
                    
                    if result: # if the animal is in the ROI
                        if self._current_trial is None: # if currently no trial is set
                            self._current_trial = trial
                            self._trial_count[trial] += 1
                            print(trial, self._trial_count[trial])
                            self._intertrial_timer.reset()
                            
                    else: # stop trial when animal is outside of ROI and start the intertrial timer
                        if self._current_trial == trial:
                            self._current_trial = None
                            self._intertrial_timer.start()
```
The following lines are very important and pass the trial and trigger response to the different processes or the app.
```
            self._process.set_trial(self._current_trial)
            return result, response
```
That's it. `MyFirstExperiment` is ready.

### Stimulation and processes

Following the same principle from above we take the `ExampleProtocolProcess` as a basis for our own process `MyFirstProtocolProcess`.
As we inherit most of the functions from the `ProtocolProcess` class we can leave this almost unchanged. The only thing we need to do is to reference the new function `myfirst_protocol_run`.
```
class ExampleProtocolProcess(ProtocolProcess):
    """
    Class to help work with protocol function in multiprocessing with simple stimulation
    """
    def __init__(self):
        """
        Setting up the three queues and the process itself
        """
        super().__init__()
        self._protocol_process = mp.Process(target= myfirst_protocol_run, args=(self._trial_queue,)) #here
```
Now we need to actually define this function, again we are using the example as a basis:

```
def myfirst_protocol_run(condition_q: mp.Queue):
    current_trial = None
    #This will be different for different boards, here we are taking the nidaqxm library as a foundation and added some custom scripts that you can find in daq_output.py
    dmod_device1 = DigitalModDevice('Dev1/PFI0')
    dmod_device2 = DigitalModDevice('Dev1/PFI1')

    while True:
        if condition_q.full():
            current_trial = condition_q.get()
        if current_trial is not None:
            if current_trial == 'Trial_1':
                dmod_device1.toggle()
            elif current_trial == 'Trial_2':
                dmod_device2.toggle()
        else:
            dmod_device1.turn_off()
            dmod_device2.turn_off()
           
```
We are done! `MyFirstExperiment` is now complete. We hope that this helped you on your way to your own experimental design.
