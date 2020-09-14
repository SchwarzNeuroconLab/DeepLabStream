from experiments.utils.exp_setup import DlStreamConfigWriter

if __name__ == '__main__':
    config = DlStreamConfigWriter()
    config.set_default(experiment_name= 'BaseConditionalExperiment', trigger_name= 'BaseRegionTrigger', stimulation_name= 'ScreenStimulation')
    config.set_experimentor('Jens')
    config.write_ini()

