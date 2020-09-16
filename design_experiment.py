from experiments.utils.exp_setup import DlStreamConfigWriter
import click

@click.command()
@click.option('--default', 'default', is_flag=True)
def design_experiment(default):
    config = DlStreamConfigWriter()
    input_dict = dict(EXPERIMENT = None,
                      TRIGGER = None,
                      PROCESS = None,
                      STIMULATION = None)

    def get_input(input_name):
        click.echo(f'Choosing {input_name}... \n Available {input_name}s are: ' + ', '.join(config.get_available_module_names(input_name)))
        input_value = click.prompt(f'Enter a base {input_name} module',type=str)
        while not config.check_if_default_exists(input_value,input_name):
            click.echo(f'{input_name} {input_value} does not exists.')
            input_value = click.prompt(f'Enter a base {input_name} module',type=str)
        return input_value
    """Experiment"""

    input_value = get_input('EXPERIMENT')
    input_dict['EXPERIMENT'] = input_value
    if input_value == 'BaseOptogeneticExperiment':
        input_dict['STIMULATION'] = 'BaseStimulation'
        input_dict['PROCESS'] = 'BaseProtocolProcess'

    elif input_value == 'BaseTrialExperiment':
        click.echo(f'Available Triggers are: ' + ', '.join(config.get_available_module_names('TRIGGER')))
        click.echo('Note, that you cannot select the same Trigger as selected in TRIGGER.')

        input_value = click.prompt(f'Enter TRIAL_TRIGGER for BaseTrialExperiment', type=str)
        while not config.check_if_default_exists(input_value, 'TRIGGER'):
            click.echo(f'TRIGGER {input_value} does not exists.')
            input_value = click.prompt(f'Enter a base TRIGGER module', type=str)

        input_dict['TRIAL_TRIGGER'] = input_value
        click.echo(f'TRIAL_TRIGGER for BaseTrialExperiment set to {input_value}.')

    """TRIGGER"""

    input_value = get_input('TRIGGER')
    input_dict['TRIGGER'] = input_value


    """Process"""

    if input_dict['PROCESS'] is None:
        input_value = get_input('PROCESS')
        input_dict['PROCESS'] = input_value

    """STIMULATION"""
    if input_dict['STIMULATION'] is None:
        input_value = get_input('STIMULATION')
        input_dict['STIMULATION'] = input_value

    """Setting Process Type"""

    if input_dict['EXPERIMENT'] == 'BaseTrialExperiment':
        input_dict['PROCESS_TYPE'] = 'trial'
    elif input_dict['STIMULATION'] == 'BaseStimulation':
        input_dict['PROCESS_TYPE'] = 'switch'
    elif input_dict['STIMULATION'] == 'ScreenStimulation' or input_dict['STIMULATION'] == 'RewardDispenser':
        input_dict['PROCESS_TYPE'] = 'supply'


    if input_dict['EXPERIMENT'] == 'BaseTrialExperiment':
        config.import_default(experiment_name=input_dict['EXPERIMENT'], trigger_name=input_dict['TRIGGER'],
                                  process_name=input_dict['PROCESS'], stimulation_name=input_dict['STIMULATION'],
                              trial_trigger_name=input_dict['TRIAL_TRIGGER'])

    else:
        config.import_default(experiment_name=input_dict['EXPERIMENT'], trigger_name=input_dict['TRIGGER'],
                                  process_name=input_dict['PROCESS'], stimulation_name=input_dict['STIMULATION'])

    if 'PROCESS_TYPE' in input_dict.keys():
        config._change_parameter(module_name=input_dict['PROCESS'],parameter_name='TYPE',
                                 parameter_value=input_dict['PROCESS_TYPE'])


    if click.confirm('Do you want to set parameters as well (Not recommended)? \n Note, that you can change them in the created file later.'):
        current_config = config.get_current_config()
        ignore_list = ['EXPERIMENT', 'BaseProtocolProcess']
        inner_ignore_list = ['EXPERIMENTER', 'PROCESS', 'STIMULATION', 'TRIGGER', 'DEBUG']
        try:
            for module in current_config.keys():
                parameter_dict = config.get_parameters(module)
                if module not in ignore_list:
                    for input_key in parameter_dict.keys():
                        if input_key not in inner_ignore_list:
                            click.echo(f'Default {input_key} is: ' + str(parameter_dict[input_key]))
                            input_value = click.prompt(f'Enter new value: ',type=str)
                            config._change_parameter(module_name=module,parameter_name=input_key,
                                                     parameter_value=input_value)
        except:
            click.echo('Failed to set individual parameters. Please change them later in the config file...')
    else:
        click.echo('Skipping parameters. Experiment config will be created with default values...')
    """Finish up"""
    # Name of experimentor
    experimenter = click.prompt('Enter an experimenter name',type=str)
    click.echo(f'Experimenter set to {experimenter}.')
    config.set_experimenter(experimenter)

    click.echo('Current modules are:\n BaseExperiment: {}\n Trigger: {}\n Process: {} \n Stimulation: {}'.format(
        input_dict['EXPERIMENT'],
        input_dict['TRIGGER'],
        input_dict['PROCESS'],
        input_dict['STIMULATION']))


    if click.confirm('Do you want to continue?'):
        config.write_ini()
        click.echo('Config was created. It can be found in experiments/configs')


if __name__ == '__main__':
    design_experiment()
