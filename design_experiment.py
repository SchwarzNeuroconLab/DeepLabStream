from experiments.utils.exp_setup import DlStreamConfigWriter
import click


def design_experiment():

    config = DlStreamConfigWriter()
    input_dict = dict(EXPERIMENT = None,
                      TRIGGER = None,
                      PROCESS = None,
                      STIMULATION = None)
    available_process_types = ['switch', 'supply']

    for input_type in input_dict.keys():
        if input_dict['EXPERIMENT'] == 'BaseOptogeneticExperiment' and input_type == 'STIMULATION':
            input_dict[input_type] = 'BaseStimulation'

        click.echo(f'Available {input_type}s are: ' + ', '.join(config.get_available_module_names(input_type)))
        input_value = click.prompt(f'Enter a base {input_type} module', type=str)

        while not config.check_if_default_exists(input_value, input_type):
            click.echo(f'{input_type} {input_value} does not exists.')
            input_value = click.prompt(f'Enter a base {input_type} module', type=str)

        input_dict[input_type] = input_value
        click.echo(f'{input_type} set to {input_value}.')

    if input_dict['EXPERIMENT'] == 'BaseTrialExperiment':
        valid_input = False
        click.echo(f'Available Triggers are: ' + ', '.join(config.get_available_module_names('TRIGGER')))
        click.echo('Note, that you cannot select the same Trigger as selected in TRIGGER.')

        input_value = click.prompt(f'Enter TRIAL_TRIGGER for BaseTrialExperiment', type=str)
        while not config.check_if_default_exists(input_value, 'TRIGGER'):
            click.echo(f'TRIGGER {input_value} does not exists.')
            input_value = click.prompt(f'Enter a base TRIGGER module', type=str)

        input_dict['TRIAL_TRIGGER'] = input_value
        click.echo(f'TRIAL_TRIGGER for BaseTrialExperiment set to {input_value}.')


    # TODO: Make adaptive for multiple Triggers
    if input_dict['EXPERIMENT'] == 'BaseTrialExperiment':
        config.import_default(experiment_name=input_dict['EXPERIMENT'], trigger_name=input_dict['TRIGGER'],
                              process_name=input_dict['PROCESS'], stimulation_name=input_dict['STIMULATION'],
                              trial_trigger_name=input_dict['TRIAL_TRIGGER'])

        config._change_parameter(module_name=input_dict['EXPERIMENT'], parameter_name='TRIAL_TRIGGER',
                             parameter_value=input_dict['TRIAL_TRIGGER'])

        if input_dict['PROCESS'] == 'BaseProtocolProcess':
            config._change_parameter(module_name=input_dict['PROCESS'], parameter_name='TYPE',
                                     parameter_value='trial')

    if input_dict['EXPERIMENT'] == 'BaseOptogeneticExperiment':
        config.import_default(experiment_name=input_dict['EXPERIMENT'], trigger_name=input_dict['TRIGGER'],
                              process_name=input_dict['PROCESS'], stimulation_name=input_dict['STIMULATION'])

        if input_dict['PROCESS'] == 'BaseProtocolProcess':
            config._change_parameter(module_name=input_dict['PROCESS'], parameter_name='TYPE',
                                     parameter_value='switch')


    else:
        config.import_default(experiment_name=input_dict['EXPERIMENT'], trigger_name=input_dict['TRIGGER'],
                              process_name=input_dict['PROCESS'], stimulation_name=input_dict['STIMULATION'])

    if input_dict['PROCESS'] == 'BaseProtocolProcess' and input_dict['EXPERIMENT'] != 'BaseTrialExperiment' and\
            input_dict['EXPERIMENT'] != 'BaseOptogeneticExperiment':
        valid_input = False

        click.echo(f'Available types are: ' + ', '.join(available_process_types))
        input_value = click.prompt(f'Enter a type for BaseProtocolProcess', type=str)
        if input_value in available_process_types:
            valid_input = True
        while not valid_input:
            click.echo(f'{input_type} {input_value} does not exists.')
            input_value = click.prompt(f'Enter a valid type for BaseProtocolProcess', type=str)
            if input_value in available_process_types:
                valid_input = True

        input_dict['PROCESS_TYPE'] = input_value
        click.echo(f'BaseProtocolProcess type set to {input_value}.')
        config._change_parameter(module_name = input_dict['PROCESS'], parameter_name = 'TYPE',
                                 parameter_value = input_dict['PROCESS_TYPE'])


    #Name of experimentor
    experimenter = click.prompt('Enter an experimenter name', type = str)
    click.echo(f'Experimenter set to {experimenter}.')
    config.set_experimenter(experimenter)

    click.echo('Current modules are:\n BaseExperiment: {}\n Trigger: {}\n Process: {} \n Stimulation: {}'.format(input_dict['EXPERIMENT'],
                                                                                                                 input_dict['TRIGGER'],
                                                                                                                 input_dict['PROCESS'],
                                                                                                                 input_dict['STIMULATION']))

    if click.confirm('Do you want to continue?'):
        config.write_ini()
        click.echo('Config was created. It can be found in experiments/configs')


if __name__ == '__main__':
    design_experiment()

