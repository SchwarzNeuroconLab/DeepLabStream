from experiments.utils.exp_setup import DlStreamConfigWriter
import click


def design_experiment():

    config = DlStreamConfigWriter()
    input_dict = dict(EXPERIMENT = None,
                      TRIGGER = None,
                      PROCESS = None,
                      STIMULATION = None)

    for input_type in input_dict.keys():
        click.echo(f'Available {input_type}s are: ' + ', '.join(config.get_available_module_names(input_type)))
        input_value = click.prompt(f'Enter a base {input_type} module', type=str)

        while not config.check_if_default_exists(input_value, input_type):
            click.echo(f'{input_type} {input_value} does not exists.')
            input_value = click.prompt(f'Enter a base {input_type} module', type=str)

        input_dict[input_type] = input_value
        click.echo(f'{input_type} set to {input_value}.')

    config.import_default(experiment_name= input_dict['EXPERIMENT'], trigger_name=  input_dict['TRIGGER'], process_name=  input_dict['PROCESS'],stimulation_name=  input_dict['STIMULATION'])

    #Name of experimentor
    experimenter = click.prompt('Enter an experimenter name', type = str)
    click.echo(f'Experimenter set to {experimenter}.')
    config.set_experimentor(experimenter)

    click.echo('Current modules are:\n BaseExperiment: {}\n Trigger: {}\n Process: {} \n Stimulation: {}'.format(input_dict['EXPERIMENT'],
                                                                                                                 input_dict['TRIGGER'],
                                                                                                                 input_dict['PROCESS'],
                                                                                                                 input_dict['STIMULATION']))

    if click.confirm('Do you want to continue?'):
        config.write_ini()
        click.echo('Config was created. It can be found in experiments/configs')


if __name__ == '__main__':
    design_experiment()

