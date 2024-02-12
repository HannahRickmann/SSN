from experiments.experiment import Experiment

from tqdm.auto import tqdm

def try_all_possibilities():
    max_int = 10
    length = 13224708

    experiment = Experiment(50)
    # if in \experiments\results there is no file 'data_combinations_max_int_{max_int}_len_{length}', then execute
    # experiment.construct_data(max_int) 

    experiment.read_constructed_data(max_int, length)

    for id in tqdm(range(length)): # Iterate through all possibilities constructed before and try experiment
        experiment.choose_experiment(id)
        experiment.run(5)

def try_random_experiment():
    for i in tqdm(range(10000)): # Iterate through random experiments
        experiment = Experiment(i)
        experiment.generate_data(4)
        experiment.run(10)

def try_custom_experiment():
    experiment = Experiment(17)
    experiment.read_custom_data('custom_experiment')
    #experiment.run(5, save = False)
    #experiment.print()
    experiment.analyse_possible_active_sets()

def try_previous_random_experiment():
    experiment = Experiment(8832)
    experiment.read_custom_data('random_experiment_dim_4')
    experiment.run(5, save = False)
    experiment.print()

try_previous_random_experiment()