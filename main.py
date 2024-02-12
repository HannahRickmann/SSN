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
    for i in tqdm(range(100000)): # Iterate through random experiments
        experiment = Experiment(i)
        experiment.generate_data(3)
        experiment.run(10)

def try_custom_experiment():
    experiment = Experiment(1)
    experiment.read_custom_data
    experiment.run(10)