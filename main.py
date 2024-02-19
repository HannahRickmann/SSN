from experiments.experiment import Experiment

from tqdm.auto import tqdm
import itertools
import numpy as np
import json

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
    experiment = Experiment(26)
    experiment.read_custom_data('custom_experiment')
    # experiment.QP.u = np.array([np.inf] * 3)
    experiment.run(5, save = False, method="newton")
    experiment.print()
    # experiment.analyse_possible_active_sets()

def try_multiplying_constant():
    c = 2
    experiment = Experiment(17)
    experiment.read_custom_data('custom_experiment')
    experiment.run(5, save = False)
    experiment.print()
    experiment.QP.A = c * experiment.QP.A
    experiment.QP.b = c * experiment.QP.b
    experiment.run(5, save = False)
    experiment.print()

def try_different_bounds():
    exp_nr = 17
    all_combinations_u = list(itertools.product(range(-20,12), repeat=3))
    print(len(all_combinations_u))
    experiment = Experiment(exp_nr)
    experiment.read_custom_data('custom_experiment')
    u_list = []
    for idx, u in tqdm(enumerate(all_combinations_u)):
        experiment.id = idx
        experiment.name = f'upper_bound_exp_{exp_nr}_nr_{idx}'
        experiment.QP.u = np.array(u)
        experiment.run(5, save = True)
        if experiment.residuals[-1] != 0:
            u_list.append(u)
    with open(f'./experiments/results/upper_bound_exp_{exp_nr}.json', 'w') as json_file:
        json.dump(u_list, json_file)


def try_previous_random_experiment():
    numbers = [987, 2564, 2706, 3639, 3898, 4621, 5327, 5437, 6773, 7481, 9356, 10160, 12071, 14409, 21671, 22182, 23639, 24005, 26253, 26760, 27667, 28483, 28818, 29495]
    for number in numbers:
        experiment = Experiment(number)
        experiment.read_custom_data('random_experiment_dim_3')
        experiment.run(5, save = False)
        print(f'experiment number: {number}')
        experiment.print_active_sets()

try_random_experiment()