from experiments.experiment import Experiment

from tqdm.auto import tqdm
import itertools
import numpy as np
import json
import os
from datetime import datetime

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

def try_previous_random_experiment(time):
    n = 3

    file_list = os.listdir(f'./experiments/results/{time}/')
    file_list = [file_name for file_name in file_list if f'random_experiment_dim_{n}' in file_name]

    numbers = []
    for file_name in file_list:
        start_index = file_name.find('nr_') + 3
        end_index = file_name.find('.json')
        numbers.append(file_name[start_index:end_index])

    for number in numbers:
        experiment = Experiment(number)
        experiment.read_custom_data('random_experiment_dim_3', folder=f'results/{time}')
        experiment.run(5, save = False)
        print(f'experiment number: {number}')
        experiment.analyse_active_set_cycle()

def try_multiplying_constant(exp_nr, c):
    experiment = Experiment(exp_nr)
    experiment.read_custom_data('custom_experiment')
    experiment.run(5, save = False)
    experiment.print()
    experiment.QP.A = c * experiment.QP.A
    experiment.QP.b = c * experiment.QP.b
    experiment.run(5, save = False)
    experiment.print()

def try_different_bounds(exp_nr):
    all_combinations_u = list(itertools.product(np.linspace(-3, 1, 20), repeat=3)) #40
    # all_combinations_u = list(itertools.product(range(-20,12), repeat=3))
    print(len(all_combinations_u))
    experiment = Experiment(exp_nr)
    experiment.read_custom_data('custom_experiment')
    u_list = []
    for idx, u in tqdm(enumerate(all_combinations_u)):
        experiment.id = idx
        experiment.name = f'upper_bound_exp_{exp_nr}_nr_{idx}'
        experiment.QP.u = np.array(u)
        experiment.run(5, save = False)
        if experiment.residuals[-1] != 0:
            u_list.append(u)
    with open(f'./experiments/results/upper_bound_exp_{exp_nr}.json', 'w') as json_file:
        json.dump(u_list, json_file)

def try_random_experiment(amount, n, upper_bound = False):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for i in tqdm(range(amount)): # Iterate through random experiments
        experiment = Experiment(i)
        experiment.current_time = current_time
        experiment.generate_data(n, upper_bound)
        experiment.run(10)

def try_custom_experiment(exp_nr, analyse=False):
    experiment = Experiment(exp_nr)
    experiment.read_custom_data('custom_experiment')
    # experiment.QP.u = np.array([np.inf] * 3)
    experiment.x_0 = np.array([-13,-9,0,0])
    experiment.run(5, save = False) #, method="newton")
    experiment.print()
    if analyse:
        experiment.analyse_possible_active_sets()
        experiment.analyse_active_set_cycle()


# try_multiplying_constant(30, 2)
# try_different_bounds(17)
        
try_previous_random_experiment('2024-03-11_10-50-45')

# try_random_experiment(100, n=2, upper_bound=True)
# try_custom_experiment(30, analyse=True)