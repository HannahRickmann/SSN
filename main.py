from tqdm.auto import tqdm
import itertools
import numpy as np
import json
import os
from datetime import datetime

from experiments.experiment import Experiment

def try_all_possibilities():
    """
    Iterate through all pre-constructed possibilities of the data, where every number x is |x| <= 10 and run the experiment.
    If no constructed data file exists, then first construct the data with experiment.construct_data(max_int).
    """
    max_int = 10
    length = 13224708
    experiment = Experiment(50)
    experiment.read_constructed_data(max_int, length)

    # Iterate through all possibilities and run the experiment
    for id in tqdm(range(length)):
        experiment.choose_experiment(id)
        experiment.run(5)

def try_multiplying_constant(exp_nr, c):
    """
    Run the experiment with a given number, multiply the data by a constant, and re-run the experiment.
    """
    experiment = Experiment(exp_nr)

    # Read custom data and run the initial experiment
    experiment.read_custom_data('custom_experiment')
    experiment.run(5, save = False)
    experiment.print()

    # Multiply the data by the constant and re-run the experiment
    experiment.QP.A = c * experiment.QP.A
    experiment.QP.b = c * experiment.QP.b
    # experiment.QP.l = c * experiment.QP.u
    # experiment.QP.u = np.array([np.inf] * experiment.QP.n)
    experiment.run(5, save = False)
    experiment.print()

def try_different_bounds(exp_nr):
    """
    Try different upper bounds for the experiment and save the bounds that do not converge.
    """
    # all_combinations_u = list(itertools.product(np.linspace(-3, 1, 20), repeat=3)) #40
    all_combinations_u = list(itertools.product(range(-20,12), repeat=3))
    print(len(all_combinations_u))
    experiment = Experiment(exp_nr)

    # Read custom data for the experiment
    experiment.read_custom_data('custom_experiment')
    u_list = []

    # Iterate through all combinations of upper bounds
    for idx, u in tqdm(enumerate(all_combinations_u)):
        experiment.id = idx
        experiment.name = f'upper_bound_exp_{exp_nr}_nr_{idx}'
        experiment.QP.u = np.array(u)
        experiment.run(5, save = False)

        # If the residuals do not converge, save the upper bound
        if experiment.residuals[-1] != 0:
            u_list.append(u)

    # Save the upper bounds to a file
    with open(f'./experiments/results/upper_bound_exp_{exp_nr}.json', 'w') as json_file:
        json.dump(u_list, json_file)

def try_possible_active_set_cycles(n, n_cycle):
    """
    Test possible active set cycles for the experiment.

    Parameters:
    n (int): The dimension of the problem.
    n_cycle (int): Number of cycles to test.
    """
    experiment = Experiment(1)
    experiment.test_possible_cycles(n, n_cycle)

def try_random_experiment(amount, n, lower_bound = False):
    """
    Generate and run random experiments.

    Parameters:
    amount (int): Number of random experiments to run.
    n (int): Dimension of the problem.
    lower_bound (bool): Whether to include lower bounds in the experiment.
    """
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    for i in tqdm(range(amount)):
        experiment = Experiment(i)
        experiment.current_time = current_time
        experiment.generate_data(n, lower_bound)
        experiment.run(10, save=True)

def try_previous_random_experiment(time):
    """
    Re-run and analyze previously generated random experiments.

    Parameters:
    time (str): The timestamp of the previous experiments.
    """
    n = 3

    # Get the list of experiment files from the specified directory
    file_list = os.listdir(f'./experiments/results/{time}/')
    file_list = [file_name for file_name in file_list if f'random_experiment_dim_{n}' in file_name]
    
    # Extract the experiment numbers from the file names
    numbers = []
    for file_name in file_list:
        start_index = file_name.find('nr_') + 3
        end_index = file_name.find('.json')
        numbers.append(file_name[start_index:end_index])

    # Re-run and analyze each experiment
    for number in numbers:
        experiment = Experiment(number)
        experiment.read_custom_data('random_experiment_dim_3', folder=f'results/{time}')
        experiment.run(5, save = False)
        print(f'experiment number: {number}')
        # experiment.print()
        # experiment.analyse_possible_active_sets()
        experiment.analyse_active_set_cycle()

def try_custom_experiment(exp_nr, analyse=False):#
    """
    Run and optionally analyse a custom experiment.
    """
    experiment = Experiment(exp_nr)
    experiment.read_custom_data('custom_experiment')
    experiment.run(5, save = False) #, method="newton")
    experiment.print()

    # If analyse is True, analyze the active sets and cycles
    if analyse:
        experiment.analyse_possible_active_sets()
        experiment.analyse_active_set_cycle()


# Examples of function calls
# try_multiplying_constant(17, -1)
# try_different_bounds(31)
# try_random_experiment(100000, n=3, lower_bound=False)
# try_previous_random_experiment('2024-04-02_17-41-42')
# try_custom_experiment(17, analyse=True)
# try_possible_active_set_cycles(3,3)