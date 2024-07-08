from qpsolver.quadratic_program import QuadraticProgram as QP
from qpsolver.solver import Solver

import numpy as np
import itertools
import json
import os

class Experiment:
    def __init__(self, id: int):
        """
        Initialize an experiment with an ID, which functions as the experiment number.
        """
        self.id = id
        self.name = ""
        self.comment = ""
        self.current_time = "other"

        # After initializing, you have to choose or generate data for the experiment.
        # Call read_custom_data, generate_data, or read_constructed_data to set up a Quadratic Program and a Solver.

    def read_custom_data(self, experiment_type, folder = 'input'):
        """
        Read custom data from a JSON file and set up the experiment with a Quadratic Program and a Solver.
        """
        self.name = f'{experiment_type}_nr_{self.id}'
        with open(f'./experiments/{folder}/{experiment_type}_nr_{self.id}.json', 'r') as json_file:
            data = json.load(json_file)
        A = data['A']
        b = data['b']
        u = data['u']

        if 'comment' in data:
            self.comment = data['comment']

        # Set up the Quadratic Program with or without lower bounds
        if 'l' in data:
            l = data['l']
            self.QP = QP(A=np.array(A), b=np.array(b), u=np.array(u), l=np.array(l))
        else:
            l = None
            self.QP = QP(A=np.array(A), b=np.array(b), u=np.array(u))
        self.solver = Solver(self.QP)
        
        # Initial guess for the solver
        if 'x_0' in data:
            self.x_0 = np.array(data['x_0'])
        else:
            self.x_0 = np.zeros(len(b)*2)
        

    def generate_data(self, n, lower_bound = False):
        """
        Generate random data for the experiment.
        """
        self.name = f"random_experiment_dim_{n}_nr_{self.id}"

        # Generate a positive definite matrix A
        #temp = np.random.randint(-3, 3, (n, n))
        temp = np.random.uniform(-3, 3, size=(n, n))
        A = np.dot(temp, temp.transpose())
        while(np.linalg.det(A) <= 0.0000001): # ensure matrix to be non singular
            #temp = np.random.randint(-3, 3, (n, n))
            temp = np.random.uniform(-3, 3, size=(n, n))
            A = np.dot(temp, temp.transpose())

        #b = np.random.randint(-10, 10, n)
        b = np.random.uniform(-10, 10, size=n)
        #u = np.random.randint(-10, 10, n)
        u = np.random.uniform(-10, 10, size=n)

        if lower_bound:
            #diff = np.random.randint(-10, 0, n)
            diff = np.random.uniform(-10, 0, size=n)
            l = u + diff
        else:
            l = np.array([-np.inf] * n)

        self.x_0 = np.zeros(n*2)

        # Set up the Quadratic Program and Solver
        self.QP = QP(A=np.array(A), b=np.array(b), u=np.array(u), l = np.array(l))
        self.solver = Solver(self.QP)

    def construct_data(self, max_int):
        """
        Construct all possible input data within certain restrictions, where every number is less than or equal to max_int.
        """
        # Construct matrices where global convergence is not yet proven.
        all_combinations_A = list(itertools.product(range(1,max_int + 1), repeat=3)) # all entries are positive, so no M-matrix

        # Criteria for positive definiteness and not cone-preserving
        def positive_definite(combination):
            return combination[0]*combination[2] - combination[1]**2 > 0

        def not_cone_preserving(combination):
            return combination[0] < combination[1] or combination[2] < combination[1]

        valid_combinations_A = [comb for comb in all_combinations_A if positive_definite(comb) and not_cone_preserving(comb)]
        all_combinations_b_u = list(itertools.product(range(-1*max_int,max_int + 1), repeat=4)) # length: (2*max_int + 1)**4
        all_combinations = list(itertools.product(valid_combinations_A, all_combinations_b_u))

        l = [list(combination[0] + combination[1]) for combination in all_combinations]
        length = len(l)

        # Save constructed data to a JSON file
        with open(f'./experiments/results/data_combinations_max_int_{max_int}_len_{length}.json', 'w') as json_file:
            json.dump(l, json_file)

    def read_constructed_data(self, max_int, length):
        """
        Read constructed data for the experiment.

        Parameters:
        max_int (int): The maximum integer value for the data.
        length (int): The total number of possibilities that were constructed.
        """
        with open(f'./experiments/results/data_combinations_max_int_{max_int}_len_{length}.json', 'r') as json_file:
            l = json.load(json_file)
        self.combinations = l

    def choose_experiment(self, id):
        """
        Choose one of the constructed data possibilities.
        """
        self.id = id
        self.name = f"constructed_experiment_dim_{2}_nr_{self.id}"
        combination = self.combinations[id]

        A = np.array([[combination[0], combination[1]], [combination[1], combination[2]]])
        b = np.array([combination[3], combination[4]])
        u = np.array([combination[5], combination[6]])

        self.x_0 = np.zeros(4)

        # Set up the Quadratic Program and Solver
        self.QP = QP(A=np.array(A), b=np.array(b), u=np.array(u))
        self.solver = Solver(self.QP)
        
    def run(self, max_iterations, method='pdas', save = True):
        """
        Run the experiment using the solver to generate iterates and test convergence afterwards.
        """
        self.iterates = self.solver.solve(self.x_0, max_iterations, method)
        self.residuals = self.solver.test_convergence(self.iterates)
        if save:
            self.save_experiment()

    def analyse_possible_active_sets(self):
        """
        Analyze all possible ways to divide the index set into two disjoint sets.
        """
        lower_bound = True
        if (self.QP.l == -np.inf).any():
            lower_bound = False

        # Generate all possible ways to divide index set into two disjoint sets
        best_x = None
        best_mu = None
        best_residual = np.Infinity
        best_index_set = []

        all_partitions = self.get_possible_index_partitions(n=self.QP.n, lower_bound=lower_bound)

        # try all possible active/inactive set combinations
        for partition in all_partitions:
            active_plus = partition[0]
            active_minus = partition[1]
            inactive = partition[2]
            
            x = self.solver.pdas_get_x(active_plus, active_minus, inactive)
            mu = self.solver.pdas_get_mu(x, active_plus, active_minus, inactive)

            residual = np.absolute(np.sum(self.QP.KKT_condition(x, mu)))
            next_index_set = self.QP.get_active_indices(x, mu)

            # if current active set leads to better solution than current best, update
            if residual < best_residual:
                best_residual = residual
                best_x = x
                best_mu = mu
                best_index_set = [active_plus, active_minus, inactive]
            
            print(f'Index sets: {[active_plus, active_minus, inactive]}')
            print(f"x: {x}")
            print(f"mu: {mu}")
            print(f"residual: {residual}")
            print(f"next index set: {next_index_set}")
            print("-"*24)

        print(f'Optimal index sets: {best_index_set}')
        print(f"x: {best_x}")
        print(f"mu: {best_mu}")
        print(f"residual: {best_residual}")
        print("-"*24)

    def get_possible_index_partitions(self, n, lower_bound = True):
        """
        Generate all possible ways to divide the index set into two disjoint sets.
        """
        all_partitions = []
        for i in range(n + 1):
            for combination in itertools.combinations(range(n), i):
                set1 = list(combination)
                set2 = [x for x in range(n) if x not in set1]
                for j in range(len(set2) + 1):
                    for combo in itertools.combinations(set2, j):
                        set2_1 = list(combo)
                        set2_2 = [x for x in set2 if x not in set2_1]
                        all_partitions.append((set1, set2_1, set2_2))
        if not lower_bound:
            all_partitions = [partition for partition in all_partitions if len(partition[1]) == 0]
        return all_partitions

    def analyse_active_set_cycle(self):
        """
        Analyze the active set cycle to check the sequence of active sets.
        """
        lower_bound = True
        if (self.QP.l == -np.inf).any():
            lower_bound = False
        all_partitions = self.get_possible_index_partitions(n=self.QP.n, lower_bound=lower_bound)
        partition = all_partitions[0]
        output_string = f"{partition}"
        while len(all_partitions) != 0:
            active_plus = partition[0]
            active_minus = partition[1]
            inactive = partition[2]
            
            x = self.solver.pdas_get_x(active_plus, active_minus, inactive)
            mu = self.solver.pdas_get_mu(x, active_plus, active_minus, inactive)
            next_index_set = self.QP.get_active_indices(x, mu)

            output_string += f" --> {next_index_set}"

            all_partitions.remove(partition)
            if len(all_partitions) == 0:
                break
            if next_index_set in all_partitions:
                partition = next_index_set
            else:
                partition = all_partitions[0]
                output_string += f"\n{partition}"
        
        print(output_string)
        print("-"*24)

    def test_possible_cycles(self, n, n_cycle):
        """
        Test for possible active set cycles.
        """
        self.QP = QP(A=np.zeros((n,n)), b=np.zeros(n), u=np.zeros(n))
        all_partitions = self.get_possible_index_partitions(self.QP.n, lower_bound=False)
        
        all_active_sets = [partition[0] for partition in all_partitions]
        print(f'In total we have {len(all_active_sets)} active sets and these are \n{all_active_sets}.')
        amount = 0
        for active_sets in itertools.combinations(all_active_sets, n_cycle):
            amount += 1
            self.test_active_set_cycle(n, n_cycle, active_sets)
        print(f'The total amount of theoretical possible active set cycles is: {amount}')

    def test_active_set_cycle(self, n, n_cycle, active_sets: list[list[int]]):
        """
        Test a specific active set cycle.
        """
        indices = range(0,n)
        inactive_sets = [[i for i in indices if i not in active_set] for active_set in active_sets]

        for idx,a in enumerate(active_sets):
            l1 = set(inactive_sets[idx % n_cycle]) & set(active_sets[(idx-1)%n_cycle]) & set(active_sets[(idx+1)%n_cycle])
            l2 = set(active_sets[idx%n_cycle]) & set(inactive_sets[(idx-1)%n_cycle]) & set(inactive_sets[(idx+1)%n_cycle])
            l = l1 | l2
            if len(l) == 0:
                print(active_sets)
                print(f'Contradicts Necessary Condition in the {idx+ 1}. case.')
                return  
        print(active_sets)
        print('Necessary Condition is fulfilled.')
        return
        
    
    def print_iterates(self):
        for i in range(0, len(self.iterates)):
            print(f"x  {i}: ", np.around(self.iterates[i][0], 2))
            print(f"mu {i}: ", np.around(self.iterates[i][1], 2))
            print("-"*24)

    def print_solution(self):
        print(f"x*: ", np.around(self.iterates[-1][0], 2))
        print(f"mu*: ", np.around(self.iterates[-1][1], 2))
        print("-"*24)
    
    def print_active_sets(self):
        active = lambda i : self.QP.get_active_indices(self.iterates[i][0], self.iterates[i][1])
        active_list = [active(i) for i in range(0, len(self.iterates))]
        print(f'Index sets: {active_list}')

    def print_residuals(self):
        print(self.residuals)

    def print(self):
        print(self.name, ': ', self.comment)
        self.print_QP()
        self.print_iterates()
        self.print_active_sets()
        self.print_residuals()
        print('\n')

    def print_QP(self):
        self.QP.print()

    def save_experiment(self, only_no_convergence = True):
        # Save the experiment results if the method did not converge
        directory = f'./experiments/results/{self.current_time}/'
        os.makedirs(directory, exist_ok=True)
        if only_no_convergence:
            save = (self.residuals[-1] != 0)
        else:
            save = True

        iterates = [[x.tolist(), mu.tolist()] for x, mu in self.iterates]
    
        if save:
            d = {#'A': self.QP.A.tolist(), 
                 #'b': self.QP.b.tolist(), 
                 #'u': self.QP.u.tolist(),
                 'iterates': iterates, 
                 'residuals': self.residuals,
                 }
            if not np.any(np.isneginf(self.QP.l)):
                d['l'] = self.QP.l.tolist(),
            with open(f'./experiments/results/{self.current_time}/{self.name}.json', 'w') as json_file:
                json.dump(d, json_file)