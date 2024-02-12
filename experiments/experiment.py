from qpsolver.quadratic_program import QuadraticProgram as QP
from qpsolver.solver import Solver

import numpy as np
import pandas as pd
import itertools
import json

class Experiment:
    def __init__(self, id: int):
        self.id = id
        self.name = ""
        self.comment = ""

    def read_custom_data(self):
        self.name = f"custom_experiment_nr_{self.id}"
        df = pd.read_csv('./experiments/custom_data.csv')
        A = eval(df.loc[df['id'] == self.id, 'A'].values[0])
        b = eval(df.loc[df['id'] == self.id, 'b'].values[0])
        u = eval(df.loc[df['id'] == self.id, 'u'].values[0])
        self.comment = df.loc[df['id'] == self.id, 'comment'].values[0]
        if df.loc[df['id'] == self.id, 'mode'].values[0] == 'BOX':
            l = eval(df.loc[df['id'] == self.id, 'l'].values[0])
        else:
            l = None
        x_0 = eval(df.loc[df['id'] == self.id, 'x0'].values[0])
        self.x_0 = np.array(x_0)
        if l == None:
            self.QP = QP(A=np.array(A), b=np.array(b), u=np.array(u))
        else:
            self.QP = QP(A=np.array(A), b=np.array(b), u=np.array(u), l=np.array(l))
        self.solver = Solver(self.QP)

    def generate_data(self, n):
        self.name = f"random_experiment_dim_{n}_nr_{self.id}"
        temp = np.random.randint(-50, 50, (n, n))
        #temp = np.random.rand(n, n)
        A = np.dot(temp, temp.transpose())
        while(np.linalg.det(A) <= 0.0000001): # ensure matrix to be non singular
            temp = np.random.randint(-50, 50, (n, n))
            #temp = np.random.rand(n, n)
            A = np.dot(temp, temp.transpose())
        b = np.random.randint(-100, 100, n)
        #b = np.random.rand(n)
        u = np.random.randint(-100, 100, n)
        #u = np.random.rand(n)
        self.x_0 = np.zeros(n*2)
        self.QP = QP(A=np.array(A), b=np.array(b), u=np.array(u))
        self.solver = Solver(self.QP)

    def construct_data(self, max_int):
        all_combinations_A = list(itertools.product(range(1,max_int + 1), repeat=3))

        def positive_definite(combination):
            return combination[0]*combination[2] - combination[1]**2 > 0

        def not_cone_preserving(combination):
            return combination[0] < combination[1] or combination[2] < combination[1]

        valid_combinations_A = [comb for comb in all_combinations_A if positive_definite(comb) and not_cone_preserving(comb)]

        all_combinations_b_u = list(itertools.product(range(-1*max_int,max_int + 1), repeat=4)) # length: (2*max_int + 1)**4

        all_combinations = list(itertools.product(valid_combinations_A, all_combinations_b_u))
        l = [list(combination[0] + combination[1]) for combination in all_combinations]
        length = len(l)
        with open(f'./experiments/results/data_combinations_max_int_{max_int}_len_{length}.json', 'w') as json_file:
            json.dump(l, json_file)

    def read_constructed_data(self, max_int, length):
        with open(f'./experiments/results/data_combinations_max_int_{max_int}_len_{length}.json', 'r') as json_file:
            l = json.load(json_file)
        self.combinations = l

    def choose_experiment(self, id):
        self.id = id
        self.name = f"constructed_experiment_dim_{2}_nr_{self.id}"
        combination = self.combinations[id]

        A = np.array([[combination[0], combination[1]], [combination[1], combination[2]]])
        b = np.array([combination[3], combination[4]])
        u = np.array([combination[5], combination[6]])

        self.x_0 = np.zeros(4)
        self.QP = QP(A=np.array(A), b=np.array(b), u=np.array(u))
        self.solver = Solver(self.QP)
        
    def run(self, max_iterations, method='pdas'):
        self.iterates = self.solver.solve(self.x_0, max_iterations, method)
        self.residuals = self.solver.test_convergence(self.iterates)
        self.save_experiment()
    
    def print_iterates(self):
        for i in range(0, len(self.iterates)):
            print("x  {}: ".format(i), np.around(self.iterates[i][0], 2))
            print("mu {}: ".format(i), np.around(self.iterates[i][1], 2))
            print("------------------------")
    
    def print_residuals(self):
        print(self.residuals)

    def print(self):
        print(self.name, ': ', self.comment)
        self.print_QP()
        self.print_iterates()
        self.print_residuals()
        print('\n')

    def print_QP(self):
        self.QP.print()

    def save_experiment(self):
        if self.residuals[-1] != 0:
            d = {'A': self.QP.A.tolist(), 
                 'b': self.QP.b.tolist(), 
                 'u': self.QP.u.tolist(),
                 #'iterates': self.iterates, 
                 #'residuals': self.residuals,
                 }
            with open(f'./experiments/results/{self.name}.json', 'w') as json_file:
                json.dump(d, json_file)