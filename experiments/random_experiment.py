from qpsolver.quadratic_program import QuadraticProgram as QP
from qpsolver.solver import Solver

import numpy as np
import json

class RandomExperiment:
    def __init__(self, n, number):
        self.n = n
        self.number = number
        self.generate_data(n)

    def generate_data(self, n):
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

    def print_QP(self):
        self.QP.print()

    def print(self):
        print('This is experiment is random')
        self.print_QP()
        self.print_iterates()
        self.print_residuals()

    def save_experiment(self):
        if self.residuals[-1] != 0:
            d = {'A': self.QP.A.tolist(), 
                 'b': self.QP.b.tolist(), 
                 'u': self.QP.u.tolist(),
                 #'iterates': self.iterates, 
                 #'residuals': self.residuals,
                 }
            with open(f'./experiments/results/random_experiment_dim_{self.n}_nr_{self.number}.json', 'w') as json_file:
                json.dump(d, json_file)
