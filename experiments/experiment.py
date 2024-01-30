from qpsolver.quadratic_program import QuadraticProgram as QP
from qpsolver.solver import Solver

import numpy as np
import pandas as pd
import os
import ast

class Experiment:
    def __init__(self, number: int):
        self.number = number
        self.read_data(number)

    def read_data(self, number: int):
        df = pd.read_csv('./experiments/data.csv')
        A = ast.literal_eval(df.loc[df['number'] == number, 'A'].values[0])
        b = ast.literal_eval(df.loc[df['number'] == number, 'b'].values[0])
        u = ast.literal_eval(df.loc[df['number'] == number, 'u'].values[0])
        self.comment = df.loc[df['number'] == number, 'comment'].values[0]
        if df.loc[df['number'] == number, 'mode'].values[0] == 'BOX':
            l = ast.literal_eval(df.loc[df['number'] == number, 'l'].values[0])
        else:
            l = None
        x_0 = ast.literal_eval(df.loc[df['number'] == number, 'x0'].values[0])
        self.x_0 = np.array(x_0)
        if l == None:
            self.QP = QP(A=np.array(A), b=np.array(b), u=np.array(u))
        else:
            self.QP = QP(A=np.array(A), b=np.array(b), u=np.array(u), l=np.array(l))
        self.solver = Solver(self.QP)

    def run(self, max_iterations):
        self.iterates = self.solver.solve(self.x_0, max_iterations)
        self.residuals = self.solver.test_convergence(self.iterates)
    
    def print_iterates(self):
        for i in range(0, len(self.iterates)):
            print("x  {}: ".format(i), np.around(self.iterates[i][0], 2))
            print("mu {}: ".format(i), np.around(self.iterates[i][1], 2))
            print("------------------------")
    
    def print_residuals(self):
        print(self.residuals)

    def print(self):
        print('This is experiment number {}:'.format(self.number), self.comment)
        self.print_iterates()
        self.print_residuals()