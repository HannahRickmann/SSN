import numpy as np
import numpy.linalg as la
from qpsolver.quadratic_program import QuadraticProgram

class Solver:
    def __init__(self, QP: QuadraticProgram):
        self.QP = QP

    def solve(self, x_0, max_iter):
        def F(x):
            x, mu = np.split(x,2)
            return self.QP.KKT_condition(x, mu)

        def G(x):
            x, mu = np.split(x,2)
            return self.QP.KKT_jacobian(x, mu)
        
        iterates = self.newton(F,G, x_0, max_iter)
        return [np.split(x, 2) for x in iterates]
    
    def newton(self, F, G, x0, max_iter):
        iterates = [x0]
        xn = x0
        for n in range(0, max_iter):
            delta = la.solve(G(xn), -F(xn))
            xn = xn + delta
            iterates.append(xn)
        return iterates
    
    def test_convergence(self, iterates):
        return [np.around(np.sum(self.QP.KKT_condition(x, mu)), 2) for [x,mu] in iterates]