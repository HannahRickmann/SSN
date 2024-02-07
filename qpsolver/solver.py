import numpy as np
import numpy.linalg as la
from qpsolver.quadratic_program import QuadraticProgram

class Solver:
    def __init__(self, QP: QuadraticProgram):
        self.QP = QP

    def solve(self,x0, max_iterations, method: str):
        if method == 'newton':
            return self.solve_newton(x0, max_iterations)
        elif method == 'pdas':
            return self.solve_pdas(x0, max_iterations)
        else:
            print('No valid method. Choose "newton" or "pdsa".')

    def solve_newton(self, x_0, max_iterations):
        def F(x):
            x, mu = np.split(x,2)
            return self.QP.KKT_condition(x, mu)

        def G(x):
            x, mu = np.split(x,2)
            return self.QP.KKT_jacobian(x, mu)
        
        iterates = self.newton(F,G, x_0, max_iterations)
        return [np.split(x, 2) for x in iterates]
    
    def newton(self, F, G, x0, max_iterations):
        iterates = [x0]
        xn = x0
        for _ in range(0, max_iterations):
            delta = la.solve(G(xn), -F(xn))
            xn = xn + delta
            iterates.append(xn)
        return iterates
    
    def solve_pdas(self, x0, max_iterations):
        x, mu = np.split(x0,2)
        xn = [np.array(x), np.array(mu)]
        iterates = [xn]
        for _ in range(0, max_iterations):
            xn = self.pdas_update(xn)
            iterates.append(xn)
        return iterates

    def pdas_update(self, xn):
        x, mu = xn
        n = len(x)
        active = self.QP.get_active_indices(x, mu)
        print(active)
        inactive = self.QP.get_inactive_indices(x, mu, active)
        
        if len(active) == 0:
            x = self.pdas_get_x_no_active(inactive)
            mu = self.pdas_get_mu_no_active(inactive)
        elif len(inactive) == 0:
            x = self.pdas_get_x_no_inactive(active)
            mu = self.pdas_get_mu_no_inactive(active)
        else:
            x = self.pdas_get_x_regular(active, inactive, n)
            mu = self.pdas_get_mu_regular(active, inactive, x, n)
        
        return [np.array(x), np.array(mu)]
    
    def pdas_get_x_regular(self, active, inactive, n):
        left = self.QP.A[inactive,:][:,inactive]
        right = self.QP.b[inactive] - self.QP.A[inactive,:][:, active].dot(self.QP.u[active])
        x_i = la.solve(left, right)
        x_a = self.QP.u[active]
        x = [x_i[inactive.index(j)] if j in inactive else x_a[active.index(j)] for j in range(n)]
        return x
    
    def pdas_get_x_no_inactive(self, active):
        return self.QP.u[active].tolist()
    
    def pdas_get_x_no_active(self, inactive):
        left = self.QP.A[inactive,:][:,inactive]
        right = self.QP.b[inactive]
        x = la.solve(left, right)
        return x.tolist()
    
    def pdas_get_mu_regular(self, active, inactive, x, n):
        x = np.array(x)
        left = np.identity(len(active))
        right = self.QP.b[active] - self.QP.A[active,:][:,active].dot(self.QP.u[active]) - self.QP.A[active,:][:,inactive].dot(x[inactive])
        mu_a = la.solve(left, right)
        mu_i = np.zeros(len(inactive))

        mu = [mu_i[inactive.index(j)] if j in inactive else mu_a[active.index(j)] for j in range(n)]
        return mu
    
    def pdas_get_mu_no_active(self, inactive):
        return np.zeros(len(inactive)).tolist()
    
    def pdas_get_mu_no_inactive(self, active):
        left = np.identity(len(active))
        right = self.QP.b[active] - self.QP.A[active,:][:,active].dot(self.QP.u[active])
        mu = la.solve(left, right)
        return mu.tolist()
    
    def test_convergence(self, iterates):
        return [np.around(np.sum(self.QP.KKT_condition(x, mu)), 2) for [x,mu] in iterates]