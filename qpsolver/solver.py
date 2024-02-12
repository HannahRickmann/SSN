import numpy as np
import numpy.linalg as la
from qpsolver.quadratic_program import QuadraticProgram

class Solver:
    def __init__(self, QP: QuadraticProgram):
        # Initialize the Solver with a QuadraticProgram instance
        self.QP = QP

    def solve(self,x0, max_iterations, method: str):
        # Solve the optimization problem using the specified method
        if method == 'newton':
            return self.solve_newton(x0, max_iterations)
        elif method == 'pdas':
            return self.solve_pdas(x0, max_iterations)
        else:
            print('No valid method. Choose "newton" or "pdsa".')

    def solve_newton(self, x_0, max_iterations):
        # Here the function and generalized gradient is set, which should be used by the newton method
        # Then the optimization problem is solved using Newton's method
        def F(x):
            x, mu = np.split(x,2)
            return self.QP.KKT_condition(x, mu)

        def G(x):
            x, mu = np.split(x,2)
            return self.QP.KKT_jacobian(x, mu)
        
        iterates = self.newton(F,G, x_0, max_iterations)
        return [np.split(x, 2) for x in iterates]
    
    def newton(self, F, G, x0, max_iterations):
        # Perform Newton's method for solving nonlinear systems of equations
        iterates = [x0]
        xn = x0
        for _ in range(0, max_iterations):
            delta = la.solve(G(xn), -F(xn))
            xn = xn + delta
            iterates.append(xn)
        return iterates
    
    def solve_pdas(self, x0, max_iterations):
        # Solve the optimization problem using the Primal-Dual Active Set method
        x, mu = np.split(x0,2)
        xn = [np.array(x), np.array(mu)]
        iterates = [xn]
        for _ in range(0, max_iterations):
            xn = self.pdas_update(xn)
            iterates.append(xn)
        return iterates

    def pdas_update(self, xn):
        # Perform a single iteration of the Primal-Dual Active Set method
        x, mu = xn
        active = self.QP.get_active_indices(x, mu)
        inactive = self.QP.get_inactive_indices(active)
        
        x, mu = self.pdas_get_x_mu(active, inactive)
        
        return [np.array(x), np.array(mu)]
    
    def pdas_get_x_mu(self, active, inactive):
        n = self.QP.n
        if len(active) == 0:
            x = self.pdas_get_x_no_active(inactive)
            mu = self.pdas_get_mu_no_active(inactive)
        elif len(inactive) == 0:
            x = self.pdas_get_x_no_inactive(active)
            mu = self.pdas_get_mu_no_inactive(active)
        else:
            x = self.pdas_get_x_regular(active, inactive, n)
            mu = self.pdas_get_mu_regular(active, inactive, x, n)
        return x, mu
    
    def pdas_get_x_regular(self, active, inactive, n):
        # Compute the next iteration of x for the Primal-Dual Active Set method
        left = self.QP.A[inactive,:][:,inactive]
        right = self.QP.b[inactive] - self.QP.A[inactive,:][:, active].dot(self.QP.u[active])
        x_i = la.solve(left, right) # on inactive indices, x is set such that Ax = b
        x_a = self.QP.u[active] # on active indices, the next x will be set to the upper bound
        x = [x_i[inactive.index(j)] if j in inactive else x_a[active.index(j)] for j in range(n)]
        return x
    
    def pdas_get_x_no_inactive(self, active):
        # Compute the next iteration of x when there are no inactive constraints
        return self.QP.u[active].tolist() # x is set to the upper bound
    
    def pdas_get_x_no_active(self, inactive):
        # Compute the next iteration of x when there are no active constraints
        left = self.QP.A[inactive,:][:,inactive]
        right = self.QP.b[inactive]
        x = la.solve(left, right) # x is set such that Ax = b
        return x.tolist()
    
    def pdas_get_mu_regular(self, active, inactive, x, n):
        # Compute the next iteration of mu for the Primal-Dual Active Set method
        x = np.array(x)
        left = np.identity(len(active))
        right = self.QP.b[active] - self.QP.A[active,:][:,active].dot(self.QP.u[active]) - self.QP.A[active,:][:,inactive].dot(x[inactive])
        mu_a = la.solve(left, right) # on active indices, mu is set such that Ax + mu = b
        mu_i = np.zeros(len(inactive)) # on inactive indices, mu in set to zero

        mu = [mu_i[inactive.index(j)] if j in inactive else mu_a[active.index(j)] for j in range(n)]
        return mu
    
    def pdas_get_mu_no_active(self, inactive):
        # Compute the next iteration of mu when there are no active constraints
        return np.zeros(len(inactive)).tolist() # on inactive indices, mu in set to zero
    
    def pdas_get_mu_no_inactive(self, active):
        # Compute the next iteration of mu when there are no inactive constraints
        left = np.identity(len(active))
        right = self.QP.b[active] - self.QP.A[active,:][:,active].dot(self.QP.u[active])
        mu = la.solve(left, right) # on active indices, mu is set such that Ax + mu = b
        return mu.tolist()
    
    def test_convergence(self, iterates):
        # Calculate the residuals of the KKT condition for every iterates
        # Here, the KKT conditions work as residual to the optimal solution, because we want to find the root of the respective function
        return [np.around(np.sum(self.QP.KKT_condition(x, mu)), 2) for [x,mu] in iterates]