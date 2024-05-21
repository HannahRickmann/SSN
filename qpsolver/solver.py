import numpy as np
import numpy.linalg as la
from qpsolver.quadratic_program import QuadraticProgram

class Solver:
    def __init__(self, QP: QuadraticProgram):
        # Initialize the Solver with a QuadraticProgram instance
        self.QP = QP
        self.converged = False

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
            if self.test_convergence([xn])[-1] == 0:
                self.converged = True
                break
        return iterates

    def pdas_update(self, xn):
        # Perform a single iteration of the Primal-Dual Active Set method
        x, mu = xn
        active_plus, active_minus, inactive = self.QP.get_active_indices(x, mu)
        
        x = self.pdas_get_x(active_plus, active_minus, inactive)
        mu = self.pdas_get_mu(x, active_plus, active_minus, inactive)
        
        return [x, mu]
    
    def pdas_get_x(self, active_plus, active_minus, inactive):
        # Compute the next iteration of x for the Primal-Dual Active Set method
        active = active_plus + active_minus

        left = self.QP.A[inactive,:][:,inactive]
        right = self.QP.b[inactive] - self.QP.A[inactive,:][:, active_plus].dot(self.QP.u[active_plus]) - self.QP.A[inactive,:][:, active_minus].dot(self.QP.l[active_minus])
        x_i = la.solve(left, right) # on inactive indices, x is set such that Ax = b
        x_a = list(self.QP.u[active_plus]) + list(self.QP.l[active_minus]) # on active indices, the next x will be set to the upper/lower bound
        return np.array([x_i[inactive.index(j)] if j in inactive else x_a[active.index(j)] for j in range(self.QP.n)])
    
    def pdas_get_mu(self, x, active_plus, active_minus, inactive):
        # Compute the next iteration of mu based on next x for the Primal-Dual Active Set method
        active = active_plus + active_minus

        left = np.identity(len(active))
        right = self.QP.b[active] - self.QP.A[active,:][:,inactive].dot(x[inactive]) - self.QP.A[active,:][:,active_plus].dot(x[active_plus]) - self.QP.A[active,:][:,active_minus].dot(x[active_minus]) 
        mu_a= la.solve(left, right)
        return np.array([0 if j in inactive else mu_a[active.index(j)] for j in range(self.QP.n)])
    
    def test_convergence(self, iterates):
        # Calculate the residuals of the KKT condition for every iterates
        # Here, the KKT conditions work as residual to the optimal solution, because we want to find the root of the respective function
        return [np.around(np.sum(self.QP.KKT_condition(x, mu)), 2) for [x,mu] in iterates]