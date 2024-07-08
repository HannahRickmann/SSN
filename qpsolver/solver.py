import numpy as np
import numpy.linalg as la
from qpsolver.quadratic_program import QuadraticProgram

class Solver:
    def __init__(self, QP: QuadraticProgram):
        """
        Initialize the Solver with a QuadraticProgram instance.
        """
        self.QP = QP
        self.converged = False

    def solve(self, x0, max_iterations, method: str):
        """
        Solve the optimization problem using the PDASA or Newton method.

        Parameters:
        x0 (numpy.ndarray): Initial guess for the solution.
        max_iterations (int): Maximum number of iterations.
        method (str): The method to use for solving ('newton' or 'pdas').
        """
        if method == 'newton':
            return self.solve_newton(x0, max_iterations)
        elif method == 'pdas':
            return self.solve_pdas(x0, max_iterations)
        else:
            print('No valid method. Choose "newton" or "pdsa".')

    def solve_newton(self, x_0, max_iterations):
        """
        Solve the optimization problem using Newton's method.
        """
        def F(x):  # Function for Newton method
            x, mu = np.split(x,2)
            return self.QP.KKT_condition(x, mu)

        def G(x):  # Generalized gradient of respective function
            x, mu = np.split(x,2)
            return self.QP.KKT_jacobian(x, mu)
        
        iterates = self.newton(F,G, x_0, max_iterations)
        return [np.split(x, 2) for x in iterates]
    
    def newton(self, F, G, x0, max_iterations):
        """
        Perform Newton's method for solving nonlinear systems of equations.
        """
        iterates = [x0]
        xn = x0
        lr = 1  # Learning rate: Set for example to 0.01 or 1
        for _ in range(0, max_iterations):
            delta = la.solve(G(xn), -F(xn))
            xn = xn + lr*delta
            lr += 0.01  # Increment learning rate but cap at 1
            if lr > 1:
                lr = 1
            iterates.append(xn)
        return iterates
    
    def solve_pdas(self, x0, max_iterations):
        """
        Solve the optimization problem using the Primal-Dual Active Set method.
        """
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
        """
        Perform a single iteration of the Primal-Dual Active Set method.
        """
        x, mu = xn
        active_plus, active_minus, inactive = self.QP.get_active_indices(x, mu)
        
        x = self.pdas_get_x(active_plus, active_minus, inactive)
        mu = self.pdas_get_mu(x, active_plus, active_minus, inactive)
        
        return [x, mu]
    
    def pdas_get_x(self, active_plus, active_minus, inactive):
        """
        Compute the next iteration of x for the Primal-Dual Active Set method.

        Parameters:
        active_plus (list): Indices where the upper bound is active.
        active_minus (list): Indices where the lower bound is active.
        inactive (list): Indices where neither bound is active.
        """
        active = active_plus + active_minus

        left = self.QP.A[inactive,:][:,inactive]
        right = self.QP.b[inactive] - self.QP.A[inactive,:][:, active_plus].dot(self.QP.u[active_plus]) - self.QP.A[inactive,:][:, active_minus].dot(self.QP.l[active_minus])
        x_i = la.solve(left, right)  # Solve Ax = b for inactive indices
        x_a = list(self.QP.u[active_plus]) + list(self.QP.l[active_minus]) # Set x to bounds for active indices
        return np.array([x_i[inactive.index(j)] if j in inactive else x_a[active.index(j)] for j in range(self.QP.n)])
    
    def pdas_get_mu(self, x, active_plus, active_minus, inactive):
        """
        Compute the next iteration of mu based on next x for the Primal-Dual Active Set method.

        Parameters:
        x (numpy.ndarray): The updated solution vector x.
        active_plus (list): Indices where the upper bound is active.
        active_minus (list): Indices where the lower bound is active.
        inactive (list): Indices where neither bound is active.
        """
        active = active_plus + active_minus

        left = np.identity(len(active))
        right = self.QP.b[active] - self.QP.A[active,:][:,inactive].dot(x[inactive]) - self.QP.A[active,:][:,active_plus].dot(x[active_plus]) - self.QP.A[active,:][:,active_minus].dot(x[active_minus]) 
        mu_a= la.solve(left, right)

        return np.array([0 if j in inactive else mu_a[active.index(j)] for j in range(self.QP.n)])
    
    def test_convergence(self, iterates):
        """
        Calculate the residuals of the KKT condition for every iterate.
        Convergence is achieved when residuals decrease and reach zero.
        """
        return [np.around(np.linalg.norm(self.QP.KKT_condition(x, mu), ord=1), 2) for [x,mu] in iterates]