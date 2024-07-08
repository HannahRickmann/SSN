import numpy as np

class QuadraticProgram:
    def __init__(self, A, b, u, l = np.array(None)):
        """
        Initialize the quadratic program with matrix A, vector b, upper bound u,
        and optionally lower bound l (default is None).
        """
        self.A = A
        self.b = b
        self.n = b.shape[0]
        if l.any() == None:
            self.l = np.array([-np.inf] * self.n) # If lower bound not provided, set it to negative infinity
        else:
            self.l = l
        self.u = u

    def KKT_condition (self, x, mu):
        """
        Compute the Karush-Kuhn-Tucker (KKT) conditions for the quadratic program,
        using the nonlinear complementarity function min(x,y)
        """
        # Calculate the gradient of the Lagrange function
        lagrange = self.A.dot(x) - self.b + mu 

        # Calculate the complementarity condition
        complementarity = mu - np.maximum(0, mu + (x - self.u)) - np.minimum(0, mu + (x - self.l))
        return np.concatenate((lagrange, complementarity))

    def KKT_jacobian(self, x, mu):
        """
        Compute the Jacobian matrix of the KKT conditions.

        """
        # Get the index partitions based on (x, mu)
        active_plus, active_minus, inactive = self.get_active_indices(x, mu)

        # Initialize identity matrices for active and inactive sets
        id_active = np.zeros((self.n, self.n), int)
        for i in active_plus:
            id_active[i,i] = -1
        for i in active_minus:
            id_active[i,i] = -1
        id_inactive = np.zeros((self.n, self.n), int)
        for i in inactive:
            id_inactive[i,i] = 1

        # Combine active and inactive identity matrices
        id = np.concatenate((id_active, id_inactive), axis=1)

        # Create the Jacobian matrix
        m = np.concatenate((self.A, np.identity(self.n)), axis=1)
        return np.concatenate((m, id), axis=0)
    
    def get_active_indices(self, x, mu):
        """
        Determine the active indices based on the current solution x and Lagrange multiplier mu.
        """
        active_plus = [i for i in range(0, self.n) if (mu[i] + x[i] - self.u[i] > 0)]
        active_minus = [i for i in range(0, self.n) if (mu[i] + x[i] - self.l[i] < 0)]
        inactive = [i for i in range(0, self.n) if i not in active_plus and i not in active_minus]
        return active_plus, active_minus, inactive
    
    def print(self):
        print('A = ', self.A)
        print('b = ', self.b)
        print('u = ', self.u)
        print('l = ', self.l)