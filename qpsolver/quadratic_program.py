import numpy as np

class QuadraticProgram:
    def __init__(self, A, b, u, l = np.array(None)):
        # Initialize the quadratic program with matrices A, vector b, upper bound u,
        # and optionally lower bound l (default is None).
        self.A = A
        self.b = b
        self.n = b.shape[0]
        if l.any() == None:
            self.l = np.array([-np.inf] * self.n) # If lower bound not provided, set it to negative infinity
        else:
            self.l = l
        self.u = u

    def KKT_condition (self, x, mu):
        # Compute the Karush-Kuhn-Tucker conditions
        # this is an equivalent formulation using a nonlinear complementarity function
        lagrange = self.A.dot(x) - self.b + mu
        complementarity = mu - np.maximum(0, mu + (x - self.u)) - np.minimum(0, mu + (x - self.l))
        return np.concatenate((lagrange, complementarity))

    def KKT_jacobian(self, x, mu):
        # Compute the Jacobian matrix of the Karush-Kuhn-Tucker conditions
        active = self.get_active_indices(x, mu)
        inactive = self.get_inactive_indices(active)
        id_active = np.zeros((self.n, self.n), int)
        for i in active:
            id_active[i,i] = -1
        id_inactive = np.zeros((self.n, self.n), int)
        for i in inactive:
            id_inactive[i,i] = 1
        id = np.concatenate((id_active, id_inactive), axis=1)
        m = np.concatenate((self.A, np.identity(self.n)), axis=1)
        return np.concatenate((m, id), axis=0)
    
    def get_active_indices(self, x, mu):
        # Determine active indices based on the current solution x and Lagrange multiplier mu
        return [i for i in range(0, self.n) if (mu[i] + x[i] - self.u[i] > 0) or (mu[i] + x[i] - self.l[i] < 0)]
    
    def get_inactive_indices(self, active):
        # Determine inactive indices based on the active indices
        # inactive = {1, ..., n} \ active
        return [i for i in range(0, self.n) if i not in active]
    
    def print(self):
        print('A = ', self.A)
        print('b = ', self.b)
        print('u = ', self.u)
        print('l = ', self.l)