import numpy as np

class QuadraticProgram:
    def __init__(self, A, b, u, l = np.array(None)):
        self.A = A
        self.b = b
        self.n = b.shape[0]
        if l.any() == None:
            self.l = np.array([-np.inf] * self.n) # Upper bound constrained
        else:
            self.l = l
        self.u = u

    def KKT_condition (self, x, mu):
        lagrange = self.A.dot(x) - self.b + mu
        complementarity = mu - np.maximum(0, mu + (x - self.u)) - np.minimum(0, mu + (x - self.l))
        return np.concatenate((lagrange, complementarity))

    def KKT_jacobian(self, x, mu):
        active = [i for i in range(0, self.n) if (mu[i] + x[i] - self.u[i] > 0) or (mu[i] + x[i] - self.l[i] < 0)]
        inactive = [i for i in range(0, self.n) if i not in active]
        id_active = np.zeros((self.n, self.n), int)
        for i in active:
            id_active[i,i] = -1
        id_inactive = np.zeros((self.n, self.n), int)
        for i in inactive:
            id_inactive[i,i] = 1
        id = np.concatenate((id_active, id_inactive), axis=1)
        m = np.concatenate((self.A, np.identity(self.n)), axis=1)
        return np.concatenate((m, id), axis=0)