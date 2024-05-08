import numpy as np
import matplotlib.pyplot as plt

import skfem as fem
from skfem.helpers import dot, grad  # helpers make forms look nice

import skfem.visuals.matplotlib

from experiments.experiment import Experiment
from qpsolver.quadratic_program import QuadraticProgram as QP
from qpsolver.solver import Solver

@fem.BilinearForm
def a(u, v, _):
    return dot(grad(u), grad(v))

@fem.LinearForm
def L2d(v, w):
    x, y = w.x  # global coordinates
    f = np.sin(np.pi * x) * np.sin(np.pi * y)
    return f * v

@fem.LinearForm
def L1d(v, w):
    f = 10
    return f * v

def create_tensor_mesh(x, y):
    x_range = np.linspace(0, 1, num=x)  # Example: x points in the x-direction
    y_range = np.linspace(0, 1, num=y)   # Example: y points in the y-direction

    # Create the triangular mesh using init_tensor() with these ranges
    return fem.MeshTri().init_tensor(x_range, y_range)

def create_line_mesh(x):
    return fem.MeshLine(np.linspace(0, 1, x))

def create_tri_basis(mesh):
    return fem.Basis(mesh, fem.ElementTriP1()) # piecewise linear

def create_line_basis(mesh):
    return fem.Basis(mesh, fem.ElementLineP1()) # piecewise linear

def get_condensed_stiffness_matrix(mesh, basis, a, L):
    A = a.assemble(basis)
    b = L.assemble(basis)
    A, b = fem.enforce(A, b, D=mesh.boundary_nodes())
    return A.toarray(), b

def perturbe_full_mesh(mesh):
    nodes = mesh.p.T
    h = nodes[1][1] - nodes[0][1]
    for node in list(mesh.interior_nodes()):
        nodes[node] += np.random.uniform(-h/4, h/4)
    triangles = mesh.t
    return fem.MeshTri(nodes.T, triangles)

def perturbe_one_node_mesh(mesh):
    nodes = mesh.p.T
    h = nodes[1][1] - nodes[0][1]
    node = mesh.interior_nodes()[-1]
    nodes[node] += h*2/3
    triangles = mesh.t
    return fem.MeshTri(nodes.T, triangles)

def discretize_2d_problem():
    mesh = create_tensor_mesh(8, 8)
    plt.subplots(figsize=(5,5))
    skfem.visuals.matplotlib.draw(mesh, ax=plt.gca())  # gca: "get current axis"
    plt.show()
    
    mesh = perturbe_one_node_mesh(mesh)
    plt.subplots(figsize=(5,5))
    skfem.visuals.matplotlib.draw(mesh, ax=plt.gca())  # gca: "get current axis"
    plt.show()

    basis = create_tri_basis(mesh)
    A, b = get_condensed_stiffness_matrix(mesh, basis, a, L2d)
    return A, b

def discretize_1d_problem(resolution):
    mesh = create_line_mesh(resolution)

    basis = create_line_basis(mesh)
    A, b = get_condensed_stiffness_matrix(mesh, basis, a, L1d)
    return A,b

def solve(A, b, u, max_iter, analyse=False):
    experiment = Experiment(1)
    experiment.QP = QP(A=A, b=b, u=u)
    experiment.solver = Solver(experiment.QP)
    experiment.x_0 = np.zeros(b.shape[0]*2)
    experiment.run(max_iter, save = False) #, method="newton")
    experiment.print_QP()
    experiment.print_solution()
    experiment.print_residuals()
    if analyse:
        experiment.analyse_possible_active_sets()
        experiment.analyse_active_set_cycle()
    return experiment

def plot_solution(x, u):
    plt.plot(points, -x, label='Numerical Solution')
    plt.plot(points, -u, label='Obstacle Function', linestyle='dashed')
    plt.ylim(-1, 1)
    plt.xlabel('domain')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    A, b = discretize_1d_problem(50)
    n = b.shape[0]
    points = np.linspace(0, 1, n)
    c = 0.6 # shift of obstacle, the more positive the higher is the obstacle
    g = lambda x: (5*x-2.5)**4 - (5*x - 2.5)**2 + c
    u = g(points)

    experiment = solve(A, b, u, 36)

    x = experiment.iterates[-1][0]
    plot_solution(x, u)
    


