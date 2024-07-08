import numpy as np
import math
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import skfem as fem
from skfem.helpers import dot, grad  # helpers make forms look nice

import skfem.visuals.matplotlib

from experiments.experiment import Experiment
from qpsolver.quadratic_program import QuadraticProgram as QP
from qpsolver.solver import Solver

# Define the bilinear form for the weak form of the PDE
@fem.BilinearForm
def a(u, v, _):
    return dot(grad(u), grad(v))

# Define the linear form for the right-hand side of the PDE
@fem.LinearForm
def L(v, w):
    f = -15
    return f * v

def build_obstacle_box_2d(n, height):
    """
    Build obstacle as a box in the middle of a square mesh.

    Returns: 
    Array representing the obstacle heights in the mesh for each node.
    """
    l = []
    for i in range(1,n+1):
      if i/(n+1) < 0.25 or i/(n+1) > 0.75:
        l.extend([-np.inf]*n)
        continue
      for j in range(1, n+1):
        if j/(n+1) < 0.25 or j/(n+1) > 0.75:
          l.append(-np.inf)
        else:
          l.append(height)
    return np.array(l)

def is_m_matrix(matrix):
    """
    Check if a matrix is an M-matrix (diagonal positive, non diagonale non positive, inverse non negative).
    """
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        print('No square matrix.')
        return False
    n = matrix.shape[0]
    
    # Check if the diagonal elements are positive
    if not all(matrix[i][i] > 0 for i in range(n)):
        print('Diagonal not positive.')
        return False
    
    # Check if all non-diagonal elements are non-positive
    if not all(matrix[i][j] <= 0 for i in range(n) for j in range(n) if i != j):
        print('Nondiagonal entries not non positive.')
        return False
    
    # Check if the inverse has only positive entries
    try:
        inverse = np.linalg.inv(matrix)
        if not all(entry >= 0 for entry in inverse.flatten()):
            print('Inverse not positive.')
            return False
    except np.linalg.LinAlgError:
        # Matrix is singular, inverse doesn't exist
        print('Matrix singular and has no inverse.')
        return False
    
    return True

def rho_smaller_one(matrix):
    """
    Check if rho, defined in Theorem 4.11, is smaller than one.
    """
    rho = [0] * 5
    n = matrix.shape[0]

    # Iterate through all possible active/inactive set partitions
    for i in range(n + 1):
        for combination in itertools.combinations(range(n), i):
            I = list(combination)
            A = [x for x in range(n) if x not in I]
            rho_temp = [0] * 5

            # Evaluate rho_1 to rho_5
            if len(A) == 0:
                matrix_I_inv = np.linalg.inv(matrix[I,:][:,I])
                rho_temp[0] = np.linalg.norm(matrix_I_inv, ord=1)
            elif len(I) == 0:
                rho_temp[2] = np.linalg.norm(matrix[A,:][:,A] - np.identity(len(A)), ord=1)
            else:
                matrix_I_inv = np.linalg.inv(matrix[I,:][:,I])
                rho_temp[0] = np.linalg.norm(matrix_I_inv, ord=1)
                matrix_temp = np.dot(matrix_I_inv, matrix[I,:][:,A])
                rho_temp[1] = np.linalg.norm(matrix_temp, ord=1)
                rho_temp[2] = np.linalg.norm(matrix[A,:][:,A] - np.identity(len(A)), ord=1)
                rho_temp[3] = np.linalg.norm(np.dot(matrix[A,:][:,I], matrix_I_inv), ord=1)
                rho_temp[4] = np.linalg.norm(np.dot(matrix[A,:][:,I], matrix_temp), ord=1)

            # Update rho and check condition
            rho = np.maximum(rho, rho_temp)
            if 2*max(rho[0], rho[1], rho[2] + rho[4], rho[3]) >= 1:
                print(f'Index partition: I = {I}, A = {A}')
                print(f'Corresponding values of rho: {rho_temp}')
                print(f'Resulting values of rho: {rho}')
                return False
    return True

def conditions_p_matrix(matrix):
    """
    Check conditions for P-matrix in Theorem 4.6.
    """
    rho = 0
    n = matrix.shape[0]

    # Iterate through all possible active/inactive set partitions
    for i in range(n + 1):
        for combination in itertools.combinations(range(n), i):
            I = list(combination)
            A = [x for x in range(n) if x not in I]
            if len(A) != 0 and len(I) != 0:
                matrix_I_inv = np.linalg.inv(matrix[I,:][:,I])
                matrix_temp = np.dot(matrix_I_inv, matrix[I,:][:,A])
                rho_temp = np.linalg.norm(np.maximum(matrix_temp, 0), ord=1)
                rho = np.maximum(rho, rho_temp)
                if rho >= 1:
                    print(f'Index partition: I = {I}, A = {A}')
                    print(f'Corresponding values of rho: {rho_temp}')
                    print(f'Resulting values of rho: {rho}')
                    return False
    return True

def check_if_small_perturbation(matrix):
    """
    Check if the matrix is a small perturbation of an M-matrix.
    """
    rho = 0
    A_original = np.matrix([[ 4, -1,  0, -1,  0,  0,  0,  0,  0],
                            [-1,  4, -1,  0, -1,  0,  0,  0,  0],
                            [ 0, -1,  4,  0,  0, -1,  0,  0,  0],
                            [-1,  0,  0,  4, -1,  0, -1,  0,  0],
                            [ 0, -1,  0, -1,  4, -1,  0, -1,  0],
                            [ 0,  0, -1,  0, -1,  4,  0,  0, -1],
                            [ 0,  0,  0, -1,  0,  0,  4, -1,  0],
                            [ 0,  0,  0,  0, -1,  0, -1,  4, -1],
                            [ 0,  0,  0,  0,  0, -1,  0, -1,  4]])
    K = A_original - matrix
    n = matrix.shape[0]

    # Iterate through all possible active/inactive set partitions
    for i in range(n + 1):
        for combination in itertools.combinations(range(n), i):
            I = list(combination)
            A = [x for x in range(n) if x not in I]
            if len(I) != 0:
                matrix_I_inv = np.linalg.inv(A_original[I,:][:,I])
                rho_temp = np.linalg.norm(np.dot(matrix_I_inv, K[I,:][:,I]), ord=1)
                rho = np.maximum(rho, rho_temp)
                if rho >= 0.5:
                    print(f'Index partition: I = {I}, A = {A}')
                    print(f'Corresponding values of rho: {rho_temp}')
                    print(f'Resulting values of rho: {rho}')
                    return False
    return True

def create_tensor_mesh(x, y):
    """
    Create a 2D triangular mesh using a tensor product of linear spaces.

    Parameters:
    x/y: Integer, number of points in the x/y-direction
    """
    x_range = np.linspace(0, 1, num=x)  # Equidistant point positions in x-direction
    y_range = np.linspace(0, 1, num=y)  # Equidistant point positions in x-direction

    return fem.MeshTri().init_tensor(x_range, y_range)

def perturb_full_mesh(mesh):
    """
    Perturb the complete interior nodes of a mesh by adding a random perturbation.
    """
    nodes = mesh.p.T
    h = nodes[1][1] - nodes[0][1]
    for node in list(mesh.interior_nodes()):
        nodes[node] += np.random.uniform(-h/4, h/4)
    triangles = mesh.t
    return fem.MeshTri(nodes.T, triangles)

def perturb_full_mesh_2(mesh):
    """
    Perturb all nodes, such that grid is shifted and triangles with angles over 90° exist.
    """
    nodes = mesh.p.T
    resolution = math.sqrt(nodes.shape[0])
    h = nodes[1][1] - nodes[0][1]
    for node in range(0, nodes.shape[0]):
        nodes[node] += h*(node//resolution)
    triangles = mesh.t
    return fem.MeshTri(nodes.T, triangles)

def perturb_one_node_mesh(mesh):
    """
    Perturb the most upper right node such that we enforce two triangles with angles >180°
    """
    nodes = mesh.p.T
    h = nodes[1][1] - nodes[0][1]
    node = mesh.interior_nodes()[-1]
    nodes[node] += h*2/3
    triangles = mesh.t
    return fem.MeshTri(nodes.T, triangles)

def create_line_mesh(x):
    """
    Create a 1D line mesh.
    """
    return fem.MeshLine(np.linspace(0, 1, x))

def create_tri_basis(mesh):
    """
    Create a basis for a triangular mesh using piecewise linear elements.
    """
    return fem.Basis(mesh, fem.ElementTriP1())

def create_line_basis(mesh):
    """
    Create a basis for a line mesh using piecewise linear elements.
    """
    return fem.Basis(mesh, fem.ElementLineP1()) # piecewise linear

def get_condensed_stiffness_matrix(mesh, basis, a, L):
    """
    Get the discretized stiffness matrix and load vector.
    """
    A = a.assemble(basis)
    b = L.assemble(basis)
    A, b = fem.enforce(A, b, D=mesh.boundary_nodes())  # Enforce boundary condition.
    # A, b, c, d = fem.condense(A, b, D=mesh.boundary_nodes())  # Optional condense
    return A.toarray(), b

def discretize_2d_problem(resolution, perturb = False, show_mesh = True):
    """
    Discretize a 2D problem.

    Parameters:
    resolution (int): Number of elements along each direction.
    perturb (bool): Whether to perturb the mesh.
    show_mesh (bool): Whether to visualize the mesh.
    """
    # Create a 2D mesh
    mesh = create_tensor_mesh(resolution, resolution)
    if show_mesh:
        plt.subplots(figsize=(5,5))
        skfem.visuals.matplotlib.draw(mesh, ax=plt.gca())  # gca: "get current axis"
        # plt.savefig(f'gird_{resolution}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Perturb the mesh if required
    if perturb:
        mesh = perturb_full_mesh_2(mesh)
        if show_mesh:
            plt.subplots(figsize=(5,5))
            skfem.visuals.matplotlib.draw(mesh, ax=plt.gca())  # gca: "get current axis"
            # plt.savefig(f'gird_perturbed_{resolution}.png', dpi=300, bbox_inches='tight')
            plt.show()

    basis = create_tri_basis(mesh)
    A, b = get_condensed_stiffness_matrix(mesh, basis, a, L)
    return A, b, mesh


def discretize_1d_problem(resolution):
    """
    Discretize a 1D problem.

    Parameters:
    resolution (int): Number of nodes along the line.
    """
    mesh = create_line_mesh(resolution)
    basis = create_line_basis(mesh)
    A, b = get_condensed_stiffness_matrix(mesh, basis, a, L)
    return A, b, mesh


def solve(A, b, u, l = None, max_iter = 10):
    """
    Solve a constrained QP defined by A, b, u, and l.
    Create and experiment object of class experiment and solve the problem with the PDASA
        
    Returns:
        Experiment: Experiment object with solving result, such as iterates, solution, and residuals.
    """
    experiment = Experiment(1)
    experiment.name = f'obstacle_problem_{A.shape}'

    if l is None:  # If no lower bound is provided
        experiment.QP = QP(A=A, b=b, u=u)
    else:
        experiment.QP = QP(A=A, b=b, u=u, l=l)

    experiment.solver = Solver(experiment.QP)
    experiment.x_0 = np.zeros(b.shape[0]*2)  # Initial guess
    experiment.run(max_iter, save = False, method='newton')

    #experiment.print_solution()  
    #experiment.print_residuals()  # If last element is zero, the solver found the solution and converged.
    experiment.save_experiment(only_no_convergence=False) 
    return experiment

def plot_solution_1d_box(x, obstacle, obstacle_u, points):
    """"
    Plot a 1D solution with two obstacles (upper and lower constraint).
    
    Parameters:
    x: Numerical solution.
    obstacle: Lower obstacle function.
    obstacle_u: Upper obstacle function.
    points: Domain points.
    """
    plt.plot(points, x, label='Numerical Solution')
    plt.plot(points, obstacle, label='Obstacle Function', linestyle='dashed')
    plt.plot(points, obstacle_u, label='Obstacle Function Upper', linestyle='dashed')
    plt.ylim(-2, 2)
    plt.legend()
    plt.xlabel('domain', fontsize=12)
    plt.ylabel('vertical position of membrane', fontsize=12)
    plt.savefig('1d_obstacle_box.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_solution_1d(x, obstacle, points):
    """
    Plot a 1D solution with one obstacle.
    
    Parameters:
    x: Numerical solution.
    obstacle: Obstacle function.
    points: Domain points.
    """
    # plt.figure(figsize=(13,5))
    plt.plot(points, x, label='Numerical Solution')
    plt.plot(points, obstacle, label='Obstacle Function', linestyle='dashed')
    plt.plot([0,1], [0,0], 'o', color='#1f77b4')
    plt.ylim(-1, 1)
    plt.legend(fontsize=10)
    plt.xlabel('domain', fontsize=12)
    plt.ylabel('vertical position of membrane', fontsize=12)
    plt.savefig('1d_obstacle.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_solution_2d(mesh, x):
    """
    Plot a 2D solution on the original mesh.
    
    Parameters:
    mesh: The 2D mesh.
    x: Solution values.
    """
    plt.subplots(figsize=(6,5))
    skfem.visuals.matplotlib.plot(mesh, x, cmap='plasma', colorbar=True, ax=plt.gca())
    plt.xlabel('domain') 
    plt.ylabel('domain')
    plt.savefig(f'2d_obstacle.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_solution_3d(mesh, x, obstacle_l, obstacle_u):
    """
    Plot a 2D solution in 3D with upper and lower obstacles.
    
    Args:
    mesh: 2D mesh.
    x: Solution values.
    obstacle_l: Lower obstacle function values.
    obstacle_u: Upper obstacle function values.
    """
    X = mesh.p[0, :]
    Y = mesh.p[1, :]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, obstacle_u, cmap='summer', alpha=0.3)  # Plot upper obstacle
    ax.plot_trisurf(X, Y, obstacle_l, cmap='summer', alpha=0.5)  # Plot lower obstacle
    ax.plot_trisurf(X, Y, x, cmap='plasma')  # Plot solution
    ax.set_zlim(-1, 1)
    # ax.view_init(elev=15)
    ax.view_init(elev=12, azim=20)
    ax.set_xlabel('domain')
    ax.set_ylabel('domain')
    ax.set_zlabel('vertical position of membrane')
    plt.savefig(f'3d_obstacle.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_residuals(residuals):
    """
    Plot residuals over iterations.
    """
    print(f'residuals: {residuals}')
    plt.plot(range(0, len(residuals)), [0]*(len(residuals)), linestyle='--', color='lightgrey')
    plt.plot(range(0, len(residuals)), residuals, linestyle=':', marker='o')
    #plt.ylim(-1, 15)
    plt.xlabel('iteration', fontsize=12)
    plt.ylabel('residual', fontsize=12)
    plt.savefig('residuals.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_upper_lower(grid_size, obstacle_height, dimension = 1, distance = 1):
    """
    Discretizes and solves a 1D or 2D obstacle problem with upper and lower bound, and plots the solution.

    Parameters:
    grid_size (int): Number of elements along each direction for mesh generation.
    obstacle_height (float): Height of the obstacle function.
    dimension (int): Dimension of the problem (1 for 1D, 2 for 2D). Default is 1.
    distance (float): Distance parameter used in obstacle function for 2D problems. Default is 1.
    """
    print('Generating mesh and basis to discretize problem...')
    # Depending on the dimension specified, discretize either a 1D or 2D problem.
    if dimension == 1:
        A, b, mesh = discretize_1d_problem(grid_size)
    else:
        A, b, mesh = discretize_2d_problem(grid_size, perturb=False)

    # Commented lines to check various properties of the stiffness matrix A.

    #print('Check requirements of global convergence results...')
    #print(f'rho is smaller than one: {rho_smaller_one(A)}')
    #print(A)
    
    #print(f'A is an M-matrix: {is_m_matrix(A)}')
    #print(f'A is a suitable P-matrix: {conditions_p_matrix(A)}')
    #print(f'A is a small perturbation of M-matrix: {check_if_small_perturbation(A)}')

    # Define upper and lower obstacles based on the problem dimension.
    n = b.shape[0]
    if dimension == 1:
        points = np.linspace(0, 1, n)
        g = lambda x: 2*((4*x - 2)**2 - (4*x-2)**4) + obstacle_height
        h = lambda x: -2*((4*x - 1.5)**2 - (4*x-1.5)**4) + obstacle_height + distance
        l = g(points) # 1d obstacle
        u = h(points)
        #u = np.array([1] * n)
    else:
        x_points = np.linspace(0, 1, grid_size)
        y_points = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x_points, y_points)
        # u = np.array([np.inf] * n)
        # l = build_obstacle_box_2d(grid_size, obstacle_height)  # box obstacle
        # g = lambda x, y: -0.2*(x + 0.5) - 0.2*(y + 0.5)  # linear obstacle
        g = lambda x, y: 2*((4*x - 2)**2 - (4*x-2)**4 + (4*y - 2)**2 - (4*y-2)**4) + obstacle_height  # quartic obstacle
        l = g(X, Y).flatten()
        l = np.maximum(l, -1)  # Ensure lower obstacle is not below -1.

        h = lambda x, y: -2*((4*x - 1.5)**2 - (4*x-1.5)**4 + (4*y - 1.5)**2 - (4*y-1.5)**4) + obstacle_height + distance
        u = h(X,Y).flatten()
        u = np.minimum(u, 1)  # Ensure upper obstacle is not above 1.

    print('Solve discretized problem...')
    experiment = solve(A, b, u, l=l, max_iter=40)
    # experiment.print_iterates()
    experiment.print_residuals()

    print('Plot solution...')
    x = experiment.iterates[-1][0]  # Final iterate of the solution.

    if dimension == 1:
        plot_solution_1d_box(x, l, u, points)
        plot_residuals(experiment.residuals)
    else:
        plot_solution_2d(mesh, x)
        plot_solution_3d(mesh, x, l, u)

    if experiment.solver.converged:
        return True
    return False

def run_one_constraint(grid_size, obstacle_height):
    """
    Discretizes and solves a 1D obstacle problem with one constraint and plots the solution.

    Parameters:
    grid_size (int): Number of nodes along the line for mesh generation.
    obstacle_height (float): Height of the obstacle function.
    """
    print('Generating mesh and basis to discretize problem...')
    A, b, mesh = discretize_1d_problem(grid_size)

    # Commented lines to check if A is an M-matrix.
    #print(f'A is an M-matrix: {is_m_matrix(A)}')
    #print(b)
    
    # Set up upper obstacle function for 1D case.
    n = b.shape[0]
    u = np.array([np.inf] * n)

    # Define lower obstacle function for 1D case based on obstacle_height.
    points = np.linspace(0, 1, n)
    g = lambda x: 2*((4*x - 2)**2 - (4*x-2)**4) + obstacle_height
    l = g(points)

    print('Solve discretized problem...')
    experiment = solve(A, b, u, l, max_iter=6)

    print('Plot solution...')
    # print(f'first iterate: {experiment.iterates[0]}')
    # for i in range(len(experiment.iterates)):
    #    x = experiment.iterates[i][0]
    #    plot_solution_1d(x, l, points)
    x = experiment.iterates[-1][0]  # Final iterate of the solution.
    plot_solution_1d(x, l, points)
    print(experiment.residuals)
    # plot_residuals(experiment.residuals)
    return x,l, points


if __name__ == '__main__':
    obstacle_height = -0.5
    grid_size = 50
    run_one_constraint(grid_size, obstacle_height)
    # run_upper_lower(grid_size, obstacle_height, dimension=2, distance=1.5)
        
