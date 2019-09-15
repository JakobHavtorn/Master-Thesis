import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import IPython


def fitness_shaping2(returns):
    """Computes the fitness shaped returns.

    Performs the fitness rank transformation used for CMA-ES.
    Reference: Natural Evolution Strategies [2014]
    
    Args:
        returns (np.array): Returns of evaluated perturbed models.
    
    Returns:
        np.array: Shaped returns
    """
    assert type(returns) == np.ndarray
    n = len(returns)
    sorted_indices = np.argsort(-returns)
    u = np.zeros(n)
    for k in range(n):
        u[sorted_indices[k]] = np.max([0, np.log(n / 2 + 1) - np.log(k + 1)])
    return u / np.sum(u) - 1 / n

def fitness_shaping(returns):
    assert type(returns) == np.ndarray
    N = len(returns)
    ranks = np.argsort(-returns)
    u = np.zeros(N)
    for k in range(N):
        u[ranks[k]] = np.max([0, np.log(N / 2 + 1) - np.log(k + 1)])
    return u / np.sum(u) - 1 / N


if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 12})
    N = 10
    mean = 1.5
    sd = 2
    fitnesses = mean + sd * np.random.randn(N)
    utilities = fitness_shaping(fitnesses)
    ranks = (-fitnesses).argsort()
    fitnesses = fitnesses[ranks]
    utilities = utilities[ranks]

    f, ax = plt.subplots()
    ax.bar(range(1, N+1), fitnesses)
    ax.set_xlabel('rank' + r'$(f(x_i))}$')
    ax.set_ylabel(r'$f(x_i)$')
    f.savefig('../graphics/fitness-transform/fitnesses.pdf', bbox_inches='tight')

    f, ax = plt.subplots()
    ax.bar(range(1, N+1), utilities)
    ax.set_xlabel('rank' + r'$(f(x_i))}$')
    ax.set_ylabel(r'$u_i$')
    f.savefig('../graphics/fitness-transform/transform.pdf', bbox_inches='tight')





