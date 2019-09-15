
# Python version of: https://gist.github.com/davidbarber/16708b9135f13c9599f754f4010a0284
# as per blog post: https://davidbarber.github.io/blog/2017/04/03/variational-optimisation/
# also see https://www.reddit.com/r/MachineLearning/comments/63dhfc/r_evolutionary_optimization_as_a_variational/

from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import os, sys
import IPython


# Objective functions
def himmelblau(x):
    """Himmelblau function, quartic function with four global minima and one local maximum.
    """
    f = (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2
    g = np.array([x[0] * (4 * x[0]**2 + 4 * x[1] - 42) + 2 * x[1]**2 - 14,
                  2 * (x[0] + x[1])**2 + 2 * (2 * x[1]**3 - x[1]**2 - 13 * x[1] - 11)])
    return (f, g)


def easom(x):
    """Easom function, very flat with single narrow minimum.
    f(pi,pi) = -1
    - 100 < x < 100
    """
    f = - np.cos(x[0]) * np.cos(x[1]) * np.exp(- (x[0] - np.pi)**2 - (x[1] - np.pi)**2 )
    g = np.array([np.cos(x[1]) * (np.sin(x[1]) + 2 * (x[0] - np.pi) * np.cos(x[0])) * np.exp(- (x[0] - np.pi)**2 - (x[1] - np.pi)**2 ),
                  np.cos(x[0]) * (np.sin(x[0]) + 2 * (x[1] - np.pi) * np.cos(x[1])) * np.exp(- (x[0] - np.pi)**2 - (x[1] - np.pi)**2 )])
    return (f, g)


def mccormick(x):
    """McCormick function, 
    # f(x,y)=\sin \left(x+y\right)+\left(x-y\right)^{2}-1.5x+2.5y+1
    """
    f = np.sin(x[0] + x[1]) + (x[0] - x[1])**2 - 1.5 * x[0] + 2.5 * x[1] + 1
    g = np.array([-1.5 + 2 * x[0] - 2 * x[1] + np.cos(x[0] + x[1]),
                   2.5 - 2 * x[0] + 2 * x[1] + np.cos(x[0] + x[1])])
    return (f, g)


def fitness_shaping(returns):
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

if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 12})
    functions = [himmelblau]# [himmelblau, easom, mccormick]
    x_inits = [np.array([0, 0]), np.array([1, 3]), np.array([-2, 2])]
    for function, x_init in zip(functions, x_inits):
        # Create the dataset:
        D = 2   # Dimension of the data
        Nloops = 60  # number of iterations

        # Plot the function surface:
        N = 200
        x_low = -5
        x_high = 5
        x1 = np.linspace(x_low,x_high, N)
        x2 = x1
        X1, X2 = np.meshgrid(x1, x2)
        fsurf = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                fsurf[j,i] = function(np.array([x1[i], x2[j]]))[0]
        
        def plot_convergence(x_hist):
            fig, ax = plt.subplots()
            ax.contourf(X1, X2, fsurf, 30, cmap=cm.coolwarm)
            ax.scatter(x_hist[0, :], x_hist[1, :], s=2, c='r')
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')

        ################################################################################

        # standard gradient descent:
        eta =  0.01  # learning rate
        x = x_init
        x_hist = []
        for i in range(Nloops):
            x_hist.append(x)
            #plot3(W(2),W(1),E(W,x,y)+0.1,'y.','markersize',20);
            x = x - eta * function(x)[1]
        x_hist = np.array(x_hist).T

        plot_convergence(x_hist)
        plt.savefig('../graphics/var-opt-conv-test/GD-' + function.__name__ + '-convergence.pdf', bbox_inches='tight')

        ################################################################################

        # Variational Optimisation:
        for method in ['ES', 'VO-R', 'VO-N']: #, 'VO-N-FULL'
            print(method + '-' + function.__name__)
            n1 = '../graphics/var-opt-conv-test/' + method + '-' + function.__name__ + '-sigma.pdf'
            n2 = '../graphics/var-opt-conv-test/' + method + '-' + function.__name__ + '-f.pdf'
            n3 = '../graphics/var-opt-conv-test/' + method + '-' + function.__name__ + '-convergence.pdf'

            np.random.seed(42)

            Nsamples = 200  # number of samples
            eta_mu = 100
            eta_beta = 10
            sigma = np.array([1])  # initial standard deviation of the Gaussian
            beta = 2 * np.log(sigma)  # parameterise the standard variance
            mu = x_init  # initial mean of the Gaussian
            sigma_hist = np.array(sigma)
            f_hist = np.zeros(Nloops)
            f = np.zeros(Nsamples)
            mu_hist = [mu]
            scale = 1/sigma**2  # Scale for VO natural gradient
            for i in range(Nloops):
                f_hist[i] = function(mu)[0]  # error value
                # epsilon = sigma * np.random.randn(Nsamples, D)  # draw samples
                epsilon = np.random.randn(Nsamples, D)  # draw samples
                
                # Evaluate fitness
                g = np.zeros((D,))  # initialise the gradient for the mean mu
                gbeta = 0  # initialise the gradient for the standard deviation (beta par)
                for j in range(Nsamples):
                    # For eps~N(0,1)
                    f[j] = function(mu + sigma * epsilon[j,:])[0]  # function value (error)
                f = fitness_shaping(f)

                # Update gradient
                for j in range(Nsamples):
                    # Vanilla ES
                    if method == 'ES':
                        g += epsilon[j, :]*f[j] / (np.exp(0.5 * beta) * Nsamples)
                        gbeta = 0

                    # VO regular gradient
                    if method == 'VO-R':
                        g += epsilon[j, :]*f[j] / (np.exp(0.5 * beta) * Nsamples)
                        gbeta += f[j] * (epsilon[j, :] @ epsilon[j, :].transpose() - D) / (2*Nsamples)

                    # VO natural gradient
                    if method == 'VO-N':
                        g += scale * np.exp(0.5 * beta) * epsilon[j, :]*f[j] / Nsamples
                        gbeta += f[j] * (epsilon[j, :] @ epsilon[j, :].transpose() - D) / Nsamples

                mu = mu - eta_mu * g            # Stochastic gradient descent for the mean
                beta = beta - eta_beta * gbeta  # Stochastic gradient descent for the variance par
                sigma = np.exp(0.5 * beta)
                mu_hist.append(mu)
                sigma_hist = np.append(sigma_hist, sigma)

            mu_hist = np.array(mu_hist).T

            fig, ax = plt.subplots()
            ax.plot(sigma_hist)
            ax.set_xlabel('Iteration, i')
            ax.set_ylabel(r'$\sigma_i$')
            fig.savefig(n1, bbox_inches='tight')

            fig, ax = plt.subplots()
            ax.plot(f_hist, label=r'$f(x_i)$')
            ax.plot([0, len(f_hist)], [fsurf.min(), fsurf.min()], label=r'$f(x^*)$')
            # ax.set_yscale('log')
            ax.legend()
            ax.set_xlabel('Iteration, i')
            ax.set_ylabel(r'$f$')
            fig.savefig(n2, bbox_inches='tight')

            plot_convergence(mu_hist)
            plt.savefig(n3, bbox_inches='tight')
            plt.close("all")
