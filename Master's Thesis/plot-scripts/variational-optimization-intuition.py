import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import IPython


def sinc(x):
    return -np.sinc(x)


def sinc_quantized(x, n=5):
    return np.floor(-np.sinc(x) * n) / n


def narrow_vs_wide(x):
    f = np.zeros(x.shape)
    f[x < 0.5] = 1 / 5 * (x[x < 0.5] + 4.5) * (x[x < 0.5] - 0.5)
    f[x >= 0.5] = 26 * (x[x >= 0.5] - 2) * (x[x >= 0.5] - 2.5)
    f[f > 0] = 0
    return f


def generate_data(function, batch_size, mu, sigma):
    # Random standard normal numbers
    rand_no = np.random.randn(1, batch_size)[0]
    # Create array with different mu and sigma
    rand_array = np.add.outer(np.multiply.outer(rand_no, sigma), mu)
    # Compute expectation
    expected_value = np.mean(function(rand_array), 0)
    # Plottig related code
    MU, SIGMA = np.meshgrid(mu, sigma)
    return (expected_value, MU, SIGMA)


def plot(function, functionlabel, expected_value, MU, SIGMA, step=False):
    fig, ax = plt.subplots()
    if step:
        plt.step(MU[0, :], function(MU[0, :]))
    else:
        plt.plot(MU[0, :], function(MU[0, :]))
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(functionlabel)
    plt.savefig('../graphics/var-opt-intu/variational-optimization-function-' + function.__name__ + '.pdf', bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.contourf(MU, SIGMA, expected_value, 17, cmap=cm.coolwarm)
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\sigma$')
    plt.savefig('../graphics/var-opt-intu/variational-optimization-contour-' + function.__name__ + '.pdf', bbox_inches='tight')

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(SIGMA, MU, expected_value, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'$\mu$')
    ax.set_zlabel(functionlabel)
    ax.view_init(azim=40)
    plt.savefig('../graphics/var-opt-intu/variational-optimization-surface-' + function.__name__ + '.pdf', bbox_inches='tight')


if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 12})
    batch_size = 15000                   # For representative results 10000
    N = 200                              # For smooth curves 200
    mu = np.linspace(-5, 5, N)
    sigma = np.linspace(0, 2, N)

    expected_value, MU, SIGMA = generate_data(narrow_vs_wide, batch_size, mu, sigma)
    plot(narrow_vs_wide, 'Narrow and wide minima', expected_value, MU, SIGMA)
    print("Narrow vs wide done")

    expected_value, MU, SIGMA = generate_data(sinc, batch_size, mu, sigma)
    plot(sinc, r'$-\sinc(\mu)$', expected_value, MU, SIGMA)
    print("Sinc done")

    expected_value, MU, SIGMA = generate_data(sinc_quantized, batch_size, mu, sigma)
    plot(sinc_quantized, r'$-\sinc(\mu)$ discretized', expected_value, MU, SIGMA, step=True)
    print("Sinc quantized done")

    # fig = plt.figure()
    # ax = fig.add_subplot(221, projection='3d')
    # surf = ax.plot_surface(SIGMA, MU, expected_value, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # ax = fig.add_subplot(222)
    # ax.plot(expected_value[0,:])

    # ax = fig.add_subplot(223)
    # ax.contourf(MU, SIGMA, expected_value, cmap=cm.coolwarm)
