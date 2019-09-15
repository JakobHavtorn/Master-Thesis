import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math

matplotlib.rcParams.update({'font.size': 12})

mu1 = 0
mu2 = 10
sigma = 100
x = np.linspace(mu1 - 3.5*sigma, mu2 + 3.5*sigma, 200)

f, ax = plt.subplots()
ax.plot(x,mlab.normpdf(x, mu1, sigma), label=r'$\mathcal{N}(x|0,100^2)$')
ax.plot(x,mlab.normpdf(x, mu2, sigma), label=r'$\mathcal{N}(x|10,100^2)$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\mathcal{N}(x|\mu,\sigma^2)$')
ax.legend()
f.savefig('../graphics/gaussian-pdfs/S1.pdf', bbox_inches='tight')
plt.close(f)

mu1 = 0
mu2 = 0.1
sigma = 0.1
x = np.linspace(mu1 - 3.5*sigma, mu2 + 3.5*sigma, 200)

f, ax = plt.subplots()
ax.plot(x,mlab.normpdf(x, mu1, sigma), label=r'$\mathcal{N}(x|0,0.1^2)$')
ax.plot(x,mlab.normpdf(x, mu2, sigma), label=r'$\mathcal{N}(x|0.1,0.1^2)$')
ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$\mathcal{N}(x|\mu,\sigma^2)$')
ax.legend()
f.savefig('../graphics/gaussian-pdfs/S2.pdf', bbox_inches='tight')
plt.close(f)

