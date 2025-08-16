#import math

import matplotlib.pyplot as plt
import numpy as np

#import torch
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

#from torch.utils.data import Dataset

plt.style.use('bmh')


def matern_covariance(n, length_scale, nu, sigma2=1.0, jitter=1e-9):
    """
    Return an n x n Matérn covariance matrix over integer index positions 0…n-1.
    """
    # Build kernel with free variance
    k = ConstantKernel(constant_value=sigma2) * Matern(length_scale=length_scale, nu=nu)

    # Locations in index space → shape (n,1)
    X = np.arange(n, dtype=float).reshape(-1, 1)

    # Evaluate and stabilize (to later draw samples, need a positive-definite Σ. 
    # Floating-point rounding can produce a single negative eigenvalue.)
    K = k(X)              # equivalent to k(X, X)
    np.fill_diagonal(K, K.diagonal() + jitter)
    return K

"""
def rbf_kernel(i, j, sigma, *args):
    return math.exp(-((i - j) ** 2) / (2 * sigma ** 2))
def fbm_kernel(t, s, sigma2=1.0, H=0.75):
    return 0.5 * sigma2 * (abs(t)**(2*H) + abs(s)**(2*H) - abs(t-s)**(2*H))
"""

# Sample from gaussian process 5 times and plot result

N = [1, 2, 4, 8, 16, 32, 64]
scales = [3, 5, 10, 20, 50, 100]
nu = [0.5, 1, 3, 5, 10, 50]
T = 100
samples = []
for n in nu:
    k = matern_covariance(T, 10, n)
    sample = np.random.multivariate_normal(np.ones(T) * 2, k)
    samples.append(sample)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
for i in range(len(nu)):
    ax.plot(samples[i], label=f'{nu[i]}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Gaussian Process Samples')
    ax.legend()

plt.show()



def generate_arma(N,                 # length of series to return
                  phi=None,          # list/array of AR coefficients φ₁…φ_p
                  theta=None,        # list/array of MA coefficients θ₁…θ_q
                  sigma=1.0,         # innovation std-dev
                  mu=0.0,            # unconditional mean
                  burnin=100):       # extra steps to reach stationarity
    """
    Simulate an ARMA(p,q) process:
        X_t = μ
              + Σ_{k=1}^p φ_k (X_{t-k} - μ)
              + ε_t
              + Σ_{ℓ=1}^q θ_ℓ ε_{t-ℓ},
        ε_t ~ N(0, σ²)
    """
    phi   = np.asarray(phi or [])
    theta = np.asarray(theta or [])
    p, q  = len(phi), len(theta)
    m     = max(p, q, 1)
    T     = N + (burnin + m)

    # innovations (ε) and output buffer long enough for burn-in
    eps = np.random.normal(scale=sigma, size=T)
    x   = np.zeros(T)

    for t in range(m, T):
        ar_part = 0.0
        if p:
            ar_part = np.dot(phi, x[t - np.arange(1, p + 1)] - mu)

        ma_part = 0.0
        if q:
            ma_part = np.dot(theta, eps[t - np.arange(1, q + 1)])

        x[t] = mu + ar_part + eps[t] + ma_part

    # discard burn-in and initial lags
    return x[m + burnin:]


p3 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
p4 = [0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0.1]
p5 = [0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5]

test1 = generate_arma(50, p3, [], 0.1)
test2 = generate_arma(50, p4, [], 0.1)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(test1, label='Process 1')
ax.plot(test2, label='Process 2')
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.legend()
plt.show()