import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy
import scipy.spatial
import scipy.optimize as opt
import pandas as pd

data = np.genfromtxt('data_part_B.csv', delimiter=',')
x_i = data[:,0]
y_i = data[:,1]
delta_i = data[:,2]
x_i_truth = x_i - delta_i
D = np.diag(delta_i)  # D is the diagonal matrix of delta_i
def gaussian_kernel(X,Xprime, gamma=2):
    dists = scipy.spatial.distance.cdist(X,Xprime,metric='sqeuclidean')
    return np.exp(-gamma*dists)

def covariance(x, gamma, delta=None, noise_var=0):
    x = x.reshape(-1, 1)

    diff = x - x.T 
    sq_dist = diff**2
    K = np.exp(-gamma * sq_dist)
    
    if delta is not None:
        D = np.diag(delta)

        DK = -2 * gamma * diff * K

        DDK = (2 * gamma - 4 * (gamma**2) * sq_dist) * K
        
        Ky = K + D @ DDK @ D - DK.T @ D - D @ DK 
    else:
        Ky = K

    return Ky + (noise_var**2) * np.eye(len(x))

def negLogLikelihood(theta, x, y, delta):
    noise_y, gamma = theta[0], theta[1]
    Ky = covariance(x, gamma, delta, noise_y)
    
    # Stable computation using Cholesky
    L = scipy.linalg.cholesky(Ky, lower=True)
    # Compute the inverse using Cholesky factors
    alpha = scipy.linalg.solve_triangular(L.T, scipy.linalg.solve_triangular(L, y, lower=True))
    # Put the terms together
    term1 = 0.5 * y.T @ alpha
    term2 = 0.5 * np.log(np.linalg.det(Ky))
    term3 = 0.5 * len(y) * np.log(2 * np.pi)
    return (term1 + term2 + term3)

def optimize_params(ranges, kernel, Ngrid, x, y, delta):
    opt_params = opt.brute(negLogLikelihood, ranges,args=(x, y, delta), Ns=Ngrid, finish=None)
    noise_var = opt_params[0]
    theta = opt_params[1:]
    return noise_var, theta

def conditional(X, y, X_star, noise_var, gamma, delta_i):
    X = X.reshape(-1, 1)
    X_star = X_star.reshape(-1, 1)
    D = np.diag(delta_i)

    # Usual kernels
    K_xx_pure = gaussian_kernel(X, X, gamma)
    K_xs = gaussian_kernel(X, X_star, gamma)
    K_ss = gaussian_kernel(X_star, X_star, gamma)

    # First derivative kernel: K1(X, X_star)
    # K1_ij = -2 * gamma * (x_i - x_star_j) * K_ij
    diff_star = X - X_star.T
    K1_xs = -2 * gamma * diff_star * K_xs

    # Ky must be the same matrix used in NLL (the B.2.2 derivative-corrected one)
    Ky = covariance(X, gamma, delta_i, noise_var)
    Ky_inv = np.linalg.inv(Ky)

    # CORRECTED cross-covariance for the Taylor model
    # Cov(y, f_star) = K_xs - D @ K1_xs
    K_y_star = K_xs - D @ K1_xs

    mu_star = K_y_star.T @ Ky_inv @ y
    sigma_star = K_ss - K_y_star.T @ Ky_inv @ K_y_star

    return mu_star, sigma_star

kernel = gaussian_kernel
ranges = ((0.01, 1.0), (0.1, 10.0))  # noise_var, gamma
Ngrid = 10
noise_var, theta = optimize_params(ranges, kernel, Ngrid, x_i, y_i, delta_i)
print("optimal params:", noise_var, theta)

def plot_target():
  f = lambda x: -x**2 + 2* 1/(1 + np.exp(-10*x))
  x = np.linspace(-1,1, num=100)
  plt.plot(x, f(x))
  plt.show()

def plot_estimate(noise_var, theta, x_i, y_i, title=""):
  f = lambda x: -x**2 + 2* 1/(1 + np.exp(-10*x))
  x_star = np.linspace(-1,1, num=100)
  mu_star, sigma_star = conditional(x_i, y_i, x_star, noise_var, theta, delta_i)
  mu_star.shape
  print("sigma_star diag:", np.diag(sigma_star))
  lower_error_bound = mu_star - np.sqrt(np.diag(sigma_star))*1.96
  upper_error_bound = mu_star + np.sqrt(np.diag(sigma_star))*1.96
  plt.plot(x_star, f(x_star), label="ground truth")
  plt.plot(x_star, mu_star, label="mean")
  plt.scatter(x_i, y_i, label="data")
  plt.fill_between(x_star, lower_error_bound, upper_error_bound, alpha=0.5, label="95% confidence interval")
  plt.grid(True, which='both', linestyle='--', linewidth=1)
  plt.title(title)
  plt.legend()
  plt.savefig("gp_estimate.png")
  plt.show()
  plt.close()

plot_estimate(noise_var, theta, x_i, y_i, title="Estimate with optimized parameters")
# Plot likelihood surface
noise_var_range = np.linspace(0.001, 10.0, 20)
gamma_range = np.linspace(0.1, 10.0, 20)
NLL_values = np.zeros((len(noise_var_range), len(gamma_range)))
for i, nv in enumerate(noise_var_range):
    for j, g in enumerate(gamma_range):
        NLL_values[i, j] = negLogLikelihood((nv, g), x_i, y_i, delta_i)
plt.figure()
plt.contourf(gamma_range, noise_var_range, NLL_values, levels=50, cmap='viridis')
plt.colorbar(label='Negative Log-Likelihood')
plt.xlabel('Gamma')
plt.ylabel('Noise Variance')
plt.title('Negative Log-Likelihood Surface')
plt.savefig("nll_surface.png")
plt.close()
