import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy
import scipy.spatial
import scipy.optimize as opt
import pandas as pd

data = np.genfromtxt('C:/Users/simon/OneDrive/Skrivebord/TestPythonPML/data_part_B.csv', delimiter=',')
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
        
        Ky = K + D @ DDK @ D - DK @ D - D @ (-DK) 
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
    term2 = np.sum(np.log(np.diag(L)))
    term3 = 0.5 * len(y) * np.log(2 * np.pi)
    return (term1 + term2 + term3).item()

def optimize_params(ranges, kernel, Ngrid, x, y, delta):
    opt_params = opt.brute(negLogLikelihood, ranges,args=(x, y, delta), Ns=Ngrid, finish=None)
    noise_var = opt_params[0]
    theta = opt_params[1:]
    return noise_var, theta

def conditional(X, y, X_star, noise_var, eta, kernel):
    # Ensure correct shapes
    X = X.reshape(-1, 1)
    X_star = X_star.reshape(-1, 1)

    K_xx = kernel(X, X, eta)
    K_xs = kernel(X, X_star, eta)
    K_ss = kernel(X_star, X_star, eta)
    # Ky = covariance(X, eta, noise_var=noise_var)
    Ky = K_xx + noise_var**2 * np.eye(len(y))
    Ky_inv = np.linalg.inv(Ky)

    mu_star = K_xs.T @ Ky_inv @ y
    sigma_star = K_ss - K_xs.T @ Ky_inv @ K_xs

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
  mu_star, sigma_star = conditional(x_i, y_i, x_star, noise_var, theta, kernel)
  mu_star.shape
  print("sigma_star diag:", np.diag(sigma_star))
  lower_error_bound = mu_star - np.sqrt(np.diag(sigma_star))*1.96
  upper_error_bound = mu_star + np.sqrt(np.diag(sigma_star))*1.96
  plt.plot(x_star, f(x_star), label="ground truth")
  plt.plot(x_star, mu_star, label="mean")
  plt.scatter(x_i, y_i, label="data")
  plt.fill_between(x_star, lower_error_bound, upper_error_bound, alpha=0.5, label="95% confidence interval")
  plt.title(title)
  plt.legend()
  plt.savefig("gp_estimate.png")
  plt.show()
  plt.close()

plot_estimate(noise_var, theta, x_i, y_i, title="Estimate with optimized parameters")