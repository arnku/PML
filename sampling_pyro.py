from math import gamma
from torch.distributions.multivariate_normal import MultivariateNormal as TorchMVN
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.spatial
import scipy.optimize as opt
import pandas as pd
from pyro.infer.mcmc import NUTS, MCMC, HMC
import arviz

data = np.genfromtxt('C:/Users/simon/OneDrive/Skrivebord/TestPythonPML/data_part_B.csv', delimiter=',')
x_i = data[:,0]
y_i = data[:,1]
delta_i = data[:,2]
DELTA = delta_i * np.eye(len(delta_i))
device = torch.device("cpu")
print(f"Using device: {device}")
n = 100
gamma = 1.2
sigma_var = 0.67
# sigma_delta should be the prior std dev for delta.
# delta_i are realizations of noise, so we use their std dev as the prior scale.
# If delta_i contains negative values, it cannot be used as scale directly.
sigma_delta = torch.tensor(np.std(delta_i), dtype=torch.float64, device=device)

X = torch.linspace(-1, 1, n, dtype=torch.float64, device=device).unsqueeze(1)
X_col = torch.tensor(x_i.reshape(-1,1), dtype=torch.float64, device=device)
Y_col = torch.tensor(y_i.reshape(-1,1), dtype=torch.float64, device=device)
# D_tensor unused


def gaussian_kernel(X,Xprime):
    if not isinstance(X, torch.Tensor): X = torch.tensor(X, device=device)
    if not isinstance(Xprime, torch.Tensor): Xprime = torch.tensor(Xprime, device=device)
    dists = torch.cdist(X, Xprime, p=2)**2
    return torch.exp(-gamma*dists)

def d_gaussian_kernel(X,Xprime):
    if not isinstance(X, torch.Tensor): X = torch.tensor(X, device=device)
    if not isinstance(Xprime, torch.Tensor): Xprime = torch.tensor(Xprime, device=device)
    
    # Difference matrix (X - Xprime.T)
    # X is (N,1), Xprime is (M,1) => diff is (N,M)
    diff = X - Xprime.T
    dists = diff**2
    K = torch.exp(-gamma*dists)
    
    return -2*gamma*diff*K

def dd_gaussian_kernel(X,Xprime):
    if not isinstance(X, torch.Tensor): X = torch.tensor(X, device=device)
    if not isinstance(Xprime, torch.Tensor): Xprime = torch.tensor(Xprime, device=device)
    dists = torch.cdist(X, Xprime, p=2)**2
    return gaussian_kernel(X, Xprime) * (2*gamma -4*gamma**2 * dists)

def log_posterior(delta):
    """
    delta: shape (N,)
    """
    K = gaussian_kernel(X_col, X_col)
    K_dd = dd_gaussian_kernel(X_col, X_col)
    K_d = d_gaussian_kernel(X_col, X_col)
    
    D_mat = torch.diag(delta)
    # Correct Ky formula: K + D*KK*D - K'D - D*(-K') = K + D*KK*D - 2*K'*D (if K' is dK/dx)
    # d_gaussian_kernel returns dK/dx. 
    # Ky = K + D @ K_dd @ D - K_d @ D - D @ (-K_d).T
    # Note: K_d is symmetric? No. K(x, x') depends on |x-x'|^2. dK/dx = -dK/dx'.
    # K_d = dK/dx. dK/dx' = -K_d.
    # We need to act on training inputs.
    # The approximation is f(x_obs) approx f(x_true) + delta * f'(x_obs)
    # cov(y) = cov(f(x_obs) - delta * f'(x_obs)) 
    #        = K(x,x) + delta * cov(f', f') * delta - cov(f, f')*delta - delta*cov(f', f)
    #        = K + D * d2K/dxdx' * D - K_d * D - D * K_d^T
    # But here K_d is dK/dx (w.r.t first arg).
    # so cov(f, f') = - dK/dx' = dK/dx.
    # Term -cov(f, f')*delta = - dK/dx * D.
    # Term -delta*cov(f', f) = - D * (dK/dx)^T.
    
    C = K + D_mat @ K_dd @ D_mat - K_d @ D_mat - D_mat @ K_d.T
    
    # Add noise variance and jitter
    C = C + (sigma_var**2 + 1e-6) * torch.eye(len(y_i), device=device)

    # log likelihood p(y | Δ)
    ll = torch.distributions.MultivariateNormal(
        torch.zeros(len(y_i), device=device), covariance_matrix=C
    ).log_prob(Y_col.squeeze())

    # log prior p(Δ)
    lp = torch.distributions.Normal(
        torch.zeros(len(delta), device=device),
        sigma_delta
    ).log_prob(delta).sum()

    return ll + lp


def potential_fn_wrapper(z):
    return -log_posterior(z['x'])

def sample_likelihood(log_likelihood_fn, init_samples, warmup_steps, num_samples, step_size):
    
    # Potential energy function = - log_prob
    num_chains = 4
    mcmc_kernel = NUTS(
        potential_fn=potential_fn_wrapper,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        full_mass=False,
        step_size = step_size
    )
    mcmc = MCMC(
        mcmc_kernel,
        initial_params={'x':init_samples}, 
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
        
    )
    mcmc.run()
    arviz_data = arviz.from_pyro(mcmc)
    samples = mcmc.get_samples(group_by_chain=False)['x']
    #todo: need arviz
    return samples, arviz_data

def conditional(x_input, y_input, x_star, noise_var, theta, kernel_fn):
    
    # Ensure inputs are shaped correctly for kernel (N, D)
    x_in_col = x_input.reshape(-1, 1)
    x_star_col = x_star.reshape(-1, 1)

    # Calculate kernels using the provided kernel_fn (handles device placement internally)
    # Convert results back to numpy/cpu for scipy
    K_xx = kernel_fn(x_in_col, x_in_col).detach().cpu().numpy()
    K_x_star_x = kernel_fn(x_star_col, x_in_col).detach().cpu().numpy()
    K_x_star_x_star = kernel_fn(x_star_col, x_star_col).detach().cpu().numpy()

    # Add noise variance
    K_xx = K_xx + noise_var * np.eye(len(x_input))

    # Solve for mean: mu = K_*X (K_XX + noise)^-1 y
    L = scipy.linalg.cholesky(K_xx, lower=True)
    alpha = scipy.linalg.cho_solve((L, True), y_input)
    mu_star = K_x_star_x @ alpha

    # Solve for variance: cov = K_** - K_*X (K_XX + noise)^-1 K_X*
    # v = L^-1 K_X*^T  => v^T v = K_*X (L^-T L^-1) K_X* = K_*X K_XX^-1 K_X*
    v = scipy.linalg.solve_triangular(L, K_x_star_x.T, lower=True)
    sigma_star = K_x_star_x_star - v.T @ v

    return mu_star, sigma_star

def plot_estimate(noise_var, theta, x_i, y_i):
  f = lambda x: -x**2 + 2* 1/(1 + np.exp(-10*x))
  x_star = np.linspace(-1,1, num=100)
  # Uses global gaussian_kernel
  mu_star, sigma_star = conditional(x_i, y_i, x_star, noise_var, theta, gaussian_kernel)
  
  lower_error_bound = mu_star - np.sqrt(np.diag(sigma_star))*1.96
  upper_error_bound = mu_star + np.sqrt(np.diag(sigma_star))*1.96
  
  plt.figure()
  plt.plot(x_star, f(x_star), label="ground truth")
  plt.plot(x_star, mu_star, label="mean")
  plt.scatter(x_i, y_i, label="data")
  # Extract diagonal for fill_between if sigma_star is matrix
  # The formula used diag above, so assuming we want marginal variance
  plt.fill_between(x_star, lower_error_bound, upper_error_bound, alpha=0.5, label="95% confidence interval")
  plt.legend()
  plt.savefig("gp_estimate.png")
  plt.close()

# Remove broken global code block
# C = ...
# L = ...
# ...
# def gp_posterior ... 

def conditional_fixed(delta_sample):
    D_mat = torch.diag(delta_sample)
    
    K_xx = gaussian_kernel(X_col, X_col)
    K_dd = dd_gaussian_kernel(X_col, X_col)
    K_d = d_gaussian_kernel(X_col, X_col)
    
    Ky = K_xx + D_mat @ K_dd @ D_mat - K_d @ D_mat - D_mat @ K_d.T
    Ky = Ky + (sigma_var**2 + 1e-6) * torch.eye(X_col.shape[0], device=device)
    
    L = torch.linalg.cholesky(Ky)
    alpha = torch.cholesky_solve(Y_col, L)
    
    K_xs = gaussian_kernel(X_col, X)
    K_ss = gaussian_kernel(X, X)
    
    dK_xs = d_gaussian_kernel(X_col, X)
    K_xs_mod = K_xs - D_mat @ dK_xs
    
    mu_star = K_xs_mod.T @ alpha
    
    v = torch.linalg.solve_triangular(L, K_xs_mod, upper=False)
    sigma_star = K_ss - v.T @ v
    
    return mu_star.squeeze(), torch.diag(sigma_star)

if __name__ == "__main__":
    num_chains = 4
    init_samples = torch.randn(num_chains, len(delta_i), dtype=torch.float64, device=device) * sigma_delta
    
    samples,arviz_data = sample_likelihood(log_posterior, init_samples, 1000, 2000, 0.01)
    print("MCMC done. Arviz summary:", arviz.summary(arviz_data))

    flat_samples = samples.reshape(-1, len(delta_i))
    
    #print the mean and covariance estimates
    print("mean est.", torch.mean(flat_samples,axis=0).detach().cpu().numpy())
    
    # Plotting
    import matplotlib.pyplot as plt
    import matplotlib 
    matplotlib.use('agg')
    
    # Plot samples
    plt.figure()
    samples_np = flat_samples.detach().cpu().numpy()
    if samples_np.shape[1] >= 2:
        plt.scatter(samples_np[:,0],samples_np[:,1],label="samples (dim 0 vs 1)")
        plt.xlabel("$x_0$")
        plt.ylabel("$x_1$")
    plt.legend()
    plt.savefig("samples_test.png")
    plt.close()

    # Posterior predictive check
    print("Generating posterior predictions...")
    indices = torch.randint(0, len(flat_samples), (50,))
    f_preds = []
    
    for i, idx in enumerate(indices):
        d = flat_samples[idx]
        m, v_diag = conditional_fixed(d)
        f_preds.append(m)
        
    f_preds = torch.stack(f_preds)
    mean_f = torch.mean(f_preds, dim=0).detach().cpu().numpy()
    std_f = torch.std(f_preds, dim=0).detach().cpu().numpy()
    
    x_star_np = X.squeeze().cpu().numpy()
    f = lambda x: -x**2 + 2* 1/(1 + np.exp(-10*x))
    plt.figure()
    plt.plot(x_star_np, mean_f, label="Posterior Mean", color='blue')
    plt.plot(x_star_np, f(x_star_np), label="Ground Truth", color='green')
    plt.fill_between(x_star_np, mean_f - 1.96*std_f, mean_f + 1.96*std_f, alpha=0.3, color='blue', label="95% CI")
    
    # Plot data
    plt.scatter(x_i, y_i, color='red', label="Data")
    plt.title("GP Posterior with Input Noise Uncertainty")
    plt.legend()
    plt.savefig("pyro_posterior.png")
    plt.close()
    print("Done.")
    # Scatter plot of delta_9 vs delta_10 meaning the x_9 and x_10 noise components
    plt.figure()
    delta_9 = flat_samples[:,9].detach().cpu().numpy()
    delta_10 = flat_samples[:,10].detach().cpu().numpy()
    plt.scatter(delta_9, delta_10, alpha=0.5)
    plt.xlabel("delta_9")
    plt.ylabel("delta_10")
    # PLot true values of delta_9 and delta_10 (-0.25,0.25):
    plt.scatter([-0.25],[0.25],color='red',label="true value",s=100,marker='x')
    plt.legend()
    plt.savefig("samples_delta9_delta10.png")
    plt.close()

    