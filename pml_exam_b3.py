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
torch.manual_seed(123)
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

    C = K + D_mat @ K_dd @ D_mat - K_d @ D_mat - D_mat @ K_d.T
    
    C = C + (sigma_var**2 + 1e-6) * torch.eye(len(y_i), device=device)

    ll = torch.distributions.MultivariateNormal(
        torch.zeros(len(y_i), device=device), covariance_matrix=C
    ).log_prob(Y_col.squeeze())

    lp = torch.distributions.Normal(
        torch.zeros(len(delta), device=device),
        sigma_delta
    ).log_prob(delta).sum()

    return ll + lp


def potential_fn_wrapper(z):
    return -log_posterior(z['x'])

def sample_likelihood(init_samples, warmup_steps, num_samples, step_size):
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
        num_chains=num_chains
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
    # Fix: Covariance expansion is additive: K(x,x*) + delta * d/dx K(x,x*)
    K_xs_mod = K_xs + D_mat @ dK_xs
    
    mu_star = K_xs_mod.T @ alpha
    
    v = torch.linalg.solve_triangular(L, K_xs_mod, upper=False)
    sigma_star = K_ss - v.T @ v
    
    return mu_star.squeeze(), torch.diag(sigma_star)

if __name__ == "__main__":
    num_chains = 4
    init_samples = torch.randn(num_chains, len(delta_i), dtype=torch.float64, device=device) * sigma_delta
    # Sample from posterior
    samples,arviz_data = sample_likelihood(init_samples, 1000, 2000, 0.01)
    print("MCMC done. Arviz summary:", arviz.summary(arviz_data))

    flat_samples = samples.reshape(-1, len(delta_i))
    
    #print the mean and covariance estimates
    print("mean est.", torch.mean(flat_samples,axis=0).detach().cpu().numpy())
    
    # Plotting
    import matplotlib.pyplot as plt
    import matplotlib 
    matplotlib.use('agg')
    plt.figure()
    arviz.plot_trace(arviz_data)
    arviz.plot_rank(arviz_data)
    plt.savefig("trace_plots_pyro.png")
    plt.close()
    # Plot delta9 and delta10 against true values
    plt.figure()
    plt.scatter([-0.25],[0.25], color='red', label='true values', s=100)
    # Plot delta9 and delta10 from markov chain:
    plt.scatter(x=flat_samples[:,9].detach().cpu().numpy(), y=flat_samples[:, 10].detach().cpu().numpy(), alpha=0.1, label='MCMC samples')
    plt.xlabel('delta9')
    plt.ylabel('delta10')
    plt.legend()
    plt.savefig("delta9_delta10_scatter_pyro.png")
    plt.close()
    # Put simulated delta into conditional GP and plot
    sample_idx = np.random.choice(flat_samples.shape[0], size=1)
    delta_sample = flat_samples[sample_idx,:].squeeze()
    mu_star, sigma_star = conditional_fixed(delta_sample)
    print("Estimated mean and variance:",mu_star,sigma_star)
    print("Estimated mean and variance for delta9 and delta10:",mu_star[9],sigma_star[9],mu_star[10],sigma_star[10])
    x_star = X.detach().cpu().numpy().squeeze()
    lower_error_bound = mu_star.detach().cpu().numpy() - 1.96 * torch.sqrt(sigma_star).detach().cpu().numpy()
    upper_error_bound = mu_star.detach().cpu().numpy() + 1.96 * torch.sqrt(sigma_star).detach().cpu().numpy()
    plt.figure()
    f = lambda x: -x**2 + 2* 1/(1 + np.exp(-10*x))
    plt.plot(x_star, f(x_star), label="ground truth")
    plt.plot(x_star, mu_star.detach().cpu().numpy(), label="mean for GP estimate")
    plt.fill_between(x_star, lower_error_bound, upper_error_bound, alpha=0.5, label="95% confidence interval")
    plt.scatter(x_i, y_i, label="data")
    plt.grid(True, which='both', linestyle='--', linewidth=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("GP estimate with sampled delta")
    plt.legend()
    plt.savefig("gp_estimate_sampled_delta_pyro.png")
    plt.close()
