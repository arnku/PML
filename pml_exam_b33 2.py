import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from pyro.infer.mcmc import NUTS, MCMC
import arviz
import torch.distributions as dist

device = torch.device("cpu")
torch.manual_seed(123)

data = np.genfromtxt('data_part_B.csv', delimiter=',')
x_i_raw = torch.tensor(data[:, 0], dtype=torch.float64, device=device)
y_i_raw = torch.tensor(data[:, 1], dtype=torch.float64, device=device)
true_deltas = torch.tensor(data[:, 2], dtype=torch.float64, device=device)

X_col = x_i_raw.unsqueeze(1)
Y_col = y_i_raw.unsqueeze(1)

sigma_x = 0.1   
sigma_y = 0.12   
gamma = 3.4   

def gaussian_kernel(X, Xp):
    dists = torch.cdist(X, Xp, p=2)**2
    return torch.exp(-gamma * dists)

def d_gaussian_kernel(X, Xp):
    diff = X - Xp.T
    K = gaussian_kernel(X, Xp)
    return -2 * gamma * diff * K

def dd_gaussian_kernel(X, Xp):
    dists = torch.cdist(X, Xp, p=2)**2
    K = gaussian_kernel(X, Xp)
    return K * (2 * gamma - 4 * gamma**2 * dists)

def log_posterior(delta):
    K = gaussian_kernel(X_col, X_col)
    K1 = d_gaussian_kernel(X_col, X_col)
    K2 = dd_gaussian_kernel(X_col, X_col)
    
    D = torch.diag(delta)
    
    C = K + D @ K2 @ D + K1 @ D + D @ K1.T

    C = C + (sigma_y**2 + 1e-6) * torch.eye(len(y_i_raw), device=device)
    
    ll = dist.MultivariateNormal(torch.zeros(len(y_i_raw), device=device), covariance_matrix=C).log_prob(Y_col.squeeze())

    lp = dist.Normal(0, sigma_x).log_prob(delta).sum()
    
    return ll + lp

def potential_fn_wrapper(z):
    return -log_posterior(z['delta'])

def sample_likelihood(init_samples, warmup_steps, num_samples, step_size):
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
        initial_params={'delta':init_samples}, 
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains
    )
    mcmc.run()
    return mcmc


def conditional_f_marginal(samples, x_grid):
    x_grid_col = x_grid.unsqueeze(1)
    K_ss = gaussian_kernel(x_grid_col, x_grid_col)
    
    mu_list = []
    var_list = []
    
    idx = np.random.choice(len(samples), size=100, replace=False)
    
    for i in idx:
        delta = samples[i]
        D = torch.diag(delta)
        
        K = gaussian_kernel(X_col, X_col)
        K1 = d_gaussian_kernel(X_col, X_col)
        K2 = dd_gaussian_kernel(X_col, X_col)
        Ky = K + D @ K2 @ D + K1 @ D + D @ K1.T + (sigma_y**2 + 1e-6) * torch.eye(len(y_i_raw))
        
        L = torch.linalg.cholesky(Ky)

        K_xs = gaussian_kernel(X_col, x_grid_col)
        K1_xs = d_gaussian_kernel(X_col, x_grid_col)
        K_cross = K_xs - D @ K1_xs
        
        alpha = torch.cholesky_solve(Y_col, L)
        mu_s = K_cross.T @ alpha
        
        v = torch.linalg.solve_triangular(L, K_cross, upper=False)
        var_s = torch.diag(K_ss) - torch.sum(v**2, dim=0)
        
        mu_list.append(mu_s.squeeze())
        var_list.append(torch.clamp(var_s, min=0.0))

    mu_stack = torch.stack(mu_list)
    var_stack = torch.stack(var_list)
    
    mu_marginal = mu_stack.mean(dim=0)
    var_marginal = var_stack.mean(dim=0) + mu_stack.var(dim=0)
    
    return mu_marginal, var_marginal

if __name__ == "__main__":
    base_delta = torch.zeros(len(y_i_raw), dtype=torch.float64, device=device)
    

    init_delta = base_delta.unsqueeze(0).repeat(4, 1)
    mcmc = sample_likelihood(init_delta, 1000, 2000, 0.01)
    samples = mcmc.get_samples()['delta']
    
    plt.figure(figsize=(6, 6))
    plt.scatter(samples[:, 9], samples[:, 10], alpha=0.1, label='MCMC Samples')
    plt.scatter(-0.25, 0.25, color='red', s=100, label='True Values (-0.25, 0.25)')
    plt.xlabel('$\Delta_9$')
    plt.ylabel('$\Delta_{10}$')
    plt.legend()
    plt.title('D5: Posterior Samples of Measurement Errors')
    plt.grid(True, which='both', linestyle='--', linewidth=1)
    plt.savefig("delta9_delta10_scatter_pyro.png",dpi=300)
    plt.close()
    # 
    x_grid = torch.linspace(-1, 1, 100, dtype=torch.float64)
    mu, var = conditional_f_marginal(samples, x_grid)
    std = torch.sqrt(var)
    
    plt.figure(figsize=(10, 5))
    f_true = lambda x: -x**2 + 2 * (1 / (1 + np.exp(-10 * x))) #
    plt.plot(x_grid, f_true(x_grid), 'k--', label='True Function')
    plt.plot(x_grid, mu.detach(), 'b', label='Marginal Mean')
    plt.fill_between(x_grid, (mu - 1.96*std).detach(), (mu + 1.96*std).detach(), color='blue', alpha=0.2, label='95% CI')
    plt.scatter(x_i_raw, y_i_raw, c='red', s=10, alpha=0.5, label='Noisy Data')
    plt.title('D5: Marginal Posterior Predictive $p(\mathcal{\Delta}|X,y,\theta,\sigma_y^2,\sigma_x^2)$')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=1)
    plt.savefig("marginal_posterior_predictive.png",dpi=300)
    plt.close()
    # Plot Delta posterior
    plt.figure()
    arviz.plot_trace(arviz.from_pyro(mcmc), var_names=["delta"])
    plt.savefig("trace_plots_pyro.png",dpi=300)
    plt.close()

    print(arviz.summary(arviz.from_pyro(mcmc), var_names=["delta"]))
    # Plot trace plots
    arviz.plot_rank(arviz.from_pyro(mcmc), var_names=["delta"])
    plt.savefig("rank_plots_pyro.png",dpi=300)
    plt.close()