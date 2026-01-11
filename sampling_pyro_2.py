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

data = np.genfromtxt('C:/Users/simon/OneDrive/Skrivebord/TestPythonPML/data_part_B.csv', delimiter=',')
x_i = data[:,0]
y_i = data[:,1]
delta_i = data[:,2]
DELTA = delta_i * np.eye(len(delta_i))
device = torch.device("cpu")
print(f"Using device: {device}")
n = 100
gamma = 1.009
X = torch.linspace(-1, 1, n, device=device).unsqueeze(1)
X_col = torch.tensor(x_i.reshape(-1,1), dtype=torch.float64, device=device)
DELTA = torch.tensor(DELTA, dtype=torch.float64, device=device)

def gaussian_kernel(X,Xprime, gamma=2):
    if not isinstance(X, torch.Tensor): X = torch.tensor(X, device=device)
    if not isinstance(Xprime, torch.Tensor): Xprime = torch.tensor(Xprime, device=device)
    dists = torch.cdist(X, Xprime, p=2)**2
    return torch.exp(-gamma*dists)

def d_gaussian_kernel(X,Xprime, gamma=2):
    if not isinstance(X, torch.Tensor): X = torch.tensor(X, device=device)
    if not isinstance(Xprime, torch.Tensor): Xprime = torch.tensor(Xprime, device=device)
    dists = torch.cdist(X, Xprime, p=2)**2
    # dist = sqrt(dists). But formula uses sqrt(dist) * exp(). 
    # original: -2*gamma*np.sqrt(dist)*gaussian_kernel(X, Xprime, gamma) where dist is sqeuclidean
    # Wait, original code:
    # dist = scipy.spatial.distance.cdist(Metric='sqeuclidean') -> returns squared distances
    # return -2*gamma*np.sqrt(dist)*gaussian_kernel
    # np.sqrt(squared_distances) = euclidian_distance. 
    return -2*gamma*torch.sqrt(dists)*gaussian_kernel(X, Xprime, gamma)

def dd_gaussian_kernel(X,Xprime, gamma=2):
    if not isinstance(X, torch.Tensor): X = torch.tensor(X, device=device)
    if not isinstance(Xprime, torch.Tensor): Xprime = torch.tensor(Xprime, device=device)
    dists = torch.cdist(X, Xprime, p=2)**2
    return gaussian_kernel(X, Xprime, gamma) * (2*gamma -4*gamma**2 * dists)

def negLogLikelihood(theta, x, y, D):
    noise_y = theta[0]
    gamma = theta[1:]
    # Use normalized y
    N = len(y)

    # Calculate K
    # Ensure X is (N,1) for kernel computation
    X_col = x.reshape(-1,1)
    # variance of y|D, X, theta, sigma_y^2
    K = gaussian_kernel(X_col, X_col, gamma) + D@dd_gaussian_kernel(X_col, X_col,gamma)@D.T - 2*d_gaussian_kernel(X_col, X_col, gamma)@D.T
    Ky = K + noise_y**2 * np.eye(N)

    # Calculate Negative Log Likelihood
    # NLL = 0.5 * y.T @ Ky^-1 @ y + 0.5 * log|Ky| + N/2 * log(2pi)

    try:
        Ky_inv = np.linalg.inv(Ky)
        term1 = 0.5 * y.T @ Ky_inv @ y

        # Use slogdet for numerical stability
        sign, logdet = np.linalg.slogdet(Ky)
        if sign <= 0:
             return 1e10
        term2 = 0.5 * logdet

        term3 = 0.5 * N * np.log(2*np.pi)
        return term1 + term2 + term3
    except np.linalg.LinAlgError:
        return 1e10

class LikelihoodWrapper:
    def __init__(self, log_like_fn):
        self.log_like_fn = log_like_fn
    def __call__(self, params):
        return self.log_like_fn(params['x'])

def sample_likelihood(log_likelihood_fn, init_samples, warmup_steps, num_samples, step_size):
    
    potential_fn = LikelihoodWrapper(log_likelihood_fn)
    num_chains = init_samples.shape[0]
    mcmc_kernel = NUTS(
        potential_fn=potential_fn,
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
    samples = mcmc.get_samples(group_by_chain=False)['x']
    #todo: need arviz
    return samples


if __name__ == "__main__":
    #define sampling from a 5D normal distribution
    K = gaussian_kernel(X,X,gamma)
    K_d = d_gaussian_kernel(X,X,gamma)
    K_dd = dd_gaussian_kernel(X,X,gamma)
    D = len(X_col)
    C = gaussian_kernel(X_col, X_col, gamma) + DELTA@dd_gaussian_kernel(X_col, X_col,gamma)@DELTA.T - 2*d_gaussian_kernel(X_col, X_col, gamma)@DELTA.T
    # Force PD using eigenvalue decomposition
    e, v = torch.linalg.eigh(C)
    e = torch.where(e < 1e-4, torch.tensor(1e-4, dtype=torch.float64, device=device), e)
    C = v @ torch.diag(e) @ v.T

    cond_number = torch.max(e) / torch.min(e)
    print(f"Condition number of C: {cond_number.item():.2e}")

    normal_dist = TorchMVN(torch.zeros(D,dtype=torch.float64, device=device), covariance_matrix = C)

    #we want to use 4 parallel chains so we crease 4 initial samples
    init_samples = torch.randn(4,D,dtype=torch.float64, device=device)
    samples = sample_likelihood(normal_dist.log_prob, init_samples, 1000, 1000, 0.1)


    #print the mean and covariance estimates
    print("mean est.", torch.mean(samples,axis=0).detach().cpu().numpy())
    print("cov est.", torch.cov(samples.T).detach().cpu().numpy())
    print("cov truth.", C.detach().cpu().numpy())

    #now some plotting
    import matplotlib.pyplot as plt
    import matplotlib 
    matplotlib.use('agg')
    #plot the first two dimensions as scatter plot
    plt.figure()
    samples_np = samples.detach().cpu().numpy()
    plt.scatter(samples_np[:,0],samples_np[:,1],label="samples")
    plt.xlabel("$x_0$")
    plt.xlabel("$x_1$")
    plt.legend()
    plt.savefig("samples_test.png")
    plt.close()


