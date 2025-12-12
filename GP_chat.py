import pyro
import torch
import pyro.distributions as pdist
import torch.distributions as tdist
import arviz
import numpy as np

import matplotlib.pyplot as plt

from torch.distributions import constraints


class MyDensity(pdist.TorchDistribution):
    # The integration interval
    support = constraints.interval(-3.0, 3.0)
    # Constraint for the starting value used in the sampling
    # (should be within integration interval)
    arg_constraints = {"start": support}

    def __init__(self, start=torch.tensor(-3.0)):
      # start = starting value for HMC sampling, default 0
      self.start = start
      super(pdist.TorchDistribution, self).__init__()

    def sample(self, sample_shape=torch.Size()):
        # This is only used to start the HMC sampling
        # It simply returns the starting value for the sampling
        return self.start

    def log_prob(self, x):
        density = torch.exp(-x**2/2)*(torch.sin(x)**2+3*torch.cos(x)**2*torch.sin(7*x)**2+1)
        log_prob = torch.log(density)
        return log_prob

def model():
    my_density = MyDensity()
    samples = pyro.sample("x", my_density)
    return samples

nuts_kernel = pyro.infer.NUTS(model)
mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=100, warmup_steps=50, num_chains=2)
mcmc.run()
samples = mcmc.get_samples()['x']
#Calculate E[x^2]
expectation_x2 = torch.mean(samples**2)
data = arviz.from_pyro(mcmc)
summary = arviz.summary(data)
print("Chain E[x^2]:", expectation_x2)
print("Chain Summary:\n", summary)

def HMC(n_samples: int) -> torch.Tensor:
    # Collect results in a Python list to avoid concat/dtype issues
    results = []
    for i in range(5):
        # Reseed to get different MCMC runs
        pyro.set_rng_seed(int(np.random.randint(0, 2**31 - 1)))
        pyro.clear_param_store()
        nuts_kernel = pyro.infer.NUTS(model)
        mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=n_samples, warmup_steps=200, num_chains=2)
        mcmc.run()
        samples = mcmc.get_samples()["x"].detach().cpu()
        expected_val = torch.mean(samples**2)
        results.append(expected_val.item())
    return torch.tensor(results, dtype=torch.float64)

n_samples_list = [10,100,1000]
# Replace the previous aggregation (which produced a single scalar mean/std)
# with per-n_samples statistics so plotting x and y sizes match.
means = []
stds = []
for n_samples in n_samples_list:
    expected_vals = HMC(n_samples)  # tensor of shape (num_repeats,), e.g. (5,)
    means.append(expected_vals.mean().item())
    stds.append(expected_vals.std(unbiased=False).item())

means = np.array(means)
stds = np.array(stds)

print("Means of E[x^2]:", means)
print("Standard Deviations of E[x^2]:", stds)

print(means)
print(stds)

plt.figure(figsize=(10, 6))
plt.errorbar(n_samples_list, means, yerr=stds, fmt='o-', capsize=5, capthick=2)
plt.xlabel('Number of Samples')
plt.ylabel('E[x²]')
plt.title('Expected Value of x² vs Number of Samples')
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.show()

