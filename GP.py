import pyro
import torch
import pyro.distributions as pdist
import torch.distributions as tdist
import arviz
import numpy as np


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
    expected_val_tensor = torch.tensor([])
    for i in range(2):
        nuts_kernel = pyro.infer.NUTS(model)
        mcmc = pyro.infer.MCMC(nuts_kernel, num_samples=n_samples, warmup_steps=200, num_chains=2)
        mcmc.run()
        samples = mcmc.get_samples()["x"].detach().cpu()
        expected_val = torch.mean(samples**2)
        expected_val_tensor = torch.cat((expected_val_tensor, expected_val.unsqueeze(0)), dim=0)
    return expected_val_tensor

n_samples_list = [10,100,1000]
expected_val_list = torch.tensor([])
for n_samples in n_samples_list:
    expected_vals = HMC(n_samples)
    expected_val_list = torch.cat((expected_val_list, expected_vals), dim=0)
std = torch.std(expected_val_list,dim=0)
mean = torch.mean(expected_val_list,dim=0)
print("Mean of E[x^2]:", mean.item())
print("Standard Deviation of E[x^2]:", std.item())