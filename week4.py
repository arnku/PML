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
mcmc1 = pyro.infer.MCMC(nuts_kernel, num_samples=100, warmup_steps=50)
mcmc2 = pyro.infer.MCMC(nuts_kernel, num_samples=100, warmup_steps=50)
mcmc1.run()
mcmc2.run()
samples1 = mcmc1.get_samples()['x']
samples2 = mcmc2.get_samples()['x']
#Calculate E[x^2]
expectation_x2_1 = torch.mean(samples1**2)
data1 = arviz.from_pyro(mcmc1)
expectation_x2_2 = torch.mean(samples2**2)
data2 = arviz.from_pyro(mcmc2)
summary1 = arviz.summary(data1)
summary2 = arviz.summary(data2)
print("Chain 1 E[x^2]:", expectation_x2_1)
print("Chain 2 E[x^2]:", expectation_x2_2)
print("Chain 1 Summary:\n", summary1)
print("Chain 2 Summary:\n", summary2)
def HMC(n_samples):
    expected_val1_tensor = torch.tensor([])
    expected_val2_tensor = torch.tensor([])
    for i in range(5):
        nuts_kernel = pyro.infer.NUTS(model)
        mcmc1 = pyro.infer.MCMC(nuts_kernel, num_samples=n_samples, warmup_steps=200)
        mcmc2 = pyro.infer.MCMC(nuts_kernel, num_samples=n_samples, warmup_steps=200)
        mcmc1.run()
        mcmc2.run()
        samples1 = mcmc1.get_samples()["x"].detach().cpu()
        samples2 = mcmc2.get_samples()["x"].detach().cpu()
        expected_val1 = torch.mean(samples1**2)
        expected_val2 = torch.mean(samples2**2)
        expected_val1_tensor = torch.cat((expected_val1_tensor, expected_val1.unsqueeze(0)), dim=0)
        expected_val2_tensor = torch.cat((expected_val2_tensor, expected_val2.unsqueeze(0)), dim=0)
    return [expected_val1_tensor, expected_val2_tensor]
n_samples_list = [10,100,1000]
expected_val_list_1 = torch.tensor([])
expected_val_list_2 = torch.tensor([])
for n_samples in n_samples_list:
    expected_vals = HMC(n_samples)
    expected_val_list_1 = torch.cat((expected_val_list_1, expected_vals[0]), dim=0)
    expected_val_list_2 = torch.cat((expected_val_list_2, expected_vals[1]), dim=0)
std1 = torch.std(expected_val_list_1,dim=0)
mean1 = torch.mean(expected_val_list_1,dim=0)
print("Mean of E[x^2]:", mean1)
print("Standard Deviation of E[x^2]:", std1)
std2 = torch.std(expected_val_list_2,dim=0)
mean2 = torch.mean(expected_val_list_2,dim=0)
print("Mean of E[x^2]:", mean2)
print("Standard Deviation of E[x^2]:", std2)
