import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.spatial
import scipy.optimize as opt
import pandas as pd

a = np.linspace(0,1,100)
b = np.linspace(0,1,100)
gamma = 0.5
def k(x,y):
    return np.exp(-gamma*(a[x]-b[y])**2)
K_S = np.fromfunction(k,(len(a),len(b)), dtype=int)
K_S
samples = np.random.multivariate_normal(mean=np.zeros(100), cov=K_S,size=100)

def gaussian_kernel(X,Xprime, gamma=2):
    dists = scipy.spatial.distance.cdist(X,Xprime,metric='sqeuclidean')
    return np.exp(-gamma*dists)

def special_kernel(X,Xprime, eta):
    a = eta[0]
    b = eta[1]
    K = (1+X@Xprime.T)**2 + a * np.multiply.outer(np.sin(2*np.pi*X.reshape(-1)+b),np.sin(2*np.pi*Xprime.reshape(-1)+b))
    return K
#load and normalize Mauna Loa data 
data = np.genfromtxt('co2_mm_mlo.csv', delimiter=',')
#10 years of data for learning
X = data[:120,2]-1958
y_raw = data[:120,3]
y_mean = np.mean(y_raw)
y_std = np.sqrt(np.var(y_raw))
y = (y_raw-y_mean)/y_std
#the next 5 years for prediction
X_predict = data[120:180,2]-1958
y_predict = data[120:180,3]

# B) todo: implement this
def negLogLikelihood(params, kernel):
    noise_y = params[0]
    eta = params[1:]
    
    # Use normalized y
    N = len(y)
    
    # Calculate K
    # Ensure X is (N,1) for kernel computation
    X_col = X.reshape(-1,1)
    K = kernel(X_col, X_col, eta)
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
    
def optimize_params(ranges, kernel, Ngrid):
    opt_params = opt.brute(lambda params: negLogLikelihood(params, kernel), ranges, Ns=Ngrid, finish=None)
    noise_var = opt_params[0]
    eta = opt_params[1:]
    return noise_var, eta

# B) todo: implement the posterior distribution, i.e. the distribution of f^star
def conditional(X, y, X_star, noise_var, eta, kernel):
    # Ensure correct shapes
    X = X.reshape(-1, 1)
    X_star = X_star.reshape(-1, 1)
    
    K_xx = kernel(X, X, eta)
    K_xs = kernel(X, X_star, eta)
    K_ss = kernel(X_star, X_star, eta)
    
    Ky = K_xx + noise_var**2 * np.eye(len(y))
    Ky_inv = np.linalg.inv(Ky)
    
    mu_star = K_xs.T @ Ky_inv @ y
    sigma_star = K_ss - K_xs.T @ Ky_inv @ K_xs
    
    return mu_star, sigma_star

# C) todo: adapt this
kernel = special_kernel # todo: change to new kernel
ranges = ((0.01, 1), (0.01, 5), (0, 1)) # todo: change to the new parameters

Ngrid = 10
noise_var, eta = optimize_params(ranges, kernel, Ngrid)
print("optimal params:", noise_var, eta)

# B) todo: use the learned GP to predict on the observations at X_predict
prediction_mean_gp, Sigma_gp = conditional(X, y, X_predict, noise_var, eta, kernel)
var_gp = np.diag(Sigma_gp) # We only need the diagonal term of the covariance matrix for the plots.
var_gp = np.maximum(var_gp, 0) # Ensure variance is non-negative

#plotting code for your convenience
plt.figure(dpi=400,figsize=(6,3))
plt.plot(X + 1958, y_raw, color='blue', label='training data')
plt.plot(X_predict + 1958, y_predict, color='red', label='test data')
yout_m =prediction_mean_gp*y_std + y_mean
yout_v =var_gp*y_std**2
plt.plot(X_predict + 1958, yout_m, color='black', label='gp prediction')
plt.plot(X_predict + 1958, yout_m+1.96*yout_v**0.5, color='grey', label='GP uncertainty')
plt.plot(X_predict + 1958, yout_m-1.96*yout_v**0.5, color='grey')
plt.xlabel("year")
plt.ylabel("co2(ppm)")
plt.legend()
plt.tight_layout()
plt.show()
