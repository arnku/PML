
import torch
import numpy as np

device = torch.device("cpu")
gamma = 0.1

def gaussian_kernel(X,Xprime):
    if not isinstance(X, torch.Tensor): X = torch.tensor(X, device=device)
    if not isinstance(Xprime, torch.Tensor): Xprime = torch.tensor(Xprime, device=device)
    dists = torch.cdist(X, Xprime, p=2)**2
    return torch.exp(-gamma*dists)

def d_gaussian_kernel(X,Xprime):
    if not isinstance(X, torch.Tensor): X = torch.tensor(X, device=device)
    if not isinstance(Xprime, torch.Tensor): Xprime = torch.tensor(Xprime, device=device)
    diff = X - Xprime.T
    dists = diff**2
    K = torch.exp(-gamma*dists)
    return -2*gamma*diff*K

def dd_gaussian_kernel(X,Xprime):
    if not isinstance(X, torch.Tensor): X = torch.tensor(X, device=device)
    if not isinstance(Xprime, torch.Tensor): Xprime = torch.tensor(Xprime, device=device)
    dists = torch.cdist(X, Xprime, p=2)**2
    return gaussian_kernel(X, Xprime) * (2*gamma -4*gamma**2 * dists)

def check_psd(N=5, M=3):
    torch.manual_seed(42)
    X_col = torch.randn(N, 1, dtype=torch.float64, device=device)
    X = torch.randn(M, 1, dtype=torch.float64, device=device)
    delta = torch.randn(N, dtype=torch.float64, device=device) * 0.1
    # delta = torch.zeros(N, dtype=torch.float64, device=device) 
    
    D_mat = torch.diag(delta)
    
    K_xx = gaussian_kernel(X_col, X_col)
    K_dd = dd_gaussian_kernel(X_col, X_col)
    K_d = d_gaussian_kernel(X_col, X_col)

    # Ky formula from file
    # Ky = K_xx + D_mat @ K_dd @ D_mat - K_d @ D_mat - D_mat @ K_d.T
    # Wait, check if the file used + or -
    # I will test both signatures.
    
    # CASE 1: The correct derivation: K + D K'' D - M D + D M
    # Note: K_d = M.
    # So: K + D K'' D - K_d @ D + D @ K_d
    # And since K_d.T = - K_d
    # D @ K_d = - D @ K_d.T
    # So: K + D K'' D - K_d @ D - D @ K_d.T
    # This matches the "minuses" version.
    
    Ky = K_xx + D_mat @ K_dd @ D_mat - K_d @ D_mat - D_mat @ K_d.T
    
    # Check Symmetry
    print(f"Ky symmetric? {(Ky - Ky.T).abs().max()}")
    
    # Check PSD
    eig = torch.linalg.eigvalsh(Ky)
    print(f"Ky min eig: {eig.min()}")
    
    # Now check full joint matrix of [g(X_col), f(X)]
    # Cross term: K_xs_mod = K_xs + D @ dK_xs
    K_xs = gaussian_kernel(X_col, X)
    dK_xs = d_gaussian_kernel(X_col, X)
    K_xs_mod = K_xs + D_mat @ dK_xs
    
    K_ss = gaussian_kernel(X, X)
    
    # Build huge matrix
    # [ Ky       K_xs_mod ]
    # [ K_xs_mod.T  K_ss  ]
    
    Joint = torch.cat([
        torch.cat([Ky, K_xs_mod], dim=1),
        torch.cat([K_xs_mod.T, K_ss], dim=1)
    ], dim=0)
    
    eig_joint = torch.linalg.eigvalsh(Joint)
    print(f"Joint min eig: {eig_joint.min()}")

    # Calculate conditional variance
    L = torch.linalg.cholesky(Ky + 1e-6 * torch.eye(N))
    v = torch.linalg.solve_triangular(L, K_xs_mod, upper=False)
    sigma_star = K_ss - v.T @ v
    print(f"Min sigma_star: {torch.diag(sigma_star).min()}")
    
    return Joint

print("Checking Correct Formula:")
check_psd()

print("\nChecking Formula with PLUSES (as seen in attachment 1):")
def check_pluses(N=5, M=3):
    X_col = torch.randn(N, 1, dtype=torch.float64, device=device)
    X = torch.randn(M, 1, dtype=torch.float64, device=device)
    delta = torch.randn(N, dtype=torch.float64, device=device) * 0.1
    D_mat = torch.diag(delta)
    K_xx = gaussian_kernel(X_col, X_col)
    K_dd = dd_gaussian_kernel(X_col, X_col)
    K_d = d_gaussian_kernel(X_col, X_col)

    Ky = K_xx + D_mat @ K_dd @ D_mat + K_d @ D_mat + D_mat @ K_d.T
    
    eig = torch.linalg.eigvalsh(Ky)
    print(f"Ky min eig: {eig.min()}")
    
    K_xs = gaussian_kernel(X_col, X)
    dK_xs = d_gaussian_kernel(X_col, X)
    K_xs_mod = K_xs + D_mat @ dK_xs
    K_ss = gaussian_kernel(X, X)
    
    Joint = torch.cat([
        torch.cat([Ky, K_xs_mod], dim=1),
        torch.cat([K_xs_mod.T, K_ss], dim=1)
    ], dim=0)
    
    eig_joint = torch.linalg.eigvalsh(Joint)
    print(f"Joint min eig: {eig_joint.min()}")
    
    # Calculate conditional variance
    try:
        L = torch.linalg.cholesky(Ky + 1e-6 * torch.eye(N))
        v = torch.linalg.solve_triangular(L, K_xs_mod, upper=False)
        sigma_star = K_ss - v.T @ v
        print(f"Min sigma_star: {torch.diag(sigma_star).min()}")
    except:
        print("Cholesky failed")

check_pluses()
