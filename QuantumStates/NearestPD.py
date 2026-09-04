# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 19:35:09 2018

@author: Simon
"""
#%%

import numpy as np
from numpy import linalg as la


import numpy as np
from scipy.optimize import minimize

def nearestPD_old(A):
    """Find the nearest positive-definite matrix to input"""
    d = np.shape(A)[0]
    a = 0  #accumulator for the negative eigenvalues
    k = 0 #iterator on the eigenvalues
    (eigvals,P) = la.eigh(A) #A = P*eigvals*P'
    eigvals = np.real(eigvals)
    while eigvals[k] + a/(d - k) < 0:
        a += eigvals[k]
        eigvals[k] = 0
        k += 1
    eigvals[k:] += (a/(d - k)) * np.ones((d - k,))
    A1 = np.dot(np.dot(P,np.diag(eigvals)),la.inv(P))
    return A1/np.trace(A1)

def nearestPD(rho_pseudo):
    dim = rho_pseudo.shape[0]
    
    # Helper: Reconstruct rho from a parameter vector 'x'
    # x contains real and imag parts of the lower triangular T
    def params_to_rho(x):
        # Reshape vector back to matrices
        size = dim * dim
        T_real = x[:size].reshape((dim, dim))
        T_imag = x[size:].reshape((dim, dim))
        
        # Enforce lower triangular
        mask = np.tril(np.ones((dim, dim)))
        T = (T_real + 1j * T_imag) * mask
        
        TTd = T @ T.conj().T
        # Avoid division by zero
        tr = np.trace(TTd).real
        if tr < 1e-9: tr = 1.0 
        return TTd / tr

    # Objective Function: || rho(x) - rho_pseudo ||^2
    def objective(x):
        rho_est = params_to_rho(x)
        diff = rho_est - rho_pseudo
        # Frobenius norm squared
        return np.real(np.trace(diff.conj().T @ diff))

    # Initial guess: Identity matrix
    x0_real = np.eye(dim).flatten()
    x0_imag = np.zeros((dim, dim)).flatten()
    x0 = np.concatenate([x0_real, x0_imag])

    # Optimize using L-BFGS-B (good for smooth functions)
    res = minimize(objective, x0, method='L-BFGS-B')
    
    return params_to_rho(res.x)

# Example Usage
# rho_final = closest_psd_cholesky_scipy(rho_pseudo)
