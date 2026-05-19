# utils/metrics.py
import numpy as np
import torch

def calculate_mae(y_true, y_pred):

    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true, y_pred):

    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_crps_gaussian(y_true, mu, var):

    from scipy.stats import norm
    std = np.sqrt(np.clip(var, 1e-6, None))
    z = (y_true - mu) / std
    
    # CRPS analytical formula for Gaussian
    crps = std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return np.mean(crps)

def calculate_mace(y_true, mu, var, num_bins=10):

    std = np.sqrt(np.clip(var, 1e-6, None))
    expected_p = np.linspace(0.1, 0.9, num_bins)
    observed_p = []
    
    from scipy.stats import norm
    for p in expected_p:
        # Calculate the theoretical bound for probability p
        bound = norm.ppf((1 + p) / 2)
        lower = mu - bound * std
        upper = mu + bound * std
        
        # Calculate empirical coverage
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        observed_p.append(coverage)
        
    mace = np.mean(np.abs(expected_p - np.array(observed_p)))
    return mace