# shot_allocation.py
import numpy as np

def allocate_shots_opt(coeffs, sigma_pauli, total_shots, min_shot=1):
    coeffs = np.asarray(coeffs, dtype=float)
    sigma_pauli = np.asarray(sigma_pauli, dtype=float)
    n = len(coeffs)
    weights = np.abs(coeffs) * (sigma_pauli + 1e-12)
    if weights.sum() <= 0:
        alloc = np.ones(n, dtype=int) * (total_shots // n)
    else:
        frac = weights / weights.sum()
        alloc = np.maximum(min_shot, np.round(frac * total_shots).astype(int))
    diff = int(total_shots - alloc.sum())
    i = 0
    while diff != 0:
        idx = i % n
        alloc[idx] += 1 if diff > 0 else -1
        diff = int(total_shots - alloc.sum()); i += 1
    return alloc
