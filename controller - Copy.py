# controller.py
import os
import numpy as np
import pandas as pd
from optimizer import SPSA
from surrogate import EnsembleCalibratedSurrogate
from shot_allocation import allocate_shots_opt
from fidelity import estimate_fidelity_drop

def energy_from_paulis(pauli_vals, coeffs):
    E = 0.0
    for p, c in coeffs.items():
        E += c * pauli_vals[p]
    return float(E)

def compute_sigmaE(coeffs_array, sigma_pauli_array, samplingVar_array=None):
    if samplingVar_array is None:
        samplingVar_array = np.zeros_like(sigma_pauli_array)
    var_terms = (coeffs_array ** 2) * (sigma_pauli_array ** 2 + samplingVar_array)
    return float(np.sqrt(np.sum(var_terms)))

# Try to load pauli_h2 if available
try:
    from pauli_h2 import pauli_list as _pauli_list_gen, coeffs as _coeffs_gen
    maxlen = max(len(p) for p in _pauli_list_gen)
    pauli_list_4q = [p.ljust(maxlen, 'I') for p in _pauli_list_gen]
    coeffs_4q = {p.ljust(maxlen, 'I'): v for p, v in _coeffs_gen.items()}
except Exception:
    pauli_list_4q = ['ZIII', 'IZII', 'IIZI', 'IIIZ']
    coeffs_4q = {'ZIII': -0.5, 'IZII': -0.25, 'IIZI': 0.2, 'IIIZ': -0.1}

def run_aqse(pauli_terms, coeffs_dict, n_qubits=None, iters=40, shots_per_eval=512,
             trust_threshold=0.07, bootstrap_points=6, surrogate_models=6,
             rf_estimators=50, gate_error_two_q=0.02, pruning_threshold=0.04,
             entangler_active=True, seed=0, use_qiskit_backend=None):
    """
    Runs AQSE. Handles bootstrap_points==0 gracefully and keeps parameter dims consistent.
    Returns: (df, summary)
    """
    # --- Backend selection at runtime ---
    if use_qiskit_backend is None:
        use_qiskit_backend = os.getenv('AQSE_USE_QISKIT', '0') == '1'
    measurement_backend = "simulator"
    if use_qiskit_backend:
        try:
            from measurement_qiskit import measure_paulis as qiskit_measure_paulis
            measurement_backend = "qiskit"
        except Exception as e:
            import warnings
            warnings.warn(f"Qiskit backend requested but import failed: {e}. Falling back to simulator.", UserWarning)
            from measurement_nosim import measure_paulis as nosim_measure_paulis
            use_qiskit_backend = False
            measurement_backend = "simulator"
    else:
        from measurement_nosim import measure_paulis as nosim_measure_paulis
        measurement_backend = "simulator"

    # Log which backend is being used
    import warnings
    warnings.warn(f"[AQSE] Measurement backend in use: {measurement_backend}", UserWarning)

    np.random.seed(int(seed))
    pauli_terms = list(pauli_terms)
    if n_qubits is None:
        n_qubits = len(pauli_terms[0])
    n_paulis = len(pauli_terms)
    coeffs_array = np.array([coeffs_dict[p] for p in pauli_terms])
    surrogate = EnsembleCalibratedSurrogate(n_models=surrogate_models, rf_estimators=rf_estimators)
    X_list = []
    Y_list = []
    backend_info = None
    for i in range(int(max(0, bootstrap_points))):
        th = np.random.randn(n_qubits)
        if use_qiskit_backend:
            pvals, backend_info = qiskit_measure_paulis(
                th, pauli_terms, shots_per_term=shots_per_eval,
                entangler=entangler_active, seed=int(seed+i), return_backend_info=True
            )
        else:
            pvals = nosim_measure_paulis(
                th, pauli_terms, shots_per_term=shots_per_eval,
                entangler=entangler_active, seed=int(seed+i)
            )
        X_list.append(th)
        Y_list.append([pvals[p] for p in pauli_terms])
    if len(X_list) > 0:
        X = np.vstack(X_list)
        Y = np.vstack(Y_list)
        surrogate.fit(X, Y)
    else:
        surrogate.is_trained = False
        surrogate.X = None
        surrogate.Y = None
    rows = []
    cumulative_shots = int(max(0, bootstrap_points) * shots_per_eval)
    baseline_shots = iters * shots_per_eval
    prev_energy = None
    optim = SPSA(np.zeros(n_qubits))
    for it in range(1, iters+1):
        def eval_fn(th):
            th = np.asarray(th).reshape(-1)
            if th.size != n_qubits:
                if th.size < n_qubits:
                    th = np.concatenate([th, np.zeros(n_qubits - th.size)])
                else:
                    th = th[:n_qubits]
            mean, std = surrogate.predict_mean_std(th.reshape(1, -1))
            mean = np.asarray(mean).reshape(1, -1)
            std = np.asarray(std).reshape(1, -1)
            sigmaE = compute_sigmaE(coeffs_array, std.flatten())
            if float(sigmaE) < trust_threshold:
                return float((mean.flatten() * coeffs_array).sum())
            if use_qiskit_backend:
                pvals, _ = qiskit_measure_paulis(
                    th, pauli_terms, shots_per_term=shots_per_eval,
                    entangler=entangler_active, seed=int(seed+it), return_backend_info=True
                )
            else:
                pvals = nosim_measure_paulis(
                    th, pauli_terms, shots_per_term=shots_per_eval,
                    entangler=entangler_active, seed=int(seed+it)
                )
            return energy_from_paulis(pvals, coeffs_dict)
        theta, _, _ = optim.step(eval_fn)
        theta = np.asarray(theta).reshape(-1)
        if theta.size != n_qubits:
            if theta.size < n_qubits:
                theta = np.concatenate([theta, np.zeros(n_qubits - theta.size)])
            else:
                theta = theta[:n_qubits]
        mean_vec, std_vec = surrogate.predict_mean_std(theta.reshape(1, -1))
        mean_vec = np.asarray(mean_vec).flatten()
        std_vec = np.asarray(std_vec).flatten()
        if mean_vec.size == 0:
            mean_vec = np.zeros(n_paulis)
        if std_vec.size == 0:
            std_vec = np.full(n_paulis, 0.25)
        samplingVar_default = (1.0 - mean_vec**2) / max(1, shots_per_eval)
        sigmaE = compute_sigmaE(coeffs_array, std_vec, samplingVar_default)
        use_surrogate = float(sigmaE) < trust_threshold
        measured_flag = False
        shots_used = 0
        E_used = None
        if use_surrogate:
            E_used = float((mean_vec * coeffs_array).sum())
        else:
            alloc = allocate_shots_opt(coeffs_array, std_vec, shots_per_eval)
            pvals = {}
            for p, s in zip(pauli_terms, alloc):
                s_int = int(round(float(s)))
                if use_qiskit_backend:
                    res, backend_info = qiskit_measure_paulis(
                        theta, [p], shots_per_term=max(1, s_int),
                        entangler=entangler_active, seed=int(seed+it), return_backend_info=True
                    )
                else:
                    res = nosim_measure_paulis(
                        theta, [p], shots_per_term=max(1, s_int),
                        entangler=entangler_active, seed=int(seed+it)
                    )
                pvals[p] = res[p]
            E_used = energy_from_paulis(pvals, coeffs_dict)
            measured_flag = True
            shots_used = int(sum([int(round(float(x))) for x in alloc]))
            cumulative_shots += shots_used
            X_new = theta.reshape(1, -1)
            Y_new = np.array([[pvals[p] for p in pauli_terms]])
            if surrogate.is_trained and surrogate.X is not None:
                X_cat = np.vstack([surrogate.X, X_new])
                Y_cat = np.vstack([surrogate.Y, Y_new])
            else:
                X_cat = X_new
                Y_cat = Y_new
            surrogate.fit(X_cat, Y_cat)
        n_two_q = 1 if entangler_active else 0
        fidelity_drop = estimate_fidelity_drop(n_two_q, gate_error=gate_error_two_q)
        prune = False
        if prev_energy is not None:
            prune = abs(E_used - prev_energy) < pruning_threshold
        prev_energy = E_used
        rows.append({
            "iter": int(it),
            "theta": theta.tolist(),
            "used_energy": float(E_used) if E_used is not None else None,
            "surrogate_sigmaE": float(sigmaE),
            "used_surrogate": bool(use_surrogate),
            "measured_flag": bool(measured_flag),
            "shots_used_this_iter": int(shots_used),
            "cumulative_shots": int(cumulative_shots),
            "fidelity_drop": float(fidelity_drop),
            "pruned_this_iter": bool(prune),
            "entangler_active": bool(entangler_active)
        })
    print(f"[DEBUG] Number of rows collected: {len(rows)}")
    print(f"[DEBUG] First row: {rows[0] if rows else 'No rows'}")
    df = pd.DataFrame(rows)
    summary = {
        "final_energy": float(df["used_energy"].iloc[-1]) if len(df) > 0 and df["used_energy"].notnull().any() else None,
        "total_shots_baseline": int(baseline_shots),
        "total_shots_aqse": int(cumulative_shots),
        "shots_saved_pct": float(max(0.0, min(100.0, 100.0 * (1 - cumulative_shots / baseline_shots)))) if baseline_shots > 0 else None,
        "backend_info": backend_info  # Add backend info to summary
    }
    return df, summary
