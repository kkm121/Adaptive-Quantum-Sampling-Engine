# measurement_nosim.py
# Simulator-only n-qubit measurement (analytic statevector + sampling).
import numpy as np

# Pauli matrices
I = np.array([[1,0],[0,1]], complex)
X = np.array([[0,1],[1,0]], complex)
Y = np.array([[0,-1j],[1j,0]], complex)
Z = np.array([[1,0],[0,-1]], complex)

def kron(*mats):
    out = np.array([[1]], complex)
    for m in mats:
        out = np.kron(out, m)
    return out

def RY(theta):
    return np.array([[np.cos(theta/2), -np.sin(theta/2)],
                     [np.sin(theta/2),  np.cos(theta/2)]], complex)

def build_ansatz_state(theta, entangler=True):
    """
    Build n-qubit ansatz statevector.
    - theta: iterable of length n_qubits (one RY per qubit)
    - entangler: apply chain CNOTs only if True
    Returns statevector (dtype=complex).
    """
    theta = np.asarray(theta, dtype=float)
    n = theta.size
    if n < 1:
        raise ValueError("theta must contain at least one parameter (n_qubits >= 1)")

    # initial state as complex for clarity and safety
    state = np.zeros(2**n, dtype=complex)
    state[0] = 1.0

    # apply single-qubit RYs by Kronecker product
    U = kron(*[RY(t) for t in theta])
    state = U @ state

    # apply entangler only if explicitly requested
    if entangler and n >= 2:
        for i in range(n-1):
            P0 = np.array([[1,0],[0,0]], complex)
            P1 = np.array([[0,0],[0,1]], complex)
            ops0 = []
            ops1 = []
            for q in range(n):
                if q == i:
                    ops0.append(P0)
                    ops1.append(P1)
                elif q == i+1:
                    ops0.append(I)
                    ops1.append(X)
                else:
                    ops0.append(I)
                    ops1.append(I)
            U0 = kron(*ops0)
            U1 = kron(*ops1)
            CNOT_full = U0 + U1
            state = CNOT_full @ state

    return state

def density_from_state(state):
    psi = state.reshape(-1,1)
    return psi @ psi.conj().T

def pauli_matrix(pauli_str):
    mapping = {'I': I, 'X': X, 'Y': Y, 'Z': Z}
    if not all(ch in mapping for ch in pauli_str):
        raise ValueError(f"Invalid Pauli string '{pauli_str}'. Allowed chars: I,X,Y,Z")
    mats = [mapping[ch] for ch in pauli_str]
    return kron(*mats)

def measure_paulis(theta, pauli_list, shots_per_term=512, entangler=True, seed=None, depolarize_p=0.0):
    """
    Measure a list of Pauli strings on the ansatz defined by `theta`.
    - Ensures shots_per_term > 0 (otherwise raises ValueError)
    - Ensures every pauli_str length == n_qubits (otherwise raises ValueError)
    - Only applies depolarizing noise if depolarize_p > 0
    - Only applies entangler if entangler is True
    Returns: dict {pauli_str: sampled_expectation}
    """
    # validate inputs
    if not isinstance(shots_per_term, (int, np.integer)) or shots_per_term <= 0:
        raise ValueError(f"shots_per_term must be a positive integer (got {shots_per_term})")

    theta = np.asarray(theta, dtype=float)
    n = theta.size
    if n < 1:
        raise ValueError("theta must have length >= 1 (n_qubits)")

    # ensure all Pauli strings match n_qubits
    for p in pauli_list:
        if len(p) != n:
            raise ValueError(f"Pauli string length mismatch: expected length {n}, but got '{p}' (len={len(p)})")

    rng = np.random.default_rng(seed)

    # build state and optional entangler
    state = build_ansatz_state(theta, entangler=entangler)

    # density matrix
    rho = state.reshape(-1,1) @ state.reshape(1,-1).conj()

    # apply depolarizing noise only if requested
    if depolarize_p is not None and depolarize_p > 0.0:
        dim = rho.shape[0]
        rho = (1.0 - depolarize_p) * rho + depolarize_p * np.eye(dim, dtype=complex) / dim

    results = {}
    for p in pauli_list:
        M = pauli_matrix(p)
        # exact expectation
        exp_exact = float(np.real_if_close(np.trace(rho @ M)))
        # simulate sampling noise (two-outcome binomial)
        p_plus = (1.0 + exp_exact) / 2.0
        # Clamp p_plus to [0, 1] to avoid ValueError
        p_plus = min(max(p_plus, 0.0), 1.0)
        counts_plus = rng.binomial(shots_per_term, p_plus)
        exp_sampled = (counts_plus - (shots_per_term - counts_plus)) / shots_per_term
        results[p] = float(exp_sampled)

    return results
