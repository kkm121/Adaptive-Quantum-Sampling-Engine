# test_qiskit_backend.py
import os
from measurement_qiskit import measure_paulis
from pauli_h2 import pauli_list

# Ensure AQSE_USE_QISKIT=1 and QISKIT_IBM_TOKEN are set in your shell before running this.
# Optionally set QISKIT_BACKEND to the name of a real device to force it.

theta = [0.1, 0.2, 0.3, 0.4]  # example angles
shots = 1024

print("Running a quick measurement test...")
res = measure_paulis(theta, pauli_list[:4], shots_per_term=shots, entangler=True, seed=42)
print("Measured expectations:", res)
