# fidelity.py
import numpy as np

def estimate_fidelity_drop(num_two_q_gates, gate_error=0.01, duration_us=1.0, T2=80.0, readout_error=0.02):
    f_gate = np.exp(- num_two_q_gates * gate_error)
    f_time = np.exp(- duration_us / T2)
    f_readout = 1.0 - readout_error
    fidelity = f_gate * f_time * f_readout
    drop = 1.0 - fidelity
    return float(drop)
