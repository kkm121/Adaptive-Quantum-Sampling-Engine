# AQSE Hackathon Prototype 

This is a ready-to-run Adaptive Quantum Sampling Engine (AQSE) demo optimized for hackathon / presentation.

## What's included
- AQSE controller: SPSA optimizer, surrogate (RF ensemble + calibration), shot allocation, pruning.
- Simulator-only analytic measurement (no Qiskit required). Works fast for up to ~6 qubits.
- Streamlit UI for interactive demos, CSV export.

## Quick setup (recommended - minimal)
1. Create a project directory and paste all files.
2. Create virtualenv & install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Windows: .venv\\Scripts\\activate
   install-numpy,pandas,scikit-lear,matplotlib,streamlit,plotly,qiskit-ibm-runtime,qiskit
