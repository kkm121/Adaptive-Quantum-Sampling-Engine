#!/usr/bin/env python3
# compare_and_report.py
# Runs AQSE vs baseline (multi-seed), saves summary + plots.

import os
import argparse
import traceback
import pandas as pd
import matplotlib.pyplot as plt

from controller import run_aqse, pauli_list_4q, coeffs_4q

def run_mode(mode, seed, iters, shots, outdir, use_qiskit_backend=False):
    """
    Run either 'aqse' or 'baseline'.
    Guarantees it returns (df, summary) or raises an exception.
    """
    n_qubits = 4
    gate_err = 0.02
    pruning_threshold = 0.04

    if mode == 'aqse':
        df, summary = run_aqse(
            pauli_terms=pauli_list_4q,
            coeffs_dict=coeffs_4q,
            n_qubits=n_qubits,
            iters=iters,
            shots_per_eval=shots,
            trust_threshold=0.07,
            bootstrap_points=8,
            surrogate_models=6,
            rf_estimators=50,
            gate_error_two_q=gate_err,
            pruning_threshold=pruning_threshold,
            entangler_active=True,
            seed=seed,
            use_qiskit_backend=use_qiskit_backend
        )
        return df, summary

    elif mode == 'baseline':
        df, summary = run_aqse(
            pauli_terms=pauli_list_4q,
            coeffs_dict=coeffs_4q,
            n_qubits=n_qubits,
            iters=iters,
            shots_per_eval=shots,
            trust_threshold=-1.0,   # never trust surrogate -> always measure
            bootstrap_points=1,     # at least one bootstrap sample to avoid vstack errors
            surrogate_models=1,
            rf_estimators=1,
            gate_error_two_q=gate_err,
            pruning_threshold=pruning_threshold,
            entangler_active=True,
            seed=seed,
            use_qiskit_backend=False  # Always use simulator for baseline
        )
        return df, summary
    else:
        raise ValueError(f"Unknown mode: {mode}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n-seeds', type=int, default=3)
    p.add_argument('--iters', type=int, default=40)
    p.add_argument('--shots', type=int, default=512)
    p.add_argument('--outdir', type=str, default='results')
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = []
    for mode in ('aqse','baseline'):
        for s in range(args.n_seeds):
            seed = 1000 + s
            print(f"Running {mode} seed {seed} ...")
            try:
                df, summary = run_mode(mode, seed, args.iters, args.shots, args.outdir)
                rows.append({
                    'mode': mode, 'seed': seed,
                    'final_energy': summary.get('final_energy'),
                    'total_shots_baseline': summary.get('total_shots_baseline'),
                    'total_shots_aqse': summary.get('total_shots_aqse'),
                    'shots_saved_pct': summary.get('shots_saved_pct'),
                    'error': None
                })
            except Exception as e:
                tb = traceback.format_exc()
                print(f"Error running {mode} seed {seed}: {e}\n{tb}")
                rows.append({'mode': mode, 'seed': seed, 'error': str(e),
                             'final_energy': None, 'total_shots_baseline': None,
                             'total_shots_aqse': None, 'shots_saved_pct': None})

    df_all = pd.DataFrame(rows)
    df_all.to_csv(os.path.join(args.outdir, 'summary.csv'), index=False)

    # plots (only when numeric fields exist)
    try:
        plt.figure(figsize=(6,4))
        df_all.boxplot(column='final_energy', by='mode')
        plt.title('Final energy by mode'); plt.suptitle(''); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'final_energy_box.png'))
    except Exception:
        print("Could not create final_energy_box.png (maybe missing numeric data).")

    try:
        plt.figure(figsize=(6,4))
        df_all.boxplot(column='total_shots_aqse', by='mode')
        plt.title('Total shots by mode'); plt.suptitle(''); plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, 'total_shots_box.png'))
    except Exception:
        print("Could not create total_shots_box.png (maybe missing numeric data).")

    print("Done. Summary and plots in", args.outdir)

if __name__ == '__main__':
    main()
