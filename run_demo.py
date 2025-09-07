# run_demo.py
from controller import run_aqse, pauli_list_4q, coeffs_4q
if __name__ == "__main__":
    df, summary = run_aqse(pauli_list_4q, coeffs_4q, iters=40, shots_per_eval=512, seed=1)
    df.to_csv("aqse_demo_results.csv", index=False)
    print("Saved aqse_demo_results.csv")
    print(summary)
