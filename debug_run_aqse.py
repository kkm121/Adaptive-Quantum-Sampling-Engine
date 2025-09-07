# debug_run_aqse.py
import os
import traceback
import sys

# Attempt to import your run function from compare_and_report
try:
    from compare_and_report import run_mode
except Exception:
    try:
        from compare_and_report import main as run_mode
    except Exception:
        run_mode = None

if run_mode is None:
    print("Could not import run_mode from compare_and_report. Adjust import inside debug_run_aqse.py.")
    sys.exit(1)

mode = 'aqse'
seed = 1000
iters = 40
shots = 512
outdir = "results_debug"

print(f"Running debug: mode={mode} seed={seed} iters={iters} shots={shots}")
try:
    df, summary = run_mode(mode, seed, iters, shots, outdir=outdir)
    print("Run completed successfully. Summary:")
    print(summary)
except Exception:
    print("Run raised an exception! Full traceback below:\n")
    traceback.print_exc()
    print("\nEnvironment details:")
    print("IBMQ_INSTANCE=", os.environ.get('IBMQ_INSTANCE'))
    print("QISKIT_BACKEND=", os.environ.get('QISKIT_BACKEND'))
    print("AQSE_USE_QISKIT=", os.environ.get('AQSE_USE_QISKIT'))
    print("QISKIT_IBM_TOKEN set? ", bool(os.environ.get('QISKIT_IBM_TOKEN')))
    sys.exit(1)
