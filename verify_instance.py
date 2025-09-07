# verify_instance.py — robust helper to show what the token/instance can access
import os, traceback
from qiskit_ibm_runtime import QiskitRuntimeService

def try_connect(token, instance):
    tried = []
    for ch in ("ibm_cloud", "ibm_quantum_platform", None):
        try:
            if ch is None:
                svc = QiskitRuntimeService(token=token, instance=instance)
            else:
                svc = QiskitRuntimeService(channel=ch, token=token, instance=instance)
            return svc, ch, None
        except Exception as e:
            tried.append((ch, str(e)))
    return None, None, tried

def main():
    token = os.getenv("QISKIT_IBM_TOKEN")
    instance = os.getenv("IBMQ_INSTANCE", None)
    if not token:
        print("ERROR: QISKIT_IBM_TOKEN not set. Set it and re-run.")
        return

    svc, ch, tried = try_connect(token, instance)
    if svc is None:
        print("Could not create QiskitRuntimeService. Attempts:")
        for c, msg in tried:
            print(f"  channel={c}: {msg}")
        return

    print("Connected using channel:", ch, " instance:", instance)
    try:
        raw = svc.backends()
        print("Raw backend entries returned:", len(raw))
        # print some details; robustly handle strings vs objects
        for entry in raw[:200]:
            if isinstance(entry, str):
                print(" - (string) ", entry)
            else:
                try:
                    cfg = entry.configuration()
                    st = entry.status()
                    print(f" - {entry.name():25s} simulator={getattr(cfg,'simulator',False)} operational={getattr(st,'operational','unknown')} pending={getattr(st,'pending_jobs','unknown')}")
                except Exception:
                    # fallback for weird shapes
                    print(" - (object) repr:", repr(entry)[:200])
    except Exception as e:
        print("Error when calling service.backends():", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
