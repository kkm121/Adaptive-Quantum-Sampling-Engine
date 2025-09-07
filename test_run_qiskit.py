# test_run_qiskit.py  (robust + transpile)
import os
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.compiler import transpile

# detect sampler versions
try:
    from qiskit_ibm_runtime import SamplerV2 as SamplerV2
except Exception:
    SamplerV2 = None
try:
    from qiskit_ibm_runtime import Sampler as SamplerV1
except Exception:
    SamplerV1 = None

def resolve_backend_entry(service, entry):
    try:
        if hasattr(entry, "name") and callable(getattr(entry, "name")):
            return entry
    except Exception:
        pass
    if isinstance(entry, str):
        try:
            return service.backend(entry)
        except Exception as e:
            try:
                for cand in service.backends():
                    if isinstance(cand, str) and cand == entry:
                        return service.backend(cand)
                    else:
                        try:
                            if cand.name() == entry:
                                return cand
                        except Exception:
                            continue
            except Exception:
                pass
            raise RuntimeError(f"Could not resolve backend name '{entry}': {e}")
    raise RuntimeError(f"Unrecognized backend entry: {repr(entry)}")

def create_sampler(service, backend):
    if SamplerV2 is not None:
        try:
            return SamplerV2(mode=backend)
        except Exception:
            try:
                return SamplerV2(session=service)
            except Exception:
                pass
    if SamplerV1 is not None:
        try:
            return SamplerV1(backend=backend)
        except Exception:
            try:
                return SamplerV1(session=service, backend=backend)
            except Exception:
                pass
    raise RuntimeError("No compatible Sampler constructor found for installed qiskit-ibm-runtime.")

def main():
    token = os.getenv("QISKIT_IBM_TOKEN")
    instance = os.getenv("IBMQ_INSTANCE", None)
    forced = os.getenv("QISKIT_BACKEND", None)
    if not token:
        raise RuntimeError("QISKIT_IBM_TOKEN not set in environment.")

    svc = None
    for ch in ("ibm_cloud", "ibm_quantum_platform", None):
        try:
            if ch is None:
                svc = QiskitRuntimeService(token=token, instance=instance)
            else:
                svc = QiskitRuntimeService(channel=ch, token=token, instance=instance)
            print("Connected using channel:", ch)
            break
        except Exception as e:
            print("Channel", ch, "failed:", e)
    if svc is None:
        raise RuntimeError("Could not create QiskitRuntimeService")

    raw = svc.backends()
    print("Backends raw entries:", len(raw))
    resolved = []
    for entry in raw[:50]:
        if isinstance(entry, str):
            print(" - (string) ", entry)
            try:
                resolved.append(resolve_backend_entry(svc, entry))
            except Exception as ex:
                print("   -> could not resolve:", ex)
        else:
            try:
                print(" - (object) ", entry.name())
                resolved.append(entry)
            except Exception:
                print(" - (object repr) ", repr(entry))
    backend = None
    if forced:
        try:
            backend = resolve_backend_entry(svc, forced)
        except Exception as e:
            print("Forced backend could not be resolved:", e)
            backend = None
    if backend is None:
        if resolved:
            backend = resolved[0]
        else:
            raise RuntimeError("No backends available to select")
    backend = resolve_backend_entry(svc, backend) if isinstance(backend, str) else backend
    try:
        name = backend.name()
    except Exception:
        name = str(backend)
    try:
        cfg = backend.configuration()
        is_sim = getattr(cfg, "simulator", None)
    except Exception:
        is_sim = None
    print("Selected backend:", name, " simulator=", is_sim)

    sampler = create_sampler(svc, backend)

    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)

    try:
        transpiled = transpile([qc], backend=backend, optimization_level=1)[0]
        print("Successfully transpiled circuit to backend target.")
    except Exception as e:
        print("Transpile failed:", e)
        transpiled = qc

    print("Submitting trivial circuit (256 shots)...")
    job = sampler.run([transpiled], shots=256)
    res = job.result()
    print("Result repr:", repr(res)[:1000])

    try:
        if hasattr(res, "__len__") and len(res) > 0:
            first = res[0]
            if hasattr(first, "get_counts"):
                print("Counts:", first.get_counts())
            else:
                data = getattr(first, "data", None)
                print("First entry data repr (truncated):", repr(data)[:400])
        else:
            if hasattr(res, "get_counts") and callable(res.get_counts):
                print("Counts (top-level):", res.get_counts())
            else:
                print("Result returned but could not extract counts; see repr above.")
    except Exception as e:
        print("Exception while extracting counts:", e)

if __name__ == "__main__":
    main()
