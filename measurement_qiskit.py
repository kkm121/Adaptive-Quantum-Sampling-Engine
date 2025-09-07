# measurement_qiskit.py
"""
Robust Qiskit measurement helper.
- Transpiles circuits to the selected backend target before submitting to Sampler.
- Robustly extracts counts (handles counts, memory lists, probability arrays, and DataBin/BitArray).
- Prints a single debug repr if extraction fails so logs aren't spammy.
"""

import os
import warnings
import numpy as np
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.compiler import transpile

# Try both Sampler versions
try:
    from qiskit_ibm_runtime import SamplerV2 as SamplerV2
except Exception:
    SamplerV2 = None
try:
    from qiskit_ibm_runtime import Sampler as SamplerV1
except Exception:
    SamplerV1 = None

LAST_QISKIT_BACKEND_INFO = {"name": None, "is_simulator": None}


def _create_sampler(service, backend):
    """Create a Sampler instance tolerant to different runtime versions."""
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
    raise RuntimeError("Could not construct a Sampler for installed qiskit-ibm-runtime.")


def _resolve_backend_entry(service, entry):
    """Resolve a service.backends() entry or a string into a backend object."""
    try:
        if hasattr(entry, "name") and callable(getattr(entry, "name")):
            return entry
    except Exception:
        pass
    if isinstance(entry, str):
        try:
            return service.backend(entry)
        except Exception:
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
            raise RuntimeError(f"Could not resolve backend name '{entry}'")
    raise RuntimeError(f"Unrecognized backend entry type: {type(entry)}")


def _get_service_and_backend():
    token = os.getenv("QISKIT_IBM_TOKEN", None)
    if not token:
        raise RuntimeError("QISKIT_IBM_TOKEN not set; cannot use real backend.")
    instance = os.getenv("IBMQ_INSTANCE", None)
    svc = None
    tried = []
    for ch in ("ibm_cloud", "ibm_quantum_platform", None):
        try:
            if ch is None:
                svc = QiskitRuntimeService(token=token, instance=instance)
            else:
                svc = QiskitRuntimeService(channel=ch, token=token, instance=instance)
            break
        except Exception as e:
            tried.append((ch, str(e)))
            svc = None
    if svc is None:
        msg = "Failed to create QiskitRuntimeService. Attempts:\n" + "\n".join(f" channel={c}: {m}" for c, m in tried)
        raise RuntimeError(msg)

    backend_name_env = os.getenv("QISKIT_BACKEND", None)
    backend_obj = None
    if backend_name_env:
        try:
            backend_obj = _resolve_backend_entry(svc, backend_name_env)
        except Exception as e:
            print(f"[measurement_qiskit] Requested backend '{backend_name_env}' not resolved: {e}. Will try auto-selection.")

    if backend_obj is None:
        try:
            # Try real devices first
            raw_list = list(svc.backends(simulator=False, operational=True))
            candidates = []
            for ent in raw_list:
                try:
                    candidates.append(_resolve_backend_entry(svc, ent))
                except Exception:
                    continue
            # Fallback: try simulators if no real devices
            if not candidates:
                raw_sim = list(svc.backends(simulator=True, operational=True))
                for ent in raw_sim:
                    try:
                        candidates.append(_resolve_backend_entry(svc, ent))
                    except Exception:
                        continue
            # Fallback: try any backend
            if not candidates:
                raw_all = list(svc.backends())
                for ent in raw_all:
                    try:
                        candidates.append(_resolve_backend_entry(svc, ent))
                    except Exception:
                        continue
            if not candidates:
                raise RuntimeError("No backends (real or simulator) returned from QiskitRuntimeService(). Check your IBM Quantum account, API key, and instance access.")
            try:
                candidates.sort(key=lambda b: getattr(b.status(), "pending_jobs", 0))
            except Exception:
                pass
            backend_obj = candidates[0]
        except Exception as e:
            raise RuntimeError(f"Auto-selection of backend failed: {e}")

    try:
        cfg = backend_obj.configuration()
        is_sim = bool(getattr(cfg, "simulator", False))
    except Exception:
        is_sim = False

    try:
        backend_name = backend_obj.name()
    except Exception:
        backend_name = getattr(backend_obj, "name", None) or str(backend_obj)

    print(f"[measurement_qiskit] Using backend: {backend_name}  simulator={is_sim}")
    global LAST_QISKIT_BACKEND_INFO
    LAST_QISKIT_BACKEND_INFO = {"name": backend_name, "is_simulator": is_sim}
    return svc, backend_obj


# ----------------- circuit helpers -----------------
def _ansatz(theta, n_qubits=4, entangler=True):
    qc = QuantumCircuit(n_qubits)
    for i, t in enumerate(theta):
        qc.ry(float(t), i % n_qubits)
    if entangler and n_qubits >= 2:
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
    return qc


def _pauli_to_measure_circuit(p, n_qubits=4):
    qc = QuantumCircuit(n_qubits, n_qubits)
    for i, ch in enumerate(p):
        if ch == 'X':
            qc.h(i)
        elif ch == 'Y':
            qc.sdg(i); qc.h(i)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def _expectation_from_counts(p, counts, shots):
    if not counts or shots <= 0:
        return 0.0
    exp = 0.0
    for bitstr, c in counts.items():
        bstr = str(bitstr)
        bits = bstr[::-1]
        val = 1.0
        for i, ch in enumerate(p):
            if ch == 'I':
                continue
            if i >= len(bits):
                bit = 0
            else:
                try:
                    bit = int(bits[i])
                except Exception:
                    bit = 0
            v = -1.0 if bit == 1 else 1.0
            val *= v
        exp += val * (c / shots)
    return float(exp)


# ----------------- helpers to convert alternate result shapes -----------------
def _probs_to_counts(prob_array, n_qubits, shots):
    """Convert probability-like array of length 2**n_qubits into counts dict."""
    counts = {}
    try:
        for idx, p in enumerate(prob_array):
            if p is None:
                continue
            p = float(p)
            if p <= 0.0:
                continue
            c = int(round(p * shots))
            if c <= 0:
                continue
            b = format(idx, 'b').zfill(n_qubits)[::-1]
            counts[b] = counts.get(b, 0) + c
    except Exception:
        return {}
    return counts


def _databin_to_counts(data, n_qubits, shots):
    """
    Attempt to decode DataBin / BitArray-like objects returned by primitives.
    Tries:
      - .get_counts()
      - .get_bitstrings()
      - .get_int_counts()
      - numpy array shapes (shots x n_qubits) or flat arrays
      - to01() / tobytes() / tolist() fallbacks
    Returns counts dict mapping little-endian bitstrings -> integer counts.
    """
    try:
        c = getattr(data, "c", data)
    except Exception:
        c = data

    # 1) direct get_counts() if available
    try:
        if hasattr(c, "get_counts") and callable(getattr(c, "get_counts")):
            counts = c.get_counts()
            if isinstance(counts, dict) and counts:
                return counts
    except Exception:
        pass

    # 2) get_bitstrings() -> list of bitstrings (one per shot)
    try:
        if hasattr(c, "get_bitstrings") and callable(getattr(c, "get_bitstrings")):
            bitlist = c.get_bitstrings()
            bitlist = list(bitlist)
            if bitlist:
                counts = {}
                for b in bitlist:
                    s = str(b)
                    counts[s] = counts.get(s, 0) + 1
                return counts
    except Exception:
        pass

    # 3) get_int_counts() -> dict(int->count)
    try:
        if hasattr(c, "get_int_counts") and callable(getattr(c, "get_int_counts")):
            int_counts = c.get_int_counts()
            if isinstance(int_counts, dict) and int_counts:
                counts = {}
                for intval, cnt in int_counts.items():
                    try:
                        key = format(int(intval), 'b').zfill(n_qubits)[::-1]
                    except Exception:
                        key = str(intval)
                    counts[key] = counts.get(key, 0) + int(cnt)
                return counts
    except Exception:
        pass

    # 4) numpy-friendly: array-like 2D (shots x n_qubits) or flat array length shots*n_qubits
    try:
        import numpy as _np
        arr = _np.asarray(c)
        if arr.size:
            if arr.ndim == 2 and (arr.shape[1] == n_qubits or arr.shape[0] == n_qubits):
                if arr.shape[1] == n_qubits:
                    rows = arr
                else:
                    rows = arr.T
                counts = {}
                for row in rows:
                    bits = "".join(str(int(x)) for x in row)[::-1]
                    counts[bits] = counts.get(bits, 0) + 1
                return counts
            if arr.ndim == 1 and arr.size >= (n_qubits):
                L = arr.size
                if L >= shots * n_qubits:
                    L_use = shots * n_qubits
                else:
                    L_use = L - (L % n_qubits)
                counts = {}
                for i in range(0, L_use, n_qubits):
                    seg = arr[i:i + n_qubits]
                    bits = "".join(str(int(x)) for x in seg)[::-1]
                    counts[bits] = counts.get(bits, 0) + 1
                if counts:
                    return counts
    except Exception:
        pass

    # 5) to01()/tobytes()/tobytes()+bit-chunking fallback
    try:
        if hasattr(c, "to01") and callable(getattr(c, "to01")):
            bitstr = c.to01()
            if bitstr:
                L = len(bitstr)
                expected = shots * n_qubits
                counts = {}
                if L >= expected:
                    for i in range(0, expected, n_qubits):
                        seg = bitstr[i:i + n_qubits]
                        counts[seg[::-1]] = counts.get(seg[::-1], 0) + 1
                else:
                    for i in range(0, L, n_qubits):
                        seg = bitstr[i:i + n_qubits]
                        counts[seg[::-1]] = counts.get(seg[::-1], 0) + 1
                if counts:
                    return counts
        if hasattr(c, "tobytes") and callable(getattr(c, "tobytes")):
            b = c.tobytes()
            bitstr = "".join(f"{byte:08b}" for byte in b)
            if bitstr:
                L = len(bitstr)
                expected = shots * n_qubits
                counts = {}
                if L >= expected:
                    for i in range(0, expected, n_qubits):
                        seg = bitstr[i:i + n_qubits]
                        counts[seg[::-1]] = counts.get(seg[::-1], 0) + 1
                else:
                    for i in range(0, L, n_qubits):
                        seg = bitstr[i:i + n_qubits]
                        counts[seg[::-1]] = counts.get(seg[::-1], 0) + 1
                if counts:
                    return counts
    except Exception:
        pass

    # 6) list/iterable fallback
    try:
        if hasattr(c, "__iter__"):
            lst = list(c)
            if lst:
                if all(isinstance(x, (int, bool)) for x in lst[:min(20, len(lst))]):
                    counts = {}
                    L = len(lst)
                    if L >= shots * n_qubits:
                        L_use = shots * n_qubits
                    else:
                        L_use = L - (L % n_qubits)
                    for i in range(0, L_use, n_qubits):
                        seg = lst[i:i + n_qubits]
                        bits = "".join(str(int(x)) for x in seg)[::-1]
                        counts[bits] = counts.get(bits, 0) + 1
                    if counts:
                        return counts
                if all(isinstance(x, str) and set(x) <= {"0", "1"} for x in lst[:min(20, len(lst))]):
                    counts = {}
                    for s in lst:
                        counts[s] = counts.get(s, 0) + 1
                    return counts
    except Exception:
        pass

    # Nothing decoded: print small helpful diagnostic and return {}
    try:
        sample_attrs = [a for a in dir(c) if not a.startswith("_")]
    except Exception:
        sample_attrs = None
    print("[measurement_qiskit] _databin_to_counts: unable to decode DataBin/BitArray payload. "
          "num_shots=", shots, " num_bits=", n_qubits, " sample_attrs=", sample_attrs[:40] if sample_attrs else None)
    return {}


def _try_convert_result_to_counts(res_entry, n_qubits, shots):
    """Try to find counts/probabilities in many common result shapes."""
    data = getattr(res_entry, "data", None)
    if isinstance(data, dict):
        for key in ("counts", "value_counts", "measurement_counts"):
            if key in data and isinstance(data[key], dict):
                return data[key]
        for key in ("probabilities", "probs", "prob", "quasi_dists", "values"):
            if key in data:
                val = data[key]
                if isinstance(val, dict):
                    return val
                if hasattr(val, "__len__"):
                    return _probs_to_counts(val, n_qubits, shots)

    if data is not None:
        try:
            if hasattr(data, "c") or hasattr(data, "num_shots") or hasattr(data, "num_bits"):
                conv = _databin_to_counts(data, n_qubits, shots)
                if conv:
                    return conv
        except Exception:
            pass

    mem = getattr(res_entry, "memory", None)
    if isinstance(mem, (list, tuple)):
        counts = {}
        for b in mem:
            s = str(b)
            counts[s] = counts.get(s, 0) + 1
        return counts

    for attr in ("counts", "value_counts", "probabilities", "probs", "quasi_dists"):
        val = getattr(res_entry, attr, None)
        if val is None:
            continue
        if isinstance(val, dict):
            return val
        if hasattr(val, "__len__"):
            try:
                candidate = val[0] if hasattr(val[0], "__len__") else val
            except Exception:
                candidate = val
            return _probs_to_counts(candidate, n_qubits, shots)

    if isinstance(res_entry, dict):
        for key in ("counts", "probabilities", "probs", "value_counts"):
            if key in res_entry:
                v = res_entry[key]
                if isinstance(v, dict):
                    return v
                if hasattr(v, "__len__"):
                    return _probs_to_counts(v, n_qubits, shots)

    return {}


def _extract_counts_from_result(result_entry):
    """Existing best-effort extractor (keeps compatibility with older shapes)."""
    try:
        if hasattr(result_entry, "get_counts") and callable(result_entry.get_counts):
            return result_entry.get_counts()
    except Exception:
        pass
    try:
        data = getattr(result_entry, "data", None)
        if data is not None:
            meas = getattr(data, "meas", None)
            if meas is not None:
                if hasattr(meas, "get_counts") and callable(meas.get_counts):
                    return meas.get_counts()
                if isinstance(meas, dict) and "counts" in meas:
                    return meas["counts"]
            if hasattr(data, "get_counts") and callable(data.get_counts):
                return data.get_counts()
            if isinstance(data, dict):
                for key in ("counts", "meas", "value_counts"):
                    if key in data and isinstance(data[key], dict):
                        return data[key]
    except Exception:
        pass

    try:
        if isinstance(result_entry, dict):
            for key in ("counts", "value_counts", "measurement_counts"):
                if key in result_entry and isinstance(result_entry[key], dict):
                    return result_entry[key]
    except Exception:
        pass

    return {}


# ----------------- main measurement function -----------------
def measure_paulis(theta, pauli_terms, shots_per_term=1024, entangler=True, seed=None, return_backend_info=False):
    """
    Measure Pauli strings using Qiskit runtime sampler (or device).
    Returns dict {pauli_str: expectation_estimate}.
    """
    # Create service/backend and sampler once
    service, backend = _get_service_and_backend()
    sampler = _create_sampler(service, backend)

    # Prepare circuits
    theta = np.asarray(theta, dtype=float)
    if not pauli_terms:
        return {}
    n_qubits = len(pauli_terms[0])
    prep = _ansatz(theta, n_qubits=n_qubits, entangler=entangler)

    # Print backend info just once
    try:
        backend_name = backend.name() if hasattr(backend, "name") and callable(getattr(backend, "name")) else str(backend)
    except Exception:
        backend_name = str(backend)
    try:
        cfg_sim = getattr(getattr(backend, 'configuration', lambda: None)(), 'simulator', None)
    except Exception:
        cfg_sim = None
    print(f"[measurement_qiskit] Using backend: {backend_name}  simulator={cfg_sim}")

    results = {}
    printed_repr_debug = False

    for p in pauli_terms:
        meas = _pauli_to_measure_circuit(p, n_qubits=n_qubits)
        qc = prep.compose(meas, front=True)

        # Transpile to backend target
        try:
            transpiled_list = transpile([qc], backend=backend, optimization_level=1)
            transpiled = transpiled_list[0]
        except Exception as e:
            warnings.warn(f"[measurement_qiskit] transpile failed: {e}. Submitting original circuit (may be rejected).", UserWarning)
            transpiled = qc

        shots = int(os.getenv("QISKIT_SHOTS", str(shots_per_term)))
        try:
            job = sampler.run([transpiled], shots=shots)
            res = job.result()
        except Exception as e:
            print(f"[ERROR] Sampler job failed for Pauli {p}: {e}")
            results[p] = 0.0
            continue

        # Try extracting counts with existing extractor (fast)
        counts = {}
        try:
            if hasattr(res, "__len__") and len(res) > 0:
                counts = _extract_counts_from_result(res[0]) or {}
            else:
                counts = _extract_counts_from_result(res) or {}
        except Exception:
            counts = {}

        # If empty, attempt conversions (prob arrays, memory lists, DataBin, etc.)
        if not counts:
            if not printed_repr_debug:
                print("[measurement_qiskit] debug: result repr (truncated):", repr(res)[:2000])
                printed_repr_debug = True
            try:
                if hasattr(res, "__len__") and len(res) > 0:
                    counts = _try_convert_result_to_counts(res[0], n_qubits, shots) or {}
                else:
                    counts = _try_convert_result_to_counts(res, n_qubits, shots) or {}
            except Exception:
                counts = {}

        # Last resort: top-level get_counts
        if not counts:
            try:
                if hasattr(res, "get_counts") and callable(res.get_counts):
                    counts = res.get_counts() or {}
            except Exception:
                counts = {}

        if not counts:
            print(f"[ERROR] No counts returned for Pauli {p}. Returning expectation=0.0.")
            warnings.warn("[measurement_qiskit] Could not extract counts. Returning 0.0 for this Pauli.", UserWarning)

        results[p] = _expectation_from_counts(p, counts, shots)
        print(f"[DEBUG] Pauli: {p}, Counts: {counts}, Expectation: {results[p]}")

    if return_backend_info:
        return results, dict(LAST_QISKIT_BACKEND_INFO)
    return results
