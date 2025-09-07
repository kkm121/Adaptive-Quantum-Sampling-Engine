# app.py
"""
AQSE demo (mock simulator)
Run: pip install -r requirements.txt
     streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error

# ----------------------------
# Helper: mock physics / generator
# ----------------------------
def ground_truth_energy(molecule: str = "H2"):
    # Simple example ground-truth energies (toy numbers, feel free to adjust)
    gt_map = {
        "H2": -1.137,   # approximate STO-3G H2
        "LiH": -7.882,  # illustrative
        "MockMol": -2.345
    }
    return gt_map.get(molecule, -1.137)

def gen_vqe_curve(gt, iters, a=0.8, k=8.0, noise_level=0.02, seed=0):
    rng = np.random.default_rng(seed)
    xs = np.arange(1, iters+1)
    # Exponential decay + noise
    energies = gt + a * np.exp(-xs / k) + rng.normal(0, noise_level, size=iters)
    return energies

def gen_aqse_curve(gt, iters, a=0.8, k=8.0, noise_level=0.02,
                   surrogate_on=True, pruning_on=True, seed=0):
    rng = np.random.default_rng(seed + 12345)
    xs = np.arange(1, iters+1)
    # Parameters that reflect AVQSE improvements
    # Make AQSE converge faster (bigger decay rate) and lower noise when surrogate/pruning are on
    alpha = 0.75 if surrogate_on else 1.0      # smaller amplitude if surrogate guides better
    beta = 0.7 if surrogate_on else 1.0        # effective faster decay
    noise_factor = 0.6 if surrogate_on else 1.0
    prune_noise_relief = 0.85 if pruning_on else 1.0
    energies = gt + a * alpha * np.exp(-xs / (k * beta)) \
               + rng.normal(0, noise_level * noise_factor * prune_noise_relief, size=iters)
    # small deterministic smoothing to look nicer
    energies = pd.Series(energies).rolling(3, min_periods=1).mean().values
    return energies

# ----------------------------
# Shot logic (mock)
# ----------------------------
def simulate_measurement_decisions(iters, shots_per_eval, surrogate_on, trust_threshold,
                                   base_surrogate_std, noise_level, seed=0, pruning_on=False):
    rng = np.random.default_rng(seed + 987)
    decisions = []
    cumulative_shots = 0
    # surrogate_std decays as iterations increase (simulating learning)
    for i in range(1, iters+1):
        # surrogate uncertainty model (toy): base * exp(-i/decay) + noise
        decay = 12.0
        surrogate_std = base_surrogate_std * np.exp(-i / decay) + 0.002 * rng.random()
        # decide: if surrogate_on and surrogate_std < trust_threshold -> skip measurement
        if surrogate_on and surrogate_std < trust_threshold:
            measured = False
            shots_used = 0
        else:
            measured = True
            # pruning reduces shots (circuit shorter -> fewer logical "shots" effort)
            prune_factor = 0.8 if pruning_on else 1.0
            shots_used = int(max(1, round(shots_per_eval * prune_factor)))
        cumulative_shots += shots_used
        decisions.append({
            "iter": i,
            "surrogate_std": surrogate_std,
            "measured": measured,
            "shots_used": shots_used,
            "cumulative_shots": cumulative_shots
        })
    return pd.DataFrame(decisions)

# ----------------------------
# Compute metrics & assemble run
# ----------------------------
def run_demo(molecule, iters, shots_per_eval, noise_level,
             surrogate_on, pruning_on, trust_threshold, seed=0):
    gt = ground_truth_energy(molecule)
    # realistic-looking constants
    a = 0.9
    k = max(4.0, iters / 6.0)
    base_surrogate_std = 0.25 * noise_level * 5.0  # scale relative to noise
    
    vqe = gen_vqe_curve(gt, iters, a=a, k=k, noise_level=noise_level, seed=seed)
    aqse = gen_aqse_curve(gt, iters, a=a, k=k, noise_level=noise_level,
                          surrogate_on=surrogate_on, pruning_on=pruning_on, seed=seed)
    
    decision_df = simulate_measurement_decisions(iters, shots_per_eval, surrogate_on, trust_threshold,
                                                 base_surrogate_std, noise_level, seed=seed, pruning_on=pruning_on)
    # Build points: if measured -> measured energy from AQSE curve (simulate noise further),
    # if not measured -> surrogate predicted = aqse value + small predictive noise
    rng = np.random.default_rng(seed + 42)
    predicted = []
    measured_vals = []
    used_avqse = []
    for idx, row in decision_df.iterrows():
        i = int(row["iter"]) - 1
        if row["measured"]:
            # "Quantum measurement" gives aqse[i] plus some measurement sampling noise
            meas_noise = rng.normal(0, noise_level * 0.6)
            val = aqse[i] + meas_noise
            measured_vals.append(val)
            predicted.append(np.nan)
            used_avqse.append(val)
        else:
            # surrogate predicted (we show predicted vs later measured maybe)
            pred_noise = rng.normal(0, max(1e-4, row["surrogate_std"]*0.6))
            pred_val = aqse[i] + pred_noise  # surrogate tends to approximate AQSE
            predicted.append(pred_val)
            measured_vals.append(np.nan)
            used_avqse.append(pred_val)
    decision_df["predicted_val"] = predicted
    decision_df["measured_val"] = measured_vals
    decision_df["used_val"] = used_avqse
    
    # Compose iteration-level dataframe
    df = pd.DataFrame({
        "iter": np.arange(1, iters+1),
        "vqe": vqe,
        "aqse_oracle": aqse,
        "used_energy": decision_df["used_val"].values,
        "measured_flag": decision_df["measured"].values,
        "shots_this_iter": decision_df["shots_used"].values,
        "cumulative_shots": decision_df["cumulative_shots"].values,
        "surrogate_std": decision_df["surrogate_std"].values,
        "predicted_val": decision_df["predicted_val"].values,
        "measured_val": decision_df["measured_val"].values
    })
    
    # summary metrics
    final_vqe = float(df["vqe"].iloc[-1])
    final_aqse = float(df["used_energy"].iloc[-1])
    total_shots_vqe = shots_per_eval * iters
    total_shots_aqse = int(df["cumulative_shots"].iloc[-1])
    shots_saved_pct = 100.0 * (1 - total_shots_aqse / max(1, total_shots_vqe))
    mae_surrogate = np.nan
    if df["measured_val"].notna().sum() > 0 and df["predicted_val"].notna().sum() > 0:
        # compare predicted points to measured points where both exist later (toy)
        # For demo, compute MAE between measured_val (non-nan) and oracle aqse at those indices
        measured_idx = df["measured_val"].dropna().index
        # surrogate MAE using predicted where available vs oracle
        pred_idx = df["predicted_val"].dropna().index
        # approximate: compute MAE(predicted_val at pred_idx vs aqse at pred_idx)
        if len(pred_idx) > 0:
            mae_surrogate = mean_absolute_error(df.loc[pred_idx, "aqse_oracle"], df.loc[pred_idx, "predicted_val"])
    else:
        mae_surrogate = None

    summary = {
        "gt": gt,
        "final_vqe": final_vqe,
        "final_aqse": final_aqse,
        "total_shots_vqe": total_shots_vqe,
        "total_shots_aqse": total_shots_aqse,
        "shots_saved_pct": shots_saved_pct,
        "mae_surrogate": mae_surrogate
    }
    return df, summary

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AQSE Demo", layout="wide", initial_sidebar_state="expanded")
st.title("AQSE — Adaptive Quantum Sampling Engine (Demo)")
st.markdown(
    """
Interactive demo (mock/simulated) showing Baseline VQE vs AQSE (surrogate + pruning).
This is a frontend-ready demo to visualize how AQSE reduces quantum measurements while preserving energy accuracy.
"""
)

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    molecule = st.selectbox("Molecule", options=["H2", "LiH", "MockMol"], index=0)
    iters = st.slider("Iterations (optimizer steps)", min_value=8, max_value=120, value=40, step=4)
    shots_per_eval = st.selectbox("Shots per evaluation (baseline)", options=[128, 256, 512, 1024], index=2)
    noise_level = st.slider("Noise level (simulated)", min_value=0.0, max_value=0.12, value=0.03, step=0.01)
    surrogate_on = st.checkbox("Use Surrogate (AQSE)", value=True)
    pruning_on = st.checkbox("Use Pruning (AQSE)", value=True)
    trust_threshold = st.slider("Surrogate trust threshold (smaller=trust more)", min_value=0.01, max_value=0.5, value=0.06, step=0.01)
    seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    run_button = st.button("Run Demo")

# Prefill run on load
if "last_run" not in st.session_state:
    st.session_state.last_run = None

if run_button or st.session_state.last_run is None:
    df, summary = run_demo(molecule, iters, shots_per_eval, noise_level,
                           surrogate_on, pruning_on, trust_threshold, seed=seed)
    st.session_state.last_run = (df, summary, {"molecule": molecule, "iters": iters,
                                               "shots_per_eval": shots_per_eval,
                                               "noise_level": noise_level,
                                               "surrogate_on": surrogate_on,
                                               "pruning_on": pruning_on,
                                               "trust_threshold": trust_threshold,
                                               "seed": seed})
else:
    df, summary, params = st.session_state.last_run

# Top metrics cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Ground truth (GT)", f"{summary['gt']:.6f} Hartree")
col2.metric("Final energy (Baseline VQE)", f"{summary['final_vqe']:.6f} Hartree")
col3.metric("Final energy (AQSE)", f"{summary['final_aqse']:.6f} Hartree")
col4.metric("Shots saved", f"{summary['shots_saved_pct']:.1f}%")

st.markdown("---")

# Main charts: Energy vs Iteration, Cumulative shots, Surrogate scatter
left, right = st.columns([3, 2])

# Energy vs Iteration
with left:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["iter"], y=df["vqe"], mode="lines+markers", name="Baseline VQE",
                             line=dict(color="royalblue")))
    fig.add_trace(go.Scatter(x=df["iter"], y=df["used_energy"], mode="lines+markers", name="AQSE (used values)",
                             line=dict(color="orange")))
    # ground truth line
    fig.add_trace(go.Scatter(x=[df["iter"].min(), df["iter"].max()], y=[summary["gt"], summary["gt"]],
                             mode="lines", name="Ground truth", line=dict(color="green", dash="dash")))
    fig.update_layout(title="Energy vs Iteration", xaxis_title="Iteration", yaxis_title="Energy (Hartree)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

# Cumulative shots
with right:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df["iter"], y=np.arange(1, len(df)+1) * shots_per_eval,
                              mode="lines", name="Baseline cumulative shots",
                              line=dict(color="royalblue")))
    fig2.add_trace(go.Scatter(x=df["iter"], y=df["cumulative_shots"],
                              mode="lines", name="AQSE cumulative shots",
                              line=dict(color="orange")))
    fig2.update_layout(title="Cumulative Shots vs Iteration", xaxis_title="Iteration", yaxis_title="Cumulative shots")
    st.plotly_chart(fig2, use_container_width=True)

    # Surrogate diagnostics: small scatter predicted vs measured if available
    measured_points = df[df["measured_val"].notna()]
    predicted_points = df[df["predicted_val"].notna()]
    # show small table
    st.markdown("**Surrogate diagnostics**")
    if len(predicted_points) > 0:
        mae = mean_absolute_error(predicted_points["aqse_oracle"], predicted_points["predicted_val"])
        st.write(f"Surrogate MAE (on predicted points): {mae:.6f} Hartree")
    else:
        st.write("Surrogate made no predictions (threshold too strict)")

st.markdown("---")

# Surrogate vs measured scatter (wide)
st.subheader("Predicted vs Measured (surrogate reliability)")
scatter_df = df.copy()
scatter_df_plot = scatter_df[["iter", "predicted_val", "measured_val", "aqse_oracle"]].melt(id_vars=["iter", "aqse_oracle"],
                                                                                             value_vars=["predicted_val", "measured_val"],
                                                                                             var_name="type", value_name="value").dropna()
if not scatter_df_plot.empty:
    fig3 = px.scatter(scatter_df_plot, x="aqse_oracle", y="value", color="type",
                      title="Predicted / Measured vs Oracle AQSE", labels={"aqse_oracle": "AQSE Oracle", "value": "Predicted/Measured"})
    fig3.add_shape(type="line", x0=scatter_df_plot["aqse_oracle"].min(), y0=scatter_df_plot["aqse_oracle"].min(),
                   x1=scatter_df_plot["aqse_oracle"].max(), y1=scatter_df_plot["aqse_oracle"].max(),
                   line=dict(color="black", dash="dash"))
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("No predicted/measured pairs to show (try different trust threshold or toggles).")

st.markdown("---")

# Show underlying table & download
st.subheader("Iteration table (sample)")
st.dataframe(df.head(60))

st.download_button("Download results CSV", df.to_csv(index=False).encode('utf-8'), "aqse_results.csv")

st.markdown("### Notes for judges")
st.write(
    """
    This is a **mock/simulated demo** meant to visualize behavior of AQSE vs baseline VQE.
    - The demo generates believable curves and decisions so you can show how surrogate/trimming reduce measurements.
    - For real experiments swap the mock generator for real QNode evaluations and plug in measured Pauli expectations and calibration data.
    """
)
