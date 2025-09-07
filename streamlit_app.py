# streamlit_app.py
import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from controller import run_aqse, pauli_list_4q, coeffs_4q
from compare_and_report import run_mode
import matplotlib.pyplot as plt

st.set_page_config(page_title="AQSE Hackathon", layout="wide")
st.title("AQSE — Adaptive Quantum Sampling Engine")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

with st.sidebar:
    st.header("Run settings")
    use_real_backend = st.checkbox("Use real IBM Quantum backend (enter token below)", value=False)
    ibmq_token = ""
    if use_real_backend:
        ibmq_token = st.text_input("IBM Quantum API token (QISKIT_IBM_TOKEN)", type="password")
        if ibmq_token:
            os.environ['QISKIT_IBM_TOKEN'] = ibmq_token
            os.environ['AQSE_USE_QISKIT'] = '1'
        else:
            st.warning("Enter API token to enable real backend.")
    else:
        os.environ['AQSE_USE_QISKIT'] = '0'

    preset = st.selectbox("Preset", ["Fast Demo", "Balanced", "High Accuracy"])
    if preset == "Fast Demo":
        iters = st.slider("Iterations", 8, 80, 40, 4)
        shots_per_eval = st.selectbox("Baseline shots per eval", [128,256,512], index=2)
        trust_threshold = st.slider("Surrogate trust threshold", 0.01, 0.2, 0.07, 0.01)
        bootstrap_points = st.slider("Bootstrap measured points", 2, 10, 6)
        surrogate_models = st.selectbox("Surrogate ensemble size", [6,8,10], index=0)
    elif preset == "Balanced":
        iters = st.slider("Iterations", 8, 160, 80, 4)
        shots_per_eval = st.selectbox("Baseline shots per eval", [128,256,512,1024], index=2)
        trust_threshold = st.slider("Surrogate trust threshold", 0.005, 0.2, 0.05, 0.005)
        bootstrap_points = st.slider("Bootstrap measured points", 4, 12, 8)
        surrogate_models = st.selectbox("Surrogate ensemble size", [8,10,12], index=0)
    else:
        iters = st.slider("Iterations", 8, 320, 160, 8)
        shots_per_eval = st.selectbox("Baseline shots per eval", [256,512,1024,2048], index=1)
        trust_threshold = st.slider("Surrogate trust threshold", 0.001, 0.1, 0.02, 0.001)
        bootstrap_points = st.slider("Bootstrap measured points", 6, 20, 10)
        surrogate_models = st.selectbox("Surrogate ensemble size", [10,12,16], index=0)

    gate_error = st.slider("2Q gate error (simulated)", 0.0, 0.05, 0.02, 0.005)
    pruning_thresh = st.slider("Pruning fidelity threshold", 0.0, 0.2, 0.04, 0.01)
    run_button = st.button("Run AQSE")

    # compare controls
    compare_seeds = st.slider('Compare: number of seeds', 1, 10, 3)
    compare_iters = st.slider('Compare: iterations', 8, 160, 40, 4)
    compare_shots = st.selectbox('Compare: shots per eval', [128,256,512,1024], index=2)
    compare_button = st.button('Run AQSE vs Baseline (multi-seed)')

    st.markdown("---")
    st.header("Saved results")
    show_existing = st.checkbox("Show existing result traces/summary", value=True)
    available_traces = [f for f in os.listdir(RESULTS_DIR) if f.startswith("trace_") and f.endswith(".csv")]
    chosen_trace = st.selectbox("Select a trace CSV to view", options=["(none)"] + available_traces)
    chosen_summary = st.selectbox("Select summary CSV to view", options=["(none)", "summary.csv"] + [f for f in os.listdir(RESULTS_DIR) if f.endswith(".csv") and f not in available_traces])

pauli_list = pauli_list_4q
coeffs = coeffs_4q

def plot_run_df(df, title_prefix="Run"):
    """Produce a multi-panel Plotly figure summarizing a single run df (per-iteration)."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=2, subplot_titles=[
        f"{title_prefix}: Used Energy vs Iter",
        f"{title_prefix}: Cumulative Shots vs Iter",
        f"{title_prefix}: Shots used this iter vs Iter",
        f"{title_prefix}: Surrogate sigmaE vs Iter",
        f"{title_prefix}: Fidelity drop vs Iter"
        # Removed: used_surrogate / pruned vs Iter
    ])

    # Used Energy vs Iter
    try:
        fig.add_trace(go.Scatter(x=df["iter"], y=df["used_energy"], mode='lines+markers', name="Used Energy"),
                      row=1, col=1)
        fig.update_xaxes(title_text="Iteration", row=1, col=1)
        fig.update_yaxes(title_text="Energy", row=1, col=1)
    except Exception:
        pass

    # Cumulative Shots vs Iter
    try:
        fig.add_trace(go.Scatter(x=df["iter"], y=df["cumulative_shots"], mode='lines+markers', name="Cumulative Shots"),
                      row=1, col=2)
        fig.update_xaxes(title_text="Iteration", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Shots", row=1, col=2)
    except Exception:
        pass

    # Shots used this iter vs Iter
    try:
        fig.add_trace(go.Bar(x=df["iter"], y=df["shots_used_this_iter"], name="Shots used this iter"),
                      row=2, col=1)
        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_yaxes(title_text="Shots This Iter", row=2, col=1)
    except Exception:
        pass

    # Surrogate sigmaE vs Iter
    try:
        fig.add_trace(go.Scatter(x=df["iter"], y=df["surrogate_sigmaE"], mode='lines+markers', name="Surrogate sigmaE"),
                      row=2, col=2)
        fig.update_xaxes(title_text="Iteration", row=2, col=2)
        fig.update_yaxes(title_text="Surrogate sigmaE", row=2, col=2)
    except Exception:
        pass

    # Fidelity drop vs Iter
    try:
        fig.add_trace(go.Scatter(x=df["iter"], y=df["fidelity_drop"], mode='lines+markers', name="Fidelity drop"),
                      row=3, col=1)
        fig.update_xaxes(title_text="Iteration", row=3, col=1)
        fig.update_yaxes(title_text="Fidelity Drop", row=3, col=1)
    except Exception:
        pass

    # Removed: used_surrogate / pruned vs Iter

    fig.update_layout(height=900, width=1100, title_text=title_prefix)
    return fig

def display_summary_table_from_rows(rows_df):
    """Show aggregated metrics for compare runs with useful summary plots."""
    df = rows_df.copy()
    print("[DEBUG] Raw summary DataFrame:")
    print(df)
    if "error" in df.columns:
        df = df[df["error"].isnull() | df["error"].astype(str).str.len()==0]

    # Group total_shots_baseline and total_shots_aqse into a single column 'total_shots'
    # Use total_shots_aqse for aqse mode, total_shots_baseline for baseline
    df['total_shots'] = np.where(
        df['mode'] == 'aqse',
        df['total_shots_aqse'],
        df['total_shots_baseline']
    )

    # Remove the old columns for display
    display_cols = [c for c in df.columns if c not in ('total_shots_baseline', 'total_shots_aqse')]
    st.write("Aggregated metrics by mode:")
    st.dataframe(df[display_cols])

# Run single AQSE
if run_button:
    with st.spinner("Running AQSE…"):
        try:
            df, summary = run_aqse(pauli_list, coeffs, n_qubits=4,
                                   iters=iters, shots_per_eval=shots_per_eval,
                                   trust_threshold=trust_threshold, bootstrap_points=bootstrap_points,
                                   surrogate_models=surrogate_models, rf_estimators=50,
                                   gate_error_two_q=gate_error, pruning_threshold=pruning_thresh,
                                   entangler_active=True, seed=1,
                                   use_qiskit_backend=use_real_backend)
            st.success("Run complete.")
            st.metric("Final energy", summary.get("final_energy", "n/a"))
            st.metric("Total shots (AQSE)", summary.get("total_shots_aqse", "n/a"))
            st.metric("Shots saved (%)", f'{summary.get("shots_saved_pct", "n/a"):.2f}' if summary.get("shots_saved_pct") is not None else "n/a")
            # Show backend info if present
            backend_info = summary.get("backend_info")
            if backend_info:
                st.info(f"Qiskit backend: {backend_info.get('name','?')} | Simulator: {backend_info.get('is_simulator','?')}")
            st.dataframe(df)

            # show multi-panel plot
            fig = plot_run_df(df, title_prefix="AQSE Run (seed=1)")
            st.plotly_chart(fig, use_container_width=True)

            # Optionally, save as static PNG for pipeline (not shown in UI)
            try:
                compact_fig = plot_run_df(df, title_prefix="AQSE Run (seed=1)")
                compact_fig.savefig(os.path.join(RESULTS_DIR, f"trace_seed1.png"), dpi=150)
            except Exception:
                pass

        except Exception as e:
            st.error(f"Run failed: {e}")

# Multi-seed compare (AQSE vs baseline)
if compare_button:
    rows = []
    with st.spinner("Running multi-seed comparison..."):
        for mode in ('aqse','baseline'):
            for s in range(compare_seeds):
                seed = 1000 + s
                try:
                    # Pass use_real_backend for AQSE, False for baseline
                    use_backend = use_real_backend if mode == 'aqse' else False
                    dfm, summary = run_mode(mode, seed, compare_iters, compare_shots, outdir=RESULTS_DIR, use_qiskit_backend=use_backend)
                    rows.append({'mode': mode, 'seed': seed, **summary})
                    # save trace for convenience
                    try:
                        dfm.to_csv(os.path.join(RESULTS_DIR, f"trace_{mode}_seed{seed}.csv"), index=False)
                    except Exception:
                        pass
                except Exception as e:
                    rows.append({'mode': mode, 'seed': seed, 'error': str(e)})
    df_all = pd.DataFrame(rows)
    # persist summary
    df_all.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    st.success('Comparison complete.')
    display_summary_table_from_rows(df_all)

    # Additional visualizations with matplotlib (for robustness)
    try:
        st.subheader("Final Energy by Mode (matplotlib)")
        fig, ax = plt.subplots(figsize=(6,4))
        df_all.boxplot(column='final_energy', by='mode', ax=ax)
        ax.set_title('Final energy by mode')
        ax.set_xlabel('Mode')
        ax.set_ylabel('Final Energy')
        plt.suptitle('')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not create final_energy_box: {e}")

    try:
        st.subheader("Total Shots by Mode (AQSE values, matplotlib)")
        fig, ax = plt.subplots(figsize=(6,4))
        df_all.boxplot(column='total_shots_aqse', by='mode', ax=ax)
        ax.set_title('Total shots (AQSE) by mode')
        ax.set_xlabel('Mode')
        ax.set_ylabel('Total Shots (AQSE)')
        plt.suptitle('')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not create total_shots_box: {e}")

    # Additional: Histogram of Final Energy
    try:
        st.subheader("Histogram of Final Energy (all modes)")
        fig, ax = plt.subplots(figsize=(6,4))
        for mode in df_all['mode'].unique():
            vals = df_all[df_all['mode'] == mode]['final_energy'].dropna()
            ax.hist(vals, bins=10, alpha=0.5, label=mode)
        ax.set_xlabel('Final Energy')
        ax.set_ylabel('Count')
        ax.set_title('Histogram of Final Energy')
        ax.legend()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not create final_energy_hist: {e}")

    # Additional: Mean Final Energy vs. Seed (line plot, improved)
    try:
        st.subheader("Final Energy vs. Seed")
        fig, ax = plt.subplots(figsize=(6,4))
        for mode in df_all['mode'].unique():
            mode_df = df_all[df_all['mode'] == mode].sort_values('seed')
            # Ensure seeds are integers for x-axis
            seeds = mode_df['seed'].astype(int)
            ax.plot(seeds, mode_df['final_energy'], marker='o', label=mode)
        ax.set_xlabel('Seed')
        ax.set_ylabel('Final Energy')
        ax.set_title('Final Energy vs. Seed')
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not create final_energy_vs_seed: {e}")

# Show existing results / chosen trace
if show_existing:
    if chosen_summary and chosen_summary != "(none)":
        path = os.path.join(RESULTS_DIR, chosen_summary)
        if os.path.exists(path):
            try:
                df_sum = pd.read_csv(path)
                st.subheader(f"Contents of {chosen_summary}")
                st.dataframe(df_sum)
            except Exception as e:
                st.error(f"Could not read {chosen_summary}: {e}")
        else:
            st.warning(f"{chosen_summary} not found in {RESULTS_DIR}.")

    if chosen_trace and chosen_trace != "(none)":
        trace_path = os.path.join(RESULTS_DIR, chosen_trace)
        if os.path.exists(trace_path):
            try:
                dft = pd.read_csv(trace_path)
                st.subheader(f"Trace: {chosen_trace}")
                st.dataframe(dft)
                # plot trace (per-iteration plot)
                fig = plot_run_df(dft, title_prefix=chosen_trace.replace(".csv",""))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not read/plot {chosen_trace}: {e}")
        else:
            st.warning(f"{chosen_trace} not found in {RESULTS_DIR}.")

if not run_button and not compare_button:
    st.info("Configure settings and click Run AQSE or Run AQSE vs Baseline. You can also view saved traces in the 'Saved results' panel.")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_flag_timeline(df):
    d = df.sort_values("iter").reset_index(drop=True).copy()

    # Ensure expected columns exist and are int
    for col in ("used_surrogate", "pruned_this_iter", "shots_used_this_iter"):
        if col not in d.columns:
            d[col] = 0
    d["used_surrogate"] = d["used_surrogate"].fillna(0).astype(int)
    d["pruned_this_iter"] = d["pruned_this_iter"].fillna(0).astype(int)
    d["shots_used_this_iter"] = d["shots_used_this_iter"].fillna(0).astype(int)

    # Identify measurement events (spikes in shots)
    median_shots = int(np.median(d["shots_used_this_iter"].replace(0, np.nan).dropna() or [0]) or 0)
    shots_thresh = max(1, median_shots * 3)
    measurement_idxs = d.index[d["shots_used_this_iter"] >= shots_thresh].tolist()
    measurement_iters = d.loc[measurement_idxs, "iter"].tolist()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("used_surrogate (Yes/No)", "pruned_this_iter (Yes/No)")
    )

    # Top: used_surrogate (step + markers)
    fig.add_trace(
        go.Scatter(
            x=d["iter"],
            y=d["used_surrogate"],
            mode="lines+markers",
            line=dict(shape="hv", width=2, color="green"),
            marker=dict(size=8),
            name="used_surrogate",
            hovertemplate="iter: %{x}<br>used_surrogate: %{y}<br>shots: %{customdata}",
            customdata=d[["shots_used_this_iter"]].values
        ),
        row=1, col=1
    )

    # Bottom: pruned_this_iter (step + markers)
    fig.add_trace(
        go.Scatter(
            x=d["iter"],
            y=d["pruned_this_iter"],
            mode="lines+markers",
            line=dict(shape="hv", width=2, color="orange"),
            marker=dict(symbol="x", size=8),
            name="pruned_this_iter",
            hovertemplate="iter: %{x}<br>pruned: %{y}<br>shots: %{customdata}",
            customdata=d[["shots_used_this_iter"]].values
        ),
        row=2, col=1
    )

    # Shade measurement events
    for it in measurement_iters:
        fig.add_vrect(
            x0=it-0.4, x1=it+0.4,
            fillcolor="LightSalmon", opacity=0.12, layer="below", line_width=0,
            row="all", col=1
        )

    # Y-axis formatting (0/1 -> No/Yes)
    fig.update_yaxes(row=1, col=1, tickmode="array", tickvals=[0,1], ticktext=["No","Yes"], range=[-0.2, 1.2])
    fig.update_yaxes(row=2, col=1, tickmode="array", tickvals=[0,1], ticktext=["No","Yes"], range=[-0.2, 1.2])
    fig.update_xaxes(title_text="Iteration", row=2, col=1)
    fig.update_layout(
        height=380,
        showlegend=False,
        margin=dict(l=60, r=20, t=60, b=50),
        title_text="used_surrogate / pruned timeline (shaded = heavy measurement events)",
    )
    return fig
