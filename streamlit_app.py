import matplotlib.pyplot as plt

def plot_run_df(df, title_prefix="Run"):
    """Produce a multi-panel Matplotlib figure summarizing a single run df (per-iteration)."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    fig.suptitle(title_prefix, fontsize=14)

    # Used Energy vs Iter
    if "used_energy" in df:
        axes[0,0].plot(df["iter"], df["used_energy"], marker='o')
        axes[0,0].set_title("Used Energy vs Iter")
        axes[0,0].set_xlabel("Iteration")
        axes[0,0].set_ylabel("Energy")

    # Cumulative Shots vs Iter
    if "cumulative_shots" in df:
        axes[0,1].plot(df["iter"], df["cumulative_shots"], marker='o')
        axes[0,1].set_title("Cumulative Shots vs Iter")
        axes[0,1].set_xlabel("Iteration")
        axes[0,1].set_ylabel("Cumulative Shots")

    # Shots used this iter vs Iter
    if "shots_used_this_iter" in df:
        axes[1,0].bar(df["iter"], df["shots_used_this_iter"])
        axes[1,0].set_title("Shots used this iter vs Iter")
        axes[1,0].set_xlabel("Iteration")
        axes[1,0].set_ylabel("Shots This Iter")

    # Surrogate sigmaE vs Iter
    if "surrogate_sigmaE" in df:
        axes[1,1].plot(df["iter"], df["surrogate_sigmaE"], marker='o')
        axes[1,1].set_title("Surrogate sigmaE vs Iter")
        axes[1,1].set_xlabel("Iteration")
        axes[1,1].set_ylabel("Surrogate sigmaE")

    # Fidelity drop vs Iter
    if "fidelity_drop" in df:
        axes[2,0].plot(df["iter"], df["fidelity_drop"], marker='o')
        axes[2,0].set_title("Fidelity drop vs Iter")
        axes[2,0].set_xlabel("Iteration")
        axes[2,0].set_ylabel("Fidelity Drop")

    # Hide unused subplot (bottom-right)
    axes[2,1].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig
