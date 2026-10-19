"""Figures for the scalability experiment (results/tables/scale.csv)."""
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *  # noqa

MC = {"monolithic": "tab:blue", "rowgen": "tab:red", "admm": "tab:green", "ph": "tab:purple"}
ML = {"monolithic": "Monolithic LP (Gurobi, 8 threads)", "rowgen": "Tail-scenario row generation (exact)",
      "admm": "ADMM / operator splitting (rel. gap ≤ 1e-4)", "ph": "Bundled progressive hedging, 8 procs (rel. gap ≤ 1e-4)"}

if __name__ == "__main__":
    df = pd.read_csv(TABLES / "scale.csv")
    if (TABLES / "scale_large.csv").exists():
        df = pd.concat([df, pd.read_csv(TABLES / "scale_large.csv")])
    df = df[df.status != -1].drop_duplicates(subset=["N", "S", "method"], keep="last")
    Ns = sorted(df.N.unique())
    fig, axes = plt.subplots(1, len(Ns), figsize=(4.2 * len(Ns), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, N in zip(axes, Ns):
        sub = df[df.N == N]
        for m in ["monolithic", "rowgen", "admm", "ph"]:
            g = sub[sub.method == m].sort_values("S")
            if len(g):
                ax.plot(g.S, g.time_s, marker="o", color=MC[m], label=ML[m])
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"N = {N} assets"); ax.set_xlabel("scenarios S"); ax.grid(alpha=.3, which="both")
    axes[0].set_ylabel("wall-clock seconds"); axes[0].legend(fontsize=7)
    fig.suptitle("Mean-CVaR LP (α = 5%, long-only, 5% position cap, 25% sector cap): solve time vs. problem size")
    fig.tight_layout(); fig.savefig(FIGS / "scale_time.png", dpi=150); plt.close(fig)
    # speed-up of row generation and active-set size
    piv = df.pivot_table(index=["N", "S"], columns="method", values="time_s")
    if "monolithic" in piv and "rowgen" in piv:
        piv["speedup_rowgen"] = piv["monolithic"] / piv["rowgen"]
    act = df[df.method == "rowgen"].set_index(["N", "S"])[["active_frac", "n_pos_u", "iters"]]
    tab = piv.join(act)
    tab.round(3).to_csv(TABLES / "scale_summary.csv")
    print(tab.round(3).to_string())
    # convergence paths for the largest instance that has both admm and ph histories
    for (N, S) in [(max(Ns), int(df.S.max()))]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ref = df[(df.N == N) & (df.S == S) & (df.method == "monolithic")].obj
        ref = float(ref.iloc[0]) if len(ref) else None
        for m in ["admm", "ph"]:
            f = TABLES / f"scale_hist_{m}_N{N}_S{S}.json"
            if f.exists() and ref:
                h = pd.DataFrame(json.load(open(f)))
                ax.plot(h.time, (h.obj - ref) / abs(ref), color=MC[m], label=ML[m])
        if ref:
            mono_t = float(df[(df.N == N) & (df.S == S) & (df.method == "monolithic")].time_s.iloc[0])
            ax.axvline(mono_t, color=MC["monolithic"], ls="--", label="monolithic LP solve time")
            rg = df[(df.N == N) & (df.S == S) & (df.method == "rowgen")]
            if len(rg):
                ax.axvline(float(rg.time_s.iloc[0]), color=MC["rowgen"], ls="--", label="row generation solve time")
        ax.set_yscale("log"); ax.set_xlabel("seconds"); ax.set_ylabel("relative objective gap"); ax.grid(alpha=.3, which="both")
        ax.set_title(f"Convergence of first-order / decomposition methods, N={N}, S={S}"); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(FIGS / f"scale_convergence_N{N}_S{S}.png", dpi=150); plt.close(fig)
