"""Figures and compact tables from the one-at-a-time sensitivity sweep.

usage: python plot_sensitivity.py --universe ff49
"""
import argparse
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *  # noqa
from analyze_run import ORDER, COLORS

LABEL = {"lookback": "Estimation window (trading days)", "gamma": "Risk aversion γ (MV family)",
         "kappa": "Robustness κ (Robust-MV)", "alpha": "Tail level α (CVaR models)",
         "c_cvar": "Risk weight c (Mean-CVaR)", "turnover_max": "Turnover cap per rebalance (Σ|Δx|)",
         "cost": "Proportional cost (one-way)", "tc_scale": "Cost term in optimizer (0 = ignored, 12 = annualized)"}
AFFECTS = {"gamma": ["MV-sample", "MV-shrink", "GMV-LW", "Robust-MV"], "kappa": ["Robust-MV"],
           "alpha": ["Mean-CVaR", "Min-CVaR"], "c_cvar": ["Mean-CVaR"]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--universe", default="ff49"); a = ap.parse_args()
    df = pd.read_csv(TABLES / f"sensitivity_{a.universe}.csv")
    dims = [d for d in LABEL if d in df.dim.unique()]
    for metric in ["Sharpe", "MaxDD", "CVaR5_daily", "AnnTurnover"]:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8)); axes = axes.ravel()
        for ax, dim in zip(axes, dims):
            sub = df[df.dim == dim]
            for s in [s for s in ORDER if s in sub.Strategy.unique()]:
                if dim in AFFECTS and s not in AFFECTS[dim] and s != "EW":
                    continue
                g = sub[sub.Strategy == s].sort_values("value")
                ax.plot(g.value.astype(str), g[metric], marker="o", label=s, color=COLORS[s], lw=1.2)
            ax.set_title(LABEL[dim], fontsize=9); ax.grid(alpha=.3); ax.tick_params(labelsize=8)
        for ax in axes[len(dims):]:
            ax.axis("off")
        axes[0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"{a.universe}: {metric} under one-at-a-time parameter changes (base: window 504d, γ=5, κ=2, α=5%, c=0.25, turnover≤50%, 10 bps)", fontsize=11)
        fig.tight_layout(); fig.savefig(FIGS / f"{a.universe}_sens_{metric}.png", dpi=140); plt.close(fig)
    # compact pivot tables
    for metric in ["Sharpe", "MaxDD", "CVaR5_daily", "AnnTurnover", "CAGR"]:
        pv = df.pivot_table(index=["dim", "value"], columns="Strategy", values=metric)[[s for s in ORDER if s in df.Strategy.unique()]]
        pv.round(4).to_csv(TABLES / f"{a.universe}_sens_{metric}.csv")
    print(df.pivot_table(index=["dim", "value"], columns="Strategy", values="Sharpe")[ORDER].round(3).to_string())
