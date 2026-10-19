"""One-at-a-time sensitivity analysis around the base configuration.

usage: python run_sensitivity.py --universe ff49 [--dims lookback gamma ...]
Appends one row per (variant, strategy) to results/tables/sensitivity_<universe>.csv
"""
import argparse, json
import pandas as pd
from config import *  # noqa
from run_backtests import run_all

GRID = {
    "lookback": [252, 504, 756, 1260],
    "gamma": [1.0, 2.5, 5.0, 10.0, 20.0],
    "kappa": [0.5, 1.0, 2.0, 3.0],
    "alpha": [0.01, 0.05, 0.10],
    "c_cvar": [0.10, 0.25, 0.50],
    "turnover_max": [2.0, 0.50, 0.25, 0.10],
    "cost": [0.0, 0.001, 0.0025, 0.005],
    "tc_scale": [0.0, 12.0],          # 0.0 = costs charged in the backtest but ignored by the optimizer
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="ff49")
    ap.add_argument("--dims", nargs="*", default=list(GRID))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default=None, help="output csv name (default sensitivity_<universe>.csv)")
    a = ap.parse_args()
    out = TABLES / (a.out or f"sensitivity_{a.universe}.csv")
    rows = []
    for dim in a.dims:
        for val in GRID[dim]:
            cfg = dict(BASE); cfg[dim] = val
            tag = f"sens_{dim}={val}"
            print(f"== {a.universe} {tag}", flush=True)
            _, summ = run_all(a.universe, cfg, tag, n_jobs=a.jobs, save=False)
            summ = summ.reset_index().rename(columns={"index": "Strategy"})
            summ.insert(0, "value", val); summ.insert(0, "dim", dim)
            rows.append(summ)
            pd.concat(rows).to_csv(out, index=False)
    print("saved", out)
