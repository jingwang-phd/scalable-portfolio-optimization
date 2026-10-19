"""Run the walk-forward backtest for all strategies on one universe and one configuration.

usage: python run_backtests.py --universe ff49 [--tag base] [--set lookback=252 gamma=10 ...]
Outputs results/runs/<universe>__<tag>/ : returns.parquet, gross.parquet, turnover.parquet,
costs.parquet, nassets.parquet, weights_<strategy>.parquet, summary.csv, config.json
"""
import argparse, json, time
import numpy as np, pandas as pd
from joblib import Parallel, delayed, parallel_config
from config import *  # noqa
from portlib import data, metrics
from portlib.backtest import run_backtest


def load_universe(name):
    u = UNIVERSES[name]
    if name == "ff49":
        R, sec = data.load_ff49(u["start"])
    else:
        R, sec = data.load_sp500(u["start"])
    return R, sec, u


def _run(strategy, R, sec, cons, cfg, u):
    res = run_backtest(R, sec, strategy, cons, cfg["lookback"], u["oos_start"], u["oos_end"],
                       freq=cfg["freq"], min_assets=u["min_assets"])
    return res


def run_all(universe, cfg, tag, strategies_list=None, n_jobs=8, save=True):
    R, sec, u = load_universe(universe)
    cons = constraints(universe, cfg)
    strats = strategies_list or strategies(cfg)
    t0 = time.time()
    with parallel_config(backend="loky", inner_max_num_threads=1):   # one BLAS thread per worker (Gurobi already uses 1)
        results = Parallel(n_jobs=n_jobs)(delayed(_run)(s, R, sec, cons, cfg, u) for s in strats)
    results = {r.name: r for r in results}
    rf = data.load_rf()
    summ = metrics.summary_table(results, rf)
    if save:
        out = RUNS / f"{universe}__{tag}"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({k: v.returns for k, v in results.items()}).to_parquet(out / "returns.parquet")
        pd.DataFrame({k: v.gross_returns for k, v in results.items()}).to_parquet(out / "gross.parquet")
        pd.DataFrame({k: v.turnover for k, v in results.items()}).to_parquet(out / "turnover.parquet")
        pd.DataFrame({k: v.costs for k, v in results.items()}).to_parquet(out / "costs.parquet")
        pd.DataFrame({k: v.n_assets for k, v in results.items()}).to_parquet(out / "nassets.parquet")
        for k, v in results.items():
            v.weights.to_parquet(out / f"weights_{k}.parquet")
        summ.to_csv(out / "summary.csv")
        json.dump({"universe": universe, "cfg": cfg, "tag": tag, "wall_s": time.time() - t0},
                  open(out / "config.json", "w"), indent=1)
    return results, summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="ff49")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    cfg = dict(BASE)
    for kv in a.set:
        k, v = kv.split("=")
        cfg[k] = type(BASE[k])(v) if not isinstance(BASE[k], float) else float(v)
    results, summ = run_all(a.universe, cfg, a.tag, n_jobs=a.jobs)
    pd.set_option("display.width", 200)
    print(summ.round(4).to_string())
