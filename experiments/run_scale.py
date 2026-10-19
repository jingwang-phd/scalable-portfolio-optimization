"""Layer 3 experiment: runtime of four solution methods for the mean-CVaR LP as N and S grow.

usage: python run_scale.py [--N 50 100 250 500] [--S 1000 10000 50000 100000] [--methods monolithic rowgen admm ph]
Appends to results/tables/scale.csv after every solve so partial results survive interruption.
Scenarios: circular block bootstrap (block = 5 days) of daily returns of S&P 500 constituents with
complete data over the last two years of the sample; assets are a fixed random subset of size N.
"""
import argparse, json, time, resource, sys
import numpy as np, pandas as pd
from config import *  # noqa
from portlib import data, scale

ALPHA, C = 0.05, 0.25
WINDOW = ("2024-09-01", "2026-08-31")


def universe(N, seed=0):
    R, sec = data.load_sp500("2024-01-01")
    R = R.loc[WINDOW[0]:WINDOW[1]].dropna(axis=1)
    rng = np.random.default_rng(seed)
    N = min(N, R.shape[1])
    cols = sorted(rng.choice(R.columns, size=N, replace=False).tolist())
    R = R[cols]
    G, names = data.sector_matrix(sec, R.columns)
    w_max = 0.10 if N <= 50 else 0.05
    return R.values, G, w_max, 0.25


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if sys.platform == "darwin" else r / 1e3   # bytes on macOS, KiB on Linux


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", nargs="*", type=int, default=[50, 100, 250, 500])
    ap.add_argument("--S", nargs="*", type=int, default=[1000, 10000, 50000, 100000])
    ap.add_argument("--methods", nargs="*", default=["monolithic", "rowgen", "admm", "ph"])
    ap.add_argument("--time_limit", type=float, default=1800)
    ap.add_argument("--gap", type=float, default=1e-4, help="relative gap at which admm/ph stop")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="scale.csv")
    ap.add_argument("--rowgen_init", default="ew", help="ew | subsample")
    a = ap.parse_args()
    out = TABLES / a.out
    rows = [] if not out.exists() else pd.read_csv(out).to_dict("records")
    for N in a.N:
        X, G, w_max, s_max = universe(N)
        for S in a.S:
            R = scale.block_bootstrap(X, S, block=5, seed=S + N)
            mu = R.mean(0)
            ref = None
            for meth in a.methods:
                print(f"== N={N} S={S} {meth}", flush=True)
                try:
                    if meth == "monolithic":
                        r = scale.solve_monolithic(R, mu, ALPHA, C, w_max, G, s_max, threads=a.threads, time_limit=a.time_limit)
                        ref = r["obj"]
                    elif meth == "rowgen":
                        r = scale.solve_rowgen(R, mu, ALPHA, C, w_max, G, s_max, threads=a.threads, time_limit=a.time_limit, init=a.rowgen_init)
                        if ref is None:
                            ref = r["obj"]
                    elif meth == "admm":
                        r = scale.solve_admm(R, mu, ALPHA, C, w_max, G, s_max, threads=a.threads, time_limit=a.time_limit,
                                             ref_obj=ref, gap_tol=a.gap)
                    elif meth == "ph":
                        r = scale.solve_ph(R, mu, ALPHA, C, w_max, G, s_max, n_bundles=8, n_jobs=8,
                                           time_limit=a.time_limit, ref_obj=ref, gap_tol=a.gap)
                except Exception as e:
                    r = dict(method=meth, status=-1, time=np.nan, obj=np.nan, iters=np.nan, rows=np.nan, nnz=np.nan, err=str(e))
                row = dict(N=N, S=S, method=(meth + "_sub" if (meth == "rowgen" and a.rowgen_init == "subsample") else meth), time_s=r.get("time"), obj=r.get("obj"),
                           rel_gap=(r.get("obj") - ref) / abs(ref) if (ref is not None and r.get("obj") is not None) else np.nan,
                           iters=r.get("iters"), rows=r.get("rows"), nnz=r.get("nnz"), status=r.get("status"),
                           active_frac=r.get("active_frac"), n_pos_u=r.get("n_pos_u"), rss_mb=rss_mb(), err=r.get("err", ""))
                rows.append(row)
                pd.DataFrame(rows).to_csv(out, index=False)
                if "hist" in r:
                    json.dump(r["hist"], open(TABLES / f"scale_hist_{meth}_N{N}_S{S}.json", "w"))
                print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items() if k != "err"}, flush=True)
