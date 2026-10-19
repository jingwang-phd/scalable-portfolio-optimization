"""Performance / risk metrics, bootstrap inference for Sharpe differences, stochastic dominance."""
from __future__ import annotations
import numpy as np
import pandas as pd

A = 252


def max_drawdown(r: pd.Series) -> float:
    w = (1 + r).cumprod()
    return float((w / w.cummax() - 1).min())


def realized_cvar(r: pd.Series, alpha=0.05) -> float:
    """Average of the worst alpha-fraction of daily returns (reported as a positive loss)."""
    q = np.quantile(r, alpha)
    return float(-r[r <= q].mean())


def summary(r: pd.Series, rf: pd.Series | None = None, turnover: pd.Series | None = None,
            costs: pd.Series | None = None, alpha=0.05) -> dict:
    r = r.dropna()
    ex = r - (rf.reindex(r.index).fillna(0.0) if rf is not None else 0.0)
    n_years = len(r) / A
    out = {
        "CAGR": float((1 + r).prod() ** (1 / n_years) - 1),
        "AnnMean": float(r.mean() * A),
        "AnnVol": float(r.std(ddof=1) * np.sqrt(A)),
        "Sharpe": float(ex.mean() / ex.std(ddof=1) * np.sqrt(A)),
        "Sortino": float(ex.mean() * A / (np.sqrt((np.minimum(ex, 0) ** 2).mean() * A))),
        f"CVaR{int(alpha*100)}_daily": realized_cvar(r, alpha),
        "MaxDD": max_drawdown(r),
        "Calmar": float((r.mean() * A) / abs(max_drawdown(r))),
        "Skew": float(r.skew()), "Kurt": float(r.kurt()),
        "WorstDay": float(r.min()),
    }
    if turnover is not None and len(turnover):
        out["AvgTurnover"] = float(turnover.mean())                 # per rebalance, sum|dx|
        out["AnnTurnover"] = float(turnover.sum() / n_years)
    if costs is not None and len(costs):
        out["AnnCostDrag"] = float(costs.sum() / n_years)
    return out


def summary_table(results: dict, rf=None, alpha=0.05) -> pd.DataFrame:
    rows = {}
    for name, res in results.items():
        rows[name] = summary(res.returns, rf, res.turnover, res.costs, alpha)
        rows[name]["SolveTime_s"] = res.solve_time
    return pd.DataFrame(rows).T


# -------------------------------------------------------------------------------------------
# Inference: stationary bootstrap (Politis & Romano 1994) for the difference of Sharpe ratios
# -------------------------------------------------------------------------------------------
def _stationary_bootstrap_idx(n, block_mean, rng):
    p = 1.0 / block_mean
    idx = np.empty(n, dtype=int)
    idx[0] = rng.integers(n)
    for t in range(1, n):
        idx[t] = rng.integers(n) if rng.random() < p else (idx[t - 1] + 1) % n
    return idx


def sharpe_diff_test(r1: pd.Series, r2: pd.Series, rf: pd.Series | None = None, n_boot=2000,
                     block_mean=20, seed=0):
    """Two-sided bootstrap test of H0: SR1 = SR2, using the stationary bootstrap on the *paired*
    daily series (so cross-correlation between strategies is preserved). Returns dict with the
    point estimate, bootstrap SE, 95% CI and p-value."""
    df = pd.concat([r1, r2], axis=1).dropna()
    if rf is not None:
        df = df.sub(rf.reindex(df.index).fillna(0.0), axis=0)
    x = df.values
    def sr(v):
        return v.mean(0) / v.std(0, ddof=1) * np.sqrt(A)
    d0 = np.subtract(*sr(x))
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = _stationary_bootstrap_idx(n, block_mean, rng)
        boots[b] = np.subtract(*sr(x[idx]))
    se = boots.std(ddof=1)
    centered = boots - boots.mean()
    p = float((np.abs(centered) >= abs(d0)).mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"SR_diff": float(d0), "SE": float(se), "CI95": (float(lo), float(hi)), "p_value": p}


# -------------------------------------------------------------------------------------------
# Stochastic dominance (empirical, equally likely outcomes) -- course material
# -------------------------------------------------------------------------------------------
def ssd_dominates(a: np.ndarray, b: np.ndarray, tol=1e-12) -> bool:
    """True if a second-order stochastically dominates b:  E[(eta - a)^+] <= E[(eta - b)^+] for all
    eta (checked on the union of outcomes, which suffices for step functions)."""
    grid = np.union1d(a, b)
    fa = np.array([np.maximum(g - a, 0).mean() for g in grid])
    fb = np.array([np.maximum(g - b, 0).mean() for g in grid])
    return bool(np.all(fa <= fb + tol)) and not np.allclose(fa, fb)


def fsd_dominates(a: np.ndarray, b: np.ndarray, tol=1e-12) -> bool:
    grid = np.union1d(a, b)
    Fa = np.array([(a <= g).mean() for g in grid]); Fb = np.array([(b <= g).mean() for g in grid])
    return bool(np.all(Fa <= Fb + tol)) and not np.allclose(Fa, Fb)


def dominance_matrix(returns: dict[str, pd.Series], freq="M") -> pd.DataFrame:
    """Pairwise SSD check on (monthly) returns: entry [i, j] = 1 if i SSD-dominates j."""
    mon = {k: (1 + v).resample("ME").prod() - 1 for k, v in returns.items()}
    df = pd.DataFrame(mon).dropna()
    names = list(df.columns)
    M = pd.DataFrame(0, index=names, columns=names)
    for i in names:
        for j in names:
            if i != j and ssd_dominates(df[i].values, df[j].values):
                M.loc[i, j] = 1
    return M


def period_table(results: dict, periods: dict, rf=None) -> pd.DataFrame:
    rows = []
    for pname, (s, e) in periods.items():
        for name, res in results.items():
            r = res.returns.loc[s:e]
            if len(r) < 20:
                continue
            rows.append({"Period": pname, "Strategy": name, "TotalReturn": float((1 + r).prod() - 1),
                         "AnnVol": float(r.std() * np.sqrt(A)), "MaxDD": max_drawdown(r),
                         "CVaR5_daily": realized_cvar(r), "WorstDay": float(r.min())})
    return pd.DataFrame(rows)
