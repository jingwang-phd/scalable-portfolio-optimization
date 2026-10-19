"""Walk-forward (rolling re-estimation) backtest engine.

At each rebalance date t_k (last trading day of the month):
    1. eligible universe = assets with a complete trailing window of `lookback` daily returns
    2. estimate / optimize on the window -> target weights x_k (turnover measured against the
       *drifted* holdings carried from the previous period)
    3. pay proportional cost  c * ||x_k - x_prev||_1  on day t_k+1
    4. hold with buy-and-hold drift until the next rebalance
Daily net portfolio returns, weights, turnover and costs are recorded.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .models import Constraints, Strategy
from .data import sector_matrix


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series          # daily net returns
    gross_returns: pd.Series    # daily returns before costs
    turnover: pd.Series         # per rebalance, sum |dx|
    costs: pd.Series            # per rebalance, cost paid (fraction of wealth)
    weights: pd.DataFrame       # target weights per rebalance date
    n_assets: pd.Series         # eligible universe size per rebalance
    solve_time: float


def rebalance_dates(index: pd.DatetimeIndex, start, end, freq="M"):
    idx = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    if freq == "M":
        return pd.DatetimeIndex(idx.to_series().groupby([idx.year, idx.month]).last().values)
    if freq == "W":
        iso = idx.isocalendar()
        return pd.DatetimeIndex(idx.to_series().groupby([iso.year.values, iso.week.values]).last().values)
    raise ValueError(freq)


def run_backtest(returns: pd.DataFrame, sectors: pd.Series, strategy: Strategy, cons: Constraints,
                 lookback: int, start, end, freq="M", min_assets: int = 10, verbose=False) -> BacktestResult:
    import time
    idx = returns.index
    rdates = rebalance_dates(idx, start, end, freq)
    vals = returns.values
    cols = np.array(returns.columns)
    pos = {d: i for i, d in enumerate(idx)}
    daily_net, daily_gross, dates = [], [], []
    turn, cost, nassets, wrows = {}, {}, {}, {}
    x_prev = pd.Series(dtype=float)          # drifted holdings before this rebalance
    t_solve = 0.0
    for k, d in enumerate(rdates):
        i = pos[d]
        if i + 1 - lookback < 0:
            continue
        win = vals[i + 1 - lookback:i + 1]
        elig = ~np.isnan(win).any(axis=0)
        if elig.sum() < min_assets:
            continue
        W = win[:, elig]
        names = cols[elig]
        # previous holdings aligned to the eligible universe (assets that left are liquidated)
        xp = x_prev.reindex(names).fillna(0.0).values if len(x_prev) else None
        liquidated = float(x_prev.drop(x_prev.index.intersection(names)).abs().sum()) if len(x_prev) else 0.0
        G, _ = sector_matrix(sectors, names)
        c_k = Constraints(**{**cons.__dict__, "sector_G": G if cons.s_max is not None else None})
        t0 = time.time()
        try:
            x = strategy.weights(W, c_k, xp)
        except Exception as e:  # keep previous weights if the solver fails (recorded)
            if verbose:
                print("solver failure at", d, e)
            x = xp if xp is not None else np.full(len(names), 1 / len(names))
            x = x / x.sum()
        t_solve += time.time() - t0
        tov = float(np.abs(x - (xp if xp is not None else 0.0)).sum()) + liquidated
        cst = cons.cost * tov
        turn[d], cost[d], nassets[d] = tov, cst, int(elig.sum())
        wrows[d] = pd.Series(x, index=names)
        # holding period: next trading day after d up to and including next rebalance date
        j_end = pos[rdates[k + 1]] if k + 1 < len(rdates) else len(idx) - 1
        w = x.copy()
        for j in range(i + 1, j_end + 1):
            r = vals[j, elig]
            r = np.where(np.isnan(r), 0.0, r)
            rp = float(w @ r)
            gross = rp
            net = rp - (cst if j == i + 1 else 0.0)
            daily_gross.append(gross); daily_net.append(net); dates.append(idx[j])
            w = w * (1 + r) / (1 + rp) if abs(1 + rp) > 1e-12 else w
        x_prev = pd.Series(w, index=names)
    dates = pd.DatetimeIndex(dates)
    return BacktestResult(strategy.name, pd.Series(daily_net, dates, name=strategy.name),
                          pd.Series(daily_gross, dates, name=strategy.name),
                          pd.Series(turn), pd.Series(cost), pd.DataFrame(wrows).T.fillna(0.0),
                          pd.Series(nassets), t_solve)
