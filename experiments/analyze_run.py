"""Analysis of one saved run: figures, stress-period table, Sharpe-difference bootstrap tests, SSD.

usage: python analyze_run.py --universe ff49 --tag base
"""
import argparse, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *  # noqa
from portlib import data, metrics

ORDER = ["EW", "MV-sample", "MV-shrink", "GMV-LW", "Robust-MV", "Mean-CVaR", "Min-CVaR", "Mean-SemiDev"]
COLORS = dict(zip(ORDER, plt.cm.tab10.colors))


class _Res:  # light-weight stand-in for BacktestResult when loading from disk
    def __init__(self, name, r, tov, cst):
        self.name, self.returns, self.turnover, self.costs = name, r, tov, cst
        self.solve_time = np.nan


def load_run(universe, tag):
    d = RUNS / f"{universe}__{tag}"
    ret = pd.read_parquet(d / "returns.parquet")
    tov = pd.read_parquet(d / "turnover.parquet")
    cst = pd.read_parquet(d / "costs.parquet")
    res = {k: _Res(k, ret[k].dropna(), tov[k].dropna(), cst[k].dropna()) for k in ret.columns}
    return res, ret, d


def fig_wealth(ret, out, title):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    W = (1 + ret).cumprod()
    for k in [c for c in ORDER if c in ret]:
        axes[0].plot(W.index, W[k], label=k, color=COLORS[k], lw=1.2)
        dd = W[k] / W[k].cummax() - 1
        axes[1].plot(dd.index, dd, color=COLORS[k], lw=0.9)
    axes[0].set_yscale("log"); axes[0].set_ylabel("Wealth (log scale, start = 1)"); axes[0].legend(ncol=4, fontsize=8)
    axes[0].set_title(title); axes[1].set_ylabel("Drawdown"); axes[1].grid(alpha=.3); axes[0].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_rolling_sharpe(ret, rf, out, window=756):
    ex = ret.sub(rf.reindex(ret.index).fillna(0), axis=0)
    rs = ex.rolling(window).mean() / ex.rolling(window).std() * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(11, 4))
    for k in [c for c in ORDER if c in ret]:
        ax.plot(rs.index, rs[k], label=k, color=COLORS[k], lw=1)
    ax.axhline(0, color="k", lw=.5); ax.set_title(f"Rolling {window//252}-year Sharpe ratio"); ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_risk_return(summ, out, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    for k in summ.index:
        ax.scatter(summ.loc[k, "AnnVol"], summ.loc[k, "AnnMean"], color=COLORS.get(k, "k"), s=60)
        ax.annotate(k, (summ.loc[k, "AnnVol"], summ.loc[k, "AnnMean"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Realized annual volatility"); ax.set_ylabel("Realized annual mean return (net)"); ax.set_title(title)
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_turnover(tov, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    m = tov.mean().reindex([c for c in ORDER if c in tov])
    ax.bar(m.index, m.values, color=[COLORS[k] for k in m.index]); ax.set_ylabel("Average monthly turnover  sum|dx|")
    plt.xticks(rotation=30, ha="right"); ax.grid(alpha=.3, axis="y"); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_weights(d, strategy, sectors, out):
    W = pd.read_parquet(d / f"weights_{strategy}.parquet")
    sec = sectors.reindex(W.columns).fillna("Unknown")
    SW = W.T.groupby(sec).sum().T
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.stackplot(SW.index, SW.values.T, labels=SW.columns, alpha=.85, colors=plt.cm.tab20.colors[:SW.shape[1]])
    ax.set_ylim(0, 1); ax.set_title(f"{strategy}: sector allocation over time"); ax.legend(ncol=6, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def ex_ante_vs_realized(d, universe, strategies, lookback=504):
    """Optimizer's curse diagnostic: ex-ante volatility predicted by the trailing sample covariance at
    each rebalance vs. the volatility realized over the following holding period."""
    R, _ = data.load_ff49(UNIVERSES["ff49"]["start"]) if universe == "ff49" else data.load_sp500(UNIVERSES["sp500"]["start"])
    rows = []
    for s in strategies:
        f = d / f"weights_{s}.parquet"
        if not f.exists():
            continue
        W = pd.read_parquet(f)
        dates = list(W.index)
        pred, real = [], []
        for i, dt in enumerate(dates):
            w = W.loc[dt]; w = w[w.abs() > 1e-8]
            pos = R.index.get_loc(dt)
            win = R.iloc[pos + 1 - lookback:pos + 1][w.index].dropna(axis=1)
            w = w.reindex(win.columns).fillna(0.0)
            Sig = np.cov(win.values, rowvar=False, ddof=1)
            pred.append(np.sqrt(252 * w.values @ Sig @ w.values))
            nxt = R.iloc[pos + 1: (R.index.get_loc(dates[i + 1]) + 1 if i + 1 < len(dates) else pos + 22)][w.index].fillna(0.0)
            if len(nxt) < 5:
                pred.pop(); continue
            real.append(np.sqrt(252) * (nxt.values @ w.values).std(ddof=1))
        pred, real = np.array(pred), np.array(real)
        rows.append(dict(Strategy=s, ExAnteVol=pred.mean(), RealizedVol_meanOfMonthly=real.mean(),
                         Ratio=real.mean() / pred.mean(), Corr=np.corrcoef(pred, real)[0, 1],
                         UnderestimationFreq=float((real > pred).mean())))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="ff49"); ap.add_argument("--tag", default="base")
    ap.add_argument("--nboot", type=int, default=2000)
    a = ap.parse_args()
    res, ret, d = load_run(a.universe, a.tag)
    rf = data.load_rf()
    sectors = data.load_ff49(UNIVERSES["ff49"]["start"])[1] if a.universe == "ff49" else data.load_sp500(UNIVERSES["sp500"]["start"])[1]
    pre = f"{a.universe}_{a.tag}"
    summ = metrics.summary_table(res, rf)
    summ.to_csv(TABLES / f"{pre}_summary.csv")
    title = {"ff49": "49 Fama-French industries, 1995-2026, monthly rebalancing, net of 10 bps costs",
             "sp500": "S&P 500 constituents (current), 2003-2026, monthly rebalancing, net of 10 bps costs"}[a.universe]
    fig_wealth(ret, FIGS / f"{pre}_wealth.png", title)
    fig_rolling_sharpe(ret, rf, FIGS / f"{pre}_rolling_sharpe.png")
    fig_risk_return(summ, FIGS / f"{pre}_risk_return.png", title)
    fig_turnover(pd.read_parquet(d / "turnover.parquet"), FIGS / f"{pre}_turnover.png")
    for s in ["MV-sample", "GMV-LW", "Min-CVaR"]:
        if (d / f"weights_{s}.parquet").exists():
            fig_weights(d, s, sectors, FIGS / f"{pre}_weights_{s}.png")
    # stress periods
    per = metrics.period_table(res, STRESS_PERIODS)
    per.to_csv(TABLES / f"{pre}_stress.csv", index=False)
    # Sharpe-difference tests vs EW and vs GMV-LW
    rows = []
    for bench in ["EW", "GMV-LW"]:
        for k in res:
            if k == bench:
                continue
            t = metrics.sharpe_diff_test(res[k].returns, res[bench].returns, rf, n_boot=a.nboot)
            rows.append(dict(Strategy=k, Benchmark=bench, **{kk: vv for kk, vv in t.items() if kk != "CI95"},
                             CI_lo=t["CI95"][0], CI_hi=t["CI95"][1]))
    pd.DataFrame(rows).to_csv(TABLES / f"{pre}_sharpe_tests.csv", index=False)
    # second-order stochastic dominance among monthly return distributions
    dom = metrics.dominance_matrix({k: v.returns for k, v in res.items()})
    dom.to_csv(TABLES / f"{pre}_ssd.csv")
    cfg = json.load(open(d / "config.json"))["cfg"]
    eva = ex_ante_vs_realized(d, a.universe, ORDER, lookback=cfg["lookback"])
    eva.to_csv(TABLES / f"{pre}_exante_vs_realized.csv", index=False)
    pd.set_option("display.width", 220)
    print(summ.round(3).to_string()); print(); print(per.round(3).to_string()); print()
    print(eva.round(3).to_string()); print()
    print(pd.DataFrame(rows).round(3).to_string()); print(); print("SSD matrix (row dominates column):"); print(dom)
