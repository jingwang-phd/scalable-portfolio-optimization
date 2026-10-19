"""Experiment configuration shared by all scripts."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from portlib.models import Strategy, Constraints  # noqa: E402

RESULTS = ROOT / "results"
TABLES, FIGS, RUNS = RESULTS / "tables", RESULTS / "figures", RESULTS / "runs"
for p in (TABLES, FIGS, RUNS):
    p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- universes
UNIVERSES = {
    # survivorship-free, long history; 49 industries
    "ff49": dict(start="1990-01-01", oos_start="1995-01-31", oos_end="2026-08-31",
                 w_max=0.10, s_max=0.30, min_assets=20),
    # current S&P 500 constituents; universe grows from ~350 (2003) to ~495 (2025)
    "sp500": dict(start="1999-01-01", oos_start="2003-01-31", oos_end="2026-09-30",
                  w_max=0.05, s_max=0.25, min_assets=100),
}

# ---------------------------------------------------------------- base configuration
BASE = dict(lookback=504, gamma=5.0, kappa=2.0, alpha=0.05, c_cvar=0.25, c_semi=0.75,
            turnover_max=0.50, cost=0.001, tc_scale=12.0, freq="M")


def strategies(cfg=BASE):
    g, k, a = cfg["gamma"], cfg["kappa"], cfg["alpha"]
    return [
        Strategy("EW", "ew"),
        Strategy("MV-sample", "mv", {"gamma": g}),
        Strategy("MV-shrink", "mv", {"gamma": g}, estimator="lw", mean_estimator="bs"),
        Strategy("GMV-LW", "gmv", {"gamma": g}, estimator="lw"),
        Strategy("Robust-MV", "robust_mv", {"gamma": g, "kappa": k}),
        Strategy("Mean-CVaR", "cvar", {"alpha": a, "c": cfg["c_cvar"]}),
        Strategy("Min-CVaR", "cvar", {"alpha": a, "c": 1.0}),
        Strategy("Mean-SemiDev", "semidev", {"c": cfg["c_semi"]}),
    ]


def constraints(universe, cfg=BASE):
    u = UNIVERSES[universe]
    return Constraints(w_max=u["w_max"], s_max=u["s_max"], turnover_max=cfg["turnover_max"],
                       cost=cfg["cost"], tc_scale=cfg["tc_scale"])


STRESS_PERIODS = {
    "Dot-com bust (2000-03..2002-09)": ("2000-03-24", "2002-10-09"),
    "GFC (2007-10..2009-03)": ("2007-10-09", "2009-03-09"),
    "COVID crash (2020-02-19..2020-03-23)": ("2020-02-19", "2020-03-23"),
    "2020 full year": ("2020-01-01", "2020-12-31"),
    "2022 rate shock (2022-01..2022-10)": ("2022-01-03", "2022-10-12"),
    "2025 tariff shock (2025-02-19..2025-04-08)": ("2025-02-19", "2025-04-08"),
}
