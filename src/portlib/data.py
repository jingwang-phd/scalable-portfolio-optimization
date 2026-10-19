"""Data loaders for the two universes.

Universe A -- Fama-French 49 value-weighted industry portfolios (daily, 1926-2026).
    Survivorship-free, long history (covers 2000-02, 2008, 2020, 2022). Sector groups are
    hand-mapped below so that sector-exposure constraints are meaningful.
Universe B -- current S&P 500 constituents, adjusted close from Yahoo (daily, 1999-2026).
    Realistic cross-section (N up to ~500, GICS sectors) but survivorship-biased: only firms
    that are in the index *today* are included. Used for realism and for scalability tests;
    performance numbers on this universe are therefore upward-biased for *all* strategies,
    and we report cross-strategy comparisons rather than absolute levels.
"""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"

FF49_SECTORS = {
    "Consumer Staples": ["Agric", "Food", "Soda", "Beer", "Smoke", "Hshld"],
    "Consumer Discretionary": ["Toys", "Fun", "Books", "Clths", "Autos", "Rtail", "Meals", "PerSv"],
    "Health Care": ["Hlth", "MedEq", "Drugs"],
    "Materials": ["Chems", "Rubbr", "Txtls", "BldMt", "Steel", "FabPr", "Gold", "Mines", "Coal", "Paper", "Boxes"],
    "Industrials": ["Cnstr", "Mach", "ElcEq", "Aero", "Ships", "Guns", "Trans", "Whlsl", "BusSv"],
    "Energy": ["Oil"],
    "Utilities": ["Util"],
    "Communication": ["Telcm"],
    "Information Technology": ["Hardw", "Softw", "Chips", "LabEq"],
    "Financials": ["Banks", "Insur", "RlEst", "Fin"],
    "Other": ["Other"],
}


def load_rf() -> pd.Series:
    fac = pd.read_parquet(PROC / "ff_factors_daily.parquet")
    return fac["RF"].rename("rf")


def load_ff49(start="1990-01-01", end=None):
    """Return (returns DataFrame, sector Series) for the FF49 industries, dropping any industry
    with missing data in the window."""
    R = pd.read_parquet(PROC / "ff49_vw_daily.parquet").loc[start:end]
    R = R.dropna(axis=1, how="any")
    sec = {}
    for s, inds in FF49_SECTORS.items():
        for i in inds:
            sec[i] = s
    sectors = pd.Series({c: sec[c] for c in R.columns}, name="sector")
    return R, sectors


def load_sp500(start="1999-01-01", end=None, min_obs=None):
    """Return (daily simple returns, GICS sector Series). Columns with fewer than `min_obs`
    observations are dropped. NaNs are kept (pre-listing periods) and handled by the backtester,
    which only admits assets with a complete trailing window at each rebalance date."""
    px = pd.read_parquet(RAW / "sp500_adjclose.parquet").loc[start:end]
    px = px.drop(columns=[c for c in ["SPY"] if c in px.columns])
    R = px.pct_change(fill_method=None)
    R = R.iloc[1:]
    # guard against bad ticks: clip absurd daily moves (|r| > 100%) to NaN, then forward-fill nothing
    R = R.mask(R.abs() > 1.0)
    if min_obs:
        R = R.loc[:, R.notna().sum() >= min_obs]
    cons = pd.read_csv(RAW / "sp500_constituents_wiki.csv")
    cons["ticker"] = cons["Symbol"].str.replace(".", "-", regex=False)
    sectors = cons.set_index("ticker")["GICS Sector"].reindex(R.columns).fillna("Unknown").rename("sector")
    return R, sectors


def load_spy(start="1999-01-01", end=None) -> pd.Series:
    px = pd.read_parquet(RAW / "sp500_adjclose.parquet")["SPY"].loc[start:end]
    return px.pct_change(fill_method=None).iloc[1:].rename("SPY")


def sector_matrix(sectors: pd.Series, columns) -> tuple[np.ndarray, list[str]]:
    """0/1 matrix G (n_sectors x N) mapping assets to sectors."""
    secs = sectors.reindex(columns)
    names = sorted(secs.dropna().unique().tolist())
    G = np.zeros((len(names), len(columns)))
    for j, c in enumerate(columns):
        if secs[c] in names:
            G[names.index(secs[c]), j] = 1.0
    return G, names
