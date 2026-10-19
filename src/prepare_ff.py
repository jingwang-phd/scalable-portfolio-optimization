"""Parse Kenneth French daily files into tidy parquet tables.

Outputs (data/processed):
  ff49_vw_daily.parquet  -- 49 value-weighted industry portfolio *simple* daily returns (decimal)
  ff_factors_daily.parquet -- Mkt-RF, SMB, HML, RF (decimal)
Industries with missing data (-99.99) in the sample window are dropped at load time by the
experiment code, not here.
"""
import pathlib, io, re
import pandas as pd, numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

def read_block(path, header_marker, next_marker=None, marker_is_header=False):
    txt = path.read_text(errors="ignore").splitlines()
    start = next(i for i, l in enumerate(txt) if header_marker in l) + (0 if marker_is_header else 1)
    end = len(txt)
    if next_marker:
        end = next(i for i, l in enumerate(txt) if i > start and next_marker in l)
    block = "\n".join(txt[start:end])
    df = pd.read_csv(io.StringIO(block), index_col=0)
    df = df[df.index.astype(str).str.strip().str.fullmatch(r"\d{8}")]
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m%d")
    df.columns = [c.strip() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace({-99.99: np.nan, -999: np.nan})
    return df / 100.0

ind = read_block(RAW / "49_Industry_Portfolios_Daily.csv", "Average Value Weighted Returns -- Daily",
                 "Average Equal Weighted Returns -- Daily")
fac = read_block(RAW / "F-F_Research_Data_Factors_daily.csv", "Mkt-RF", marker_is_header=True)
ind.to_parquet(OUT / "ff49_vw_daily.parquet"); fac.to_parquet(OUT / "ff_factors_daily.parquet")
print("industries", ind.shape, ind.index.min().date(), ind.index.max().date())
print("missing per industry since 1990:", ind.loc["1990":].isna().sum().sort_values(ascending=False).head(5).to_dict())
print("factors", fac.shape, fac.index.min().date(), fac.index.max().date(), fac.columns.tolist())
