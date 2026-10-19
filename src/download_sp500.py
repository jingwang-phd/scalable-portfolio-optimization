"""Download adjusted daily prices for current S&P 500 constituents (plus SPY) via yfinance.

Caveat recorded in the write-up: using *current* constituents introduces survivorship bias.
The Fama-French industry universe (download_ff.py) is survivorship-free and is used as the
primary long-history universe; the stock universe is used for realism (sector constraints,
position limits) and for the scalability experiments.
"""
import sys, time, pathlib, io
import pandas as pd, numpy as np, yfinance as yf, requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
START = "1999-01-01"

# constituent list and GICS sectors
html = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=60).text
pd.read_html(io.StringIO(html))[0].to_csv(RAW / "sp500_constituents_wiki.csv", index=False)

cons = pd.read_csv(RAW / "sp500_constituents_wiki.csv")
tickers = [s.replace(".", "-") for s in cons["Symbol"].tolist()]
tickers_all = sorted(set(tickers + ["SPY"]))
print(f"{len(tickers_all)} tickers", flush=True)

# sanity check on a tiny batch first
test = yf.download(["AAPL", "MSFT"], start="2024-01-01", end="2024-02-01", progress=False, auto_adjust=True)
if test.empty:
    sys.exit("yfinance returned empty frame -- aborting")

frames = []
B = 50
for i in range(0, len(tickers_all), B):
    batch = tickers_all[i:i + B]
    for attempt in range(3):
        try:
            df = yf.download(batch, start=START, progress=False, auto_adjust=True, threads=True, group_by="column")
            break
        except Exception as e:  # noqa
            print("retry", i, attempt, e, flush=True); time.sleep(5)
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    frames.append(close)
    print(f"batch {i//B+1}: {close.shape}", flush=True)
    time.sleep(1.0)

px = pd.concat(frames, axis=1).sort_index()
px = px.loc[:, ~px.columns.duplicated()]
px.index = pd.to_datetime(px.index).tz_localize(None)
px.to_parquet(RAW / "sp500_adjclose.parquet")
print("saved", px.shape, px.index.min(), px.index.max())
print("non-null per ticker (first/last):")
cnt = px.notna().sum().sort_values()
print(cnt.head(10)); print(cnt.tail(3))
