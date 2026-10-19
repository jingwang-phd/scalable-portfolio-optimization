"""Download the daily 49-industry portfolios and Fama-French factors from the Kenneth R. French data library."""
import io, pathlib, zipfile
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = ["49_Industry_Portfolios_daily_CSV.zip", "F-F_Research_Data_Factors_daily_CSV.zip"]

for f in FILES:
    r = requests.get(BASE + f, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for name in z.namelist():
            # normalise the file name so prepare_ff.py finds it regardless of the library's capitalisation
            target = "49_Industry_Portfolios_Daily.csv" if name.lower().startswith("49_industry") else "F-F_Research_Data_Factors_daily.csv"
            (RAW / target).write_bytes(z.read(name))
            print("saved", RAW / target)
