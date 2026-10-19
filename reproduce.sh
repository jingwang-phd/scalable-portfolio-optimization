#!/bin/bash
# Full pipeline after the data have been downloaded (see README, section 7).
set -e
cd "$(dirname "$0")/experiments"
python3 run_backtests.py --universe ff49  --tag base
python3 run_backtests.py --universe sp500 --tag base
python3 analyze_run.py --universe ff49  --tag base
python3 analyze_run.py --universe sp500 --tag base
python3 run_sensitivity.py --universe ff49
python3 plot_sensitivity.py --universe ff49
python3 run_sensitivity.py --universe sp500 --dims lookback gamma cost turnover_max alpha tc_scale
python3 plot_sensitivity.py --universe sp500
python3 test_solvers.py
python3 run_scale.py --time_limit 1200
python3 run_scale.py --N 500 --S 100000 500000 1000000 --methods monolithic rowgen admm --time_limit 2400 --out scale_large.csv
python3 run_scale.py --N 250 500 --S 50000 100000 --methods rowgen --rowgen_init subsample --out scale_rowgen_sub.csv
python3 plot_scale.py
