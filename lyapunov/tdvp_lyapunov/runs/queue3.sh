#!/bin/bash
# beta scan at L=8, D=8, full spectrum (k=2n), 2026-09-17.
# Question: does the programme stay numerically sound as beta -> 1e-2,
# where the thermofield double becomes nearly rank-1 and s_min collapses?
cd "C:/Users/charl/OneDrive/Documents/GitHub/quantum-tensor"
PY=/c/Users/charl/anaconda3/python.exe
R=lyapunov/tdvp_lyapunov/runs
OUT=C:/Users/charl/lyapunov_runs
stamp() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*"; }
for B in 1 0.1 0.01; do
    stamp "START beta=$B"
    $PY -u lyapunov/tdvp_lyapunov/run_lyapunov.py --L 8 --D 8 --beta $B --k full \
        --blocks 250 --transient 160 --dt 0.05 --store-Q-every 25 --tag k2n \
        --out-dir "$OUT" > $R/L8_D8_beta${B}_k2n.log 2>&1
    stamp "END   beta=$B (exit $?)"
done
stamp "QUEUE3 DONE"
