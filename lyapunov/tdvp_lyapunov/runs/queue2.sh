#!/bin/bash
# Second run queue (2026-09-16 evening). Everything on parallel Route A.
# h5 files go outside OneDrive; logs stay here; figures go to ../figures.
# Transient 160 steps (t = 8) for the new runs: the L=8, D=4 run showed
# lambda_max still ramping until t ~ 8, and plain TDVP steps are cheap.
cd "C:/Users/charl/OneDrive/Documents/GitHub/quantum-tensor"
PY=/c/Users/charl/anaconda3/python.exe
R=lyapunov/tdvp_lyapunov/runs
OUT=C:/Users/charl/lyapunov_runs
mkdir -p "$OUT"
stamp() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*"; }

stamp "waiting for L12_D4 (Route B, started under the first queue)"
until grep -qE "written|Traceback|Error" $R/L12_D4_beta1.log; do sleep 20; done
stamp "L12_D4 finished; plotting"
$PY lyapunov/tdvp_lyapunov/plots.py $R/L12_D4_beta1.h5 --discard 120 --clv --modes 0 1 -1 -2 -3 -4
stamp "L12_D4 plots done"

run() {   # run <tag-for-log> <plot-discard> <args...>
    local name=$1 discard=$2; shift 2
    stamp "START $name"
    $PY -u lyapunov/tdvp_lyapunov/run_lyapunov.py "$@" --route A --n-jobs 16 --out-dir "$OUT" > $R/$name.log 2>&1
    stamp "END   $name (exit $?); plotting"
    $PY lyapunov/tdvp_lyapunov/plots.py "$OUT/$name.h5" --discard $discard --clv --modes 0 1 -1 -2 -3 -4
    stamp "PLOTS $name"
}

run L8_D8_beta1         20  --L 8  --D 8  --blocks 250 --transient 160 --dt 0.05  --store-Q-every 25
run L8_D4_beta1_dt025   40  --L 8  --D 4  --blocks 600 --transient 320 --dt 0.025 --store-Q-every 50 --tag dt025
run L16_D4_beta1        20  --L 16 --D 4  --blocks 300 --transient 160 --dt 0.05  --store-Q-every 25
run L8_D12_beta1        20  --L 8  --D 12 --blocks 200 --transient 160 --dt 0.05  --store-Q-every 25

stamp "QUEUE2 DONE"
