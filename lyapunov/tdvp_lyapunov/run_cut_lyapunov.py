"""
Driver: Lyapunov spectrum of TDVP on the one-cut manifold (bond r = N - m
at the centre cut, every other bond full), by cut_benettin.

The start is the beta thermofield double evolved exactly under H_sym to
t_scramble, truncated to rank r -- the same start as run_cut_rates.py.

Run from the repo root, e.g.

    python lyapunov/tdvp_lyapunov/run_cut_lyapunov.py --L 8 --m 4 --k 64 \
        --dt 0.05 --blocks 250 --out-dir C:/Users/charl/lyapunov_runs

--k full carries all 2(N^2 - m^2 - 1) directions (L <= 6 only in practice).
--time-only times one step and prints the estimate.
"""

import argparse
import os
import sys
import time as clock

import numpy as np
import scipy.linalg as la

sys.path.insert(0, os.getcwd())

import qtensor.operators as ops
import qtensor.thermofield as tf

from lyapunov.tdvp_lyapunov.cut_matrix import mpo_cut_terms, truncate
from lyapunov.tdvp_lyapunov.cut_rates import ExactThermofield, dense_thermofield
from lyapunov.tdvp_lyapunov.cut_benettin import (
    cut_benettin, start_state, tangent_dimension,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--L', type=int, default=8)
    p.add_argument('--m', type=int, default=4)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--t-scramble', type=float, default=20.0)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--blocks', type=int, default=250)
    p.add_argument('--k', default='64', help="an integer, or 'full'")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--store-Q-every', type=int, default=25)
    p.add_argument('--tag', default='')
    p.add_argument('--out-dir', default=RUNS)
    p.add_argument('--time-only', action='store_true')
    return p.parse_args()


def main():
    a = parse()
    L, cut, d = a.L, a.L // 2 - 1, 4
    H = tf.thermofield_hamiltonian(ops.tilted_ising(N=L), asym=False)
    terms = mpo_cut_terms(H, cut)
    exact = ExactThermofield(ops.tilted_ising(N=L), L, cut)
    Psi0 = exact.evolve(dense_thermofield(L, a.beta, terms, cut), a.t_scramble)
    N = Psi0.shape[0]
    r = N - a.m
    Psi0, discarded = truncate(Psi0, r)
    psi0 = start_state(Psi0, L, d, r, cut)
    k = tangent_dimension(N, a.m) if a.k == 'full' else int(a.k)
    print(f"L={L} m={a.m} (r={r}) beta={a.beta}: start at t={a.t_scramble}, "
          f"discarded {discarded:.2e}; real tangent dimension "
          f"{tangent_dimension(N, a.m)}, k = {k}", flush=True)

    if a.time_only:
        t0 = clock.time()
        cut_benettin(psi0, H, cut, r, a.dt, 2, k, verbose=False)
        per = (clock.time() - t0) / 2
        print(f"~{per:.1f} s per step -> ~{per * a.blocks / 3600:.2f} h for {a.blocks} blocks; "
              f"Q is {k * N * N * 16 / 1e6:.0f} MB per stored copy")
        return

    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"_{a.tag}" if a.tag else ''
    path = os.path.join(a.out_dir, f"cut_L{L}_m{a.m}_beta{a.beta:g}_k{k}{tag}.h5")
    store = set(range(0, a.blocks, a.store_Q_every)) | {a.blocks - 1}
    cut_benettin(psi0, H, cut, r, a.dt, a.blocks, k, exact=exact,
                 store_path=path, store_Q_blocks=store, seed=a.seed,
                 attrs={'L': L, 'beta': a.beta, 't_scramble': a.t_scramble,
                        'discarded0': discarded})
    print(f"written {path}")


if __name__ == '__main__':
    main()
