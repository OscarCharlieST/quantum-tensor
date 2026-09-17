"""
Driver: Lyapunov spectrum of TDVP under H_sym from the uniform thermofield
double, with everything Ginelli needs stored to h5.

Run from the repo root, e.g.

    python lyapunov/tdvp_lyapunov/run_lyapunov.py --L 8 --D 4 --beta 0.1 \
        --dt 0.05 --blocks 200 --transient 160 --out-dir C:/Users/charl/lyapunov_runs

beta defaults to 0.1: hydrodynamics is a high-temperature expectation, and
the spectrum is better conditioned there (see the README temperature scan).

Output: runs/L{L}_D{D}_beta{beta}_{tag}.h5 next to this script. The
default k is n (the non-negative half of the spectrum); pass --k for
another value, e.g. --k full for all 2n.
"""

import argparse
import os
import sys
import time as clock

import numpy as np

sys.path.insert(0, os.getcwd())

import qtensor.operators as ops
import qtensor.thermofield as tf

from lyapunov.tdvp_lyapunov.frame import Frame
from lyapunov.tdvp_lyapunov.stepper import tdvp_step, exact_method, lanczos_method
from lyapunov.tdvp_lyapunov.benettin import benettin
from lyapunov.tdvp_lyapunov.tangent_generator import (
    generator, parallel_transport, propagate,
)
from lyapunov.relaxation.run_relaxation_scan import build_uniform_thermofield

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--L', type=int, default=8)
    p.add_argument('--D', type=int, default=4)
    p.add_argument('--beta', type=float, default=0.1,
                   help='inverse temperature; the default is the standing '
                        'convention, see the README')
    p.add_argument('--imag-steps', type=int, default=60)
    p.add_argument('--seed', type=int, default=0, help='rank-seeding noise RNG seed')
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--blocks', type=int, default=100)
    p.add_argument('--tau', type=int, default=1, help='TDVP steps per QR')
    p.add_argument('--transient', type=int, default=0, help='steps before tangent vectors start')
    p.add_argument('--k', default='half', help="'half' (n), 'full' (2n) or an integer")
    p.add_argument('--method', default='lanczos', choices=['lanczos', 'exact'])
    p.add_argument('--store-Q-every', type=int, default=10,
                   help='store Q and frame every this many blocks')
    p.add_argument('--tag', default='')
    p.add_argument('--out-dir', default=RUNS,
                   help='where the h5 goes; keep large runs out of OneDrive')
    p.add_argument('--time-only', action='store_true',
                   help='time one step of everything, print an estimate, exit')
    return p.parse_args()


def main():
    a = parse()
    np.random.seed(a.seed)
    psi, energy = build_uniform_thermofield(a.L, a.D, a.beta, a.imag_steps)
    H_sym = tf.thermofield_hamiltonian(ops.tilted_ising(N=a.L), asym=False)
    method = lanczos_method() if a.method == 'lanczos' else exact_method()

    frame = Frame(psi, a.D)
    n = frame.n
    k = {'half': n, 'full': 2 * n}.get(a.k, None)
    if k is None:
        k = int(a.k)
    smin = min(s.min() for s in frame.schmidt_values().values())
    print(f"L={a.L} D={a.D} beta={a.beta}: E/L = {energy / a.L:.4f}, n = {n}, k = {k}, "
          f"bond dims {frame.bond_dims()}, s_min = {smin:.1e}")

    t0 = clock.time()
    psi1 = tdvp_step(psi, H_sym, a.dt, method)
    t_step = clock.time() - t0
    t0 = clock.time()
    A = generator(frame, H_sym)
    t_gen = clock.time() - t0
    t0 = clock.time()
    T = parallel_transport(frame, Frame(psi1, a.D))
    t_tr = clock.time() - t0
    t0 = clock.time()
    Q = propagate(np.eye(2 * n, k), A, T, A, a.dt)
    t_map = clock.time() - t0
    per_step = t_step + t_gen + t_tr + t_map
    print(f"one TDVP step: {t_step * 1e3:.0f} ms; generator {t_gen:.1f} s; "
          f"transport {t_tr:.1f} s; exponential action {t_map:.1f} s")
    print(f"-> ~{per_step:.1f} s per step, ~{per_step * a.blocks * a.tau / 3600:.2f} h "
          f"for {a.blocks} blocks of {a.tau}")
    if a.time_only:
        return

    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"_{a.tag}" if a.tag else ''
    path = os.path.join(a.out_dir, f"L{a.L}_D{a.D}_beta{a.beta:g}{tag}.h5")
    store_blocks = set(range(0, a.blocks, a.store_Q_every)) | {a.blocks - 1}
    benettin(psi, H_sym, a.dt, a.blocks, k, tau=a.tau, method=method,
             max_bond_dim=a.D, transient_steps=a.transient,
             store_path=path, store_Q_blocks=store_blocks, seed=a.seed)
    print(f"written {path}")


if __name__ == '__main__':
    main()
