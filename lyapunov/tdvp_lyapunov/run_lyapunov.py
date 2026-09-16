"""
Driver: Lyapunov spectrum of TDVP under H_sym from the uniform thermofield
double, with everything Ginelli needs stored to h5.

Run from the repo root, e.g.

    python lyapunov/tdvp_lyapunov/run_lyapunov.py --L 8 --D 4 --beta 1 \
        --dt 0.05 --blocks 200 --tau 2 --transient 40

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
from lyapunov.tdvp_lyapunov.stepper import tdvp_step, tangent_map, exact_method, lanczos_method
from lyapunov.tdvp_lyapunov.benettin import benettin
from lyapunov.tdvp_lyapunov.tangent_generator import (
    generator, half_step_propagator, parallel_transport, propagate,
)
from lyapunov.relaxation.run_relaxation_scan import build_uniform_thermofield

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--L', type=int, default=8)
    p.add_argument('--D', type=int, default=4)
    p.add_argument('--beta', type=float, default=1.0)
    p.add_argument('--imag-steps', type=int, default=60)
    p.add_argument('--seed', type=int, default=0, help='rank-seeding noise RNG seed')
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--blocks', type=int, default=100)
    p.add_argument('--tau', type=int, default=1, help='TDVP steps per QR')
    p.add_argument('--transient', type=int, default=0, help='steps before tangent vectors start')
    p.add_argument('--k', default='half', help="'half' (n), 'full' (2n) or an integer")
    p.add_argument('--eps', type=float, default=1e-5)
    p.add_argument('--scheme', default='forward', choices=['forward', 'central'])
    p.add_argument('--method', default='lanczos', choices=['lanczos', 'exact'])
    p.add_argument('--route', default='B', choices=['A', 'B'])
    p.add_argument('--n-jobs', type=int, default=1, help='worker processes for route A')
    p.add_argument('--store-Q-every', type=int, default=10,
                   help='store Q and frame every this many blocks')
    p.add_argument('--tag', default='')
    p.add_argument('--out-dir', default=RUNS,
                   help='where the h5 goes; keep large runs out of OneDrive')
    p.add_argument('--time-only', action='store_true',
                   help='time one TDVP step and one tangent-map column, then exit')
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
    if a.route == 'A':
        # time a real slice of the tangent map rather than guessing overheads
        frame1 = Frame(psi1, a.D)
        cols = min(k, max(8, 2 * a.n_jobs))
        Y = np.linalg.qr(np.random.default_rng(0).normal(size=(2 * n, cols)))[0]
        if a.n_jobs > 1:
            tangent_map(frame, frame1, Y, H_sym, a.dt, method, a.eps, a.D, a.scheme, n_jobs=a.n_jobs)
        t0 = clock.time()
        tangent_map(frame, frame1, Y, H_sym, a.dt, method, a.eps, a.D, a.scheme, n_jobs=a.n_jobs)
        per_col = (clock.time() - t0) / cols
        per_step = per_col * k + t_step
        print(f"one TDVP step: {t_step * 1e3:.0f} ms; tangent map {per_col * 1e3:.0f} ms "
              f"per column with {a.n_jobs} worker(s), measured on {cols} columns")
    else:
        t0 = clock.time()
        A = generator(frame, H_sym)
        t_gen = clock.time() - t0
        t0 = clock.time()
        E = half_step_propagator(A, a.dt)
        t_exp = clock.time() - t0
        t0 = clock.time()
        T = parallel_transport(frame, Frame(psi1, a.D))
        Q = propagate(np.eye(2 * n, k), E, T, E)
        t_map = clock.time() - t0
        per_step = t_step + t_gen + t_exp + t_map
        print(f"one TDVP step: {t_step * 1e3:.0f} ms; generator {t_gen:.1f} s; "
              f"expm {t_exp:.1f} s; transport+apply {t_map:.1f} s")
    print(f"-> ~{per_step:.1f} s per step, ~{per_step * a.blocks * a.tau / 3600:.2f} h "
          f"for {a.blocks} blocks of {a.tau}")
    if a.time_only:
        return

    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"_{a.tag}" if a.tag else ''
    path = os.path.join(a.out_dir, f"L{a.L}_D{a.D}_beta{a.beta:g}{tag}.h5")
    store_blocks = set(range(0, a.blocks, a.store_Q_every)) | {a.blocks - 1}
    benettin(psi, H_sym, a.dt, a.blocks, k, tau=a.tau, method=method, eps=a.eps,
             max_bond_dim=a.D, scheme=a.scheme, transient_steps=a.transient,
             store_path=path, store_Q_blocks=store_blocks, seed=a.seed, route=a.route,
             n_jobs=a.n_jobs)
    print(f"written {path}")


if __name__ == '__main__':
    main()
