"""
Step 1 driver: the static energy-rate bracket against bond dimension, at
the two kinds of point the Lyapunov strand cares about.

  * the beta = 0.1 thermofield double itself (t = 0, nearly rank-deficient:
    the local-equilibrium state the formula is written for), and
  * points along the H_sym trajectory after the production transient
    (160 steps of dt = 0.05 with swap-even projection every step, then
    plain TDVP), where the Lyapunov spectrum is computed.

Same build as the quoted even-sector runs: 240 imaginary-time steps, no
seeding noise, --symmetrize-every 1. For each point, energy_rate.brackets
gives B_k in four variants; the summary is sum_k k^2 B_k, against the exact
infinite-temperature 8 G(0) = 4.41.

    python lyapunov/tdvp_lyapunov/run_energy_rate.py --L 8 --D 2 3 4 6 8 12 \
        [--samples 9 --sample-every 20]

writes runs/energy_rate_L{L}_D{D}.npz, one per (L, D).
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
from lyapunov.tdvp_lyapunov.stepper import tdvp_step, lanczos_method
from lyapunov.tdvp_lyapunov.symmetry import evolve, odd_amplitude
from lyapunov.tdvp_lyapunov.energy_rate import brackets, sum_k2, EXACT_SUM_K2B
from lyapunov.relaxation.run_relaxation_scan import build_uniform_thermofield

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
VARIANTS = ('B1', 'BPH', 'BPh', 'BP', 'BP_one', 'Csym', 'Cphys')
# Csym and Cphys are normalized to C_k; -4 C puts them on the bracket's
# scale, where every entry is +4.41 at beta = 0 on the TFD.
SCALE = {'Csym': -4.0, 'Cphys': -4.0}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--L', type=int, nargs='+', default=[8])
    p.add_argument('--D', type=int, nargs='+', default=[2, 3, 4, 6, 8, 12])
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--imag-steps', type=int, default=240)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--transient', type=int, default=160)
    p.add_argument('--symmetrize-every', type=int, default=1)
    p.add_argument('--samples', type=int, default=9,
                   help='trajectory points, the first at the end of the transient')
    p.add_argument('--sample-every', type=int, default=20, help='steps between them')
    p.add_argument('--out-dir', default=RUNS)
    return p.parse_args()


def record(frame, psi, t, rows):
    r = brackets(frame)
    r['t'] = t
    r['s_min'] = min(s.min() for s in frame.schmidt_values().values())
    r['a_odd'] = odd_amplitude(psi)
    rows.append(r)
    S = {v: SCALE.get(v, 1.0) * sum_k2(r[v], r['k']) for v in VARIANTS}
    print(f"   t={t:6.2f}  s_min={r['s_min']:.1e}  a_odd={r['a_odd']:.0e}  "
          f"cap_H={r['capture_H']:.4f} cap_h0={r['capture_h0']:.4f}  "
          + "  ".join(f"{v}={S[v].real:+.4f}{S[v].imag:+.3f}j" for v in VARIANTS),
          flush=True)


def run(L, D, a):
    t0 = clock.time()
    np.random.seed(0)
    psi, _ = build_uniform_thermofield(L, D, a.beta, a.imag_steps)
    H = tf.thermofield_hamiltonian(ops.tilted_ising(N=L), asym=False)
    method = lanczos_method()
    rows = []
    print(f"L={L} D={D} beta={a.beta}  (exact beta=0 sum k^2 B = {EXACT_SUM_K2B:.4f})")
    record(Frame(psi, D), psi, 0.0, rows)                       # the TFD
    psi, _ = evolve(psi, H, a.dt, a.transient, method, D, a.symmetrize_every)
    t = a.transient * a.dt
    for i in range(a.samples):
        if i:
            for _ in range(a.sample_every):
                psi = tdvp_step(psi, H, a.dt, method)
            t += a.sample_every * a.dt
        record(Frame(psi, D), psi, t, rows)
    out = {'t': np.array([r['t'] for r in rows]), 'k': rows[0]['k']}
    for key in VARIANTS + ('hk_mean',):
        out[key] = np.array([r[key] for r in rows])
    for key in ('capture_H', 'capture_h0', 's_min', 'a_odd', 'E_H'):
        out[key] = np.array([r[key] for r in rows])
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"energy_rate_L{L}_D{D}.npz")
    np.savez(path, L=L, D=D, beta=a.beta, dt=a.dt, transient=a.transient, **out)

    traj = slice(1, None)
    print(f"   -> {clock.time() - t0:.0f} s.  sum k^2 B, TFD | trajectory mean +- sd:")
    for v in VARIANTS:
        S = SCALE.get(v, 1.0) * np.array([sum_k2(B, out['k']) for B in out[v]])
        print(f"      {v:7s} {S[0].real:+.4f} | {S[traj].real.mean():+.4f} +- "
              f"{S[traj].real.std():.4f}   (Im {S[traj].imag.mean():+.4f})")
    return path


def main():
    a = parse()
    for L in a.L:
        for D in a.D:
            run(L, D, a)


if __name__ == '__main__':
    main()
