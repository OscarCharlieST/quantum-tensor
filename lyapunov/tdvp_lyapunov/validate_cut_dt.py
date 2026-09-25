"""
Time-step error of the one-cut tangent map, on a fixed trajectory.

Comparing whole runs at different dt mixes two things: the trajectory is
itself chaotic (lambda ~ 3 at L = 6), so runs at different dt decorrelate
within t ~ 2 and their exponent estimates differ by finite-time noise. Here
the trajectory is computed once at a fine step dt_f, and the same starting
vectors are carried along it with the tangent step coarsened to s * dt_f
(every s-th point). With trajectory and vectors shared, the differences in
lambda_i(T) are the tangent integrator's discretization error alone.

The trajectory's own dt error is a separate matter: the exponents are
averages over the orbit, so they need a typical orbit, not an accurate one.

Run from the repo root:
    python lyapunov/tdvp_lyapunov/validate_cut_dt.py --L 6 --m 4 --k 32 --T 10
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

from lyapunov.tdvp_lyapunov.benettin import positive_qr
from lyapunov.tdvp_lyapunov.stepper import tdvp_step, lanczos_method
from lyapunov.tdvp_lyapunov.cut_matrix import mpo_cut_terms, mps_to_matrix, truncate, CutPoint
from lyapunov.tdvp_lyapunov.cut_rates import ExactThermofield, dense_thermofield
from lyapunov.tdvp_lyapunov.cut_benettin import (
    start_state, random_tangent, generator_norm, step_map,
    to_real, from_real,
)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--L', type=int, default=6)
    p.add_argument('--m', type=int, default=4)
    p.add_argument('--k', type=int, default=32)
    p.add_argument('--T', type=float, default=10.0)
    p.add_argument('--dt-fine', type=float, default=0.00625)
    p.add_argument('--strides', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--t-scramble', type=float, default=20.0)
    return p.parse_args()


def fine_trajectory(psi0, H, dt, steps, cut):
    """Psi (normalized, dense) at every fine step, t = 0 .. steps * dt."""
    method = lanczos_method()
    psi, out = psi0, []
    for j in range(steps + 1):
        if j:
            psi = tdvp_step(psi, H, dt, method)
        Psi = mps_to_matrix(psi, cut)
        out.append(Psi / la.norm(Psi))
    return np.array(out)


def tangent_along(Psis, stride, dt_f, r, terms, Q0):
    """Benettin along Psis[::stride]: per-step log R_ii, (n_steps, k)."""
    pts = Psis[::stride]
    dt = stride * dt_f
    point = CutPoint(pts[0], r, terms)
    norm = generator_norm(point)
    Q, shape, logs = Q0.copy(), pts[0].shape, []
    for Psi in pts[1:]:
        nxt = CutPoint(Psi, r, terms)
        norm_next = generator_norm(nxt)
        Q = step_map(Q, point, nxt, dt, (norm, norm_next))
        Y, R = positive_qr(to_real(Q))
        Q = from_real(Y, shape)
        logs.append(np.log(np.diag(R)))
        point, norm = nxt, norm_next
    return np.array(logs)


def main():
    a = parse()
    L, cut = a.L, a.L // 2 - 1
    H = tf.thermofield_hamiltonian(ops.tilted_ising(N=L), asym=False)
    terms = mpo_cut_terms(H, cut)
    exact = ExactThermofield(ops.tilted_ising(N=L), L, cut)
    Psi0 = exact.evolve(dense_thermofield(L, a.beta, terms, cut), a.t_scramble)
    N = Psi0.shape[0]
    r = N - a.m
    Psi0, _ = truncate(Psi0, r)
    psi0 = start_state(Psi0, L, 4, r, cut)
    steps = int(round(a.T / a.dt_fine))
    assert all(steps % s == 0 for s in a.strides)

    t0 = clock.time()
    Psis = fine_trajectory(psi0, H, a.dt_fine, steps, cut)
    print(f"L={L} m={a.m} k={a.k} T={a.T}: fine trajectory, {steps} steps of "
          f"{a.dt_fine}, in {clock.time() - t0:.0f} s", flush=True)
    Q0 = random_tangent(CutPoint(Psis[0], r, terms), a.k)

    lam = {}
    for s in sorted(a.strides):
        t0 = clock.time()
        logs = tangent_along(Psis, s, a.dt_fine, r, terms, Q0)
        lam[s] = logs.sum(0) / a.T
        print(f"  dt = {s * a.dt_fine:.5f}: lambda[0,1,2,k/4,k/2,k-1] = "
              f"{np.round(lam[s][[0, 1, 2, a.k // 4, a.k // 2, -1]], 4)}  "
              f"sum = {lam[s].sum():+.4f}  [{clock.time() - t0:.0f} s]", flush=True)

    ref = lam[min(a.strides)]
    print(f"error against dt = {min(a.strides) * a.dt_fine:.5f} "
          f"(max over the top {a.k}, top exponent, sum of the top {a.k}):")
    prev = None
    for s in sorted(a.strides)[1:]:
        err = np.abs(lam[s] - ref).max()
        order = f"  (x{err / prev:.2f} on doubling dt)" if prev else ''
        print(f"  dt = {s * a.dt_fine:.5f}: max {err:.2e}, lambda_1 "
              f"{abs(lam[s][0] - ref[0]):.2e}, sum {abs(lam[s].sum() - ref.sum()):.2e}{order}")
        prev = err


if __name__ == '__main__':
    main()
