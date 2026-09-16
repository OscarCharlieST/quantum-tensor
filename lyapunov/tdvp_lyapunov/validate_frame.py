"""
Validation ladder, rung 1: the frame primitives.

  (a) project_to_frame(retract(psi, X, eps)) = eps * X + O(eps^2), same frame
  (b) tangent_mps(X, include_point=False) has norm |X| and is orthogonal to psi
  (c) dimension n matches sum_n (d D_{n-1} - D_n) D_n
  (d) the TDVP step map is invariant under a random gauge transformation of
      its input, and its output is phase-smooth in the input (the
      finite-difference column has an eps-plateau)

Run from the repo root:  python lyapunov/tdvp_lyapunov/validate_frame.py
"""

import os
import sys
import time as clock

import numpy as np
import scipy.linalg as la

sys.path.insert(0, os.getcwd())

import qtensor.states as states
import qtensor.operators as ops
import qtensor.thermofield as tf

from lyapunov.tdvp_lyapunov.frame import (
    Frame, project_to_frame, tangent_mps, retract, realify, complexify,
)
from lyapunov.tdvp_lyapunov.stepper import (
    tdvp_step, phase_fixed_projection, exact_method, lanczos_method,
)
from lyapunov.relaxation.run_relaxation_scan import build_uniform_thermofield


def random_gauge(psi, seed=1):
    """Same state, different tensors: insert G G^-1 on every bond."""
    rng = np.random.default_rng(seed)
    sites = sorted(psi.sites)
    T = {s: psi[s].copy() for s in sites}
    for a, b in zip(sites[:-1], sites[1:]):
        D = T[a].shape[2]
        G = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
        T[a] = T[a] @ G
        T[b] = la.inv(G) @ T[b]
    out = states.mps(T)
    return out


def main(L=6, D=4, beta=1.0):
    rng = np.random.default_rng(0)
    np.random.seed(0)
    psi, _ = build_uniform_thermofield(L, D, beta, steps=20)
    H_phys = ops.tilted_ising(N=L)
    H_sym = tf.thermofield_hamiltonian(H_phys, asym=False)

    frame = Frame(psi, D)
    bd = [1] + frame.bond_dims() + [1]
    n_expected = sum((4 * bd[i] - bd[i + 1]) * bd[i + 1] for i in range(L))
    print(f"L={L} D={D}: bond dims {bd}, n={frame.n} (expected {n_expected})")
    assert frame.n == n_expected
    smin = min(s.min() for s in frame.schmidt_values().values())
    print(f"smallest Schmidt value on any bond: {smin:.2e}")

    X = rng.normal(size=frame.n) + 1j * rng.normal(size=frame.n)
    X /= la.norm(X)

    # (b) bare tangent vector: norm and orthogonality to psi
    v = states.mps(tangent_mps(frame, X, include_point=False))
    print(f"(b) |Phi(X)| = {abs(states.overlap(v, v)):.12f}  (want 1), "
          f"<psi|Phi(X)> = {abs(states.overlap(frame.state(), v)):.2e}  (want 0), "
          f"|project(Phi(X)) - X| = {la.norm(project_to_frame(v, frame) - X):.2e}")

    # (a) retract then project, several eps
    print("(a) |project(retract(X, eps))/eps - X| :")
    for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        phi = retract(frame, X, eps, D)
        Xb = phase_fixed_projection(phi, frame) / eps
        print(f"      eps={eps:.0e}: {la.norm(Xb - X):.2e}")

    # (d) gauge invariance of the step and eps-plateau of a FD column
    dt = 0.05
    variants = [('exact', exact_method())]
    variants += [(f'lanczos eps={e:.0e}', lanczos_method(epsilon=e))
                 for e in [1e-5, 1e-8, 1e-10]]
    for name, method in variants:
        t0 = clock.time()
        out1 = tdvp_step(psi, H_sym, dt, method)
        t_step = clock.time() - t0
        out2 = tdvp_step(random_gauge(psi), H_sym, dt, method)
        ov = states.overlap(out1, out2)
        print(f"(d) {name}: step time {t_step*1e3:.1f} ms, "
              f"|<step(psi)|step(gauge psi)>| - 1 = {abs(ov) - 1:.2e}")
        frame_next = Frame(out1, D)
        cols = {}
        for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            plus = tdvp_step(retract(frame, X, eps, D), H_sym, dt, method)
            minus = tdvp_step(retract(frame, X, -eps, D), H_sym, dt, method)
            pp = phase_fixed_projection(plus, frame_next)
            pm = phase_fixed_projection(minus, frame_next)
            cols[eps] = (pp - pm) / (2 * eps)
        ref = cols[1e-5]
        for eps, c in cols.items():
            print(f"      eps={eps:.0e}: |col| = {la.norm(c):.6f}, "
                  f"|col - col(1e-5)| = {la.norm(c - ref):.2e}")


if __name__ == '__main__':
    main()
