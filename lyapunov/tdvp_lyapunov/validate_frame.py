"""
Validation rung 1: the frame primitives that everything else is built on.

  (a) project_to_frame(Phi(X)) = X, and Phi(X) has unit norm and is
      orthogonal to psi
  (b) project_to_frame(retract(psi, X, eps)) = eps X + O(eps^2)
  (c) dimension n matches sum_n (d D_{n-1} - D_n) D_n
  (d) one TDVP step is invariant under a random gauge transformation of
      its input (it must depend on the state, not its representation)

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
    Frame, project_to_frame, tangent_mps, retract,
)
from lyapunov.tdvp_lyapunov.stepper import tdvp_step, exact_method, lanczos_method
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
    return states.mps(T)


def main(L=6, D=4, beta=1.0):
    rng = np.random.default_rng(0)
    np.random.seed(0)
    psi, _ = build_uniform_thermofield(L, D, beta, steps=20)
    H_sym = tf.thermofield_hamiltonian(ops.tilted_ising(N=L), asym=False)

    frame = Frame(psi, D)
    bd = [1] + frame.bond_dims() + [1]
    n_expected = sum((4 * bd[i] - bd[i + 1]) * bd[i + 1] for i in range(L))
    print(f"(c) L={L} D={D}: bond dims {bd}, n={frame.n} (expected {n_expected})")
    assert frame.n == n_expected
    smin = min(s.min() for s in frame.schmidt_values().values())
    print(f"    smallest Schmidt value on any bond: {smin:.2e}")

    X = rng.normal(size=frame.n) + 1j * rng.normal(size=frame.n)
    X /= la.norm(X)

    v = states.mps(tangent_mps(frame, X, include_point=False))
    print(f"(a) |Phi(X)| = {abs(states.overlap(v, v)):.12f}  (want 1), "
          f"<psi|Phi(X)> = {abs(states.overlap(frame.state(), v)):.2e}  (want 0), "
          f"|project(Phi(X)) - X| = {la.norm(project_to_frame(v, frame) - X):.2e}")

    print("(b) |project(retract(X, eps))/eps - X| :")
    for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        phi = retract(frame, X, eps, D)
        print(f"      eps={eps:.0e}: {la.norm(project_to_frame(phi, frame) / eps - X):.2e}")

    dt = 0.05
    for name, method in [('exact', exact_method()), ('lanczos', lanczos_method())]:
        t0 = clock.time()
        out1 = tdvp_step(psi, H_sym, dt, method)
        t_step = clock.time() - t0
        out2 = tdvp_step(random_gauge(psi), H_sym, dt, method)
        ov = states.overlap(out1, out2)
        print(f"(d) {name}: step time {t_step * 1e3:.1f} ms, "
              f"|<step(psi)|step(gauge psi)>| - 1 = {abs(ov) - 1:.2e}")


if __name__ == '__main__':
    main()
