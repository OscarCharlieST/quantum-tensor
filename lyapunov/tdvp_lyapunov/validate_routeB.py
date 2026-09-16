"""
Route B checks (L=6, D=4, H_sym, beta=1).

  (a) w_N is orthogonal to the tangent space and to psi.
  (b) K against a finite difference of the full tangent projector:
      [P_{psi+eps v} - P_{psi-eps v}] H psi / 2eps, in the frame at psi,
      must equal conj(K X) for v = Phi(X).
  (c) the one-step map from step_matrix against the Route A finite-
      difference map, at two dt, with and without K.

Run from the repo root:  python lyapunov/tdvp_lyapunov/validate_routeB.py
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
    Frame, project_to_frame, tangent_mps, retract, frame_change, realify, complexify,
)
from lyapunov.tdvp_lyapunov.stepper import tdvp_step, tangent_map, lanczos_method
from lyapunov.tdvp_lyapunov.tangent_generator import (
    normal_residual, assemble_K, assemble_H_tan, realify_generator, step_matrix, apply_mpo,
)
from lyapunov.relaxation.run_relaxation_scan import build_uniform_thermofield


def full_projection_mps(frame, w_tensors):
    """P_full w = Phi(h) + <psi|w> psi as an MPS, for w given as tensors."""
    h = project_to_frame(w_tensors, frame)
    E = np.eye(1)
    for s in frame.sites:
        E = np.einsum('kb,pkl,pbm->lm', E, w_tensors[s], frame.A_L[s].conj())
    c = E[0, 0]
    return tangent_mps(frame, h, eps=1.0, include_point=True, point_weight=c)


def main(L=6, D=4, beta=1.0):
    rng = np.random.default_rng(1)
    np.random.seed(0)
    psi, _ = build_uniform_thermofield(L, D, beta, steps=20)
    H = tf.thermofield_hamiltonian(ops.tilted_ising(N=L), asym=False)
    method = lanczos_method()
    frame = Frame(psi, D)
    n = frame.n
    print(f"L={L} D={D}: n={n}, s_min={min(s.min() for s in frame.schmidt_values().values()):.1e}")

    # (a)
    t0 = clock.time()
    w_N, h, E = normal_residual(frame, H)
    wN_mps = states.mps(dict(w_N))
    norm_wN = np.sqrt(abs(states.overlap(wN_mps, wN_mps)))
    print(f"(a) |w_N| = {norm_wN:.4f}, |P^perp w_N| = {la.norm(project_to_frame(w_N, frame)):.1e}, "
          f"|<psi|w_N>| = {abs(states.overlap(frame.state(), wN_mps)):.1e}  [{clock.time() - t0:.1f} s]")

    # (b)
    t0 = clock.time()
    K = assemble_K(frame, w_N)
    print(f"    K assembled in {clock.time() - t0:.1f} s; |K - K^T| = {np.abs(K - K.T).max():.1e}, |K| = {la.norm(K):.3f}")
    X = rng.normal(size=n) + 1j * rng.normal(size=n)
    X /= la.norm(X)
    Hpsi_tensors = apply_mpo(frame.A_R, H)
    for eps in [1e-3, 1e-4, 1e-5]:
        proj = {}
        for sgn in (+1, -1):
            fr = Frame(retract(frame, X, sgn * eps, D), D)
            # H psi at the *same* psi, projected with the displaced frame
            proj[sgn] = project_to_frame(full_projection_mps(fr, Hpsi_tensors), frame)
        fd = (proj[+1] - proj[-1]) / (2 * eps)
        pred = np.conj(K @ X)
        print(f"(b) eps={eps:.0e}: |FD - conj(K X)| = {la.norm(fd - pred):.2e}   "
              f"(|conj(K X)| = {la.norm(pred):.3f}, |FD| = {la.norm(fd):.3f})")

    # (c)
    H_tan = assemble_H_tan(frame, H)
    for dt in [0.02, 0.01]:
        psi_next = tdvp_step(psi, H, dt, method)
        frame_next = Frame(psi_next, D)
        t0 = clock.time()
        M_A = tangent_map(frame, frame_next, np.eye(2 * n), H, dt, method, 1e-5, D, 'central')
        t_A = clock.time() - t0
        t0 = clock.time()
        w_N2, _, _ = normal_residual(frame_next, H)
        A = realify_generator(H_tan, K)
        A2 = realify_generator(assemble_H_tan(frame_next, H), assemble_K(frame_next, w_N2))
        M_B = step_matrix(frame, A, frame_next, A2, dt)
        t_B = clock.time() - t0
        A0 = realify_generator(H_tan, 0 * K)
        A20 = realify_generator(assemble_H_tan(frame_next, H), 0 * K)
        M_B0 = step_matrix(frame, A0, frame_next, A20, dt)
        O = frame_change(frame, frame_next)
        print(f"(c) dt={dt}: |M_A - M_B| = {la.norm(M_A - M_B):.2e}, without K: {la.norm(M_A - M_B0):.2e}, "
              f"|M_A - O| = {la.norm(M_A - O):.2e}   [Route A {t_A:.0f} s, Route B {t_B:.1f} s]")
        sv_A, sv_B = la.svdvals(M_A), la.svdvals(M_B)
        print(f"    top singular values: A {sv_A[:3].round(6)}, B {sv_B[:3].round(6)}")


if __name__ == '__main__':
    main()
