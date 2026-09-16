"""
Validation ladder, rungs 2 and 3.

Rung 2 -- full Hilbert space (N=4, d=4, D=16). The MPS manifold is all of
projective Hilbert space, TDVP is exact unitary evolution, so the one-step
tangent map must be an isometry: every singular value 1, every exponent 0.
Checked directly on the full 2n x 2n finite-difference map, then through a
short Benettin run to exercise the loop.

Rung 3 -- H_asym at the uniform thermofield double (L=6, D=4). The step
map's generator should be the relaxation strand's -i H_tangent: compare
the spectrum of the one-step map (after removing the change of frame) with
expm(-i H_tangent dt), and check all singular values are 1 to within the
fixed-point residual.

Run from the repo root:  python lyapunov/tdvp_lyapunov/validate_benettin.py
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
    Frame, project_to_frame, tangent_mps, realify, complexify,
)
from lyapunov.tdvp_lyapunov.stepper import tdvp_step, tangent_map, exact_method, lanczos_method
from lyapunov.tdvp_lyapunov.benettin import benettin
from lyapunov.tdvp_lyapunov.analysis import pairing_residual
from lyapunov.relaxation.run_relaxation_scan import build_uniform_thermofield
from lyapunov.relaxation import tangent_hamiltonian as tangent


def frame_change(frame_a, frame_b):
    """
    Real orthogonal (2n_b, 2n_a) matrix of the identity map between two
    frames at (numerically) the same point: column j is the coordinates in
    frame_b of frame_a's j-th real basis vector.
    """
    assert frame_a.n == frame_b.n, "frames have different bond dimensions"
    O = np.zeros((2 * frame_b.n, 2 * frame_a.n))
    for j in range(frame_a.n):
        X = np.zeros(frame_a.n, dtype=complex)
        X[j] = 1.0
        v = tangent_mps(frame_a, X, include_point=False)
        c = project_to_frame(v, frame_b)
        O[:, j] = realify(c)
        O[:, frame_a.n + j] = realify(1j * c)
    return O


def full_tangent_map(psi, H, dt, method, D, eps, scheme):
    frame = Frame(psi, D)
    psi_next = tdvp_step(psi, H, dt, method)
    frame_next = Frame(psi_next, D)
    Y = np.eye(2 * frame.n)
    M = tangent_map(frame, frame_next, Y, H, dt, method, eps, D, scheme)
    return frame, frame_next, M


def rung2(dt=0.05, eps=1e-5):
    N, D = 4, 16
    np.random.seed(0)
    psi = states.mps(states.random(N, 4, D, seed=3).tensors)
    psi.right_orthogonal(D)
    H = tf.thermofield_hamiltonian(ops.tilted_ising(N=N), asym=False)
    print(f"--- rung 2: N={N}, D={D}: full Hilbert space ---")
    t0 = clock.time()
    method = lanczos_method()      # exact expm on 1024-dim local spaces is too slow here
    frame, frame_next, M = full_tangent_map(psi, H, dt, method, D, eps, 'central')
    sv = la.svdvals(M)
    print(f"n = {frame.n} (want 4^{N}-1 = {4**N - 1}); "
          f"singular values of the one-step map in [{sv.min():.8f}, {sv.max():.8f}]  "
          f"[{clock.time() - t0:.0f} s]")
    ev = la.eigvals(M)
    print(f"|eigenvalues| - 1 max: {np.abs(np.abs(ev) - 1).max():.1e}")

    # exact generator: -i H acting on tangent space is P(-iH)P = -iH on the
    # orthogonal complement of psi; the step map should be expm of it up to
    # the change of frame.
    t0 = clock.time()
    res = benettin(psi, H, dt, n_blocks=3, k=2 * frame.n, tau=1, method=method,
                   eps=eps, max_bond_dim=D, scheme='central', verbose=False)
    lam = res['exponents']
    print(f"Benettin, 3 blocks, k=2n: max|lambda| = {np.abs(lam).max():.2e} "
          f"(finite-difference floor ~ eps^2/dt = {eps**2 / dt:.0e})  [{clock.time() - t0:.0f} s]")


def rung3(L=6, D=4, beta=1.0, dt=0.02, eps=1e-5):
    np.random.seed(0)
    psi, _ = build_uniform_thermofield(L, D, beta, steps=40)
    H_phys = ops.tilted_ising(N=L)
    H_asym = tf.thermofield_hamiltonian(H_phys, asym=True)
    print(f"--- rung 3: L={L}, D={D}, beta={beta}: H_asym fixed point ---")
    t0 = clock.time()
    frame, frame_next, M = full_tangent_map(psi, H_asym, dt, exact_method(), D, eps, 'central')
    print(f"n = {frame.n}; FD map built in {clock.time() - t0:.0f} s")

    # residual of the fixed point, and how far the step moved it
    Hpsi_norm = np.sqrt(np.real(ops.expect(frame.state(), H_asym @ H_asym)))
    moved = 1 - abs(states.overlap(frame.state(), frame_next.state()))
    print(f"|H_asym psi| = {Hpsi_norm:.2e}; 1 - |<psi|Phi_dt psi>| = {moved:.2e}")

    sv = la.svdvals(M)
    print(f"singular values in [{sv.min():.6f}, {sv.max():.6f}]  "
          f"(deviation from 1 should be ~ dt * residual)")

    # compare with the relaxation strand's generator in *this* frame
    W = H_asym.tensors
    left_envs, right_envs = tangent.build_environments(
        frame.A_L, frame.A_R, W, H_asym.l, H_asym.r, frame.sites)
    H_tan, _ = tangent.assemble_tangent_hamiltonian(
        frame.sites, frame.A_L, frame.A_R, frame.V_L, left_envs, right_envs, H_asym)
    U = la.expm(-1j * dt * H_tan)
    # realify: X -> U X  becomes  y -> [[Re U, -Im U], [Im U, Re U]] y
    U_real = np.block([[U.real, -U.imag], [U.imag, U.real]])
    O = frame_change(frame, frame_next)          # frame -> frame_next
    M_same_frame = O.T @ M                        # back into the input frame
    print(f"frame change orthogonal to {np.abs(O.T @ O - np.eye(O.shape[1])).max():.1e}")
    diff = la.norm(M_same_frame - U_real) / la.norm(U_real)
    diff_id = la.norm(M_same_frame - np.eye(2 * frame.n)) / la.norm(U_real)
    print(f"|M - expm(-i H_tan dt)| / |.| = {diff:.2e}   (vs |M - 1| = {diff_id:.2e}, "
          f"|expm - 1| = {la.norm(U_real - np.eye(2 * frame.n)) / la.norm(U_real):.2e})")
    # eigenvalues of the realified map are e^{-i omega dt} and e^{+i omega dt}
    w_fd = np.sort(np.angle(la.eigvals(M_same_frame)) / dt)
    w_ex = np.sort(np.concatenate([la.eigvalsh(H_tan), -la.eigvalsh(H_tan)]))
    print(f"generator spectra: max |omega_fd - omega_exact| = "
          f"{np.abs(w_fd - w_ex).max():.2e} (bandwidth {np.abs(w_ex).max():.2f})")


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('2', 'both'):
        rung2()
    if which in ('3', 'both'):
        rung3()
