"""
Validation rungs 2 and 3, for the Route B tangent map.

Rung 2 -- full Hilbert space (N=4, d=4, D=16). The MPS manifold is all of
projective Hilbert space, so TDVP is exact unitary evolution and the
one-step tangent map must be an isometry: every singular value 1, every
exponent 0. This exercises the generator, the transport and the QR
bookkeeping together, and is the check that the curvature term K vanishes
where it should.

Rung 3 -- H_asym at the uniform thermofield double, where H psi ~ 0. The
generator must reduce to the relaxation strand's -i H_tangent, the step map
to expm(-i H_tangent dt), and the exponents to zero. Note the P H P block
of the generator *is* assemble_tangent_hamiltonian, so that part is shared
code rather than an independent check; what rung 3 actually tests is that
K -> 0 and that the transport is the identity to the same order.

Run from the repo root:  python lyapunov/tdvp_lyapunov/validate_benettin.py [2|3]
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

from lyapunov.tdvp_lyapunov.frame import Frame, frame_change
from lyapunov.tdvp_lyapunov.stepper import tdvp_step, exact_method, lanczos_method
from lyapunov.tdvp_lyapunov.tangent_generator import generator, step_matrix
from lyapunov.tdvp_lyapunov.benettin import benettin
from lyapunov.relaxation.run_relaxation_scan import build_uniform_thermofield
from lyapunov.relaxation import tangent_hamiltonian as tangent


def one_step_map(psi, H, dt, method, D):
    """The Route B tangent map over one step, as an explicit matrix."""
    frame = Frame(psi, D)
    psi_next = tdvp_step(psi, H, dt, method)
    frame_next = Frame(psi_next, D)
    M = step_matrix(frame, generator(frame, H), frame_next, generator(frame_next, H), dt)
    return frame, frame_next, M


def rung2(dt=0.05):
    N, D = 4, 16
    np.random.seed(0)
    psi = states.mps(states.random(N, 4, D, seed=3).tensors)
    psi.right_orthogonal(D)
    H = tf.thermofield_hamiltonian(ops.tilted_ising(N=N), asym=False)
    print(f"--- rung 2: N={N}, D={D}: full Hilbert space ---")
    t0 = clock.time()
    method = lanczos_method()
    frame, _, M = one_step_map(psi, H, dt, method, D)
    sv = la.svdvals(M)
    print(f"n = {frame.n} (want 4^{N}-1 = {4 ** N - 1}); singular values of the "
          f"one-step map in [{sv.min():.8f}, {sv.max():.8f}]  [{clock.time() - t0:.0f} s]")
    ev = la.eigvals(M)
    print(f"|eigenvalues| - 1 max: {np.abs(np.abs(ev) - 1).max():.1e}")

    t0 = clock.time()
    res = benettin(psi, H, dt, n_blocks=3, k=2 * frame.n, tau=1, method=method,
                   max_bond_dim=D, verbose=False)
    lam = res['exponents']
    print(f"Benettin, 3 blocks, k=2n: max|lambda| = {np.abs(lam).max():.2e}  "
          f"[{clock.time() - t0:.0f} s]")


def rung3(L=6, D=4, beta=1.0, dt=0.02):
    np.random.seed(0)
    psi, _ = build_uniform_thermofield(L, D, beta, steps=40)
    H_asym = tf.thermofield_hamiltonian(ops.tilted_ising(N=L), asym=True)
    print(f"--- rung 3: L={L}, D={D}, beta={beta}: H_asym fixed point ---")
    t0 = clock.time()
    frame, frame_next, M = one_step_map(psi, H_asym, dt, exact_method(), D)
    print(f"n = {frame.n}; map built in {clock.time() - t0:.0f} s")

    Hpsi_norm = np.sqrt(np.real(ops.expect(frame.state(), H_asym @ H_asym)))
    moved = 1 - abs(states.overlap(frame.state(), frame_next.state()))
    print(f"|H_asym psi| = {Hpsi_norm:.2e}; 1 - |<psi|Phi_dt psi>| = {moved:.2e}")
    sv = la.svdvals(M)
    print(f"singular values in [{sv.min():.6f}, {sv.max():.6f}] "
          f"(deviation from 1 should be ~ dt * residual)")

    W = H_asym.tensors
    left_envs, right_envs = tangent.build_environments(
        frame.A_L, frame.A_R, W, H_asym.l, H_asym.r, frame.sites)
    H_tan, _ = tangent.assemble_tangent_hamiltonian(
        frame.sites, frame.A_L, frame.A_R, frame.V_L, left_envs, right_envs, H_asym)
    U = la.expm(-1j * dt * H_tan)
    U_real = np.block([[U.real, -U.imag], [U.imag, U.real]])
    O = frame_change(frame, frame_next)
    M_same_frame = O.T @ M
    diff = la.norm(M_same_frame - U_real) / la.norm(U_real)
    print(f"|M - expm(-i H_tan dt)| / |.| = {diff:.2e}  (vs |M - 1| = "
          f"{la.norm(M_same_frame - np.eye(2 * frame.n)) / la.norm(U_real):.2e})")
    w_fd = np.sort(np.angle(la.eigvals(M_same_frame)) / dt)
    w_ex = np.sort(np.concatenate([la.eigvalsh(H_tan), -la.eigvalsh(H_tan)]))
    print(f"generator spectra: max |omega - omega_exact| = "
          f"{np.abs(w_fd - w_ex).max():.2e} (bandwidth {np.abs(w_ex).max():.2f})")


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    if which in ('2', 'both'):
        rung2()
    if which in ('3', 'both'):
        rung3()
