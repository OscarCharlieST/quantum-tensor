"""
Checks on langevin.brackets.poisson_brackets. Run from the repo root:

    python langevin/validate_brackets.py

1. Full bond dimension (P = 1): A_m = i<[H, F_m]>, K_mn = i<[F_m, F_n]>,
   against dense matrices. Also K is block diagonal here, exactly.
2. Truncated D: against 2 Im <B|P|G> with P built densely from an explicit
   orthonormal basis of tangent vectors -- an independent route to the same
   projector. Reports how large the off-site blocks of K are: the reference
   code sets them to zero.
3. End to end against qtensor's own tdvp_step: evolving under G for +-dt
   changes <F_n> at the rate the brackets predict. This pins down that the
   sign and ordering conventions (-i, MPO transpose, dt) agree with the
   integrator the Langevin step actually uses.
"""

import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())

import qtensor.states as states
import qtensor.simulation.finiteTDVP as sim
import qtensor.simulation.updatemethod as methods

import qtensor.operators as ops

from langevin.models import (PAULIS, ising_mpo, with_onsite_fields, dense_state,
                             dense_onsite, dense_ising, ising_local_terms, PAPER_PARAMS)
from langevin.brackets import Frame, poisson_brackets, local_brackets, block_diagonal


def random_state(N, D, seed):
    psi = states.random(N, 2, D, seed=seed)
    psi.right_orthogonal()
    return psi


def dense_F(N):
    return [dense_onsite(PAULIS[a], j, N) for j in range(N) for a in range(3)]


def check_full_rank(N=6, seed=1):
    psi = random_state(N, 2 ** (N // 2), seed)
    H = ising_mpo(N, **PAPER_PARAMS)
    A, K, f = poisson_brackets(psi, H)
    v = dense_state(psi)
    Hd = dense_ising(N, **PAPER_PARAMS)
    Fs = dense_F(N)
    A_ex = np.array([np.real(1j * v.conj() @ (Hd @ F - F @ Hd) @ v) for F in Fs])
    K_ex = np.array([[np.real(1j * v.conj() @ (Fm @ Fn - Fn @ Fm) @ v) for Fn in Fs]
                     for Fm in Fs])
    f_ex = np.array([np.real(v.conj() @ F @ v) for F in Fs])
    print(f"[full rank N={N}] |A - i<[H,F]>|   = {np.max(abs(A - A_ex)):.2e}")
    print(f"[full rank N={N}] |K - i<[F,F]>|   = {np.max(abs(K - K_ex)):.2e}")
    print(f"[full rank N={N}] |f - <F>|        = {np.max(abs(f - f_ex)):.2e}")
    print(f"[full rank N={N}] off-site |K|     = {np.max(abs(K - block_diagonal(K))):.2e}")
    return max(np.max(abs(A - A_ex)), np.max(abs(K - K_ex)), np.max(abs(f - f_ex)))


def tangent_basis_dense(frame):
    """Dense orthonormal basis of the single-site tangent space, column by column."""
    sites = frame.sites
    cols = []
    for k, n in enumerate(sites):
        d, Dl, Dr = frame.C[n].shape
        if n == sites[-1]:
            site_vecs = np.eye(d * Dl * Dr, dtype=complex)
        else:
            V = frame.V_L[n]                           # (d*Dl, nnull)
            site_vecs = np.einsum('xa,rq->xrqa', V, np.eye(Dr)).reshape(
                d * Dl * Dr, V.shape[1] * Dr)
        for c in range(site_vecs.shape[1]):
            B = site_vecs[:, c].reshape(d, Dl, Dr)
            tensors = {m: frame.A_L[m] for m in sites[:k]}
            tensors[n] = B
            tensors.update({m: frame.A_R[m] for m in sites[k + 1:]})
            cols.append(dense_state(states.mps(tensors)))
    return np.array(cols).T


def check_truncated(N=6, D=3, seed=2):
    psi = random_state(N, D, seed)
    H = ising_mpo(N, **PAPER_PARAMS)
    frame = Frame(psi)
    A, K, f = poisson_brackets(psi, H, frame)
    Q = tangent_basis_dense(frame)
    ortho = np.max(abs(Q.conj().T @ Q - np.eye(Q.shape[1])))
    P = Q @ Q.conj().T
    v = dense_state(frame.state())
    Hv = dense_ising(N, **PAPER_PARAMS) @ v
    Fv = [F @ v for F in dense_F(N)]
    A_ex = np.array([2 * np.imag(Fm.conj() @ P @ Hv) for Fm in Fv])
    K_ex = np.array([[2 * np.imag(Fn.conj() @ P @ Fm) for Fn in Fv] for Fm in Fv])
    off = np.max(abs(K - block_diagonal(K)))
    print(f"[D={D} N={N}] tangent dim {Q.shape[1]} of {2**N}, basis orthonormality {ortho:.1e}")
    print(f"[D={D} N={N}] |A - 2Im<F|P|H>| = {np.max(abs(A - A_ex)):.2e}")
    print(f"[D={D} N={N}] |K - 2Im<F|P|F>| = {np.max(abs(K - K_ex)):.2e}")
    print(f"[D={D} N={N}] off-site |K| = {off:.2e}  (max |K| = {np.max(abs(K)):.2e})"
          "  <- zero, since P F_j psi = F_j psi; the reference code's truncation is exact")
    return max(np.max(abs(A - A_ex)), np.max(abs(K - K_ex)))


def check_against_tdvp_step(N=6, D=3, seed=3, dt=1e-4):
    psi = random_state(N, D, seed)
    H = ising_mpo(N, **PAPER_PARAMS)
    A, K, f = poisson_brackets(psi, H)
    method = methods.exact_method()
    zero = ising_mpo(N, J=0, hx=0, hy=0, hz=0)

    def rate(G):
        fs = []
        for s in (+1, -1):
            phi, _ = sim.tdvp_step(copy.deepcopy(psi), G, s * dt, method)
            fs.append(poisson_brackets(phi, H)[2])
        return (fs[0] - fs[1]) / (2 * dt)

    err_A = np.max(abs(rate(H) - A))
    errs = []
    for m in (0, 4, 8, 3 * N - 1):
        coeffs = np.zeros(3 * N)
        coeffs[m] = 1.0
        errs.append(np.max(abs(rate(with_onsite_fields(zero, coeffs)) - K[m])))
    print(f"[tdvp_step D={D}] |d<F>/dt under H   - A|    = {err_A:.2e}")
    print(f"[tdvp_step D={D}] |d<F>/dt under F_m - K[m]| = {max(errs):.2e}  (dt = {dt})")
    return max(err_A, max(errs))


def check_local_path(N=8, D=4, seed=4):
    """The RDM fast path against the projector route, at truncated D."""
    psi = random_state(N, D, seed)
    H = ising_mpo(N, **PAPER_PARAMS)
    frame = Frame(psi)
    A, K, f = poisson_brackets(psi, H, frame)
    A2, K2, f2, E2 = local_brackets(frame, *ising_local_terms(**PAPER_PARAMS))
    E = np.real(ops.expect(frame.state(), H))
    err = max(np.max(abs(A - A2)), np.max(abs(K - K2)), np.max(abs(f - f2)), abs(E - E2))
    print(f"[local path D={D} N={N}] |A|,|K|,|f|,|E| vs projector route: "
          f"{np.max(abs(A - A2)):.1e} {np.max(abs(K - K2)):.1e} "
          f"{np.max(abs(f - f2)):.1e} {abs(E - E2):.1e}")
    return err


if __name__ == '__main__':
    errs = [check_full_rank(), check_truncated(), check_against_tdvp_step(),
            check_local_path()]
    ok = errs[0] < 1e-10 and errs[1] < 1e-10 and errs[2] < 1e-5 and errs[3] < 1e-10
    print("PASS" if ok else "FAIL")
