"""
Validation of the one-cut matrix picture (cut_matrix.py) against the
Route B machinery it replaces at large bond dimension.

    1. mpo_cut_terms reproduces apply_mpo and ops.expect
    2. P is the frame's tangent projector plus the psi direction
    3. the tangent part of G in frame coordinates equals generator(frame, H)
    4. the symmetric part of that generator has spectrum +-nu_a/s_i (x2)
    5. m = 0: the generator is antisymmetric (no curvature, no exponents)
    (3 also checks the stacked generator against the term-by-term one)

Run from the repo root:  python lyapunov/tdvp_lyapunov/validate_cut.py [L ...]
"""

import os
import sys
import time as clock

import numpy as np
import scipy.linalg as la

sys.path.insert(0, os.getcwd())

import qtensor.operators as ops
import qtensor.states as states
import qtensor.thermofield as tf

from lyapunov.tdvp_lyapunov.frame import Frame, tangent_mps
from lyapunov.tdvp_lyapunov.tangent_generator import apply_mpo, generator
from lyapunov.tdvp_lyapunov.cut_matrix import (
    mpo_cut_terms, apply_H, mps_to_matrix, dense_to_mps, truncate, CutPoint,
)

D_LOC = 4


def hamiltonian(L):
    return tf.thermofield_hamiltonian(ops.tilted_ising(N=L), asym=False)


def rel(a, b):
    return la.norm(a - b) / max(la.norm(b), 1e-300)


def random_point(L, m, seed=0):
    """A random state of the one-cut manifold: dense, truncated, to MPS."""
    cut = L // 2 - 1
    N = D_LOC ** (L // 2)
    rng = np.random.default_rng(seed)
    Psi = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    Psi, _ = truncate(Psi, N - m)
    return states.mps(dense_to_mps(Psi.reshape(-1), L, D_LOC, N - m, cut)), cut, N


def frame_basis(frame, cut):
    """Dense (n, dL, dR) stack of the frame's orthonormal basis vectors."""
    n = frame.n
    B = []
    for j in range(n):
        X = np.zeros(n, dtype=complex)
        X[j] = 1.0
        B.append(mps_to_matrix(tangent_mps(frame, X, include_point=False), cut))
    return np.array(B)


def rung1(L):
    H = hamiltonian(L)
    cut = L // 2 - 1
    terms = mpo_cut_terms(H, cut)
    psi = states.random(L, D_LOC, 5, seed=L)
    Psi = mps_to_matrix(psi, cut)
    e_apply = rel(apply_H(terms, Psi), mps_to_matrix(apply_mpo(psi.tensors, H), cut))
    e_expect = abs(np.vdot(Psi, apply_H(terms, Psi)) - ops.expect(psi, H)) / abs(ops.expect(psi, H))
    print(f"  rung 1, L={L}: H Psi vs apply_mpo {e_apply:.1e}; <H> vs ops.expect {e_expect:.1e}")
    return max(e_apply, e_expect)


def rungs234(L, m, seed=0):
    t0 = clock.time()
    H = hamiltonian(L)
    psi, cut, N = random_point(L, m, seed)
    r = N - m
    terms = mpo_cut_terms(H, cut)
    frame = Frame(psi, r)
    point = CutPoint(mps_to_matrix(frame.state(), cut), r, terms)
    n = frame.n
    assert n == N * N - m * m - 1, f"n = {n}, expected {N * N - m * m - 1}"

    B = frame_basis(frame, cut)                                   # (n, N, N)
    Bm = B.reshape(n, -1).T                                        # (N^2, n)
    gram = la.norm(Bm.conj().T @ Bm - np.eye(n))

    # rung 2: P = B B^dag + psi psi^dag on random vectors, and P fixes B
    rng = np.random.default_rng(seed + 1)
    Z = rng.normal(size=(8, N, N)) + 1j * rng.normal(size=(8, N, N))
    psi_v = point.Psi.reshape(-1) / la.norm(point.Psi)
    Zm = Z.reshape(8, -1).T
    PZ_frame = Bm @ (Bm.conj().T @ Zm) + np.outer(psi_v, psi_v.conj() @ Zm)
    e_P = rel(point.P(Z).reshape(8, -1).T, PZ_frame)
    e_fix = rel(point.P(B), B)

    # rung 3: tangent part of G in frame coordinates vs Route B's A
    A_ref = generator(frame, H)
    GB = point.generator(B)
    e_fast = rel(GB, point.generator_reference(terms, B))
    C1 = Bm.conj().T @ GB.reshape(n, -1).T
    C2 = Bm.conj().T @ point.generator(1j * B).reshape(n, -1).T
    A_cut = np.block([[C1.real, C2.real], [C1.imag, C2.imag]])
    e_gen = rel(A_cut, A_ref)

    # rung 4: symmetric part -> +-nu/s, each twice, and zeros
    sym = la.eigvalsh(0.5 * (A_ref + A_ref.T))[::-1]
    rates = point.rate_spectrum()
    k = rates.size
    pred = np.concatenate([rates, np.zeros(2 * n - 2 * k), -rates[::-1]])
    e_sym = la.norm(sym - pred) / max(la.norm(sym), 1e-300)
    print(f"  L={L} m={m} (n={n}): basis Gram {gram:.1e}; rung 2 P {e_P:.1e}, "
          f"P(B)=B {e_fix:.1e}; rung 3 G vs Route B {e_gen:.1e} "
          f"(|A| {la.norm(A_ref):.1f}); rung 4 sym spectrum {e_sym:.1e} "
          f"(top rate {rates[0] if k else 0:.3f}) [{clock.time() - t0:.0f} s]")
    print(f"      stacked generator vs term-by-term reference {e_fast:.1e}")
    return max(e_P, e_fix, e_gen, e_sym, e_fast)


def rung5(L):
    H = hamiltonian(L)
    psi, cut, N = random_point(L, 0)
    terms = mpo_cut_terms(H, cut)
    point = CutPoint(mps_to_matrix(psi, cut) / np.sqrt(states.overlap(psi, psi).real), N, terms)
    rng = np.random.default_rng(5)
    Z = rng.normal(size=(6, N, N)) + 1j * rng.normal(size=(6, N, N))
    Zm = Z.reshape(6, -1)
    GZ = np.array([point.generator(z).reshape(-1) for z in Z])
    # real symmetric part of the real-linear map: Re<z_i, G z_j> + Re<z_j, G z_i>
    S = np.real(Zm.conj() @ GZ.T)
    e = la.norm(S + S.T) / la.norm(S)
    print(f"  rung 5, L={L}, m=0: |sym part| / |G| = {e:.1e}, w_N = {la.norm(point.w_N):.1e}")
    return e


def main():
    Ls = [int(x) for x in sys.argv[1:]] or [4]
    worst = 0.0
    for L in Ls:
        print(f"L = {L}")
        worst = max(worst, rung1(L), rung5(L))
        ms = [1, 2, 3, 4] if L == 4 else [2, 40]
        for m in ms:
            worst = max(worst, rungs234(L, m))
    print(f"worst relative residual: {worst:.1e}")


if __name__ == '__main__':
    main()
