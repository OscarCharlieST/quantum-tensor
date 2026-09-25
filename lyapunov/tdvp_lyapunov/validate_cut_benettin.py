"""
Validation of cut_benettin against Route B (benettin.benettin) along the
same TDVP trajectory.

    6. m = 0 (full Hilbert space): every exponent vanishes
    7. m > 0, k = 2n, both routes started from the same vectors (the
       ambient ones written in Route B's frame): cut_benettin is Route B's
       step in other coordinates, so the per-block diagonals of R, and hence
       the exponents, must agree to roundoff

Run from the repo root:  python lyapunov/tdvp_lyapunov/validate_cut_benettin.py [blocks]
"""

import os
import sys
import time as clock

import numpy as np
import scipy.linalg as la

sys.path.insert(0, os.getcwd())

import qtensor.operators as ops
import qtensor.thermofield as tf

from lyapunov.tdvp_lyapunov.benettin import benettin
from lyapunov.tdvp_lyapunov.frame import Frame
from lyapunov.tdvp_lyapunov.cut_benettin import (
    cut_benettin, start_state, tangent_dimension, make_point, random_tangent,
)
from lyapunov.tdvp_lyapunov.cut_matrix import mpo_cut_terms, truncate
from lyapunov.tdvp_lyapunov.validate_cut import frame_basis
from lyapunov.tdvp_lyapunov.cut_rates import ExactThermofield, dense_thermofield

L, CUT, D_LOC, BETA, T_SCRAMBLE, DT = 4, 1, 4, 0.1, 20.0, 0.05


def setup(m):
    H = tf.thermofield_hamiltonian(ops.tilted_ising(N=L), asym=False)
    terms = mpo_cut_terms(H, CUT)
    exact = ExactThermofield(ops.tilted_ising(N=L), L, CUT)
    Psi0 = exact.evolve(dense_thermofield(L, BETA, terms, CUT), T_SCRAMBLE)
    N = Psi0.shape[0]
    Psi0, _ = truncate(Psi0, N - m)
    return H, exact, start_state(Psi0, L, D_LOC, N - m, CUT), N


def rung6(blocks):
    H, exact, psi0, N = setup(0)
    k = tangent_dimension(N, 0)
    t0 = clock.time()
    res = cut_benettin(psi0, H, CUT, N, DT, blocks, k, exact=exact, verbose=False)
    lam = res['exponents']
    print(f"  rung 6, m=0, k={k}: max|lambda| = {np.abs(lam).max():.1e}, "
          f"min fidelity {res['fidelity'].min():.12f} [{clock.time() - t0:.0f} s]")
    return np.abs(lam).max()


def rung7(m, blocks):
    H, exact, psi0, N = setup(m)
    r = N - m
    k = tangent_dimension(N, m)
    # common starting vectors: ambient, and the same vectors in Route B's frame
    point = make_point(psi0, r, mpo_cut_terms(H, CUT), CUT)
    Q_amb = random_tangent(point, k)
    B = frame_basis(Frame(psi0, r), CUT).reshape(k // 2, -1)        # (n, N^2)
    X = B.conj() @ Q_amb.reshape(k, -1).T                           # (n, k) complex
    Q_frame = np.concatenate([X.real, X.imag])
    print(f"      starting vectors: frame round trip {la.norm(Q_frame.T @ Q_frame - np.eye(k)):.1e}")
    t0 = clock.time()
    ref = benettin(psi0, H, DT, blocks, k, max_bond_dim=r, Q0=Q_frame, verbose=False)
    t_ref = clock.time() - t0
    t0 = clock.time()
    cut = cut_benettin(psi0, H, CUT, r, DT, blocks, k, Q0=Q_amb, verbose=False)
    t_cut = clock.time() - t0
    det_ref = ref['log_diag_R'].sum(1)
    det_cut = cut['log_diag_R'].sum(1)
    lam_ref = np.sort(ref['exponents'])[::-1]
    lam_cut = np.sort(cut['exponents'])[::-1]
    print(f"  rung 7, m={m}, k={k}, {blocks} blocks: "
          f"Route B {t_ref:.0f} s, cut {t_cut:.0f} s")
    print(f"      per-block log R_ii: max |diff| "
          f"{np.abs(ref['log_diag_R'] - cut['log_diag_R']).max():.1e} "
          f"(scale {np.abs(ref['log_diag_R']).max():.1e}); log|det M| per block "
          f"{np.abs(det_ref).max():.1e} / {np.abs(det_cut).max():.1e}")
    print(f"      lambda_max {lam_ref[0]:+.4f} / {lam_cut[0]:+.4f}; "
          f"lambda_min {lam_ref[-1]:+.4f} / {lam_cut[-1]:+.4f}; "
          f"sum {lam_ref.sum():+.1e} / {lam_cut.sum():+.1e}")
    print(f"      sorted spectra: max |diff| {np.abs(lam_ref - lam_cut).max():.1e}, "
          f"rms {np.sqrt(np.mean((lam_ref - lam_cut) ** 2)):.1e}; pairing "
          f"{np.abs(lam_ref + lam_ref[::-1]).max():.1e} / {np.abs(lam_cut + lam_cut[::-1]).max():.1e}")
    return np.abs(ref['log_diag_R'] - cut['log_diag_R']).max()


def main():
    blocks = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f"L = {L}, dt = {DT}, start: beta = {BETA} TFD evolved exactly to t = {T_SCRAMBLE}")
    rung6(min(blocks, 40))
    for m in (2, 4):
        rung7(m, blocks)


if __name__ == '__main__':
    main()
