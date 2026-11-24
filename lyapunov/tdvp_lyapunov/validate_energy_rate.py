"""
Checks for energy_rate.brackets, cheapest first.

1. Dense brute force (L = 5, D = 3, a complex point after a few H_sym
   steps): every variant B1, BPH, BPh, BP, BP_one against dense vectors and
   the explicit projector V V^dag built from the frame's own basis.
2. Full rank (L = 4, D = 16): the tangent space is everything orthogonal to
   psi, so every projected variant must equal B1.
3. Against exact diagonalization (L = 8, D = 16, beta = 0.1 thermofield
   double): B1 against energy_rate_ed.thermofield_bracket at the effective
   inverse temperature of the build, beta (1 + 1/steps).

    python lyapunov/tdvp_lyapunov/validate_energy_rate.py
"""

import os
import sys

import numpy as np
import scipy.linalg as la

sys.path.insert(0, os.getcwd())

import qtensor.operators as ops
import qtensor.thermofield as tf

from lyapunov.tdvp_lyapunov.frame import Frame, tangent_mps
from lyapunov.tdvp_lyapunov.stepper import tdvp_step, lanczos_method
from lyapunov.tdvp_lyapunov.energy_rate import (
    brackets, bond_energy_plus, h_sym_mpo, J, HF, G,
)
from lyapunov.tdvp_lyapunov.energy_rate_ed import thermofield_bracket
from lyapunov.relaxation.run_relaxation_scan import build_uniform_thermofield


def dense_state(T):
    sites = sorted(T)
    v = T[sites[0]][:, 0, :]                     # (d, D)
    for s in sites[1:]:
        v = np.einsum('ia,jab->ijb', v, T[s]).reshape(-1, T[s].shape[2])
    return v[:, 0]


def dense_mpo(O, sites):
    """
    Dense matrix O[out, in] of an mpo on `sites` (missing sites are
    identities). The repo contracts W's first index with the ket
    (ops.contract_left), so W[in, out, l, r]; hence the transpose.
    """
    T = np.asarray(O.l, dtype=complex).reshape(1, 1, -1)
    for s in sorted(sites):
        if O[s] is None:
            T = np.einsum('IJb,ij->IiJjb', T, np.eye(4))
        else:
            T = np.einsum('IJa,ijab->IiJjb', T, O[s].transpose(1, 0, 2, 3))
        a, b, c, d, e = T.shape
        T = T.reshape(a * b, c * d, e)
    return np.einsum('ijb,b->ij', T, O.r)


def dense_check(L=5, D=3, steps=6):
    np.random.seed(0)
    psi, _ = build_uniform_thermofield(L, D, 0.1, 40)
    H_repo = tf.thermofield_hamiltonian(ops.tilted_ising(N=L), asym=False)
    for _ in range(steps):
        psi = tdvp_step(psi, H_repo, 0.05, lanczos_method())
    frame = Frame(psi, D)
    sites = frame.sites
    b0 = sites[0] + L // 2 - 1
    got = brackets(frame, b0)

    v = dense_state(frame.A_R)
    v /= la.norm(v)
    Hs = dense_mpo(h_sym_mpo(sites), sites)
    h = {b: dense_mpo(bond_energy_plus(b), sites) for b in sites[:-1]}
    from lyapunov.tdvp_lyapunov.analysis import energy_density_mpo
    hp = {b: dense_mpo(energy_density_mpo(b, 'phys'), sites) for b in sites[:-1]}
    X = Hs @ h[b0] - h[b0] @ Hs

    V = np.column_stack([dense_state(tangent_mps(frame, e, include_point=False))
                         for e in np.eye(frame.n, dtype=complex)])
    P = V @ V.conj().T
    conn = lambda A: A @ v - np.vdot(v, A @ v) * v
    dH = conn(Hs)
    from lyapunov.relaxation.response import single_copy_hamiltonian
    Hp = dense_mpo(single_copy_hamiltonian(sites, J=J, h=HF, g=G), sites)
    comm = lambda A, B: A @ B - B @ A
    DDs = comm(Hs, comm(Hs, h[b0]))
    DDp = comm(Hp, comm(Hp, hp[b0]))
    ref = {'B1': [], 'BPH': [], 'BPh': [], 'BP': [], 'BP_one': [], 'Csym': [], 'Cphys': []}
    for b in sites[:-1]:
        dh = conn(h[b])
        ref['Csym'].append(0.25 * np.vdot(dh, DDs @ v))
        ref['Cphys'].append(np.vdot(conn(hp[b]), DDp @ v))
        ref['B1'].append(np.vdot(dh, X @ dH))
        ref['BPH'].append(np.vdot(dh, X @ P @ dH))
        ref['BPh'].append(np.vdot(dh, P @ X @ dH))
        ref['BP'].append(np.vdot(dh, P @ X @ P @ dH))
        ref['BP_one'].append(2 * np.vdot(conn(hp[b]), P @ X @ P @ dH))
    print(f"1. dense brute force, L={L} D={D}, n={frame.n}, "
          f"|V^dag V - 1| = {la.norm(V.conj().T @ V - np.eye(frame.n)):.1e}, "
          f"<psi|V> = {la.norm(v.conj() @ V):.1e}")
    for key, r in ref.items():
        r = np.array(r)
        print(f"   {key:7s} max|diff| = {np.max(np.abs(got[key] - r)):.1e}   "
              f"(scale {np.max(np.abs(r)):.2f}, max |Im| {np.max(np.abs(r.imag)):.1e})")


def full_rank_check():
    np.random.seed(0)
    psi, _ = build_uniform_thermofield(4, 16, 0.1, 40)
    got = brackets(Frame(psi, 16))
    print("2. full rank, L=4 D=16: max|B_variant - B1| = " + ", ".join(
        f"{k} {np.max(np.abs(got[k] - got['B1'])):.1e}" for k in ('BPH', 'BPh', 'BP')),
        f"; capture_H {got['capture_H']:.12f}")


def ed_check(L=8, D=16, beta=0.1, steps=240):
    np.random.seed(0)
    psi, _ = build_uniform_thermofield(L, D, beta, steps)
    got = brackets(Frame(psi, D))
    ks, B_ed = thermofield_bracket(L, beta * (1 + 1 / steps))
    print(f"3. against ED, L={L} D={D} beta={beta}:")
    for k, a, b, cs, cp in zip(got['k'], got['B1'], B_ed, got['Csym'], got['Cphys']):
        print(f"   k={k:+d}  MPS {a.real:+.6f}{a.imag:+.0e}j   ED {b.real:+.6f}   "
              f"diff {abs(a - b):.1e}   -B1/4 {-a.real / 4:+.6f}  Csym {cs.real:+.6f}  "
              f"Cphys {cp.real:+.6f}")


if __name__ == '__main__':
    dense_check()
    full_rank_check()
    ed_check()
