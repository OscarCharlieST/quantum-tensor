"""
Step 1 of the energy-rate study: the static bracket of the formula

    dE_0/dt = -(tau/8) grad^2 beta  sum_k k^2 B_k,
    B_k     = < h_k^+  P [H, h_0]^+  P  H^+ >,

evaluated on the MPS manifold, with and without the tangent projector, at a
given state. The exact (P = 1) infinite-temperature value is
sum_k k^2 B_k = 8 G(0) = 16 (g J)^2 = 4.41 (energy_rate_ed.py, which also
fixes the -1/8: physical-copy energy, rho ~ exp(-sum beta_j h_j)).

Conventions. A^+ = A (x) 1 + 1 (x) A on the doubled site (physical copy
first, analysis._copy_kron), so [H, h_0]^+ = [H_sym, h_0^+]. Every bracket is
*connected*: h_k^+ psi and H_sym psi have their component along psi removed,
which is what the projective tangent space does automatically, so the
projected and unprojected versions differ by the projector and nothing else.
Bonds are labelled by their left site; k = bond - b0, b0 = L//2 - 1.

Four variants per k, isolating where the projector acts:

    B1   < dh_k^+ psi |   X^+   | dH psi >          exact, no projection
    BPH  < dh_k^+ psi |   X^+ P | H psi >           P on the flow only
    BPh  < dh_k^+ psi | P X^+   | dH psi >          P on the perturbation only
    BP   < dh_k^+ psi | P X^+ P | H psi >           the literal formula

plus BP_one, the literal formula with the one-sided perturbation
2 h_k (x) 1 in place of h_k^+ (a diagnostic, not the quantity of record;
on a swap-symmetric state it equals BP identically, since every other
operator in the bracket is swap-even).

The single ordering <h X H> is the whole second derivative only on the
thermofield double, because there X^+ psi = 0 (X^+ acts on psi <-> M as
[X, M], and M ~ rho^(1/2) commutes with X to O(beta)). Off it, the other
ordering contributes. So two exact (P = 1) second derivatives are also
recorded, each normalized to equal C_k = <hdot_k hdot_0>_inf at the TFD:

    Csym   (1/4) < dh_k^+ psi | [H_sym, [H_sym, h_0^+]] psi >
           symmetric perturbation dpsi = -(1/4) sum dbeta_k dh_k^+ psi,
           symmetric energy: d^2<h_0^+>/dt^2 = 2 sum_k dbeta_k Re Csym_k
    Cphys  < d(h_k (x) 1) psi | [H, [H, h_0]] (x) 1  psi >
           one-sided perturbation dpsi = -(1/2) sum dbeta_k d(h_k (x) 1) psi,
           physical energy: d^2<h_0 (x) 1>/dt^2 = sum_k dbeta_k Re Cphys_k.
           Depends on psi only through the physical reduced state, so it
           tests whether the point is still an infinite-T reference at all.

At the TFD, Csym = Cphys = -B1/4 = C_k (sum_k k^2 C_k = -2 G(0) = -1.1025).
"""

import numpy as np
import scipy.linalg as la
from ncon import ncon

import qtensor.operators as ops
from lyapunov.relaxation.tangent_hamiltonian import contract_left_mixed
from lyapunov.relaxation.response import pad_with_identity, single_copy_hamiltonian
from lyapunov.tdvp_lyapunov.analysis import energy_density_mpo
from lyapunov.tdvp_lyapunov.frame import (
    project_to_frame, tangent_mps, _left_transfer, _right_transfer,
)
from lyapunov.tdvp_lyapunov.tangent_generator import apply_mpo

J, HF, G = 1.0, 0.25, -0.525
EXACT_SUM_K2B = 16 * (G * J) ** 2          # 4.41, beta = 0


# -------------------------------------------------------------- operators

def _kc(A, which):
    return np.kron(A, np.eye(2)) if which == 'phys' else np.kron(np.eye(2), A)


def bond_energy_plus(b):
    """h_b^+ = h_b (x) 1 + 1 (x) h_b on bond (b, b+1), bond dimension 4."""
    Zp, Za = _kc(ops.pauli('z'), 'phys'), _kc(ops.pauli('z'), 'aux')
    a = sum(0.5 * (HF * _kc(ops.pauli('z'), w) + G * _kc(ops.pauli('x'), w))
            for w in ('phys', 'aux'))
    I4 = np.eye(4)
    Wl = np.zeros((4, 4, 1, 4), dtype=complex)
    Wl[:, :, 0, 0], Wl[:, :, 0, 1], Wl[:, :, 0, 2], Wl[:, :, 0, 3] = J * Zp, J * Za, I4, a
    Wr = np.zeros((4, 4, 4, 1), dtype=complex)
    Wr[:, :, 0, 0], Wr[:, :, 1, 0], Wr[:, :, 2, 0], Wr[:, :, 3, 0] = Zp, Za, a, I4
    return ops.mpo([(b, Wl), (b + 1, Wr)], np.array([1.0]), np.array([1.0]))


def h_sym_mpo(sites):
    """H_sym = H (x) 1 + 1 (x) H in complex128 (thermofield.thermofield_hamiltonian
    is complex64, so its -0.525 is -0.52499998)."""
    W = np.zeros((4, 4, 4, 4), dtype=complex)
    I4 = np.eye(4)
    W[:, :, 0, 0] = W[:, :, 3, 3] = I4
    W[:, :, 0, 1] = J * _kc(ops.pauli('z'), 'phys')
    W[:, :, 0, 2] = J * _kc(ops.pauli('z'), 'aux')
    W[:, :, 1, 3] = _kc(ops.pauli('z'), 'phys')
    W[:, :, 2, 3] = _kc(ops.pauli('z'), 'aux')
    W[:, :, 0, 3] = sum(HF * _kc(ops.pauli('z'), w) + G * _kc(ops.pauli('x'), w)
                        for w in ('phys', 'aux'))
    return ops.mpo([(s, W) for s in sorted(sites)],
                   np.array([1.0, 0, 0, 0]), np.array([0, 0, 0, 1.0]))


def full(O, sites):
    """A local MPO padded with identities to every site, for apply_mpo."""
    return ops.mpo(list(pad_with_identity(O, sites).items()), O.l, O.r)


# ------------------------------------------------------------ contractions

def _envs(bra, ket):
    sites = sorted(ket)
    L = {sites[0] - 1: np.eye(1)}
    for s in sites:
        L[s] = _left_transfer(L[s - 1], ket[s], bra[s])
    R = {sites[-1] + 1: np.eye(1)}
    for s in reversed(sites):
        R[s] = _right_transfer(R[s + 1], ket[s], bra[s])
    return L, R


def overlap(bra, ket):
    """<bra|ket> for two {site: tensor} MPS on the same sites."""
    L, _ = _envs(bra, ket)
    return L[max(ket)][0, 0]


def matrix_elements(bra, ket, mpo_at, positions):
    """<bra| O_j |ket> for every local MPO O_j = mpo_at(j), in O(N) sweeps.
    The general form of analysis.local_profile (any bra, any ket)."""
    L, R = _envs(bra, ket)
    out = np.zeros(len(positions), dtype=complex)
    for i, j in enumerate(positions):
        O = mpo_at(j)
        op_sites = sorted(O.sites)
        E = ncon((L[op_sites[0] - 1], O.l), ((-1, -2), (-3,)))
        for s in op_sites:
            E = contract_left_mixed(E, ket[s], bra[s], O[s])
        E = ncon((E, O.r), ((-1, -2, 1), (1,)))
        out[i] = np.trace(E @ R[op_sites[-1] + 1].T)
    return out


# ----------------------------------------------------------------- bracket

def brackets(frame, b0=None):
    """
    All four variants of B_k at the frame's point, for every bond k, plus
    the capture ratios that say how much of H_sym psi and h_b0^+ psi the
    tangent space holds. Returns a dict of arrays indexed by k = bond - b0.
    """
    sites = frame.sites
    bonds = sites[:-1]
    if b0 is None:
        b0 = sites[0] + len(sites) // 2 - 1
    psi = frame.A_R
    Hs = h_sym_mpo(sites)
    h0p = full(bond_energy_plus(b0), sites)

    def xplus(T):
        """X^+ T = H_sym h_0^+ T - h_0^+ H_sym T, as (coef, tensors) pieces."""
        return [(+1.0, apply_mpo(apply_mpo(T, h0p), Hs)),
                (-1.0, apply_mpo(apply_mpo(T, Hs), h0p))]

    def proj(pieces):
        return sum(c * project_to_frame(T, frame) for c, T in pieces)

    def elements(pieces, mpo_at=bond_energy_plus, mean=None):
        """<psi| O_k |sum> - <O_k>^* <psi|sum>: the connected bra."""
        mean = hk_mean if mean is None else mean
        me = sum(c * matrix_elements(psi, T, mpo_at, bonds) for c, T in pieces)
        ov = sum(c * overlap(psi, T) for c, T in pieces)
        return me - np.conj(mean) * ov

    hk_mean = matrix_elements(psi, psi, bond_energy_plus, bonds)

    # the flow: H psi, connected, and its tangent part
    Hpsi = apply_mpo(psi, Hs)
    E_H = overlap(psi, Hpsi)
    var_H = overlap(Hpsi, Hpsi).real - abs(E_H) ** 2
    xH = project_to_frame(Hpsi, frame)
    dH_pieces = [(1.0, Hpsi), (-E_H, psi)]                  # (H - <H>) psi
    w_full = [(c * cx, Tx) for c, T in dH_pieces for cx, Tx in xplus(T)]
    w_tan = xplus(tangent_mps(frame, xH, include_point=False))

    # the perturbations' tangent parts, one copy at a time
    def x_of(b, which):
        O = energy_density_mpo(b, which)
        return project_to_frame(apply_mpo(psi, full(O, sites)), frame)
    x_phys = np.array([x_of(b, 'phys') for b in bonds])
    x_aux = np.array([x_of(b, 'aux') for b in bonds])
    x_plus = x_phys + x_aux

    z_full, z_tan = proj(w_full), proj(w_tan)
    B1 = elements(w_full)

    # exact second derivatives (see module docstring)
    rev = [(c * cx, Tx) for c, T in xplus(psi)
           for cx, Tx in ((1.0, apply_mpo(T, Hs)), (-E_H, T))]     # (H - <H>) X^+ psi
    Csym = 0.25 * (elements(rev) - B1)
    Hp = single_copy_hamiltonian(sites, J=J, h=HF, g=G)
    hp0 = full(energy_density_mpo(b0, 'phys'), sites)
    def seq(T, *Os):
        """Apply Os in order (the first acts first)."""
        for O in Os:
            T = apply_mpo(T, O)
        return T
    dd =[(1.0, seq(psi, hp0, Hp, Hp)), (-2.0, seq(psi, Hp, hp0, Hp)),
          (1.0, seq(psi, Hp, Hp, hp0))]                               # [H,[H,h0]] psi
    phys_at = lambda b: energy_density_mpo(b, 'phys')
    Cphys = elements(dd, phys_at, matrix_elements(psi, psi, phys_at, bonds))

    out = {
        'k': np.array(bonds) - b0,
        'Csym': Csym,
        'Cphys': Cphys,
        'B1': B1,
        'BPH': elements(w_tan),
        'BPh': x_plus.conj() @ z_full,
        'BP': x_plus.conj() @ z_tan,
        'BP_one': 2 * x_phys.conj() @ z_tan,
        'hk_mean': hk_mean,
        'E_H': E_H,
        'capture_H': float(np.vdot(xH, xH).real / var_H),
    }
    # capture of the central perturbation: |P dh^+ psi|^2 / |dh^+ psi|^2
    i0 = list(bonds).index(b0)
    hpsi = apply_mpo(psi, h0p)
    var_h = overlap(hpsi, hpsi).real - abs(hk_mean[i0]) ** 2
    out['capture_h0'] = float(np.vdot(x_plus[i0], x_plus[i0]).real / var_h)
    return out


def sum_k2(B, k, kmax=None):
    sel = np.ones_like(k, dtype=bool) if kmax is None else (np.abs(k) <= kmax)
    return np.sum(k[sel] ** 2 * B[sel])
