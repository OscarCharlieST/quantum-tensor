"""
Reading a Lyapunov run: spectrum checks, and the spatial structure of a
tangent vector (site weights, physical-copy energy-density profile, its
cosine transform) for hunting hydrodynamic Lyapunov modes.
"""

import numpy as np
import scipy.linalg as la
from scipy.fft import dct
from ncon import ncon

import qtensor.operators as ops
from lyapunov.relaxation.tangent_hamiltonian import contract_left_mixed
from lyapunov.tdvp_lyapunov.frame import (
    tangent_mps, complexify, _left_transfer, _right_transfer,
)


# ----------------------------------------------------------------- spectrum

def pairing_residual(exponents):
    """
    lambda_i + lambda_{2n+1-i} for a full (k = 2n) spectrum: should vanish
    for a Hamiltonian flow. Returns the array of sums, sorted-pair-wise.
    """
    lam = np.sort(exponents)[::-1]
    return lam + lam[::-1]


def spectrum_summary(exponents, dt=None):
    lam = np.sort(exponents)[::-1]
    out = {'max': lam[0], 'min': lam[-1], 'sum': lam.sum(),
           'n_positive': int((lam > 0).sum()),
           'n_near_zero': int((np.abs(lam) < 1e-3).sum())}
    return out


# ------------------------------------------------------------ observables
#
# Thermofield convention throughout the repo: the physical copy is the
# first factor of the (2, 2) local index, so A on the physical copy is
# kron(A, I) and on the auxiliary copy kron(I, A).

def _copy_kron(A, which):
    return np.kron(A, np.eye(2)) if which == 'phys' else np.kron(np.eye(2), A)


def energy_density_mpo(site_l, which='phys', J=1, h=0.25, g=-0.525):
    """
    Tilted-Ising energy density on bond (site_l, site_l+1), on one copy:
    J z z + a_l + a_{l+1}, a = (h z + g x)/2, so bonds sum to H.
    """
    Z = _copy_kron(ops.pauli('z'), which)
    X = _copy_kron(ops.pauli('x'), which)
    I4 = np.eye(4)
    a = 0.5 * (h * Z + g * X)
    Wl = np.zeros((4, 4, 1, 3), dtype=complex)
    Wl[:, :, 0, 0], Wl[:, :, 0, 1], Wl[:, :, 0, 2] = J * Z, I4, a
    Wr = np.zeros((4, 4, 3, 1), dtype=complex)
    Wr[:, :, 0, 0], Wr[:, :, 1, 0], Wr[:, :, 2, 0] = Z, a, I4
    return ops.mpo([(site_l, Wl), (site_l + 1, Wr)],
                   np.array([1.0]), np.array([1.0]))


def onesite_mpo(site, pauli='z', which='phys'):
    W = np.zeros((4, 4, 1, 1), dtype=complex)
    W[:, :, 0, 0] = _copy_kron(ops.pauli(pauli), which)
    return ops.mpo([(site, W)], np.array([1.0]), np.array([1.0]))


# ------------------------------------------------------------ profiles

def local_profile(frame, y, mpo_at, positions):
    """
    delta<O_j> = 2 Re <psi| O_j |Phi(X)> for the tangent vector with real
    coordinates y, for each local MPO O_j = mpo_at(j), j in positions.

    The bra is psi (frame.A_R), the ket the bond-2D tangent MPS. Identity
    transfer matrices are built once from both ends, so the whole profile
    costs O(N) contractions.
    """
    X = complexify(y)
    ket = tangent_mps(frame, X, include_point=False)
    sites = frame.sites
    A = frame.A_R

    L = {sites[0] - 1: np.eye(1)}
    for s in sites:
        L[s] = _left_transfer(L[s - 1], ket[s], A[s])
    R = {sites[-1] + 1: np.eye(1)}
    for s in reversed(sites):
        R[s] = _right_transfer(R[s + 1], ket[s], A[s])

    out = np.zeros(len(positions))
    for i, j in enumerate(positions):
        O = mpo_at(j)
        op_sites = sorted(O.sites)
        E = ncon((L[op_sites[0] - 1], O.l), ((-1, -2), (-3,)))
        for s in op_sites:
            E = contract_left_mixed(E, ket[s], A[s], O[s])
        E = ncon((E, O.r), ((-1, -2, 1), (1,)))
        val = np.trace(E @ R[op_sites[-1] + 1].T)
        out[i] = 2 * np.real(val)
    return out


def energy_profile(frame, y, which='phys', **params):
    """Energy-density profile over bonds, on the physical or auxiliary copy."""
    bonds = frame.sites[:-1]
    return local_profile(frame, y, lambda j: energy_density_mpo(j, which, **params), bonds)


def site_weight_profile(frame, y):
    """|X^n|^2 per site (zeros at sites with no tangent directions)."""
    w = frame.site_weights(complexify(y))
    return np.array([w.get(s, 0.0) for s in frame.sites])


def cosine_transform(profile):
    """
    DCT-II (orthonormal) of a profile on an open chain. Returns (q, amplitude)
    with q_k = pi k / N_bonds, k = 0 .. N_bonds-1. The k=0 entry is the
    uniform component.
    """
    a = dct(profile, type=2, norm='ortho')
    N = len(profile)
    q = np.pi * np.arange(N) / N
    return q, a


def mode_report(frame, y, which='phys'):
    """Everything the plots need for one tangent vector, in one dict."""
    prof = energy_profile(frame, y, which)
    q, amp = cosine_transform(prof)
    return {'sites': np.array(frame.sites), 'site_weights': site_weight_profile(frame, y),
            'bonds': np.array(frame.sites[:-1]), 'energy_profile': prof,
            'q': q, 'dct': amp}


def q_weight_by_exponent(frame, V, exponents, n_bins=12, which='phys'):
    """
    Where in wavevector does the energy-profile power of the Lyapunov vectors
    sit, as a function of their exponent. For every column of V (in the
    frame), the DCT of its energy-density profile is normalized to unit
    power; vectors are binned by exponent and the normalized power averaged
    within each bin.

    Individual vectors inside a near-degenerate cluster are arbitrary up to
    rotations within the cluster, so this cluster average is the meaningful
    object, not the single-vector profiles.

    Returns edges (n_bins+1), q (N_bonds), W (n_bins, N_bonds), counts.
    """
    lam = np.asarray(exponents)
    edges = np.linspace(lam.min(), lam.max(), n_bins + 1)
    edges[-1] += 1e-12
    q = cosine_transform(np.zeros(len(frame.sites) - 1))[0]
    W = np.zeros((n_bins, len(q)))
    counts = np.zeros(n_bins, dtype=int)
    for i in range(V.shape[1]):
        prof = energy_profile(frame, V[:, i], which)
        _, a = cosine_transform(prof)
        p = a ** 2
        if p.sum() == 0:
            continue
        b = min(np.searchsorted(edges, lam[i], side='right') - 1, n_bins - 1)
        W[b] += p / p.sum()
        counts[b] += 1
    W[counts > 0] /= counts[counts > 0, None]
    return edges, q, W, counts


def template_modes(frame, H, q_values):
    """
    Not implemented. Intended: coordinates of P sum_j cos(q j) h_j psi and
    -i P sum_j cos(q j) h_j psi (local temperature shift / local time shift)
    for overlap with Lyapunov vectors. Needs a cosine-weighted extensive
    MPO builder.
    """
    raise NotImplementedError
