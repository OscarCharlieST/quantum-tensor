"""
The physical/ancilla swap symmetry: the involution, and how to hold a
trajectory to it.

The doubled physical index is i = 2 * a_phys + b_anc, so exchanging the two
copies is the on-site permutation [0, 2, 1, 3] and S = (x)_m s_m is a product
of on-site unitaries. H_sym is swap-even and the thermofield double is
swap-symmetric, so the exact flow keeps S|psi> = |psi> and the tangent space
splits into +-1 sectors that the tangent flow cannot mix.

The numerics do not keep it. Where a Schmidt value is numerically zero the
point is a singular point of the rank-D manifold, V_L and A_R past the rank
are an arbitrary isometry completion, and the step walks off the symmetric
sector. At beta = 0.1 the thermofield double is close to rank 1, so the
build *ends* in that region and only real time walks the state out of it --
the larger D, the longer it lingers and the more it picks up. Measured at
L = 8 over a transient to t = 8: a_odd = 5.8e-7 at D = 4, 1.9e-3 at D = 8,
6.8e-1 at D = 12, where 1/sqrt(2) = 0.707 is a state orthogonal to its own
swap. Chaos then amplifies whatever was injected at the Lyapunov rate.

So the symmetry has to be enforced rather than assumed, and the place to do
it is the transient, where all of the injection happens. `evolve` is that
loop. See SWAP_SYMMETRY.md for the measurements and for what is *not* worth
enforcing (projecting the tangent vectors each block: costs 4e-8, changes no
exponent).
"""

import copy

import numpy as np
import scipy.linalg as la

import qtensor.states as states

from lyapunov.tdvp_lyapunov.frame import (
    project_to_frame, tangent_mps, realify,
)
from lyapunov.tdvp_lyapunov.stepper import tdvp_step, lanczos_method


# i = 2 * a_phys + b_anc (analysis._copy_kron builds one-copy operators as
# kron(A, I) / kron(I, A), and the infinite-T seed is [1, 0, 0, 1]), so the
# swap is 2a + b -> 2b + a: an involution, hence a symmetric permutation
# matrix, and applying it to a tensor is a reindex of the physical leg.
SWAP_PERM = [0, 2, 1, 3]


# ------------------------------------------------------------------- states

def swap(phi):
    """S|phi>: exchange the two copies on every site. Takes and returns
    {site: tensor}; an mps is accepted and unwrapped."""
    tensors = phi.tensors if hasattr(phi, 'tensors') else phi
    return {site: T[SWAP_PERM, :, :].copy() for site, T in tensors.items()}


def swap_overlap(psi):
    """<psi|S|psi> / <psi|psi>."""
    return states.overlap(psi, states.mps(swap(psi))) / abs(states.overlap(psi, psi))


def odd_amplitude(psi):
    """
    |(1 - S) psi / 2|: the swap-odd amplitude of the state, the quantity
    everything else inherits. Linear in the asymmetry, unlike <psi|S|psi>,
    which is 1 - 2 a_odd^2 and so hides it. Saturates at 1/sqrt(2).
    """
    return float(np.sqrt(max(0.0, (1.0 - swap_overlap(psi).real) / 2.0)))


def enforce_swap_symmetry(psi, max_bond_dim):
    """
    Project psi onto the swap-even sector: (psi + S psi)/||.||, recompressed
    to max_bond_dim. Does not mutate psi; returns a normalized state.

    max_bond_dim is required and must be finite: the sum lives at bond 2D,
    and handing it back at that rank would silently double the dimension of
    the manifold the Lyapunov machinery is working on.

    Returns (psi_sym, a_odd, cut):
        a_odd   the odd amplitude removed, |(1 - S) psi / 2|
        cut     the fraction of the projected state lost to the
                recompression. Measured at 1e-14 even in the rank-deficient
                region; anything larger means the truncation, rather than
                the projection, is changing the state.
    """
    assert np.isfinite(max_bond_dim), (
        "enforce_swap_symmetry needs the rank to compress back to; the sum "
        "is at bond 2D")
    c = swap_overlap(psi).real
    a_odd = float(np.sqrt(max(0.0, (1.0 - c) / 2.0)))
    exact = np.sqrt(max(0.0, 2.0 + 2.0 * c))          # |psi + S psi|
    psi_sym = copy.deepcopy(psi)
    got = psi_sym.add(states.mps(swap(psi)), max_bond_dim=max_bond_dim)
    cut = float(1.0 - got / exact) if exact > 0 else np.nan
    return psi_sym, a_odd, cut


def evolve(psi, H, dt, steps, method=None, max_bond_dim=np.inf,
           symmetrize_every=0, verbose=False):
    """
    The transient: `steps` of plain TDVP under H, projecting onto the
    swap-even sector every `symmetrize_every` steps (0 = never).

    This is the loop every driver runs before it starts a tangent
    calculation, and the only place the symmetry needs enforcing: the
    injection is all in the rank-deficient early time, and a run that starts
    clean stays clean to the accuracy anything here needs. Note tdvp_step
    itself never truncates -- the bond dimension is whatever psi carries, and
    max_bond_dim is the rank the projection compresses back to.

    Returns (psi, log), log being the (n, 3) array of (t, a_odd, cut) at each
    projection, empty when symmetrize_every is 0.
    """
    if method is None:
        method = lanczos_method()
    if symmetrize_every:
        assert np.isfinite(max_bond_dim), (
            "pass max_bond_dim (the run's D) when symmetrizing")
    log = []
    for j in range(1, steps + 1):
        psi = tdvp_step(psi, H, dt, method)
        if symmetrize_every and j % symmetrize_every == 0:
            psi, a_odd, cut = enforce_swap_symmetry(psi, max_bond_dim)
            log.append((j * dt, a_odd, cut))
            if verbose:
                print(f"    t={j * dt:6.2f}  removed a_odd={a_odd:.2e}  "
                      f"compression cut={cut:.1e}", flush=True)
    return psi, np.array(log).reshape(-1, 3)


# ------------------------------------------------------------ tangent space

def swap_matrix(frame):
    """
    The (2n, 2n) real matrix Sigma of dS in the frame's coordinates. Column j
    is dS applied to the j-th real basis vector: the same construction as
    frame.frame_change, with the swapped state in place of the second point,
    so nothing crosses between canonicalization passes. S is complex-linear,
    so the columns for X and iX come from one projection: n projections in
    total, each O(N d D^3).

    Sigma is the compression P S P, so every diagnostic taken from it is
    *quadratic* in the symmetry breaking. The linear measure is
    leak_j = sqrt(1 - |Sigma e_j|^2), and it tracks a_odd.
    """
    n = frame.n
    Sig = np.zeros((2 * n, 2 * n))
    for j in range(n):
        X = np.zeros(n, dtype=complex)
        X[j] = 1.0
        c = project_to_frame(swap(tangent_mps(frame, X, include_point=False)),
                             frame)
        Sig[:, j] = realify(c)
        Sig[:, n + j] = realify(1j * c)
    return Sig


def leaks(Sig):
    """sqrt(1 - |Sigma e_j|^2) per column: the weight S takes out of the
    tangent space, and the honest (linear) measure of the breaking."""
    return np.sqrt(np.maximum(0.0, 1.0 - np.sum(Sig ** 2, axis=0)))


def sector_basis(Sig):
    """Orthonormal bases (even, odd) of the +1 and -1 eigenspaces of Sigma."""
    ev, U = la.eigh(Sig)
    return U[:, ev > 0], U[:, ev < 0]


def odd_weight(Sig, Q):
    """
    Per column of Q, the fraction of the norm in the odd sector,
    |P_- q|^2 / |q|^2 with P_- = (1 - Sigma)/2. Zero for an even vector, one
    for an odd one, ~1/2 for a generic one.
    """
    return np.clip(0.5 * (1 - np.einsum('ij,ij->j', Q, Sig @ Q)
                          / np.einsum('ij,ij->j', Q, Q)), 0.0, 1.0)
