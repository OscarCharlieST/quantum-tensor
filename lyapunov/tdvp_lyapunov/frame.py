"""
The orthonormal tangent frame at a point of the MPS manifold, and the two
primitives every Lyapunov calculation here is built on:

    project_to_frame(phi, frame)   ->  coordinates of P_psi |phi>
    retract(frame, X, eps)         ->  the MPS  psi + eps * Phi(X), compressed

Both work purely with Hilbert-space overlaps, so tensors from different
canonicalization passes never get combined. The frame itself comes from
one pass (canonicalize_and_build_environments-style), and the same
A_L/A_R/V_L are used for every overlap taken at that point.

Tangent vectors are carried as *real* coordinate vectors of length 2n,
(Re X, Im X), because the linearized TDVP flow is only real-linear.
"""

import numpy as np
import scipy.linalg as la
from ncon import ncon

import qtensor.states as states
from lyapunov.relaxation.tangent_hamiltonian import (
    build_null_space_tensor, build_centre_tensors,
)


class Frame:
    """
    Orthonormal basis of the projective tangent space at one MPS.

    Attributes
    ----------
    sites : sorted list of site labels
    A_L, A_R, C : {site: tensor}, one shared canonicalization pass
    V_L : {site: (d*Dl, n_null) array}, null-space tensors
    index_map : list of (site, start, (n_null, D_r)) -- which slice of the
        complex coordinate vector belongs to which site
    n : complex tangent dimension; real dimension is 2n
    """

    def __init__(self, psi, max_bond_dim=np.inf):
        tensors, _ = states.right_orthogonal_state(psi.tensors, max_bond_dim)
        A_L, _ = states.left_orthogonal_state(tensors, max_bond_dim)
        A_R, _ = states.right_orthogonal_state(A_L, max_bond_dim)
        self._init_from_tensors(A_L, A_R, None)

    @classmethod
    def from_tensors(cls, A_L, A_R, V_L=None):
        """
        Rebuild a frame from stored gauge tensors *without* re-canonicalizing.
        Re-running the SVD sweeps on a stored state lands in a different gauge
        wherever the Schmidt spectrum is degenerate (always, for an already
        canonical tensor), so coordinates stored against a frame are only
        meaningful with that frame's own A_L/A_R/V_L.
        """
        self = cls.__new__(cls)
        self._init_from_tensors(A_L, A_R, V_L)
        return self

    def _init_from_tensors(self, A_L, A_R, V_L):
        self.sites = sorted(A_L.keys())
        for site in self.sites:
            assert A_L[site].shape == A_R[site].shape, (
                f"canonical forms disagree on bond dimensions at site {site}: "
                f"{A_L[site].shape} vs {A_R[site].shape}")
        self.A_L, self.A_R = A_L, A_R
        self.C = build_centre_tensors(A_L, A_R)
        # bond matrices Lambda^n = (A_L^n)^dag C^n on the bond right of site n
        self.Lam = {}
        for site in self.sites[:-1]:
            d, Dl, Dr = A_L[site].shape
            self.Lam[site] = (A_L[site].reshape(d * Dl, Dr).conj().T
                              @ self.C[site].reshape(d * Dl, -1))
        if V_L is None:
            V_L = {n: build_null_space_tensor(A_L[n]) for n in self.sites}
        self.V_L = V_L

        self.index_map = []
        dim = 0
        for site in self.sites:
            n_null = self.V_L[site].shape[1]
            if n_null == 0:
                continue
            D_r = A_L[site].shape[2]
            self.index_map.append((site, dim, (n_null, D_r)))
            dim += n_null * D_r
        self.n = dim

    # ------------------------------------------------------------ helpers

    def state(self):
        """The point itself, as an mps in right-canonical form."""
        psi = states.mps(dict(self.A_R))
        psi.form, psi.c_site, psi.normalized = 'right', self.sites[0], True
        return psi

    def bond_dims(self):
        return [self.A_L[s].shape[2] for s in self.sites[:-1]]

    def schmidt_values(self):
        """
        Schmidt spectrum on every internal bond, from the bond matrices
        Lambda^n = (A_L^n)^dag C^n. The smallest of these is the conditioning
        diagnostic: both the retraction and the manifold's curvature go bad
        as it heads to zero.
        """
        return {site: la.svdvals(Lam) for site, Lam in self.Lam.items()}

    def unpack(self, X):
        """Complex coordinate vector -> {site: (n_null, D_r) block}."""
        return {site: X[start:start + a * b].reshape(a, b)
                for site, start, (a, b) in self.index_map}

    def pack(self, blocks):
        X = np.zeros(self.n, dtype=complex)
        for site, start, (a, b) in self.index_map:
            X[start:start + a * b] = blocks[site].reshape(-1)
        return X

    def site_weights(self, X):
        """|X^n|^2 per site, for a complex coordinate vector."""
        return {site: np.sum(np.abs(blk) ** 2)
                for site, blk in self.unpack(X).items()}


# ---------------------------------------------------------- real <-> complex

def realify(X):
    """Complex length-n coordinates -> real length-2n vector (Re, Im)."""
    return np.concatenate([X.real, X.imag])


def complexify(y):
    n = y.shape[0] // 2
    return y[:n] + 1j * y[n:]


# ---------------------------------------------------------- transfer matrices
#
# Mixed transfer matrices between a bra in one of the frame's gauges and an
# arbitrary ket MPS. Convention: E[ket_bond, bra_bond].

def _left_transfer(E, M_ket, A_bra):
    return ncon((E, M_ket, A_bra.conj()), ((1, 2), (3, 1, -1), (3, 2, -2)))


def _right_transfer(E, M_ket, A_bra):
    return ncon((E, M_ket, A_bra.conj()), ((1, 2), (3, -1, 1), (3, -2, 2)))


def project_to_frame(phi, frame):
    """
    Coordinates of the tangent projection of |phi> in `frame`:
    X^n[a, r] = <Phi_n(e_{a r}) | phi>.

    Gauge-invariant in phi (any representation of the same Hilbert vector
    gives the same result), so phi may come from any canonicalization
    pass. Its global phase is *not* removed. Returns the complex length-n
    vector; realify() it for Benettin.

    One left sweep and one right sweep of mixed transfer matrices, then a
    local contraction per site: O(N d D^3).
    """
    sites = frame.sites
    tensors = phi.tensors if hasattr(phi, 'tensors') else phi

    L = {sites[0] - 1: np.eye(1)}
    for site in sites:
        L[site] = _left_transfer(L[site - 1], tensors[site], frame.A_L[site])
    R = {sites[-1] + 1: np.eye(1)}
    for site in reversed(sites):
        R[site] = _right_transfer(R[site + 1], tensors[site], frame.A_R[site])

    blocks = {}
    for site, _, _ in frame.index_map:
        F = ncon((L[site - 1], tensors[site], R[site + 1]),
                 ((1, -2), (-1, 1, 2), (2, -3)))
        d, Dl, Dr = F.shape
        blocks[site] = frame.V_L[site].conj().T @ F.reshape(d * Dl, Dr)
    return frame.pack(blocks)


def overlap_with_point(phi, frame):
    """<psi|phi> for the frame's own point psi -- the component the
    projection drops. Used to phase-fix perturbed states."""
    sites = frame.sites
    tensors = phi.tensors if hasattr(phi, 'tensors') else phi
    E = np.eye(1)
    for site in sites:
        E = _left_transfer(E, tensors[site], frame.A_L[site])
    return E[0, 0]


# ------------------------------------------------------------------ retract

def tangent_mps(frame, X, eps=1.0, include_point=True, point_weight=1.0):
    """
    Tensors of point_weight * psi + eps * sum_n Phi_n(X^n) as a
    bond-dimension-2D MPS, via the block construction

        T^n = [[A_L^n, eps B^n], [0, A_R^n]],   B^n = V_L^n X^n

    with boundary vectors (1, point_weight) on the left and (0, 1) on the
    right. The path that never leaves the lower block is psi itself; drop
    it (include_point=False, left boundary (1, 0)) to get the bare tangent
    vector eps * Phi(X) instead.
    """
    blocks = frame.unpack(X)
    sites = frame.sites
    T = {}
    for site in sites:
        AL, AR = frame.A_L[site], frame.A_R[site]
        d, Dl, Dr = AL.shape
        if site in blocks:
            B = eps * (frame.V_L[site] @ blocks[site]).reshape(d, Dl, Dr)
        else:
            B = np.zeros_like(AL)
        top = np.concatenate([AL, B], axis=2)
        bot = np.concatenate([np.zeros_like(AL), AR], axis=2)
        T[site] = np.concatenate([top, bot], axis=1)     # (d, 2Dl, 2Dr)

    first, last = sites[0], sites[-1]
    l = np.array([1.0, point_weight if include_point else 0.0])
    r = np.array([0.0, 1.0])
    T[first] = ncon((l, T[first].reshape(4, 2, 1, -1)), ((1,), (-1, 1, -2, -3)))
    d, Dl2, _ = T[last].shape
    T[last] = ncon((T[last].reshape(d, Dl2, 2, 1), r), ((-1, -2, 1, -3), (1,)))
    return T


def mps_direct_sum(T1, T2):
    """
    Tensors of the Hilbert-space sum of two MPS given as {site: tensor}
    dicts on the same sites: block-diagonal in the bulk, concatenated along
    the open bond at the two edge sites.
    """
    sites = sorted(T1)
    T = {}
    for site in sites:
        a, b = T1[site], T2[site]
        d, Dl1, Dr1 = a.shape
        _, Dl2, Dr2 = b.shape
        if site == sites[0]:
            T[site] = np.concatenate([a, b], axis=2)
        elif site == sites[-1]:
            T[site] = np.concatenate([a, b], axis=1)
        else:
            blk = np.zeros((d, Dl1 + Dl2, Dr1 + Dr2), dtype=np.result_type(a, b))
            blk[:, :Dl1, :Dr1] = a
            blk[:, Dl1:, Dr1:] = b
            T[site] = blk
    return T


def frame_change(frame_a, frame_b):
    """
    Real (2n, 2n) matrix of the identity map between two frames at (nearly)
    the same point: column j holds the coordinates in frame_b of frame_a's
    j-th real basis vector. Orthogonal up to the distance between the two
    points. n projections, each O(N d D^3).
    """
    assert frame_a.n == frame_b.n, "frames have different bond dimensions"
    n = frame_a.n
    O = np.zeros((2 * n, 2 * n))
    for j in range(n):
        X = np.zeros(n, dtype=complex)
        X[j] = 1.0
        c = project_to_frame(tangent_mps(frame_a, X, include_point=False), frame_b)
        O[:, j] = realify(c)
        O[:, n + j] = realify(1j * c)
    return O


def retract(frame, X, eps, max_bond_dim):
    """
    An MPS for psi + eps * Phi(X), normalized and SVD-compressed back to
    max_bond_dim. Since Phi(X) is tangent, the compression changes the
    state only at O(eps^2). Returned right-canonical, ready for a TDVP step.
    """
    T = tangent_mps(frame, X, eps)
    psi = states.mps(T)
    # The block-sum tensors are not canonical, and a truncating sweep only
    # sees the true Schmidt values if the part of the chain behind it is
    # isometric. So: one lossless sweep first, then truncate, then two more
    # sweeps to clip the bond dimensions to both staircases (as Frame does).
    # Leaves the state right-canonical.
    psi.left_orthogonal()
    psi.right_orthogonal(max_bond_dim)
    psi.left_orthogonal(max_bond_dim)
    psi.right_orthogonal(max_bond_dim)
    for site in frame.sites:
        assert psi[site].shape == frame.A_R[site].shape, (
            f"retract changed the bond dimensions at site {site}: "
            f"{psi[site].shape} vs {frame.A_R[site].shape}")
    return psi
