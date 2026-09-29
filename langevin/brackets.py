"""
Poisson brackets on the single-site MPS manifold, for the Langevin coefficients

    A_m    = {H, F_m}      rate of change of <F_m> under the TDVP flow of H
    K_mn   = {F_m, F_n}    rate of change of <F_n> under the TDVP flow of F_m

with the convention that the TDVP flow of a generator G is psi' = -i P G psi,
so that {G, B} = d<B>/dt|_G = 2 Im <B psi| P |G psi>. At full bond dimension
(P = 1) this is i<[G, B]>, the Heisenberg rate, which is what
validate_brackets.py checks against.

P is the single-site TDVP projector
    P = sum_n  P_L^{n-1} (x) (1 - A_L^n A_L^n^dag) (x) P_R^{n+1}    n < N
      +        P_L^{N-1} (x) 1_N
written in the orthonormal coordinates Y_n = V_L^n^dag M_n (n < N), Y_N = M_N,
where M_n = <A_L..A_L e_n A_R..A_R | G psi> is the one-site effective
operator applied to the centre tensor, and V_L^n spans the complement of A_L^n.
In those coordinates <u|P|v> = sum_n Y_n(u)^dag Y_n(v), so both matrices are
imaginary parts of one Gram matrix over the 3N + 1 generators {F_m} + {H}.
This is the same decomposition as the reference code's AC-minus-C sums
(calculate_friction.m), written in the gauge-fixed form, so there is no
cancellation between the two halves.

Everything is built from ONE canonicalization pass: A_L, A_R, C and V_L are
all derived from it, never recomputed (see the gauge-reproducibility TODO in
CLAUDE.md).

Cost: O(N^2 d D^3) for the transfers of the 3N single-site operators, plus
O(N^2 D^2) memory for the right-hand transfer stacks. Fine to N ~ 30, D ~ 64.
"""

import numpy as np
import scipy.linalg as la
from ncon import ncon

import qtensor.states as states
import qtensor.operators as ops
import qtensor.simulation.updatemethod as methods

from langevin.models import PAULIS

PAULI_STACK = np.stack(PAULIS)          # (3, 2, 2), O[a, out, in]


class Frame:
    """
    One canonicalization pass of psi: A_L (sites 0..N-2), A_R (sites 1..N-1),
    centre tensors C (all sites) with psi = A_L..A_L C_n A_R..A_R for every n,
    and the complements V_L (sites 0..N-2).
    """

    def __init__(self, psi):
        tensors, _ = states.right_orthogonal_state(psi.tensors, np.inf)
        A_L, _ = states.left_orthogonal_state(tensors, np.inf)
        self.sites = sorted(A_L.keys())
        sites = self.sites
        self.A_L = {n: A_L[n] for n in sites[:-1]}
        self.C = {sites[-1]: A_L[sites[-1]]}
        self.A_R = {}
        for k in range(len(sites) - 1, 0, -1):
            n = sites[k]
            G, self.A_R[n] = states.right_orthogonal_tensor(self.C[n])
            self.C[sites[k - 1]] = A_L[sites[k - 1]] @ G
        self._V_L = None

    @property
    def V_L(self):
        """Complements of A_L, built on first use: only the projector route needs them."""
        if self._V_L is None:
            self._V_L = {}
            for n in self.sites[:-1]:
                d, Dl, Dr = self.A_L[n].shape
                Q, _ = la.qr(self.A_L[n].reshape(d * Dl, Dr), mode='full')
                self._V_L[n] = Q[:, Dr:]
        return self._V_L

    def state(self):
        """The point as a right-canonical mps (centre at the first site)."""
        tensors = dict(self.A_R)
        tensors[self.sites[0]] = self.C[self.sites[0]]
        psi = states.mps(tensors)
        psi.form, psi.c_site, psi.normalized = 'right', self.sites[0], True
        return psi

    def coordinates(self, n, M):
        """Tangent coordinates at site n of a stack of site tensors M (k, d, Dl, Dr)."""
        k, d, Dl, Dr = M.shape
        if n == self.sites[-1]:
            return M.reshape(k, -1)
        Y = np.einsum('xa,kxr->kar', self.V_L[n].conj(), M.reshape(k, d * Dl, Dr),
                      optimize=True)
        return Y.reshape(k, -1)

    def schmidt_values(self):
        """Schmidt spectrum on the bond right of each site but the last."""
        out = {}
        for n in self.sites[:-1]:
            d, Dl, Dr = self.C[n].shape
            out[n] = la.svdvals(self.C[n].reshape(d * Dl, Dr))
        return out


def _onsite_transfer_left(A, O):
    """E[a][r, r'] = sum A*[t,l,r'] O[a,t,s] A[s,l,r]  (left leg traced)."""
    return np.einsum('tlq,ats,slr->arq', A.conj(), O, A, optimize=True)


def _onsite_transfer_right(A, O):
    """G[a][l, l'] = sum A*[t,l',r] O[a,t,s] A[s,l,r]  (right leg traced)."""
    return np.einsum('tqr,ats,slr->alq', A.conj(), O, A, optimize=True)


def _propagate_left(E, A):
    """E'[k][r, r'] = sum A[s,l,r] E[k][l,l'] A*[s,l',r']."""
    if len(E) == 0:
        return np.zeros((0, A.shape[2], A.shape[2]), dtype=complex)
    return np.einsum('slr,klm,smq->krq', A, E, A.conj(), optimize=True)


def _propagate_right(G, A):
    """G'[k][l, l'] = sum A[s,l,r] G[k][r,r'] A*[s,l',r']."""
    if len(G) == 0:
        return np.zeros((0, A.shape[1], A.shape[1]), dtype=complex)
    return np.einsum('slr,krq,smq->klm', A, G, A.conj(), optimize=True)


def poisson_brackets(psi, H, frame=None):
    """
    A (3N,), K (3N, 3N), and local expectations f (3N,) = <F_m>, at psi.

    Generator index m = 3 * (position of site) + a, a in (X, Y, Z), matching
    the reference code's etak ordering. H is a qtensor mpo covering every site.
    """
    if frame is None:
        frame = Frame(psi)
    sites = frame.sites
    N = len(sites)
    nF = 3 * N
    iH = nF
    Gram = np.zeros((nF + 1, nF + 1), dtype=complex)
    f = np.zeros(nF)

    # H environments in the frame's gauges
    L_H = {sites[0] - 1: ncon((np.eye(1), H.l), ((-1, -2), (-3,)))}
    for n in sites[:-1]:
        L_H[n] = ops.contract_left(L_H[n - 1], frame.A_L[n], H[n])
    R_H = {sites[-1] + 1: ncon((np.eye(1), H.r), ((-1, -2), (-3,)))}
    for n in reversed(sites[1:]):
        R_H[n] = ops.contract_right(R_H[n + 1], frame.A_R[n], H[n])

    # Right stacks of single-site-operator transfers: right_stack[n] holds,
    # for every F on a site j > n, its transfer on the bond right of n.
    right_stack = {sites[-1]: (np.zeros((0, 1, 1), dtype=complex), [])}
    for k in range(N - 1, 0, -1):
        n, A = sites[k], frame.A_R[sites[k]]
        G_old, idx_old = right_stack[n]
        G_new = np.concatenate([_propagate_right(G_old, A),
                                _onsite_transfer_right(A, PAULI_STACK)])
        right_stack[sites[k - 1]] = (G_new, idx_old + [3 * k + a for a in range(3)])

    E, idx_left = np.zeros((0, 1, 1), dtype=complex), []
    for k, n in enumerate(sites):
        C = frame.C[n]
        blocks, idx = [], []
        # H
        MH = methods.apply_Heff_parts(C, H[n], L_H[n - 1], R_H[n + 1])
        blocks.append(MH[None])
        idx.append(iH)
        # F on this site
        M0 = np.einsum('ats,slr->atlr', PAULI_STACK, C, optimize=True)
        blocks.append(M0)
        idx += [3 * k + a for a in range(3)]
        f[3 * k:3 * k + 3] = np.real(np.einsum('tlr,atlr->a', C.conj(), M0))
        # F to the left: left transfer on bond left of n
        if len(E):
            blocks.append(np.einsum('kab,sar->ksbr', E, C, optimize=True))
            idx += idx_left
        # F to the right
        G, idx_right = right_stack[n]
        if len(G):
            blocks.append(np.einsum('slr,krq->kslq', C, G, optimize=True))
            idx += idx_right
        Y = frame.coordinates(n, np.concatenate(blocks))
        idx = np.array(idx)
        Gram[np.ix_(idx, idx)] += Y.conj() @ Y.T
        # advance the left stack past site n
        if k < N - 1:
            A = frame.A_L[n]
            E = np.concatenate([_propagate_left(E, A),
                                _onsite_transfer_left(A, PAULI_STACK)])
            idx_left = idx_left + [3 * k + a for a in range(3)]

    A_vec = 2 * np.imag(Gram[:nF, iH])
    K = 2 * np.imag(Gram[:nF, :nF]).T        # K[m, n] = 2 Im <F_n|P|F_m>
    return A_vec, K, f


"""
Fast path. A single-site operator F_j maps the manifold into itself
(exp(-i theta F_j) psi has the same bond dimensions), so F_j psi is already a
tangent vector and P F_j psi = F_j psi. Then

    A_m  = 2 Im <F_m psi|P|H psi>   = 2 Im <F_m psi|H psi>   = i<[H, F_m]>
    K_mn = 2 Im <F_n psi|P|F_m psi> = i<[F_m, F_n]>          (same site only)

-- both are *exact* Heisenberg expectation values, whatever D is. K is exactly
block diagonal, K_j[a, b] = -2 eps_abc <sigma^c_j>, so the reference code's
same-site truncation of K loses nothing, and A only needs the nearest-neighbour
reduced density matrices of a nearest-neighbour H. validate_brackets.py checks
this path against the projector one above at truncated D.
"""

LEVI_CIVITA = np.zeros((3, 3, 3))
for _a, _b, _c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    LEVI_CIVITA[_a, _b, _c], LEVI_CIVITA[_b, _a, _c] = 1, -1


def local_rdms(frame):
    """One-site rho_j (d, d) and nearest-neighbour rho_{j,j+1} (d, d, d, d) as [out.., in..]."""
    sites = frame.sites
    rho1 = [np.einsum('slr,tlr->st', frame.C[n], frame.C[n].conj()) for n in sites]
    rho2 = []
    for k, n in enumerate(sites[:-1]):
        T = np.einsum('slr,trq->stlq', frame.C[n], frame.A_R[sites[k + 1]], optimize=True)
        rho2.append(np.einsum('stlq,uvlq->stuv', T, T.conj(), optimize=True))
    return rho1, rho2


def local_brackets(frame, onsite, bond):
    """
    A (3N,), K (3N, 3N), f (3N,) and the energy, from local density matrices,
    for H = sum_j onsite_j + sum_j bond_{j,j+1}.

    onsite : (2, 2) matrix, or list of N of them
    bond   : (4, 4) matrix acting on (j, j+1), or list of N-1 of them
    """
    rho1, rho2 = local_rdms(frame)
    N = len(rho1)
    onsite = onsite if isinstance(onsite, list) else [onsite] * N
    bond = bond if isinstance(bond, list) else [bond] * (N - 1)
    I = np.eye(2)
    f = np.real(np.einsum('jst,ats->ja', np.array(rho1), PAULI_STACK)).reshape(-1)
    A = np.zeros((N, 3))
    energy = 0.0
    for j in range(N):
        h = onsite[j]
        energy += np.real(np.trace(rho1[j] @ h))
        for a, F in enumerate(PAULIS):
            A[j, a] += np.real(1j * np.trace(rho1[j] @ (h @ F - F @ h)))
    for j in range(N - 1):
        b = bond[j]
        r = rho2[j].reshape(4, 4)
        energy += np.real(np.trace(r @ b))
        for a, F in enumerate(PAULIS):
            for site, Fb in ((j, np.kron(F, I)), (j + 1, np.kron(I, F))):
                A[site, a] += np.real(1j * np.trace(r @ (b @ Fb - Fb @ b)))
    sig = f.reshape(N, 3)
    K = np.zeros((3 * N, 3 * N))
    for j in range(N):
        K[3 * j:3 * j + 3, 3 * j:3 * j + 3] = -2 * np.einsum('abc,c->ab', LEVI_CIVITA, sig[j])
    return A.reshape(-1), K, f, energy


def block_diagonal(K):
    """Keep only the same-site 3x3 blocks of K (what the reference code computes)."""
    nF = K.shape[0]
    site = np.arange(nF) // 3
    return np.where(site[:, None] == site[None, :], K, 0.0)
