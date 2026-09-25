"""
The MPS manifold with every bond full except one cut, seen as the manifold
of rank-r matrices.

With bond dimensions (d, d^2, ..., r, ..., d^2, d) -- every bond at its
staircase maximum except the one right of site `cut`, capped at r -- the
MPS constraint is exactly rank(Psi) = r for the state reshaped across that
cut, Psi of shape (d^nL, d^nR). Everything the Lyapunov machinery needs
then has a closed form in U, s, V of one SVD of Psi:

    P(Z)      = P_U Z + Z P_V - P_U Z P_V
    D_xi P(Z) = DP_U Z (1 - P_V) + (1 - P_U) Z DP_V
    DP_U      = Q_U xi Psi^+ + Psi^+^dag xi^dag Q_U
    DP_V      = Q_V xi^dag Psi^+^dag + Psi^+ xi Q_V

(Q = 1 - P). Only projectors and Psi^+ appear, so nothing depends on the
gauge. The linearized TDVP flow in the ambient space is

    G xi = -i [ P(H xi) + D_xi P(H Psi) ]

whose tangent part is Route B's -i (H_tan X + conj(K X)), and whose normal
part carries xi along as the tangent space turns. The curvature acts only
through the m x m normal core N~ = U_perp^dag H Psi V_perp, swapping the two
cross blocks U_perp^dag xi V and U^dag xi V_perp; the symmetric part of the
generator has eigenvalues +-nu_a/s_i (each twice), nu = svd(N~).

Operators and matrices use O[out, in]; composite indices are C-ordered with
the leftmost site slowest, matching MPO leg order (in, out, left, right) as
in tangent_generator.apply_mpo.
"""

import numpy as np
import scipy.linalg as la


# ------------------------------------------------------------ MPO at a cut

def mpo_cut_terms(H, cut):
    """
    (Ls, Rs) with H = sum_c Ls[c] (x) Rs[c] across the bond right of site
    `cut`: the MPO halves contracted densely, one term per MPO bond index.
    """
    sites = sorted(H.sites)
    left = [s for s in sites if s <= cut]
    right = [s for s in sites if s > cut]

    E = np.asarray(H.l, dtype=complex).reshape(1, 1, -1)        # (out, in, bond)
    for s in left:
        W = np.asarray(H[s], dtype=complex)                       # (in, out, bl, br)
        E = np.einsum('abc,ioce->aobie', E, W)
        a, o, b, i, e = E.shape
        E = E.reshape(a * o, b * i, e)
    F = np.asarray(H.r, dtype=complex).reshape(-1, 1, 1)        # (bond, out, in)
    for s in reversed(right):
        W = np.asarray(H[s], dtype=complex)
        F = np.einsum('ioce,eab->coaib', W, F)
        c, o, a, i, b = F.shape
        F = F.reshape(c, o * a, i * b)
    # contiguous: a strided L_c sends numpy's matmul off BLAS (~70x slower)
    return np.ascontiguousarray(E.transpose(2, 0, 1)), np.ascontiguousarray(F)


def apply_H(terms, X):
    """H X for X of shape (..., dL, dR): sum_c L_c X R_c^T."""
    Ls, Rs = terms
    out = np.zeros(np.broadcast_shapes(X.shape, (Ls.shape[1], Rs.shape[1])), dtype=complex)
    for Lc, Rc in zip(Ls, Rs):
        out += Lc @ X @ Rc.T
    return out


def H_norm_bound(terms):
    """||H||_2 <= sum_c ||L_c|| ||R_c||; sets Taylor substeps only."""
    Ls, Rs = terms
    return sum(la.norm(Lc, 2) * la.norm(Rc, 2) for Lc, Rc in zip(Ls, Rs))


# ------------------------------------------------------------ MPS <-> dense

def mps_to_matrix(psi, cut):
    """Psi (d^nL, d^nR) of an MPS reshaped across the bond right of `cut`."""
    tensors = psi.tensors if hasattr(psi, 'tensors') else psi
    sites = sorted(tensors)
    left = [s for s in sites if s <= cut]
    right = [s for s in sites if s > cut]

    T = tensors[left[0]][:, 0, :]
    for s in left[1:]:
        T = np.einsum('aD,dDE->adE', T, tensors[s]).reshape(-1, tensors[s].shape[2])
    S = tensors[right[-1]][:, :, 0].T
    for s in reversed(right[:-1]):
        S = np.einsum('dDE,Eb->Ddb', tensors[s], S).reshape(tensors[s].shape[1], -1)
    return T @ S


def dense_to_mps(v, L, d, max_bond_dim=np.inf, cut=None):
    """
    {site: tensor} of the dense vector v (length d^L) by one left SVD sweep.
    Every bond gets its staircase maximum min(d^(n+1), d^(L-n-1)); with
    `cut` given, the bond right of that site is capped at max_bond_dim
    (otherwise every bond is). Left-canonical, norm kept in the last tensor.
    """
    tensors = {}
    M = np.asarray(v, dtype=complex).reshape(d, -1)
    Dl = 1
    for n in range(L - 1):
        chi = min(d ** (n + 1), d ** (L - n - 1))
        if cut is None or n == cut:
            chi = min(chi, max_bond_dim)
        U, s, Vh = la.svd(M, full_matrices=False)
        U, s, Vh = U[:, :chi], s[:chi], Vh[:chi]
        tensors[n] = U.reshape(Dl, d, chi).transpose(1, 0, 2)
        M = (s[:, None] * Vh).reshape(chi * d, -1)
        Dl = chi
    tensors[L - 1] = M.reshape(Dl, d, 1).transpose(1, 0, 2)
    return tensors


def truncate(Psi, r):
    """Best rank-r approximation, renormalized, and the discarded weight."""
    U, s, Vh = la.svd(Psi, full_matrices=False)
    discarded = np.sum(s[r:] ** 2) / np.sum(s ** 2)
    out = (U[:, :r] * s[:r]) @ Vh[:r]
    return out / la.norm(out), discarded


# ------------------------------------------------------------ one point

def _dag(X):
    return np.swapaxes(X.conj(), -1, -2)


# Stacks (k, p, q) of vectors times one fixed matrix, as a single GEMM each:
# numpy's broadcast matmul loops over the stack, which is ~4x slower here.

def _lmul(A, X):
    """A @ X[j] for every j."""
    k, q, s = X.shape
    Y = A @ X.transpose(1, 0, 2).reshape(q, k * s)
    return Y.reshape(A.shape[0], k, s).transpose(1, 0, 2)


def _rmul(X, B):
    """X[j] @ B for every j."""
    k, p, q = X.shape
    return (np.ascontiguousarray(X).reshape(k * p, q) @ B).reshape(k, p, B.shape[1])


class CutPoint:
    """
    One point Psi of the rank-r manifold, from a single SVD, with H Psi and
    its normal part cached.

    Attributes: U, V (kept singular vectors), U_perp, V_perp, s (kept
    Schmidt values, descending), s_disc (the rest; ~0 on the manifold),
    HPsi, w_N = (1 - P) H Psi.
    """

    def __init__(self, Psi, r, terms):
        self.Psi = Psi
        self.r = r
        U, s, Vh = la.svd(Psi)
        # contiguous copies: numpy's matmul drops to a slow non-BLAS loop on
        # strided slices, which cost ~40x in the stacked generator
        c = np.ascontiguousarray
        self.U, self.U_perp = c(U[:, :r]), c(U[:, r:])
        V = Vh.conj().T
        self.V, self.V_perp = c(V[:, :r]), c(V[:, r:])
        self.s, self.s_disc = s[:r], s[r:]
        self.HPsi = apply_H(terms, Psi)
        self.w_N = self.HPsi - self.P(self.HPsi)

        # for the stacked generator: H with identity factors marked None,
        # and the pieces of D_xi P(H Psi) that do not depend on xi
        eye = np.eye(Psi.shape[0])
        self._H = [(None if np.allclose(Lc, eye) else Lc,
                    None if np.allclose(Rc, eye) else Rc.T.copy())
                   for Lc, Rc in zip(*terms)]
        Z = self.HPsi
        self._Y1 = self.pinv_left(Z - self.PV_right(Z))            # Psi^+ Z Q_V
        self._Y4 = self.pinv_right(Z - self.PU(Z))                 # Q_U Z Psi^+
        self._Us, self._Vh, self._Uh = self.U / self.s, c(_dag(self.V)), c(_dag(self.U))

    # -- projectors and pseudoinverse, all acting on stacks (..., dL, dR)

    def PU(self, X):
        return self.U @ (_dag(self.U) @ X)

    def PV_right(self, X):                      # X P_V
        return (X @ self.V) @ _dag(self.V)

    def P(self, X):
        A = _dag(self.U) @ X                     # (r, dR)
        return self.U @ A + self.PV_right(X) - self.U @ ((A @ self.V) @ _dag(self.V))

    def pinv_left(self, X):                      # Psi^+ X, Psi^+ = V s^-1 U^dag
        return self.V @ ((_dag(self.U) @ X) / self.s[:, None])

    def pinv_right(self, X):                     # X Psi^+
        return ((X @ self.V) / self.s[None, :]) @ _dag(self.U)

    def pinv_dag_left(self, X):                  # Psi^+^dag X, Psi^+^dag = U s^-1 V^dag
        return self.U @ ((_dag(self.V) @ X) / self.s[:, None])

    def pinv_dag_right(self, X):                 # X Psi^+^dag
        return ((X @ self.U) / self.s[None, :]) @ _dag(self.V)

    def dP(self, Xi, Z):
        """D_xi P applied to Z."""
        QU = lambda X: X - self.PU(X)
        QV = lambda X: X - self.PV_right(X)
        Xd = _dag(Xi)
        ZQV = QV(Z)
        QUZ = QU(Z)
        return (QU(Xi @ self.pinv_left(ZQV))                 # Q_U xi Psi^+ Z Q_V
                + self.pinv_dag_left(Xd @ QU(ZQV))            # Psi^+^dag xi^dag Q_U Z Q_V
                + self.pinv_dag_right(QV(QUZ) @ Xd)           # Q_U Z Q_V xi^dag Psi^+^dag
                + QV(self.pinv_right(QUZ) @ Xi))              # Q_U Z Psi^+ xi Q_V

    def generator_reference(self, terms, Xi):
        """G xi = -i [P(H xi) + D_xi P(H Psi)], term by term (validation)."""
        return -1j * (self.P(apply_H(terms, Xi)) + self.dP(Xi, self.HPsi))

    def generator(self, Xi):
        """
        G xi = -i [P(H xi) + D_xi P(H Psi)] on a stack (k, dL, dR) or a
        single matrix: 24 GEMMs per vector, batched over the stack.
        """
        single = Xi.ndim == 2
        X = Xi[None] if single else Xi
        U, V, Uh, Vh, W = self.U, self.V, self._Uh, self._Vh, self.w_N

        HX = np.zeros(X.shape, dtype=complex)
        for Lc, RcT in self._H:
            T = X if Lc is None else _lmul(Lc, X)
            HX += T if RcT is None else _rmul(T, RcT)

        A = _lmul(Uh, HX)                                          # P(H xi)
        B = _rmul(HX, V) - _lmul(U, _rmul(A, V))
        out = _lmul(U, A) + _rmul(B, Vh)

        T = _rmul(X, self._Y1)                                     # Q_U xi Psi^+ Z Q_V
        out += T - _lmul(U, _lmul(Uh, T))
        S = _lmul(self._Y4, X)                                     # Q_U Z Psi^+ xi Q_V
        out += S - _rmul(_rmul(S, V), Vh)
        out += self._weingarten(X)
        out *= -1j
        return out[0] if single else out

    def tangent_generator(self, Xi):
        """
        The tangent part of G xi for tangent xi: -i [P(H xi) + Psi^+^dag
        xi^dag w_N + w_N xi^dag Psi^+^dag]. (The other two terms of D_xi P
        are purely normal.) In a frame this is Route B's -i(H_tan X +
        conj(K X)); 18 GEMMs per vector.
        """
        single = Xi.ndim == 2
        X = Xi[None] if single else Xi
        U, V, Uh, Vh = self.U, self.V, self._Uh, self._Vh
        HX = np.zeros(X.shape, dtype=complex)
        for Lc, RcT in self._H:
            T = X if Lc is None else _lmul(Lc, X)
            HX += T if RcT is None else _rmul(T, RcT)
        A = _lmul(Uh, HX)
        B = _rmul(HX, V) - _lmul(U, _rmul(A, V))
        out = _lmul(U, A) + _rmul(B, Vh) + self._weingarten(X)
        out *= -1j
        return out[0] if single else out

    def _weingarten(self, X):
        """Psi^+^dag xi^dag w_N + w_N xi^dag Psi^+^dag on a stack."""
        W = self.w_N
        return (_lmul(self._Us, _rmul(_dag(_rmul(X, self.V)), W))
                + _rmul(_lmul(W, _dag(_lmul(self._Uh, X))) / self.s, self._Vh))

    def horizontal(self, X):
        """Remove the complex component along Psi (the directions Psi, i Psi)."""
        psi = self.Psi / la.norm(self.Psi)
        c = np.einsum('ij,...ij->...', psi.conj(), X)
        return X - c[..., None, None] * psi

    # -- curvature diagnostics

    def normal_core(self):
        """N~ = U_perp^dag H Psi V_perp, the (m, m) core of w_N."""
        return _dag(self.U_perp) @ self.HPsi @ self.V_perp

    def rate_spectrum(self):
        """
        The positive instantaneous growth rates nu_a / s_i, descending, each
        listed twice: the positive eigenvalues of the symmetric part of the
        generator. Their negatives are the rest of its nonzero spectrum.
        """
        if self.U_perp.shape[1] == 0:
            return np.zeros(0)
        nu = la.svdvals(self.normal_core())
        rates = np.sort(np.outer(nu, 1.0 / self.s).ravel())[::-1]
        return np.repeat(rates, 2)

    def rate_bounds(self):
        """(nu_1 / s_r, 2 ||N~||_* tr s^-1): the lambda_max and h_KS bounds."""
        if self.U_perp.shape[1] == 0:
            return 0.0, 0.0
        nu = la.svdvals(self.normal_core())
        return nu[0] / self.s[-1], 2 * nu.sum() * np.sum(1.0 / self.s)


# ------------------------------------------------------------ exponentials

def taylor_action(apply, t, X, norm, tol=1e-14, theta=1.0):
    """
    exp(t A) X for A given as `apply`, by scaled Taylor series in
    ceil(|t| norm / theta) substeps; `norm` bounds ||A||.
    """
    s = max(1, int(np.ceil(abs(t) * norm / theta)))
    out = X
    for _ in range(s):
        term, acc = out, out
        for k in range(1, 100):
            term = (t / s / k) * apply(term)
            acc = acc + term
            if la.norm(term) <= tol * la.norm(acc):
                break
        out = acc
    return out


def exact_evolve(terms, Psi, t, norm=None):
    """e^{-iHt} Psi, densely."""
    if norm is None:
        norm = H_norm_bound(terms)
    return taylor_action(lambda X: -1j * apply_H(terms, X), t, Psi, norm)
