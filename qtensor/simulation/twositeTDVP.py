import numpy as np
import scipy.linalg as la
from ncon import ncon

import qtensor.states as states
import qtensor.operators as ops
import qtensor.simulation.finiteTDVP as sim
import qtensor.simulation.updatemethod as methods

"""
Two-site TDVP step, with SVD truncation -- the bond-dimension-growing
companion to finiteTDVP.tdvp_step, written against the same conventions
(tensor legs, L_con/R_con environments keyed by site, right-canonical state
with its centre at the first site before and after a step).

The use it was written for: growing bond dimension out of a product state
before handing over to single-site TDVP. Starting single-site TDVP from a
zero-padded product state makes the early dynamics depend on the arbitrary
orthonormal completion the first SVD picks for the padding -- measured at
L = 12 on |+>^L under the tilted Ising chain, mid-bond entropy at t = 0.2 is
correct for D = 8, 16, 24, 48 and 7% low for D = 32. Two-site updates don't
see the padding (the discarded weight is truncated, not carried), which is
also how the reference code for arXiv:2111.06408 starts its runs.

Indexing:

    1                           1             1
    ¦        2 -- A -- 3        ¦             ¦
    L -- 3  ,     ¦      , 3 -- W -- 4 , 3 -- R
    ¦             1             ¦             ¦
    2                           2             2
"""


def apply_Heff_two_site(T, W1, W2, L, R):
    """
    (H_eff T) for a two-site tensor T[s, t, l, r] (legs as in the header).
    Explicit tensordots: an optimize=True einsum re-plans its path on every
    call, which dominated the cost at small D.
    """
    X = np.tensordot(T, L, axes=([2], [0]))           # s t q m a
    X = np.tensordot(X, W1, axes=([0, 4], [0, 2]))    # t q m u b
    X = np.tensordot(X, W2, axes=([0, 4], [0, 2]))    # q m u v c
    X = np.tensordot(X, R, axes=([0, 4], [0, 2]))     # m u v r
    return X.transpose(1, 2, 0, 3)


def krylov_expm(matvec, v, dt, max_iters=20, epsilon=1e-12):
    """
    exp(-i dt H) v for Hermitian H given as a matvec, via Lanczos with full
    (double) reorthogonalization -- see the 2026-09-16 Lanczos fix in
    updatemethod for why the second pass matters.
    """
    shape = v.shape
    norm0 = la.norm(v)
    basis = [v.reshape(-1) / norm0]
    max_iters = min(max_iters, v.size)
    T = np.zeros((max_iters + 1, max_iters + 1), dtype=complex)
    k = 1
    for j in range(max_iters):
        w = matvec(basis[j].reshape(shape)).reshape(-1)
        for i, b in enumerate(basis):
            h = np.vdot(b, w)
            T[i, j] += h
            w = w - h * b
        for b in basis:
            w = w - np.vdot(b, w) * b
        beta = la.norm(w)
        k = j + 1
        if beta < epsilon or j == max_iters - 1:
            break
        T[j + 1, j] = beta
        basis.append(w / beta)
    Tk = T[:k, :k]
    Tk = 0.5 * (Tk + Tk.conj().T)
    coeffs = la.expm(-1j * dt * Tk)[:, 0]
    out = norm0 * sum(c * b for c, b in zip(coeffs, basis[:k]))
    return out.reshape(shape)


def split_two_site(T, max_bond_dim, cutoff):
    """
    SVD split of T[s, t, l, r] into U[s, l, chi], s, V[t, chi, r], keeping at
    most max_bond_dim values and dropping those below cutoff * largest.
    """
    d1, d2, Dl, Dr = T.shape
    M = T.transpose(0, 2, 1, 3).reshape(d1 * Dl, d2 * Dr)
    U, s, V = la.svd(M, full_matrices=False)
    chi = int(min(max_bond_dim, np.sum(s > cutoff * s[0])))
    chi = max(chi, 1)
    U = U[:, :chi].reshape(d1, Dl, chi)
    V = V[:chi, :].reshape(chi, d2, Dr).transpose(1, 0, 2)
    return U, s[:chi], V


def tdvp2_step(state, operator, dt, max_bond_dim,
               method=methods.lanczos_method(epsilon=1e-10, max_iters=16),
               cutoff=1e-10, krylov_iters=20, expm=krylov_expm):
    """
    One two-site TDVP step (right sweep, left sweep, dt/2 each), truncating
    every bond to max_bond_dim. Same contract as finiteTDVP.tdvp_step: the
    state is right-canonicalized first, mutated in place, and returned in
    right-canonical form with its centre at the first site. Environments are
    not threaded between calls (bond dimensions change, so they would be
    rebuilt anyway); the second return value is None to keep the signature.

    `method` supplies the single-site backward updates of the centre tensor
    between two-site updates. `expm(matvec, v, dt, krylov_iters)` does the
    two-site forward updates; the default assumes a Hermitian operator.
    """
    sites = sorted(state.sites)
    state.right_orthogonal()
    R_con = sim.right_mpo_contractions(state, operator)
    L_con = {sites[0] - 1: ncon((np.eye(1), operator.l), ((-1, -2), (-3,)))}

    # right sweep
    for k in range(len(sites) - 1):
        n, m = sites[k], sites[k + 1]
        T = np.einsum('slr,trq->stlq', state[n], state[m])
        T = expm(lambda x: apply_Heff_two_site(x, operator[n], operator[m],
                                               L_con[n - 1], R_con[m + 1]),
                 T, dt / 2, krylov_iters)
        U, s, V = split_two_site(T, max_bond_dim, cutoff)
        state[n] = U
        L_con[n] = ops.contract_left(L_con[n - 1], U, operator[n])
        C = np.einsum('a,tar->tar', s, V)
        C = C / la.norm(C)
        if k < len(sites) - 2:
            C = method.c((C, operator[m], L_con[n], R_con[m + 1]), dt=-dt / 2)
            C = C / la.norm(C)
        state[m] = C

    # left sweep
    for k in range(len(sites) - 1, 0, -1):
        n, m = sites[k - 1], sites[k]
        T = np.einsum('slr,trq->stlq', state[n], state[m])
        T = expm(lambda x: apply_Heff_two_site(x, operator[n], operator[m],
                                               L_con[n - 1], R_con[m + 1]),
                 T, dt / 2, krylov_iters)
        U, s, V = split_two_site(T, max_bond_dim, cutoff)
        state[m] = V
        R_con[m] = ops.contract_right(R_con[m + 1], V, operator[m])
        C = np.einsum('slr,r->slr', U, s)
        C = C / la.norm(C)
        if k > 1:
            C = method.c((C, operator[n], L_con[n - 1], R_con[m]), dt=-dt / 2)
            C = C / la.norm(C)
        state[n] = C

    state.form, state.c_site = 'right', sites[0]
    state.normalized = True
    return state, None


def bonds_saturated(state, max_bond_dim, d=2):
    """True once every bond is at min(max_bond_dim, d^left, d^right)."""
    sites = sorted(state.sites)
    N = len(sites)
    for k, n in enumerate(sites[:-1]):
        full = min(max_bond_dim, d ** (k + 1), d ** (N - k - 1))
        if state[n].shape[2] < full:
            return False
    return True
