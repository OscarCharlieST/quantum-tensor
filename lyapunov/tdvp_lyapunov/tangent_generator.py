"""
Route B: the exact generator of the linearized TDVP flow in the frame at a
point, and the one-step tangent map built from it.

    DF v = -i [ P H v + (D_v P) H psi ]

In frame coordinates (X complex, length n):

    term 1:  P H Phi(X)          ->  H_tan X          (complex-linear)
    term 2:  (D_v P) H psi       ->  conj(K X)        (anti-linear)

with H_tan the tangent-projected Hamiltonian (relaxation strand) and

    K_ij = < w_N | D^2 psi[dM^(j), dM^(i)] >,   w_N = (1 - P) H psi

the overlap of the normal residual with the two-defect states. Realified
on y = (Re X, Im X):

    A = [[ Hi - Ki,   Hr - Kr ],
         [ -(Hr + Kr), Hi + Ki ]]

The two-defect state for the pair (site n with direction i, site m with
direction j) is the basis vector Phi_m(e_j) = A_L..B_m..A_R with site n's
tensor replaced by  B_n Lambda_n^{-1}  (n < m)  or  Lambda_{n-1}^{-1} B_n
(n > m); the inverse bond matrices are the manifold's curvature and are
what degrades as the smallest Schmidt value shrinks.
"""

import numpy as np
import scipy.linalg as la
from ncon import ncon

import qtensor.operators as ops
from lyapunov.relaxation import tangent_hamiltonian as tangent
from lyapunov.tdvp_lyapunov.frame import (
    project_to_frame, tangent_mps, mps_direct_sum, frame_change,
    _left_transfer, _right_transfer,
)


# ------------------------------------------------------------- H psi, w_N

def apply_mpo(tensors, H):
    """{site: tensor} of H|psi> (bond dimension D * chi); no compression."""
    out = {}
    sites = sorted(tensors)
    for site in sites:
        M, W = tensors[site], H[site]
        d, Dl, Dr = M.shape
        _, _, wDl, wDr = W.shape
        T = ncon((W, M), ((1, -1, -3, -5), (1, -2, -4)))
        if site == sites[0]:
            T = ncon((T, H.l), ((-1, -2, 3, -4, -5), (3,))).reshape(d, Dl, Dr * wDr)
        elif site == sites[-1]:
            T = ncon((T, H.r), ((-1, -2, -3, -4, 5), (5,))).reshape(d, Dl * wDl, Dr)
        else:
            T = T.reshape(d, Dl * wDl, Dr * wDr)
        out[site] = T
    return out


def normal_residual(frame, H):
    """
    w_N = (1 - P) H psi as a {site: tensor} MPS of bond dimension D(chi+2):
    H psi  minus  its tangent projection Phi(h)  minus  <H> psi.
    Also returns h (coordinates of P H psi) and <H>.
    """
    Hpsi = apply_mpo(frame.A_R, H)
    h = project_to_frame(Hpsi, frame)
    E = np.real(ops.expect(frame.state(), H))
    tangent_part = tangent_mps(frame, h, eps=-1.0, include_point=True, point_weight=-E)
    return mps_direct_sum(Hpsi, tangent_part), h, E


# ------------------------------------------------------------------- K

def assemble_K(frame, w_N):
    """
    K_ij = <w_N | D^2 psi[dM^(j), dM^(i)]>, one column at a time. For each
    ket basis vector Phi_m(e_j), one sweep of mixed transfer matrices
    against w_N gives the gradient F_n of <w_N|Phi_m(e_j)> with respect to
    every other site's tensor; contracting F_n with the allowed
    perturbations at site n yields the whole column block.
    """
    n = frame.n
    sites = frame.sites
    K = np.zeros((n, n), dtype=complex)
    Lam_inv = {s: la.inv(L) for s, L in frame.Lam.items()}

    for m, col0, (n_null_m, D_m) in frame.index_map:
        for j in range(n_null_m * D_m):
            Y = np.zeros((n_null_m, D_m), dtype=complex)
            Y.flat[j] = 1.0
            ket = tangent.build_tangent_vector(frame.A_L, frame.A_R, frame.V_L[m], m, Y)

            EL = {sites[0] - 1: np.eye(1)}
            for s in sites:
                EL[s] = _left_transfer(EL[s - 1], ket[s], w_N[s])
            ER = {sites[-1] + 1: np.eye(1)}
            for s in reversed(sites):
                ER[s] = _right_transfer(ER[s + 1], ket[s], w_N[s])

            for nn, row0, (n_null_n, D_n) in frame.index_map:
                if nn == m:
                    continue
                # F[p, a, r]: <w_N| ket with site nn -> Z > = sum F Z
                F = ncon((EL[nn - 1], w_N[nn].conj(), ER[nn + 1]),
                         ((-2, 1), (-1, 1, 2), (-3, 2)))
                d, Dl, Dr = F.shape
                if nn < m:
                    G = F @ Lam_inv[nn].T                      # B Lambda^-1
                else:
                    G = ncon((Lam_inv[nn - 1], F), ((1, -2), (-1, 1, -3)))   # Lambda^-1 B
                block = frame.V_L[nn].T @ G.reshape(d * Dl, Dr)
                K[row0:row0 + n_null_n * D_n, col0 + j] = block.reshape(-1)
    return K


# ------------------------------------------------------------ generator

def assemble_H_tan(frame, H):
    left_envs, right_envs = tangent.build_environments(
        frame.A_L, frame.A_R, H.tensors, H.l, H.r, frame.sites)
    H_tan, _ = tangent.assemble_tangent_hamiltonian(
        frame.sites, frame.A_L, frame.A_R, frame.V_L, left_envs, right_envs, H)
    return H_tan


def realify_generator(H_tan, K):
    Hr, Hi, Kr, Ki = H_tan.real, H_tan.imag, K.real, K.imag
    return np.block([[Hi - Ki, Hr - Kr], [-(Hr + Kr), Hi + Ki]])


def generator(frame, H, with_K=True):
    """Real (2n, 2n) matrix A of the linearized flow in `frame`."""
    H_tan = assemble_H_tan(frame, H)
    if with_K:
        w_N, _, _ = normal_residual(frame, H)
        K = assemble_K(frame, w_N)
    else:
        K = np.zeros_like(H_tan)
    return realify_generator(H_tan, K)


def parallel_transport(frame, frame_next):
    """
    Second-order-accurate parallel transport between the two tangent
    spaces: the polar factor (nearest orthogonal matrix) of the projection
    O = P_next Phi. The bare projection shrinks vectors by dt^2/2 * II^dag II
    (the manifold's curvature along the step); the polar factor undoes
    exactly that, and is what makes the step map below second order --
    checked against the finite-difference map: error O(dt^3), and the flow
    direction is transported covariantly to O(dt^3).
    """
    T = frame_change(frame, frame_next)
    # O^T O = 1 - dt^2 S, so Newton-Schulz T <- T (3 - T^T T)/2 converges to
    # the polar factor quadratically from an O(dt^2) start: two iterations
    # (four matmuls) leave an O(dt^8) error, against an SVD of the same size.
    I = np.eye(T.shape[1])
    for _ in range(2):
        T = T @ (1.5 * I - 0.5 * (T.T @ T))
    return T


def half_step_propagator(A, dt):
    """expm(dt/2 A); the same matrix closes one step and opens the next."""
    return la.expm(0.5 * dt * A)


def propagate(Q, E, T, E_next):
    """One step applied to the tangent vectors: E_next . T . E . Q."""
    return E_next @ (T @ (E @ Q))


def step_matrix(frame, A, frame_next, A_next, dt):
    """
    Full tangent map over one step from frame to frame_next:
    expm(dt/2 A_next) . T . expm(dt/2 A), with T the parallel transport.
    (Validation use; the Benettin loop applies the factors to Q instead.)
    """
    T = parallel_transport(frame, frame_next)
    return half_step_propagator(A_next, dt) @ T @ half_step_propagator(A, dt)
