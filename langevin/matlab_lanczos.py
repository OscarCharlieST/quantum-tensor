"""
Line-for-line port of the reference code's Krylov exponential
(General/lanczos_exp.m in github.com/AndrewHallam/Langevin), for the
'paper_exact' scheme.

    vout = lanczos_exp(H, b, dt)  ~  exp(dt H) b      (dt = -1j * step for real time)

Three-term recurrence, no reorthogonalization, m = min(20, dim) fixed, stop
when beta < machine eps, alpha taken as wtemp' * v = conj(<v|H v>). The
output has unit norm (the MATLAB code normalizes b before its final
norm(b) * ...).

Why this port matters for the typo scheme: the recurrence defines each new
vector through H v_q = beta_q v_{q+1} + alpha_q v_q + beta_{q-1} v_{q-1}, so
H V = V T + (residual in v_{m+1}) holds exactly whatever alpha is. The
conjugated alpha only makes V non-orthogonal; V exp(dt T) e_1 is still
exp(dt H) b to O((dt |H|)^m / m!), for non-Hermitian H too. qtensor's own
Krylov routines assume a Hermitian H and do not:
  - updatemethod.lanczos_*: fills the lower triangle of the projected matrix
    with the conjugate of the upper one, which flips the sign of the
    anti-Hermitian part's off-diagonal action at first order in dt;
  - twositeTDVP.krylov_expm: symmetrizes (T + T^dag)/2, which drops the
    anti-Hermitian part altogether.
For Hermitian H all three agree up to roundoff and Krylov truncation.
"""

import numpy as np
import scipy.linalg as la

from qtensor.simulation.updatemethod import TDVPMethod, apply_Heff_parts, apply_Heff_bond

EPS = np.finfo(float).eps   # MATLAB eps
M_MAX = 20


def lanczos_exp(H, b, dt, m_max=M_MAX):
    """
    lanczos_exp.m. H: matvec on flat vectors; b: flat vector; dt: the complex
    exponent (update_A/update_C pass -1j * dt).
    """
    b = np.asarray(b, dtype=complex).reshape(-1)
    n = b.size
    if n == 1:
        h = H(b) / b
        return np.exp(h * dt) * b
    m = min(m_max, n)
    b = b / la.norm(b)
    V, alpha, beta = [b], [], []
    wtemp = H(b)
    alpha.append(np.vdot(wtemp, b))              # wtemp' * v(:,1)
    w = wtemp - alpha[0] * b
    with np.errstate(divide='ignore', invalid='ignore'):
        for q in range(1, m):                    # MATLAB q = 2 .. m
            beta.append(la.norm(w))
            V.append(w / beta[-1])
            wtemp = H(V[-1])
            alpha.append(np.vdot(wtemp, V[-1]))
            w = wtemp - alpha[-1] * V[-1] - beta[-1] * V[-2]
            if abs(beta[-1]) < EPS:
                # m = q - 1: drop the vector just made from the tiny beta
                V, alpha, beta = V[:q], alpha[:q], beta[:q - 1]
                break
    T = np.diag(np.array(alpha, dtype=complex))
    if beta:
        T += np.diag(beta, 1) + np.diag(beta, -1)
    return np.column_stack(V) @ la.expm(dt * T)[:, 0]   # norm(b) = 1 here


def matlab_centre(tensors, dt, **kwargs):
    """qtensor method.c contract: exp(-i dt H_eff) C, via lanczos_exp."""
    C, W, L, R = tensors
    shape = C.shape
    out = lanczos_exp(lambda x: apply_Heff_parts(x.reshape(shape), W, L, R).reshape(-1),
                      C, -1j * dt)
    return out.reshape(shape)


def matlab_bond(tensors, dt, **kwargs):
    """qtensor method.b contract: exp(-i dt H_eff) M, via lanczos_exp."""
    M, L, R = tensors
    shape = M.shape
    out = lanczos_exp(lambda x: apply_Heff_bond(x.reshape(shape), L, R).reshape(-1),
                      M, -1j * dt)
    return out.reshape(shape)


def matlab_method():
    """TDVPMethod using the reference Krylov exponential for sites and bonds."""
    return TDVPMethod(matlab_centre, matlab_bond)


def matlab_two_site_expm(matvec, v, dt, max_iters=None, **kwargs):
    """Drop-in for twositeTDVP.krylov_expm (update_AA.m uses the same lanczos_exp)."""
    shape = v.shape
    out = lanczos_exp(lambda x: matvec(x.reshape(shape)).reshape(-1), v, -1j * dt)
    return out.reshape(shape)
