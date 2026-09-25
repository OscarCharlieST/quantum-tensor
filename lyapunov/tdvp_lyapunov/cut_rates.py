"""
Tier 0 of the one-cut study: trajectory diagnostics only, no tangent
vectors.

Along a TDVP trajectory on the rank-r manifold, every step gives the kept
Schmidt values s and the normal core N~ exactly, and from them the rate
spectrum nu_a / s_i that bounds the Lyapunov spectrum:

    lambda_max <= < nu_1 / s_r >_t,     h_KS <= < 2 ||N~||_* tr s^-1 >_t

The exact trajectory from the same (truncated) start is carried alongside
densely, so the cost of the restriction is measured directly: fidelity,
and the rank-r discarded weight the exact state would need.
"""

import time as clock

import numpy as np
import scipy.linalg as la

import qtensor.states as states
import qtensor.thermofield as tf

from lyapunov.tdvp_lyapunov.stepper import tdvp_step, lanczos_method
from lyapunov.tdvp_lyapunov.cut_matrix import (
    apply_H, mps_to_matrix, dense_to_mps, truncate, CutPoint,
    mpo_cut_terms, taylor_action, H_norm_bound,
)


def dense_thermofield(L, beta, terms, cut):
    """
    e^{-beta H_sym / 4} on the infinite-temperature double, normalized: the
    state finite_T_thermofield targets (without its one-step overshoot).
    """
    Psi = mps_to_matrix(tf.inf_T_thermofield(L, 1), cut).astype(complex)
    Psi = taylor_action(lambda X: -apply_H(terms, X), beta / 4, Psi, H_norm_bound(terms))
    return Psi / la.norm(Psi)


class ExactThermofield:
    """
    Exact H_sym evolution of a doubled state, via the physical Hamiltonian.

    With the doubled index i = 2 a_phys + b_aux on every site, the state is
    a matrix M[a, b] over the two copies and H_sym = H (x) I + I (x) H acts
    as M -> H M + M H^T, so e^{-i H_sym t}: M -> U M U^T with U = e^{-iHt}.
    One eigendecomposition of the d^L physical H makes any t exact and
    O(d^{3L}).
    """

    def __init__(self, H_phys, L, cut):
        Ls, Rs = mpo_cut_terms(H_phys, max(H_phys.sites))
        Hp = np.einsum('cab,c->ab', Ls, Rs[:, 0, 0])
        self.E, self.V = la.eigh(Hp)
        self.Hp, self.L, self.cut = Hp, L, cut

    def to_copies(self, Psi):
        L = self.L
        T = Psi.reshape((2,) * (2 * L))                       # (a0, b0, a1, b1, ...)
        return T.transpose(list(range(0, 2 * L, 2)) + list(range(1, 2 * L, 2))).reshape(
            2 ** L, 2 ** L)

    def from_copies(self, M):
        L = self.L
        T = M.reshape((2,) * (2 * L))                         # (a0..a_{L-1}, b0..)
        order = [x for n in range(L) for x in (n, L + n)]
        nL = 4 ** (self.cut + 1)
        return T.transpose(order).reshape(nL, -1)

    def evolve(self, Psi, t):
        U = (self.V * np.exp(-1j * self.E * t)) @ self.V.conj().T
        M = self.to_copies(Psi)
        return self.from_copies(U @ M @ U.T)


def cut_entropy(s):
    p = s ** 2 / np.sum(s ** 2)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def entropy_curve(exact, Psi0, ts):
    """Exact cut entropy and smallest Schmidt value along e^{-iHt} Psi0."""
    S, smin = [], []
    for t in ts:
        s = la.svdvals(exact.evolve(Psi0, t))
        S.append(cut_entropy(s)), smin.append(s[-1])
    return np.array(S), np.array(smin)


def run_rates(Psi0, r, terms, H, exact, L, cut, dt, steps, method=None, verbose=True):
    """
    TDVP from the rank-r truncation of Psi0 for `steps` steps, with the
    exact evolution of the same truncated state alongside. Returns a dict of
    per-step arrays (index 0 is the start).
    """
    if method is None:
        method = lanczos_method()
    d = H[0].shape[0]
    Psi_r, discarded0 = truncate(Psi0, r)
    psi = states.mps(dense_to_mps(Psi_r.reshape(-1), L, d, r, cut))
    m = Psi0.shape[0] - r

    out = {k: [] for k in ('t', 's', 'nu', 'w_N', 'lam_bound', 'hks_bound',
                           'S_tdvp', 'S_exact', 'fidelity', 'energy',
                           'discarded_exact', 's_disc')}
    t_wall = clock.time()
    for j in range(steps + 1):
        if j:
            psi = tdvp_step(psi, H, dt, method)
        Psi_ex = exact.evolve(Psi_r, j * dt)
        Psi = mps_to_matrix(psi, cut)
        point = CutPoint(Psi / la.norm(Psi), r, terms)
        nu = la.svdvals(point.normal_core()) if m else np.zeros(0)
        lam_b, hks_b = point.rate_bounds()
        s_ex = la.svdvals(Psi_ex)

        out['t'].append(j * dt)
        out['s'].append(point.s)
        out['nu'].append(nu)
        out['w_N'].append(la.norm(point.w_N))
        out['lam_bound'].append(lam_b)
        out['hks_bound'].append(hks_b)
        out['S_tdvp'].append(cut_entropy(point.s))
        out['S_exact'].append(cut_entropy(s_ex))
        out['fidelity'].append(abs(np.vdot(Psi_ex, point.Psi)) ** 2)
        out['energy'].append(np.real(np.vdot(point.Psi, point.HPsi)))
        out['discarded_exact'].append(np.sum(s_ex[r:] ** 2))
        out['s_disc'].append(la.norm(point.s_disc))
        if verbose and (j % max(1, steps // 20) == 0):
            print(f"  step {j}/{steps} t={j * dt:.2f}  s_r={point.s[-1]:.2e}  "
                  f"nu_1={nu[0] if m else 0:.2e}  lam_bound={lam_b:.2f}  "
                  f"hks_bound={hks_b:.1f}  F={out['fidelity'][-1]:.6f}  "
                  f"E={out['energy'][-1]:.6f}  [{clock.time() - t_wall:.0f} s]",
                  flush=True)
    res = {k: np.array(v) for k, v in out.items()}
    res['discarded0'] = discarded0
    return res
