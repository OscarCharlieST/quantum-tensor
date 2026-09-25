"""
Benettin for the one-cut manifold in the ambient matrix space.

Tangent vectors are carried as complex (dL, dR) matrices xi, a stack of k
of them, horizontal (orthogonal to Psi and i Psi) and tangent at the
current point. One step is Route B's scheme written in ambient
coordinates, so no frame or basis is ever built:

    xi <- e^{dt/2 A'} T e^{dt/2 A} xi

with A = Pi_h (tangent part of G) Pi_h -- in any frame, Route B's
-i(H_tan X + conj(K X)) -- and T the polar factor of the projection
O = Pi_h' P' from the old tangent space to the new, applied through the
same two Newton-Schulz iterations as tangent_generator.parallel_transport,
with O^T O = Pi_h P Pi_h' P' evaluated by projections. Every factor is
symplectic or orthogonal, so the volume is kept to roundoff.

(Integrating the full ambient flow d xi/dt = G xi with frozen G instead,
and projecting at the end of the step, is also second order, but loses
volume at O(dt^2) through the normal detour: -0.13 per unit time at
dt = 0.05, L = 4, m = 2.)

The metric is the real part of the Hilbert inner product, so QR on
(Re, Im) of the flattened matrices is QR in the Fubini-Study metric. The
trajectory is stepper.tdvp_step on the MPS, converted to Psi at each step.

Storage follows benettin.py (R, log_diag_R, t, exponents, Q/<j>), so
plots.load_run reads these files; extra per-block data: s_r, lam_bound,
hks_bound, energy, fidelity with the exact trajectory, and Psi/<j> beside
each stored Q.
"""

import time as clock

import h5py
import numpy as np
import scipy.linalg as la

import qtensor.states as states

from lyapunov.tdvp_lyapunov.benettin import positive_qr
from lyapunov.tdvp_lyapunov.stepper import tdvp_step, lanczos_method
from lyapunov.tdvp_lyapunov.cut_matrix import (
    mpo_cut_terms, mps_to_matrix, dense_to_mps, CutPoint, taylor_action,
)


# ------------------------------------------------------------ vectors

def to_real(Q):
    """(k, dL, dR) complex stack -> (2 dL dR, k) real matrix of (Re, Im) columns."""
    k = Q.shape[0]
    flat = Q.reshape(k, -1)
    return np.concatenate([flat.real, flat.imag], axis=1).T


def from_real(Y, shape):
    half = Y.shape[0] // 2
    return (Y[:half] + 1j * Y[half:]).T.reshape((Y.shape[1],) + shape)


def tangent_dimension(N, m):
    """Real dimension of the projective tangent space, 2 (N^2 - m^2 - 1)."""
    return 2 * (N * N - m * m - 1)


def random_tangent(point, k, seed=0):
    rng = np.random.default_rng(seed)
    N = point.Psi.shape
    Z = rng.normal(size=(k,) + N) + 1j * rng.normal(size=(k,) + N)
    Q, _ = positive_qr(to_real(point.horizontal(point.P(Z))))
    return from_real(Q, N)


def tangent_map(point):
    """A: xi -> Pi_h (tangent part of G xi), on horizontal tangent stacks."""
    return lambda X: point.horizontal(point.tangent_generator(X))


def generator_norm(point, iters=8, seed=0):
    """||A||, estimated by power iteration on one vector; sets substeps only."""
    A = tangent_map(point)
    rng = np.random.default_rng(seed)
    x = point.horizontal(point.P(rng.normal(size=point.Psi.shape)
                                 + 1j * rng.normal(size=point.Psi.shape)))
    x /= la.norm(x)
    nrm = 0.0
    for _ in range(iters):
        y = A(x)
        nrm = la.norm(y)
        x = y / nrm
    return 1.2 * nrm


def transport(Q, point, point_next):
    """
    Polar factor of O = Pi_h' P' (old tangent space -> new) applied to Q:
    two Newton-Schulz iterations T <- T (3 - T^T T)/2 from T = O, i.e.
    O g(M) Q with M = O^T O = Pi_h P Pi_h' P' and
    g(M) = h (3/2 - M h^2 / 2),  h = 3/2 - M/2.
    """
    O = lambda X: point_next.horizontal(point_next.P(X))
    M = lambda X: point.horizontal(point.P(O(X)))
    h = lambda X: 1.5 * X - 0.5 * M(X)
    w = h(h(Q))
    return O(h(1.5 * Q - 0.5 * M(w)))


# ------------------------------------------------------------ one step

def step_map(Q, point, point_next, dt, norms=None):
    """e^{dt/2 A'} T e^{dt/2 A} Q: Route B's step, in ambient coordinates."""
    na, nb = norms if norms is not None else (generator_norm(point), generator_norm(point_next))
    Q = taylor_action(tangent_map(point), 0.5 * dt, Q, na)
    Q = transport(Q, point, point_next)
    return taylor_action(tangent_map(point_next), 0.5 * dt, Q, nb)


# ------------------------------------------------------------ Benettin

def make_point(psi, r, terms, cut):
    Psi = mps_to_matrix(psi, cut)
    return CutPoint(Psi / la.norm(Psi), r, terms)


def cut_benettin(psi0, H, cut, r, dt, n_blocks, k, method=None, exact=None,
                 store_path=None, store_Q_blocks=(), attrs=None, seed=0,
                 Q0=None, verbose=True):
    """
    Forward Benettin from the MPS psi0 (bond r at the cut, staircase
    elsewhere) under the MPO H. One QR per TDVP step (tau = 1).

    exact : optional cut_rates.ExactThermofield; if given, the fidelity of
        the TDVP trajectory with the exact evolution of psi0 is logged.
    Returns a dict like benettin.benettin's.
    """
    if method is None:
        method = lanczos_method()
    terms = mpo_cut_terms(H, cut)
    psi = psi0
    point = make_point(psi, r, terms, cut)
    shape = point.Psi.shape
    N, m = shape[0], shape[0] - r
    n = tangent_dimension(N, m) // 2
    assert k <= 2 * n, f"k = {k} exceeds the real tangent dimension {2 * n}"
    Q = random_tangent(point, k, seed) if Q0 is None else Q0.copy()
    Psi_start = point.Psi.copy()
    norm = generator_norm(point)
    if verbose:
        print(f"one-cut Benettin: N = {N}, m = {m}, r = {r}, real tangent dimension "
              f"{2 * n}, k = {k}, |G| ~ {norm:.1f}", flush=True)

    log_diag = np.zeros((n_blocks, k))
    diag = {key: np.zeros(n_blocks) for key in
            ('t', 's_r', 'lam_bound', 'hks_bound', 'energy', 'fidelity')}
    f = None
    if store_path is not None:
        f = h5py.File(store_path, 'w')
        f.attrs.update({'dt': dt, 'tau': 1, 'k': k, 'n': n, 'transient_steps': 0,
                        'N': N, 'm': m, 'r': r, 'cut': cut})
        if attrs:
            f.attrs.update(attrs)
        R_store = f.create_dataset('R', (n_blocks, k, k), dtype='f8',
                                   chunks=(1, k, k), compression='lzf')
        f.create_dataset('log_diag_R', (n_blocks, k), dtype='f8')
        for key in diag:
            f.create_dataset(key, (n_blocks,), dtype='f8')
        f.create_group('Q')
        f.create_group('Psi')
        f.attrs['blocks_done'] = 0

    t = 0.0
    t_wall = clock.time()
    for j in range(n_blocks):
        psi_next = tdvp_step(psi, H, dt, method)
        point_next = make_point(psi_next, r, terms, cut)
        norm_next = generator_norm(point_next)
        Q = step_map(Q, point, point_next, dt, (norm, norm_next))
        psi, point, norm = psi_next, point_next, norm_next
        t += dt

        Y, R = positive_qr(to_real(Q))
        Q = from_real(Y, shape)
        log_diag[j] = np.log(np.diag(R))
        lam_b, hks_b = point.rate_bounds()
        diag['t'][j], diag['s_r'][j] = t, point.s[-1]
        diag['lam_bound'][j], diag['hks_bound'][j] = lam_b, hks_b
        diag['energy'][j] = np.real(np.vdot(point.Psi, point.HPsi))
        if exact is not None:
            diag['fidelity'][j] = abs(np.vdot(exact.evolve(Psi_start, t), point.Psi)) ** 2

        if f is not None:
            R_store[j] = R
            f['log_diag_R'][j] = log_diag[j]
            for key, v in diag.items():
                f[key][j] = v[j]
            f.attrs['blocks_done'] = j + 1
            if j in store_Q_blocks:
                f['Q'].create_dataset(str(j), data=Q)
                f['Psi'].create_dataset(str(j), data=point.Psi)
            f.flush()
        if verbose:
            lam = log_diag[:j + 1].sum(0) / ((j + 1) * dt)
            print(f"block {j + 1}/{n_blocks}  t={t:.3f}  lambda_max={lam[0]:+.4f}  "
                  f"lambda_min={lam[-1]:+.4f}  sum={lam.sum():+.3e}  "
                  f"s_r={point.s[-1]:.1e}  bound={lam_b:.2f}  "
                  f"E={diag['energy'][j]:.8f}  [{clock.time() - t_wall:.0f} s]", flush=True)

    result = {'exponents': log_diag.sum(0) / (n_blocks * dt), 'log_diag_R': log_diag,
              'dt': dt, 'tau': 1, 'n': n, 'k': k, 'Q': Q, 'state': psi, **diag}
    if f is not None:
        f.create_dataset('exponents', data=result['exponents'])
        f.create_dataset('Q_current', data=Q)
        for site in psi.sites:
            f.create_dataset(f'state/{site}', data=psi[site])
        f.close()
    return result


def start_state(Psi0, L, d, r, cut):
    """The rank-r truncation of the dense Psi0 as an MPS with the staircase."""
    return states.mps(dense_to_mps(Psi0.reshape(-1), L, d, r, cut))
