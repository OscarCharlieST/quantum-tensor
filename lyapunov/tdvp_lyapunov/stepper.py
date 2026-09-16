"""
One TDVP time step as a map on the manifold, and the action of its tangent
map on a set of frame coordinates by finite differences (Route A).

The step is finiteTDVP's right-then-left sweep pair, unchanged, minus the
progress bar and history bookkeeping that make `tdvp` awkward to call one
step at a time from many perturbed clones.
"""

import copy
import os

import numpy as np
from ncon import ncon

import qtensor.simulation.finiteTDVP as sim
import qtensor.simulation.updatemethod as methods

from lyapunov.tdvp_lyapunov.frame import (
    Frame, project_to_frame, overlap_with_point, retract,
    realify, complexify,
)


def tdvp_step(psi, H, dt, method):
    """
    Phi_dt(psi): one full TDVP step (right sweep, left sweep) under the MPO H.
    Does not mutate psi; returns a new mps in right-canonical form with its
    orthogonality centre at the first site.
    """
    state = copy.deepcopy(psi)
    state.right_orthogonal()
    sites = sorted(state.sites)
    R_con = sim.right_mpo_contractions(state, H)
    L_con = {sites[0] - 1: ncon((np.eye(1), H.l), ((-1, -2), (-3,)))}
    state, L_con, _ = sim.tdvp_sweep_r(state, H, dt, L_con, R_con, method)
    state, _, _ = sim.tdvp_sweep_l(state, H, dt, L_con, R_con, method)
    # tdvp_sweep_l leaves the centre at the first site; that is the
    # right-canonical form.
    state.form, state.c_site = 'right', sites[0]
    return state


def phase_fixed_projection(phi, frame):
    """
    project_to_frame(phi) with phi's global phase aligned to the frame's
    point first, so that only the genuinely tangent part of phi - psi
    survives. The phase drift of a perturbed clone is O(eps) and lives in
    the i*psi direction the frame excludes anyway; removing it keeps the
    forward-difference error at its nominal order.
    """
    ov = overlap_with_point(phi, frame)
    phase = ov / abs(ov)
    return project_to_frame(phi, frame) / phase


def _tangent_columns(frame, frame_next, Y, H, dt, method, eps, max_bond_dim,
                     scheme):
    """Serial finite-difference images of the columns of Y (see tangent_map)."""
    k = Y.shape[1]
    out = np.zeros((2 * frame_next.n, k))
    if scheme == 'forward':
        base = phase_fixed_projection(frame_next.state(), frame_next)
    for j in range(k):
        X = complexify(Y[:, j])
        plus = tdvp_step(retract(frame, X, eps, max_bond_dim), H, dt, method)
        p_plus = phase_fixed_projection(plus, frame_next)
        if scheme == 'central':
            minus = tdvp_step(retract(frame, X, -eps, max_bond_dim), H, dt, method)
            p_minus = phase_fixed_projection(minus, frame_next)
            out[:, j] = realify((p_plus - p_minus) / (2 * eps))
        else:
            out[:, j] = realify((p_plus - base) / eps)
    return out


def _prepare_workers():
    """
    Worker processes start from a fresh interpreter, so they need the repo
    root on PYTHONPATH to unpickle references into qtensor/lyapunov.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    paths = os.environ.get('PYTHONPATH', '').split(os.pathsep)
    if root not in paths:
        os.environ['PYTHONPATH'] = os.pathsep.join([root] + [p for p in paths if p])


def tangent_map(frame, frame_next, Y, H, dt, method, eps, max_bond_dim,
                scheme='central', n_jobs=1):
    """
    Apply the tangent map of one TDVP step, [D Phi_dt]_{frame -> frame_next},
    to the columns of the real (2n, k) matrix Y, by finite differences
    through tdvp_step.

    frame_next must be the frame at Phi_dt(psi) for the *unperturbed* psi
    (frame.state()). Returns the real (2n', k) matrix of images, where n'
    is frame_next.n (equal to n whenever the bond dimensions are unchanged).

    scheme : 'central' (2k clone steps, O(eps^2) error) or
             'forward' (k clone steps, O(eps) error).
    n_jobs : worker processes. Columns are independent, so they are split
        into one contiguous chunk per worker (finer chunking was measured to
        cost more in overhead than it gains), each worker pinned to one BLAS
        thread (the tensors are far too small for threaded BLAS to help, and
        oversubscription would hurt). The pool is reused across calls, so
        its start-up cost is paid once. Output is bit-identical to serial.
        Measured on the Core Ultra 7 155H at L=8, D=4, k=303: 4.5x at 8
        workers, 5.6x at 16.
    """
    if n_jobs == 1 or Y.shape[1] < 2:
        return _tangent_columns(frame, frame_next, Y, H, dt, method, eps,
                                max_bond_dim, scheme)
    from joblib import Parallel, delayed, parallel_config
    _prepare_workers()
    k = Y.shape[1]
    bounds = np.linspace(0, k, min(k, n_jobs) + 1).astype(int)
    chunks = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    with parallel_config(backend='loky', inner_max_num_threads=1):
        parts = Parallel(n_jobs=n_jobs)(
            delayed(_tangent_columns)(frame, frame_next, Y[:, a:b], H, dt,
                                      method, eps, max_bond_dim, scheme)
            for a, b in chunks)
    return np.concatenate(parts, axis=1)


def exact_method():
    return methods.exact_method()


def lanczos_method(max_iters=16, epsilon=1e-8):
    """
    Lanczos for Lyapunov runs. The early exit at `epsilon` makes the step
    map discontinuous, but only by the neglected Krylov term, which is
    below epsilon; it cannot be removed altogether, because once the
    Krylov space is exhausted (edge sites at small D) the next vector is
    pure roundoff and dividing by its norm corrupts the whole step. So
    epsilon is set well below the default 1e-5 but well above roundoff.
    """
    return methods.lanczos_method(epsilon=epsilon, max_iters=max_iters)
