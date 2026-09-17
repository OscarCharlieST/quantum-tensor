"""
One TDVP time step as a map on the manifold: the trajectory the Lyapunov
spectrum is computed along.

This is finiteTDVP's right-then-left sweep pair, unchanged, minus the
progress bar and history bookkeeping that make `tdvp` awkward to call one
step at a time.
"""

import copy

import numpy as np
from ncon import ncon

import qtensor.simulation.finiteTDVP as sim
import qtensor.simulation.updatemethod as methods


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
