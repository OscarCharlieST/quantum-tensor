"""
One TDVP time step as a map on the manifold: the trajectory the Lyapunov
spectrum is computed along.

A thin wrapper on finiteTDVP.tdvp_step: a copy in front, so the step is a
pure function of psi, and the cached environments dropped.
"""

import copy

import qtensor.simulation.finiteTDVP as sim
import qtensor.simulation.updatemethod as methods


def tdvp_step(psi, H, dt, method):
    """
    Phi_dt(psi): one full TDVP step (right sweep, left sweep) under the MPO H.
    Does not mutate psi; returns a new mps in right-canonical form with its
    orthogonality centre at the first site.

    This is sim.tdvp_step with a copy in front and the environments thrown
    away. Rebuilding them costs ~1% of a step here, because the sweeps are
    dominated by the Lanczos applications, and a Lyapunov block spends its
    time in the generator anyway -- so the step stays a pure function of psi,
    which is what the frame bookkeeping needs.
    """
    state, _ = sim.tdvp_step(copy.deepcopy(psi), H, dt, method)
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
