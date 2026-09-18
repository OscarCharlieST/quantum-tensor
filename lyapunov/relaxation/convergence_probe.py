"""
How large must D be, and how many imaginary-time steps, before a run?

    python lyapunov/relaxation/convergence_probe.py --L 32 --D 12,16,20 \
        --steps 60,240

``||P H_asym psi*||`` is exactly zero for the true thermofield double, so it
measures how far the variational state sits from the fixed point. The point
of this module is that it needs only the tangent *vector*: it skips both the
``dim**2`` allocation of ``assemble_tangent_hamiltonian`` and the ``dim**3``
``eigh``, which are the entire cost of a real run. At L = 32, D = 16 that is
20 seconds against an estimated 4.7 hours.

So the parameters of an expensive run can be chosen by measuring rather than
by guessing. Measured 2026-09-18 (see README, "Cost and convergence"):

- **D = 16 is enough at every L tested**, because that is where the residual
  stops being limited by the manifold and starts being limited by the build.
  D = 20 and D = 24 are no better at any step count.
- **IMAG_STEPS = 60 is too low for D >= 16.** Raising it to 240 buys 25x at
  D = 16 and costs seconds. Past 240 there is only roundoff scatter.

A small residual is necessary for a converged spectrum, not sufficient --
it says psi* is close to the fixed point, not that every tangent mode is
resolved. Treat this as a cheap screen, not a proof.
"""
import sys
import time
import io
import contextlib

import numpy as np

sys.path.insert(0, '.')

import qtensor.operators as ops
import qtensor.thermofield as tf

import tangent_hamiltonian as tangent
import response as resp
from run_relaxation_scan import (build_uniform_thermofield, J, H_FIELD,
                                 G_FIELD, BETA, IMAG_STEPS)


def basis_layout(sites, A_L, V_L):
    """
    The (site, start, shape) map `assemble_tangent_hamiltonian` builds,
    without the (dim, dim) matrix that follows it.

    Kept in step with that function by construction: it is the same loop,
    and `observable_tangent_vector` only needs the layout.
    """
    layout, dim = [], 0
    for site in sites:
        n_null = V_L[site].shape[1]
        if n_null == 0:
            continue
        layout.append((site, dim, (n_null, A_L[site].shape[2])))
        dim += n_null * A_L[site].shape[2]
    return layout, dim


def probe(L, D, steps=IMAG_STEPS, beta=BETA, quiet=True):
    """(tangent dimension, ||P H_asym psi*||, seconds) at these parameters."""
    t0 = time.time()
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf) if quiet else contextlib.nullcontext():
        psi, _ = build_uniform_thermofield(L, D, beta, steps)

    H_phys = ops.tilted_ising(J=J, h=H_FIELD, g=G_FIELD, N=L)
    H_asym = tf.thermofield_hamiltonian(H_phys, asym=True)
    sites = sorted(psi.sites)

    A_L, A_R, _, _ = tangent.canonicalize_and_build_environments(
        psi, H_asym, max_bond_dim=D
    )
    V_L = {n: tangent.build_null_space_tensor(A_L[n]) for n in sites}
    centres = tangent.build_centre_tensors(A_L, A_R)
    layout, dim = basis_layout(sites, A_L, V_L)

    residual = resp.observable_tangent_vector(
        H_asym, A_L, A_R, centres, V_L, layout, sites
    )
    return dim, float(np.linalg.norm(residual)), time.time() - t0


# Fitted on the six measured runs with dim > 2000; max relative residual
# 2.6%. The memory factor is 2 arrays of dim^2 complex128 times a measured
# 1.48 for LAPACK workspace and the assembly's own peak.
EIGH_COEFF, EIGH_POWER = 2.908e-09, 2.942
MEM_BYTES_PER_ELEMENT = 1.48 * 2 * 16


def estimate(dim):
    """(eigh seconds, peak GB) for a full run at this tangent dimension."""
    return (EIGH_COEFF * dim ** EIGH_POWER,
            MEM_BYTES_PER_ELEMENT * dim ** 2 / 1024 ** 3)


def _fmt(seconds):
    if seconds < 3600:
        return '%.0f min' % (seconds / 60)
    if seconds < 86400:
        return '%.1f h' % (seconds / 3600)
    return '%.1f days' % (seconds / 86400)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    def opt(name, default):
        return argv[argv.index(name) + 1] if name in argv else default

    L = int(opt('--L', '16'))
    d_values = [int(x) for x in opt('--D', '12,16,20').split(',')]
    step_values = [int(x) for x in opt('--steps', str(IMAG_STEPS)).split(',')]

    print('L = %d, beta = %s' % (L, BETA))
    print('%4s %8s %10s %8s | %s'
          % ('D', 'dim', 'eigh', 'mem', '  '.join(
              '%14s' % ('steps=%d' % s) for s in step_values)), flush=True)
    for D in d_values:
        cells, dim = [], None
        for steps in step_values:
            dim, residual, elapsed = probe(L, D, steps)
            cells.append('%9.3e(%3.0fs)' % (residual, elapsed))
        t, gb = estimate(dim)
        print('%4d %8d %10s %6.1fGB | %s'
              % (D, dim, _fmt(t), gb, '  '.join('%14s' % c for c in cells)),
              flush=True)


if __name__ == '__main__':
    main()
