"""
Validation rung 4: the physical/ancilla swap symmetry of the tangent space.

The thermofield double is built from rho^(1/2), a symmetric matrix, and in
the vectorization psi <-> M the on-site swap |a b> -> |b a> is transposition,
S|M>> = |M^T>>. Real-time evolution under H_sym = H x I + I x H sends
M -> e^(-iHt) M e^(-iHt) (the tilted Ising H is real symmetric), which is
still symmetric, so S|psi(t)> = |psi(t)> along the whole trajectory even
though M(t) is complex. S is a product of on-site unitaries, so it maps the
bond-dimension-D manifold to itself; with the point fixed, dS is an
involution of the tangent space, splitting it into +1 and -1 sectors that
the tangent flow cannot mix.

This script builds that involution and checks it.

  (a) the point is swap-symmetric: |<psi|S|psi>| = 1.

  (b) Sigma, the (2n, 2n) real matrix of dS in the frame's coordinates:
      column j holds the coordinates of S applied to the frame's j-th real
      basis vector. Sigma^2 = 1; Sigma orthogonal (equivalently, S maps the
      tangent space into itself with no weight leaking onto psi or off the
      manifold -- this is the check that would fail if the point were not
      swap-symmetric); Sigma symmetric, so (1 +- Sigma)/2 are orthogonal
      projectors; and [Sigma, J] = 0, i.e. S is complex-linear rather than
      antiunitary, which is what leaves the symplectic pairing intact
      inside each sector.

  (c) the construction itself: with Y = Sigma X the swap of X, X + Y is
      swap-even and X - Y is swap-odd. Verified by applying the swap
      operator to the tangent MPS in Hilbert space -- not by reusing
      Sigma -- and taking the Rayleigh quotient lambda = <v, S v>/<v, v>.
      The error reported is |1 - lambda| for the even combination and
      |1 + lambda| for the odd one. A Rayleigh quotient alone does not
      prove an eigenvector (it is blind to a component of S v orthogonal
      to v), so the full residual |S v - lambda v|/|v| is reported next
      to it.

Reading the numbers. Every coordinate diagnostic here -- |1 - lambda|, the
residual, |Sigma^2 - 1| -- is *quadratic* in the symmetry breaking, because
Sigma is the compression P S P and what leaks out of the tangent space is
dropped twice. The linear measure is

    leak_j = sqrt(1 - |Sigma e_j|^2),

the part of S Phi_j that is not in the frame's span, and it tracks the
swap-odd amplitude of the state, a_odd = |(1 - S)psi / 2|. Expect
|1 - lambda| ~ leak^2. Two things drive leak, and neither is roundoff:

  1. Rank deficiency. Where a Schmidt value is numerically zero the point
     is a *singular* point of the rank-D manifold: A_R and V_L past the
     rank are an arbitrary isometry completion, so the parameterized
     tangent space is a property of the representation rather than of the
     state, and S -- which makes a different arbitrary choice -- maps it
     elsewhere. At beta = 0.1 the thermofield double is close to rank 1,
     so the imaginary-time build *ends* in this region (L=8, D=8:
     s_min = 2.4e-11) and only real-time evolution walks the state off it
     (s_min = 2.9e-2 by t = 1). Taking the frame at t < 1 gives leak ~ 0.2
     and the check fails outright; converging the build harder makes it
     worse there, because a better-converged TFD is closer to rank 1.
     This is why --transient is a *time* and not a step count.

  2. Chaotic amplification. The exact flow commutes with S, but the odd
     sector carries positive exponents, so whatever asymmetry exists grows
     exponentially: measured +0.649 at L=8, D=8, beta=0.1, against
     lambda_max = 0.615 for the same parameters. The symmetry therefore
     has a lifetime of ~1/lambda and is gone by the end of a production
     run -- any use of the sector split has to enforce it (re-symmetrize
     psi and project the vectors each block) rather than assume it.

So there is a window: long enough that the bond dimension has filled,
short enough that the drift has not taken over. t ~ 1-2 at these sizes.

Run from the repo root:
    python lyapunov/tdvp_lyapunov/validate_tangent_swap.py
    python lyapunov/tdvp_lyapunov/validate_tangent_swap.py --L 8 --D 8
    python lyapunov/tdvp_lyapunov/validate_tangent_swap.py --transient 8
"""

import argparse
import os
import sys
import time as clock

import numpy as np
import scipy.linalg as la

sys.path.insert(0, os.getcwd())

import qtensor.operators as ops
import qtensor.states as states
import qtensor.thermofield as tf

from lyapunov.tdvp_lyapunov.frame import (
    Frame, project_to_frame, tangent_mps, overlap_with_point,
    realify, complexify,
)
from lyapunov.tdvp_lyapunov.stepper import tdvp_step, lanczos_method
from lyapunov.relaxation.run_relaxation_scan import build_uniform_thermofield


# The doubled physical index is i = 2 * a_phys + b_anc (analysis._copy_kron
# builds one-copy operators as kron(A, I) / kron(I, A), and the infinite-T
# seed is [1, 0, 0, 1] = |00> + |11>). Swapping the copies is therefore the
# permutation 2a + b -> 2b + a, an involution, so the swap matrix is
# symmetric and applying it to a tensor is a reindex of the physical leg.
SWAP_PERM = [0, 2, 1, 3]


def swap(phi):
    """S|phi>: exchange the two copies on every site. Takes and returns
    {site: tensor}; an mps is accepted and unwrapped."""
    tensors = phi.tensors if hasattr(phi, 'tensors') else phi
    return {site: T[SWAP_PERM, :, :].copy() for site, T in tensors.items()}


def odd_amplitude(psi):
    """
    |(1 - S) psi / 2|: the swap-odd amplitude of the state, the quantity
    everything else inherits. Linear in the asymmetry, unlike <psi|S|psi>,
    which is 1 - 2 a_odd^2 and so hides it.
    """
    norm = abs(states.overlap(psi, psi))
    ov = states.overlap(psi, states.mps(swap(psi))) / norm
    return float(np.sqrt(max(0.0, (1.0 - ov.real) / 2.0)))


def swap_coords(frame, y):
    """
    dS applied to the real coordinate vector y, the honest way: build the
    tangent MPS, apply the swap in Hilbert space, project back into the
    same frame. Never combines tensors from two canonicalization passes --
    only the Hilbert-space overlap crosses.
    """
    phi = tangent_mps(frame, complexify(y), include_point=False)
    return realify(project_to_frame(swap(phi), frame))


def swap_matrix(frame):
    """
    The (2n, 2n) real matrix Sigma of dS in the frame's coordinates.
    Column j is dS applied to the j-th real basis vector. S is
    complex-linear, so the columns for X and iX come from one projection:
    n projections in total, each O(N d D^3), exactly as frame_change.
    """
    n = frame.n
    Sig = np.zeros((2 * n, 2 * n))
    for j in range(n):
        X = np.zeros(n, dtype=complex)
        X[j] = 1.0
        c = project_to_frame(swap(tangent_mps(frame, X, include_point=False)),
                             frame)
        Sig[:, j] = realify(c)
        Sig[:, n + j] = realify(1j * c)
    return Sig


def complex_structure(n):
    """J: multiplication by i, in real (Re, Im) coordinates."""
    J = np.zeros((2 * n, 2 * n))
    J[n:, :n] = np.eye(n)
    J[:n, n:] = -np.eye(n)
    return J


def build_point(L, D, beta, imag_steps, t_transient, dt):
    """
    The thermofield double at beta, carried off the fixed point by real-time
    TDVP under H_sym for a *time* t_transient -- the same construction
    run_lyapunov.py uses before it starts the tangent flow.

    The transient is specified as a time and not as a step count on purpose.
    What the state needs is real time: it is born on the rank-deficient
    boundary and only evolution fills its bond dimension. Halving dt at a
    fixed step count halves that time, which looks like a more accurate
    integration and is in fact the thing that breaks the swap check.
    """
    psi, energy = build_uniform_thermofield(L, D, beta, imag_steps)
    H_sym = tf.thermofield_hamiltonian(ops.tilted_ising(N=L), asym=False)
    method = lanczos_method()
    for _ in range(int(round(t_transient / dt))):
        psi = tdvp_step(psi, H_sym, dt, method)
    return psi, energy


def main(L=6, D=4, beta=0.1, imag_steps=40, t_transient=1.0, dt=0.05, ncols=6,
         tol=1e-5):
    np.random.seed(0)
    rng = np.random.default_rng(0)

    t0 = clock.time()
    psi, energy = build_point(L, D, beta, imag_steps, t_transient, dt)
    frame = Frame(psi, D)
    n = frame.n
    s_all = np.concatenate(list(frame.schmidt_values().values()))
    smin, n_tiny = s_all.min(), int((s_all < 1e-4).sum())
    print(f"L={L} D={D} beta={beta}: n={n} (2n={2 * n}), "
          f"t={t_transient:g} ({int(round(t_transient / dt))} steps of dt={dt:g}), "
          f"built in {clock.time() - t0:.1f}s")
    print(f"    s_min = {smin:.3e} over all bonds, {n_tiny} Schmidt values below 1e-4")
    if n_tiny:
        print("    WARNING: the point is close to rank-deficient, i.e. close to "
              "a singular\n             point of the manifold, where the tangent "
              "space stops being a\n             property of the state. Give it "
              "more real time with --transient.")

    # (a) -------------------------------------------------------------- point
    ov = overlap_with_point(swap(frame.state()), frame)
    a_odd = odd_amplitude(frame.state())
    print(f"\n(a) <psi|S|psi> = {ov.real:+.12f}{ov.imag:+.3e}j, "
          f"|<psi|S|psi>| = {abs(ov):.12f}  (want 1)")
    print(f"    a_odd = |(1 - S)psi/2| = {a_odd:.2e}  -- the swap-odd amplitude "
          f"of the state,\n    which grows like exp(0.65 t) once the bond "
          f"dimension has filled; everything\n    below inherits it.")

    # (b) -------------------------------------------------------------- Sigma
    t0 = clock.time()
    Sig = swap_matrix(frame)
    t_sig = clock.time() - t0
    I2n, J = np.eye(2 * n), complex_structure(n)
    # per-entry (RMS) rather than Frobenius: a Frobenius norm grows like
    # sqrt(2n) at fixed per-vector accuracy, so a fixed tolerance on it would
    # penalize larger D for nothing. These are now comparable with (c).
    rms = lambda M: la.norm(M) / np.sqrt(2 * n)
    err_inv = rms(Sig @ Sig - I2n)
    err_orth = rms(Sig.T @ Sig - I2n)
    err_symm = rms(Sig - Sig.T)
    err_J = rms(Sig @ J - J @ Sig)
    print(f"\n(b) Sigma built in {t_sig:.1f}s   (norms are per entry, |.|_F / sqrt(2n))")
    print(f"    |Sigma^2 - 1|       = {err_inv:.2e}   (involution)")
    print(f"    |Sigma^T Sigma - 1| = {err_orth:.2e}   (tangent space is S-invariant)")
    print(f"    |Sigma - Sigma^T|   = {err_symm:.2e}   (projectors (1 +- Sigma)/2 orthogonal)")
    print(f"    |[Sigma, J]|        = {err_J:.2e}   (S complex-linear, pairing survives)")

    ev = la.eigvalsh(Sig)
    n_plus = int(np.sum(ev > 0))
    print(f"    eigenvalues: max ||lambda| - 1| = {np.abs(np.abs(ev) - 1).max():.2e}; "
          f"sectors {n_plus} even / {2 * n - n_plus} odd (trace {Sig.trace():+.6f})")

    leaks = np.sqrt(np.maximum(0.0, 1.0 - np.sum(Sig ** 2, axis=0)))
    print(f"    leak = sqrt(1 - |Sigma e_j|^2): max {leaks.max():.2e}, "
          f"median {np.median(leaks):.2e}")
    print(f"      the linear measure of the breaking, and the one to quote: "
          f"the errors\n      above and in (c) are all O(leak^2) = "
          f"{leaks.max() ** 2:.1e}. Compare a_odd = {a_odd:.2e}.")

    # (c) ------------------------------------------------- X + Y and X - Y
    print("\n(c) X = k-th frame basis vector, Y = Sigma X its swap. S applied")
    print("    in Hilbert space to Phi(X +- Y), Rayleigh quotient lambda:")
    print(f"\n    {'k':>6}  {'|X+Y|':>8}  {'lambda(+)':>13}  {'|1-lambda|':>10}  "
          f"{'resid':>8}   {'|X-Y|':>8}  {'lambda(-)':>13}  {'|1+lambda|':>10}  {'resid':>8}")

    cols = [str(k) for k in np.linspace(0, 2 * n - 1, ncols, dtype=int)]
    worst = 0.0
    for label in cols + ['random']:
        if label == 'random':
            y = rng.normal(size=2 * n)
            y /= la.norm(y)
            Y = swap_coords(frame, y)
        else:
            y = np.zeros(2 * n)
            y[int(label)] = 1.0
            Y = Sig[:, int(label)]

        row = f"    {label:>6}"
        for sign, target in ((+1, +1.0), (-1, -1.0)):
            v = y + sign * Y
            nv = la.norm(v)
            if nv < min(0.5, max(1e-6, 10 * leaks.max())) * la.norm(y):
                # X was already sector-pure, so this combination is the null
                # vector: nothing to test, and its residual would be the
                # breaking level divided by itself. The threshold has to scale
                # with leak, since that is how close to null these get.
                row += f"  {nv:>8.1e}  {'(pure)':>13}  {'--':>10}  {'--':>8}"
                continue
            Sv = swap_coords(frame, v)
            lam = float(v @ Sv / (v @ v))
            res = la.norm(Sv - lam * v) / nv
            worst = max(worst, abs(lam - target), res)
            row += (f"  {nv:>8.4f}  {lam:>+13.9f}  {abs(target - lam):>10.2e}"
                    f"  {res:>8.1e}")
        print(row)

    print(f"\n    worst error over all of (c): {worst:.2e}")

    if leaks.max() > 1e-6:
        # leak >> a_odd means the frame is the problem, not the state: only a
        # representation-dependent tangent space can lose that much weight
        # while the state itself is still symmetric to within a_odd. Measured
        # ratios: 2-30 with a sound frame (the weak bond directions are merely
        # ill-conditioned), ~1e5 at a singular point.
        if n_tiny or leaks.max() > 1e3 * max(a_odd, 1e-16):
            print(f"\n    Dominant cause: the frame, not the state -- leak "
                  f"{leaks.max():.1e} >> a_odd {a_odd:.1e},\n    at s_min = "
                  f"{smin:.1e}. The point is at or next to a singular point of "
                  f"the\n    manifold, where V_L and A_R past the rank are an "
                  f"arbitrary completion\n    and the tangent space is not well "
                  f"defined. Raise --transient above ~1\n    so that real time "
                  f"fills the bond dimension first.")
        else:
            print(f"\n    Dominant cause: drift -- leak {leaks.max():.1e} is of "
                  f"order a_odd {a_odd:.1e},\n    so the frame is sound and this "
                  f"is the state's own odd component, grown\n    along the "
                  f"trajectory to t = {t_transient:g} at exp(0.65 t). Lower "
                  f"--transient, or\n    enforce the symmetry rather than "
                  f"assuming it.")

    assert abs(abs(ov) - 1) < tol, "the point is not swap-symmetric"
    assert max(err_inv, err_orth, err_symm, err_J) < tol, "Sigma is not a clean involution"
    assert worst < tol, "X +- Y are not eigenvectors of the swap"
    print("\nAll checks passed.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--L', type=int, default=6)
    p.add_argument('--D', type=int, default=4)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--imag-steps', type=int, default=40)
    p.add_argument('--transient', type=float, default=1.0,
                   help='real time (not steps) under H_sym before the frame is '
                        'taken; below ~1 the state is still rank-deficient, '
                        'above ~2 the drift dominates')
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--ncols', type=int, default=6, help='basis columns to demonstrate on')
    p.add_argument('--tol', type=float, default=1e-5,
                   help='pass/fail bar on the per-vector errors; the floor is '
                        'set by a_odd, so loosen this rather than the code when '
                        'testing a long transient or a large D')
    a = p.parse_args()
    main(a.L, a.D, a.beta, a.imag_steps, a.transient, a.dt, a.ncols, a.tol)
