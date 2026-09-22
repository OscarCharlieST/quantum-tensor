"""
Validation rung 5: does the swap sector structure survive a Lyapunov run?

Rung 4 (`validate_tangent_swap.py`) established the involution at a point:
Sigma is exact, the tangent space splits 330/276 at L=8 D=4, and the split
is preserved by the exact flow. This script asks the three questions that
decide whether the split can be *used* -- see SWAP_SYMMETRY.md, staging
items 2 and 3.

    inject   Where does the asymmetry come from? a_odd = |(1-S)psi/2| along
             the imaginary-time build and the real-time transient. The build
             is clean at every D (a_odd ~ 1e-8, the integration error); all
             of it is injected while the state sits in the rank-deficient
             region s_min << 1, where the isometry completion past the rank
             is arbitrary and S maps the representation somewhere else. How
             long the state lingers there grows steeply with D, because the
             beta = 0.1 thermofield double is close to rank 1.

    run      Post-hoc on a stored run: a_odd along the stored frames, and
             the sector content of the stored Q (and optionally the CLVs).
             Two separate findings, and they are worth keeping apart:

             - the *state* keeps its symmetry only at small D. Measured over
               the 2026-09-21 scan: L8 D4 ends at a_odd = 9e-4, L8 D8 at
               0.70 (saturated -- psi is orthogonal to S psi), L8 D12 is
               already saturated before the first block.
             - the *vectors* carry no sector label at all, at any D, and
               this is not a symmetry problem. Q is seeded at random and its
               columns converge to Lyapunov directions at the rate of the
               local spectral gap; the mean gap here is 2e-3, so the
               convergence time is ~500 against a run length of 12. Nothing
               in the bulk is converged, so nothing has a sector. Labelling
               an existing run after the fact is therefore not on -- the
               sector has to be imposed at the start.

    seeded   The soft-enforcement experiment. Three sets of k vectors are
             carried along one trajectory: even-seeded and left alone,
             even-seeded and re-projected onto (1+Sigma)/2 every block, and
             odd-seeded and re-projected. Reports the contamination of the
             free set (how well the scheme preserves the split by itself),
             the size of the projection the enforced set needs (what
             enforcement costs), and the two sectors' leading exponents.

Run from the repo root:
    python lyapunov/tdvp_lyapunov/validate_swap_flow.py inject --D 12
    python lyapunov/tdvp_lyapunov/validate_swap_flow.py run --path <run.h5>
    python lyapunov/tdvp_lyapunov/validate_swap_flow.py seeded --blocks 200
"""

import argparse
import os
import sys
import time as clock

import h5py
import numpy as np
import scipy.linalg as la

sys.path.insert(0, os.getcwd())

import qtensor.operators as ops
import qtensor.thermofield as tf

from lyapunov.tdvp_lyapunov.frame import Frame
from lyapunov.tdvp_lyapunov.stepper import tdvp_step, lanczos_method
from lyapunov.tdvp_lyapunov.tangent_generator import (
    generator, parallel_transport, propagate,
)
from lyapunov.tdvp_lyapunov.benettin import positive_qr, load_frame, ginelli_backward
from lyapunov.tdvp_lyapunov.validate_tangent_swap import swap_matrix, odd_amplitude
from lyapunov.relaxation.run_relaxation_scan import build_uniform_thermofield


# ------------------------------------------------------------------ helpers

def sector_basis(Sig):
    """Orthonormal bases of the +1 and -1 eigenspaces of Sigma."""
    ev, U = la.eigh(Sig)
    return U[:, ev > 0], U[:, ev < 0]


def odd_weight(Sig, Q):
    """
    Per column, the fraction of the norm in the odd sector,
    |P_- q|^2 / |q|^2 with P_- = (1 - Sigma)/2. Zero for an even vector,
    one for an odd one, ~1/2 for a generic one.
    """
    return np.clip(0.5 * (1 - np.einsum('ij,ij->j', Q, Sig @ Q)
                          / np.einsum('ij,ij->j', Q, Q)), 0.0, 1.0)


def decomposability(Sig, Q):
    """
    Is span(Q) a sum of an even and an odd subspace? It is iff the
    eigenvalues mu of Q^T Sigma Q are all +-1 (Q orthonormal). Returns
    max|1 - |mu|| and the (n_+, n_-) split. A subspace can decompose
    cleanly while none of its columns is pure, which is why this is the
    diagnostic that matters for a restricted run.
    """
    mu = la.eigvalsh(Q.T @ (Sig @ Q))
    return float(np.abs(1 - np.abs(mu)).max()), int((mu > 0).sum()), int((mu < 0).sum())


def s_min_of(frame):
    return min(v.min() for v in frame.schmidt_values().values())


# ------------------------------------------------------------------- inject

def inject(L, D, beta, imag_steps, dt, steps):
    """a_odd and s_min along the build and the transient."""
    H = tf.thermofield_hamiltonian(ops.tilted_ising(N=L), asym=False)
    method = lanczos_method()
    np.random.seed(0)
    t0 = clock.time()
    psi, _ = build_uniform_thermofield(L, D, beta, imag_steps)
    print(f"L={L} D={D} beta={beta}, {imag_steps} imaginary steps "
          f"({clock.time() - t0:.0f}s), then dt={dt:g} to t={steps * dt:g}")
    print(f"\n{'t':>6}  {'s_min':>9}  {'a_odd':>9}")
    marks = {2, 5, 10, 20, 40, 80, 160, steps}
    print(f"{0.0:>6.2f}  {s_min_of(Frame(psi, D)):>9.1e}  {odd_amplitude(psi):>9.2e}")
    for j in range(1, steps + 1):
        psi = tdvp_step(psi, H, dt, method)
        if j in marks:
            print(f"{j * dt:>6.2f}  {s_min_of(Frame(psi, D)):>9.1e}  "
                  f"{odd_amplitude(psi):>9.2e}", flush=True)
    print("\nThe build is symmetric to the integration error. Everything after "
          "that is\ninjected while s_min << 1 and then amplified at the "
          "Lyapunov rate.")


# ------------------------------------------------------------------ posthoc

def posthoc(path, every, sigma_blocks, clv_block):
    with h5py.File(path, 'r') as f:
        t, exps = f['t'][()], f['exponents'][()]
        stored = sorted(int(x) for x in f['Q'])
        n, k = int(f.attrs['n']), int(f.attrs['k'])
        print(f"{os.path.basename(path)}: n={n}, k={k}, {len(t)} blocks, "
              f"t = {t[0]:.2f} .. {t[-1]:.2f}")
        gaps = -np.diff(np.sort(exps)[::-1])
        print(f"    median spectral gap {np.median(gaps):.2e} -> QR converges "
              f"a bulk column in ~{1 / np.median(gaps):.0f} time units, "
              f"against T = {t[-1] - t[0]:.1f}")

        print(f"\n{'block':>6} {'t':>7} {'s_min':>9} {'a_odd':>10}")
        for j in stored[::every]:
            fr = load_frame(f['frame'][str(j)])
            print(f"{j:>6} {t[j]:>7.2f} {s_min_of(fr):>9.2e} "
                  f"{odd_amplitude(fr.state()):>10.3e}", flush=True)

        for j in sigma_blocks:
            j = stored[-1] if j < 0 else j
            fr = load_frame(f['frame'][str(j)])
            Sig = swap_matrix(fr)
            leak = np.sqrt(np.maximum(0.0, 1 - np.sum(Sig ** 2, axis=0)))
            ev = la.eigvalsh(Sig)
            Q = f['Q'][str(j)][()]
            w = odd_weight(Sig, Q)
            print(f"\nblock {j} (t={t[j]:.2f}): leak max {leak.max():.1e}, "
                  f"sectors {int((ev > 0).sum())}/{int((ev < 0).sum())}, "
                  f"|1-|lambda||max {np.abs(np.abs(ev) - 1).max():.1e}")
            print(f"    Q columns: odd weight min/median/max "
                  f"{w.min():.3f}/{np.median(w):.3f}/{w.max():.3f}; "
                  f"{int((np.minimum(w, 1 - w) < 0.01).sum())} of {len(w)} "
                  f"pure to 1%")
            print("    span(q_1..q_m) decomposable?  "
                  "(max|1-|mu|| over the m x m block; 0 = clean split)")
            for m in (8, 32, 128, min(n, k), k):
                if m > k:
                    continue
                err, np_, nm = decomposability(Sig, Q[:, :m])
                print(f"       m={m:>4}  {err:.2e}   {np_}/{nm}")

        if clv_block is not None:
            out = ginelli_backward(path, want_blocks=[clv_block])
            V = out['clv'][clv_block]
            Sig = swap_matrix(out['frame'][clv_block])
            w = odd_weight(Sig, V)
            err, np_, nm = decomposability(Sig, la.qr(V, mode='economic')[0])
            print(f"\nCLVs at block {clv_block}: odd weight min/median/max "
                  f"{w.min():.3f}/{np.median(w):.3f}/{w.max():.3f}; "
                  f"{int((np.minimum(w, 1 - w) < 0.01).sum())} of {len(w)} "
                  f"pure to 1%; span decomposability {err:.2e}")


# ------------------------------------------------------------------- seeded

def seeded(L, D, beta, imag_steps, dt, transient, n_blocks, k, report_every):
    np.random.seed(0)
    H = tf.thermofield_hamiltonian(ops.tilted_ising(N=L), asym=False)
    method = lanczos_method()
    t0 = clock.time()
    psi, _ = build_uniform_thermofield(L, D, beta, imag_steps)
    for _ in range(transient):
        psi = tdvp_step(psi, H, dt, method)
    frame = Frame(psi, D)
    Sig = swap_matrix(frame)
    even, odd = sector_basis(Sig)
    n = frame.n
    print(f"L={L} D={D} beta={beta}: n={n}, sectors {even.shape[1]} even / "
          f"{odd.shape[1]} odd, k={k} per set, {n_blocks} blocks of dt={dt:g} "
          f"from t={transient * dt:g} ({clock.time() - t0:.0f}s to here)")
    print(f"    a_odd at the start = {odd_amplitude(frame.state()):.2e}")

    rng = np.random.default_rng(0)
    seed = lambda B: la.qr(B @ rng.normal(size=(B.shape[1], k)), mode='economic')[0]
    Q = {'even_free': seed(even)}
    Q['even_enf'] = Q['even_free'].copy()
    Q['odd_enf'] = seed(odd)
    sign = {'even_free': None, 'even_enf': +1, 'odd_enf': -1}
    logd = {key: np.zeros((n_blocks, k)) for key in Q}

    print(f"\n{'blk':>4} {'t':>6} {'a_odd':>8} {'leak':>8} | "
          f"{'free w max':>10} {'free w med':>10} | {'enf w max':>9} "
          f"{'enf cut':>8} | {'lam0 free':>9} {'lam0 even':>9} {'lam0 odd':>9}")
    A = generator(frame, H)
    t_wall = clock.time()
    for j in range(n_blocks):
        psi_next = tdvp_step(psi, H, dt, method)
        frame_next = Frame(psi_next, D)
        A_next = generator(frame_next, H)
        T = parallel_transport(frame, frame_next)
        for key in Q:
            Q[key] = propagate(Q[key], A, T, A_next, dt)
        A, psi, frame = A_next, psi_next, frame_next

        Sig = swap_matrix(frame)
        row = {}
        for key, s in sign.items():
            M = Q[key]
            w = odd_weight(Sig, M)
            row[key] = w if s in (None, +1) else 1 - w
            if s is not None:
                M = 0.5 * (M + s * (Sig @ M))
                row[key + '_cut'] = la.norm(M - Q[key]) / la.norm(Q[key])
            M, R = positive_qr(M)
            Q[key], logd[key][j] = M, np.log(np.diag(R))
        if j % report_every == 0 or j == n_blocks - 1:
            lam = {key: logd[key][:j + 1].sum(0) / ((j + 1) * dt) for key in Q}
            print(f"{j:>4} {(transient + j + 1) * dt:>6.2f} "
                  f"{odd_amplitude(frame.state()):>8.1e} "
                  f"{np.sqrt(max(0.0, 1 - np.sum(Sig ** 2, axis=0).min())):>8.1e} | "
                  f"{row['even_free'].max():>10.1e} "
                  f"{np.median(row['even_free']):>10.1e} | "
                  f"{row['even_enf'].max():>9.1e} {row['even_enf_cut']:>8.1e} | "
                  f"{lam['even_free'][0]:>+9.4f} {lam['even_enf'][0]:>+9.4f} "
                  f"{lam['odd_enf'][0]:>+9.4f}  [{clock.time() - t_wall:.0f}s]",
                  flush=True)

    print(f"\ntop exponents after {n_blocks} blocks "
          f"(t = {transient * dt:g} .. {(transient + n_blocks) * dt:g}):")
    print(f"{'i':>3} {'even_free':>10} {'even_enf':>10} {'odd_enf':>10}")
    lam = {key: logd[key].sum(0) / (n_blocks * dt) for key in Q}
    for i in range(min(k, 12)):
        print(f"{i:>3} {lam['even_free'][i]:>+10.4f} {lam['even_enf'][i]:>+10.4f} "
              f"{lam['odd_enf'][i]:>+10.4f}")
    d = np.abs(lam['even_free'] - lam['even_enf']).max()
    print(f"\nlargest |even_free - even_enf| over the {k} exponents: {d:.2e}")


# ---------------------------------------------------------------------- cli

def parse():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = p.add_subparsers(dest='mode', required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--L', type=int, default=8)
    common.add_argument('--D', type=int, default=4)
    common.add_argument('--beta', type=float, default=0.1)
    common.add_argument('--imag-steps', type=int, default=240)
    common.add_argument('--dt', type=float, default=0.05)

    a = sub.add_parser('inject', parents=[common],
                       help='where the asymmetry enters')
    a.add_argument('--steps', type=int, default=160)

    b = sub.add_parser('run', help='post-hoc sector content of a stored run')
    b.add_argument('--path', required=True)
    b.add_argument('--every', type=int, default=5,
                   help='report a_odd at every n-th stored frame')
    b.add_argument('--sigma-blocks', type=int, nargs='*', default=[0, -1],
                   help='blocks at which to build Sigma (-1 = the last stored)')
    b.add_argument('--clv-block', type=int, default=None,
                   help='also decompose the CLVs at this block (Ginelli pass)')

    c = sub.add_parser('seeded', parents=[common],
                       help='soft enforcement: seed a sector and watch it')
    c.add_argument('--transient', type=int, default=160, help='TDVP steps first')
    c.add_argument('--blocks', type=int, default=200)
    c.add_argument('--k', type=int, default=32, help='vectors per set')
    c.add_argument('--report-every', type=int, default=10)
    return p.parse_args()


if __name__ == '__main__':
    a = parse()
    if a.mode == 'inject':
        inject(a.L, a.D, a.beta, a.imag_steps, a.dt, a.steps)
    elif a.mode == 'run':
        posthoc(a.path, a.every, a.sigma_blocks, a.clv_block)
    else:
        seeded(a.L, a.D, a.beta, a.imag_steps, a.dt, a.transient, a.blocks,
               a.k, a.report_every)
