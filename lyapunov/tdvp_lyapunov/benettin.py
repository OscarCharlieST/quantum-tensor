"""
Forward Benettin/QR iteration for the full Lyapunov spectrum of the TDVP
step map, storing what the Ginelli backward pass needs for covariant
Lyapunov vectors.

Conventions
-----------
Tangent vectors are real coordinate vectors of length 2n in the frame at
the current point (see frame.py). One "block" is tau TDVP steps followed by
a QR. Writing the tau-step tangent map as M_j, block j produces

    M_j Q_{j-1} = Q_j R_j,       diag(R_j) > 0

so that lambda_i = sum_j log R_j[i, i] / (T tau dt). Q_j lives in the
frame at psi_j; the frames themselves (A_L, A_R, V_L) are stored alongside
Q_j wherever Q_j is stored, because a stored coordinate vector means
nothing without the exact gauge it was written in.

Storage (h5)
------------
    R                (n_blocks, k, k)  every block
    log_diag_R       (n_blocks, k)
    t                (n_blocks,)       time after each block
    s_min            (n_blocks, N-1)   smallest Schmidt value per bond
    energy           (n_blocks,)       <H> along the trajectory
    Q/<j>            (2n, k)           at blocks j in store_Q_blocks
    frame/<j>/A_L/<site>, .../A_R/<site>, .../V_L/<site>
    state/<site>     final psi, for resuming
    Q_current        final Q
"""

import time as clock

import h5py
import numpy as np
import scipy.linalg as la

import qtensor.operators as ops

from lyapunov.tdvp_lyapunov.frame import Frame
from lyapunov.tdvp_lyapunov.stepper import tdvp_step, tangent_map, lanczos_method
from lyapunov.tdvp_lyapunov.tangent_generator import (
    generator, half_step_propagator, parallel_transport, propagate,
)


def orthonormal_random(dim, k, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = la.qr(rng.normal(size=(dim, k)), mode='economic')
    return Q


def positive_qr(Y):
    """Economic QR with the diagonal of R made positive."""
    Q, R = la.qr(Y, mode='economic')
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return Q * signs, signs[:, None] * R


def _store_frame(grp, frame):
    for name, tensors in [('A_L', frame.A_L), ('A_R', frame.A_R), ('V_L', frame.V_L)]:
        sub = grp.create_group(name)
        for site, T in tensors.items():
            sub.create_dataset(str(site), data=T)


def load_frame(grp):
    """Inverse of _store_frame: a Frame in exactly the stored gauge."""
    def read(name):
        return {int(s): grp[name][s][()] for s in grp[name]}
    return Frame.from_tensors(read('A_L'), read('A_R'), read('V_L'))


def benettin(psi0, H, dt, n_blocks, k, tau=1, method=None, eps=1e-5,
             max_bond_dim=np.inf, scheme='forward', transient_steps=0,
             store_path=None, store_Q_blocks=(), seed=0, Q0=None,
             verbose=True, route='B', n_jobs=1):
    """
    Run the forward Benettin iteration from psi0 under the MPO H.

    Parameters
    ----------
    k : number of tangent vectors carried (<= 2n). k = n covers the
        non-negative half of a +/- paired spectrum.
    tau : TDVP steps per QR.
    route : 'B' (exact generator + parallel transport, tangent_generator)
        or 'A' (finite differences through the TDVP step, stepper).
    eps, scheme : finite-difference step and scheme for route A.
    n_jobs : worker processes for route A's clone steps.
    transient_steps : steps of plain TDVP before the tangent vectors are
        switched on.
    store_path : h5 file for R, diagnostics, and Q/frames at
        store_Q_blocks (block indices).
    Q0 : optional initial (2n, k) matrix, e.g. from a previous run.

    Returns a dict with the exponents, per-block log R diagonals, times and
    diagnostics; the same data is in the h5 file if store_path was given.
    """
    if method is None:
        method = lanczos_method()
    psi = psi0
    for _ in range(transient_steps):
        psi = tdvp_step(psi, H, dt, method)
    frame = Frame(psi, max_bond_dim)
    n, N = frame.n, len(frame.sites)
    if verbose:
        print(f"tangent dimension n = {n} (real 2n = {2 * n}), k = {k}, "
              f"bond dims {frame.bond_dims()}")
    Q = orthonormal_random(2 * n, k, seed) if Q0 is None else Q0.copy()
    E = half_step_propagator(generator(frame, H), dt) if route == 'B' else None

    log_diag = np.zeros((n_blocks, k))
    times = np.zeros(n_blocks)
    s_min = np.zeros((n_blocks, N - 1))
    energy = np.zeros(n_blocks)
    R_store = None
    f = None
    if store_path is not None:
        f = h5py.File(store_path, 'w')
        f.attrs.update({'dt': dt, 'tau': tau, 'k': k, 'n': n, 'eps': eps,
                        'scheme': scheme, 'transient_steps': transient_steps,
                        'route': route})
        # R is upper triangular, so half zeros: lzf roughly halves it, cheaply
        R_store = f.create_dataset('R', (n_blocks, k, k), dtype='f8',
                                   chunks=(1, k, k), compression='lzf')
        f.create_group('Q')
        f.create_group('frame')
        # per-block summaries are written as they come, so a crash keeps them
        f.create_dataset('log_diag_R', (n_blocks, k), dtype='f8')
        f.create_dataset('t', (n_blocks,), dtype='f8')
        f.create_dataset('s_min', (n_blocks, N - 1), dtype='f8')
        f.create_dataset('energy', (n_blocks,), dtype='f8')
        f.attrs['blocks_done'] = 0

    t = transient_steps * dt
    t_wall = clock.time()
    for j in range(n_blocks):
        for _ in range(tau):
            psi_next = tdvp_step(psi, H, dt, method)
            frame_next = Frame(psi_next, max_bond_dim)
            if route == 'B':
                E_next = half_step_propagator(generator(frame_next, H), dt)
                Q = propagate(Q, E, parallel_transport(frame, frame_next), E_next)
                E = E_next
            else:
                Q = tangent_map(frame, frame_next, Q, H, dt, method, eps,
                                max_bond_dim, scheme, n_jobs=n_jobs)
            psi, frame = psi_next, frame_next
            t += dt
        Q, R = positive_qr(Q)
        log_diag[j] = np.log(np.diag(R))
        times[j] = t
        s_min[j] = [s.min() for s in frame.schmidt_values().values()]
        energy[j] = np.real(ops.expect(frame.state(), H))

        if f is not None:
            R_store[j] = R
            f['log_diag_R'][j] = log_diag[j]
            f['t'][j] = times[j]
            f['s_min'][j] = s_min[j]
            f['energy'][j] = energy[j]
            f.attrs['blocks_done'] = j + 1
            f.flush()
            if j in store_Q_blocks:
                f['Q'].create_dataset(str(j), data=Q)
                _store_frame(f['frame'].create_group(str(j)), frame)
        if verbose:
            lam = log_diag[:j + 1].sum(0) / ((j + 1) * tau * dt)
            print(f"block {j + 1}/{n_blocks}  t={t:.3f}  "
                  f"lambda_max={lam[0]:+.4f}  lambda_min={lam[-1]:+.4f}  "
                  f"s_min={s_min[j].min():.1e}  E={energy[j]:.6f}  "
                  f"[{clock.time() - t_wall:.0f} s]", flush=True)

    result = {
        'exponents': log_diag.sum(0) / (n_blocks * tau * dt),
        'log_diag_R': log_diag, 't': times, 's_min': s_min,
        'energy': energy, 'dt': dt, 'tau': tau, 'n': n, 'k': k,
        'Q': Q, 'frame': frame, 'state': psi,
    }
    if f is not None:
        f.create_dataset('exponents', data=result['exponents'])
        f.create_dataset('Q_current', data=Q)
        _store_frame(f.create_group('frame_final'), frame)
        for site in psi.sites:
            f.create_dataset(f'state/{site}', data=psi[site])
        f.close()
    return result


def running_exponents(log_diag, dt, tau):
    """lambda_i estimated from the first j blocks, for every j: (n_blocks, k)."""
    cum = np.cumsum(log_diag, axis=0)
    blocks = np.arange(1, log_diag.shape[0] + 1)[:, None]
    return cum / (blocks * tau * dt)


# ------------------------------------------------------------------ Ginelli

def ginelli_backward(store_path, want_blocks, discard_last=None, seed=0):
    """
    Covariant Lyapunov vectors from a stored forward run (Ginelli et al.,
    PRL 99, 130601 (2007)).

    The forward run gives M_j Q_{j-1} = Q_j R_j. Any covariant vector at
    block j is Q_j c for an upper-triangular coefficient matrix that
    transports backward as C_{j-1} = R_j^{-1} C_j (columns renormalized).
    Starting from a random upper-triangular C at the last block and
    iterating backward, the columns converge to the CLVs; the last
    `discard_last` blocks are the backward transient and are not trusted.

    Returns {j: (2n, k) array of CLVs in the frame stored at block j}
    for each j in want_blocks (which must all have a stored Q), plus the
    frames, in a dict.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(store_path, 'r') as f:
        R = f['R']
        n_blocks, k, _ = R.shape
        if 'blocks_done' in f.attrs:                              # partial runs
            n_blocks = int(f.attrs['blocks_done'])
        else:
            n_blocks = next((j for j in range(n_blocks) if not np.any(R[j])), n_blocks)
        if discard_last is None:
            discard_last = n_blocks // 4
        last = n_blocks - 1
        want = sorted(want_blocks)
        assert max(want) <= last - discard_last, (
            f"blocks {want} overlap the backward transient (last "
            f"{discard_last} of {n_blocks})")
        for j in want:
            assert str(j) in f['Q'], f"no Q stored at block {j}"

        C = np.triu(rng.normal(size=(k, k)))
        C /= la.norm(C, axis=0)
        out = {'clv': {}, 'frame': {}}
        for j in range(last, -1, -1):
            if j in want:
                out['clv'][j] = f['Q'][str(j)][()] @ C
                out['frame'][j] = load_frame(f['frame'][str(j)])
            if j == min(want):
                break
            # C at block j-1 from C at block j
            C = la.solve_triangular(R[j], C)
            C /= la.norm(C, axis=0)
    return out
