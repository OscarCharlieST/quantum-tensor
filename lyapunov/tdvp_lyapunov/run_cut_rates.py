"""
Driver for Tier 0 of the one-cut study: rate-spectrum bounds and the cost
of the rank restriction along TDVP trajectories, scanned over m.

The start is the beta thermofield double evolved *exactly* under H_sym to
t_scramble (dense, so free at L <= 8), then truncated to rank r = N - m at
the centre cut. Every other bond stays at its staircase maximum.

Run from the repo root, e.g.

    python lyapunov/tdvp_lyapunov/run_cut_rates.py --L 8 --ms 1 2 4 8 16 32 64 \
        --dt 0.05 --steps 200 --t-scramble 20

Output: runs/cut_rates_L{L}_m{m}_beta{beta}{tag}.h5 per m, and
figures/cut_rates_L{L}_beta{beta}{tag}_{series,summary}.png.
"""

import argparse
import os
import sys
import time as clock

import h5py
import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

import qtensor.operators as ops
import qtensor.thermofield as tf

from lyapunov.tdvp_lyapunov.cut_matrix import (
    mpo_cut_terms, apply_H,
)
from lyapunov.tdvp_lyapunov.cut_rates import (
    dense_thermofield, cut_entropy, run_rates, ExactThermofield,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
FIGS = os.path.join(HERE, 'figures')


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--L', type=int, default=8)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--t-scramble', type=float, default=20.0,
                   help='exact H_sym evolution before truncating')
    p.add_argument('--ms', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--tag', default='')
    p.add_argument('--out-dir', default=RUNS)
    p.add_argument('--time-only', action='store_true')
    return p.parse_args()


def scrambled_start(L, beta, t_scramble, terms, exact, cut):
    Psi = exact.evolve(dense_thermofield(L, beta, terms, cut), t_scramble)
    return Psi / la.norm(Psi)


def save(path, res, attrs):
    with h5py.File(path, 'w') as f:
        f.attrs.update(attrs)
        for k, v in res.items():
            f.create_dataset(k, data=v)


def plot(results, N, path_stem):
    ms = sorted(results)
    cmap = plt.get_cmap('viridis')
    col = {m: cmap(i / max(1, len(ms) - 1)) for i, m in enumerate(ms)}

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for m in ms:
        r = results[m]
        t = r['t']
        ax[0, 0].semilogy(t, r['s'][:, -1], color=col[m], label=f'm={m}')
        ax[0, 1].semilogy(t, r['nu'][:, 0], color=col[m])
        ax[0, 2].semilogy(t, r['lam_bound'], color=col[m])
        ax[1, 0].semilogy(t, r['hks_bound'], color=col[m])
        ax[1, 1].semilogy(t, np.maximum(1 - r['fidelity'], 1e-16), color=col[m])
        ax[1, 2].plot(t, r['S_tdvp'] - r['S_exact'], color=col[m])
    ax[0, 0].set_title('smallest kept Schmidt value $s_r$')
    ax[0, 1].set_title(r'$\nu_1 = \|\tilde N\|_2$')
    ax[0, 2].set_title(r'$\lambda_{\max}$ bound $\nu_1/s_r$')
    ax[1, 0].set_title(r'$h_{KS}$ bound $2\|\tilde N\|_*\,\mathrm{tr}\,s^{-1}$')
    ax[1, 1].set_title('1 - fidelity with exact')
    ax[1, 2].set_title('cut entropy, TDVP - exact')
    ax[0, 0].legend(fontsize=8)
    for a in ax.flat:
        a.set_xlabel('t')
    fig.tight_layout()
    fig.savefig(path_stem + '_series.png', dpi=120)
    plt.close(fig)

    mean = lambda k: np.array([np.mean(results[m][k][1:]) for m in ms])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].loglog(ms, mean('lam_bound'), 'o-', label=r'$\langle\nu_1/s_r\rangle$')
    ax[0].loglog(ms, [np.mean(1 / results[m]['s'][1:, -1]) for m in ms], 's--',
                 label=r'$\langle 1/s_r\rangle$')
    ax[0].loglog(ms, [np.mean(results[m]['nu'][1:, 0]) for m in ms], '^--',
                 label=r'$\langle\nu_1\rangle$')
    ax[0].set_xlabel('m')
    ax[0].legend()
    ax[0].set_title(r'$\lambda_{\max}$ bound and its factors')
    ax[1].loglog(ms, mean('hks_bound'), 'o-', label=r'$h_{KS}$ bound')
    ax[1].loglog(ms, mean('hks_bound') / (2 * np.array(ms) * (N - np.array(ms))), 's--',
                 label='per non-unitary direction')
    ax[1].set_xlabel('m')
    ax[1].legend()
    ax[1].set_title(r'$h_{KS}$ bound; $2mr$ positive rates')
    for m in ms:
        s = results[m]['s'][-1]
        ax[2].semilogy(np.arange(1, s.size + 1), s, color=col[m], label=f'm={m}')
    ax[2].set_xlabel('i')
    ax[2].set_title('kept Schmidt spectrum at the last step')
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_stem + '_summary.png', dpi=120)
    plt.close(fig)


def main():
    a = parse()
    L, cut = a.L, a.L // 2 - 1
    H = tf.thermofield_hamiltonian(ops.tilted_ising(N=L), asym=False)
    terms = mpo_cut_terms(H, cut)
    exact = ExactThermofield(ops.tilted_ising(N=L), L, cut)
    N = terms[0].shape[1]

    t0 = clock.time()
    Psi0 = scrambled_start(L, a.beta, a.t_scramble, terms, exact, cut)
    s0 = la.svdvals(Psi0)
    print(f"L={L} beta={a.beta}: exact start at t={a.t_scramble} in {clock.time() - t0:.0f} s; "
          f"S_cut = {cut_entropy(s0):.4f} (max {np.log(N):.4f}), s_max = {s0[0]:.3e}, "
          f"s_min = {s0[-1]:.3e}; E = {np.real(np.vdot(Psi0, apply_H(terms, Psi0))):.6f}",
          flush=True)

    if a.time_only:
        m = min(a.ms)
        t0 = clock.time()
        run_rates(Psi0, N - m, terms, H, exact, L, cut, a.dt, 2, verbose=False)
        per = (clock.time() - t0) / 3
        print(f"~{per:.1f} s per step -> ~{per * a.steps * len(a.ms) / 60:.0f} min "
              f"for {len(a.ms)} values of m x {a.steps} steps")
        return

    os.makedirs(a.out_dir, exist_ok=True)
    tag = f"_{a.tag}" if a.tag else ''
    results = {}
    for m in a.ms:
        print(f"m = {m} (r = {N - m})", flush=True)
        res = run_rates(Psi0, N - m, terms, H, exact, L, cut, a.dt, a.steps)
        results[m] = res
        save(os.path.join(a.out_dir, f"cut_rates_L{L}_m{m}_beta{a.beta:g}{tag}.h5"), res,
             {'L': L, 'm': m, 'r': N - m, 'beta': a.beta, 'dt': a.dt,
              't_scramble': a.t_scramble, 'steps': a.steps})
        print(f"  <lam bound> = {np.mean(res['lam_bound'][1:]):.3f}, "
              f"<hks bound> = {np.mean(res['hks_bound'][1:]):.1f}, "
              f"final 1-F = {1 - res['fidelity'][-1]:.2e}, "
              f"discarded at start {res['discarded0']:.2e}", flush=True)
    os.makedirs(FIGS, exist_ok=True)
    plot(results, N, os.path.join(FIGS, f"cut_rates_L{L}_beta{a.beta:g}{tag}"))


if __name__ == '__main__':
    main()
