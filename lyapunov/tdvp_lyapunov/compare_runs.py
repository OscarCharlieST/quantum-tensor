"""
Cross-run comparison figures: sorted half spectra against i/2n (the
extensivity collapse), for an L scan at fixed D, a D scan at fixed L, and
the dt/route check. All exponents averaged over the same time window.

    python lyapunov/tdvp_lyapunov/compare_runs.py [--t-min 11]

writes figures/compare_{L_scan,D_scan,dt_route}.png.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from lyapunov.tdvp_lyapunov.plots import load_run, FIGS, MUTED, BLUE, ORANGE, AQUA

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = 'C:/Users/charl/lyapunov_runs'     # h5 files live outside OneDrive

SCANS = {
    'L_scan_beta0.1_k2n': ('beta = 0.1, D = 4, L scan', [
        ('L = 8',  f'{RUNS}/L8_D4_beta0.1_k2n.h5'),
        ('L = 12', f'{RUNS}/L12_D4_beta0.1_k2n.h5'),
        ('L = 16', f'{RUNS}/L16_D4_beta0.1_k2n.h5'),
    ]),
    'D_scan_beta0.1_k2n': ('beta = 0.1, L = 8, D scan', [
        ('D = 4',  f'{RUNS}/L8_D4_beta0.1_k2n.h5'),
        ('D = 8',  f'{RUNS}/L8_D8_beta0.1_k2n_ns.h5'),
        ('D = 12', f'{RUNS}/L8_D12_beta0.1_k2n.h5'),
    ]),
    'dt_route': ('L = 8, D = 4: time step and route', [
        ('Route B, dt = 0.05',  f'{RUNS}/L8_D4_beta1.h5'),
        ('Route A, dt = 0.025', f'{RUNS}/L8_D4_beta1_dt025.h5'),
    ]),
}


def windowed_exponents(path, t_min, t_max=np.inf):
    run = load_run(path)
    m = (run['t'] > t_min) & (run['t'] <= t_max)
    ld = run['log_diag'][m]
    lam = ld.sum(0) / (m.sum() * run['tau'] * run['dt'])
    return np.sort(lam)[::-1], run['n'], run['t'][m][[0, -1]]


def plot_scan(title, entries, t_min, t_max=np.inf):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for color, (label, path) in zip([BLUE, ORANGE, AQUA], entries):
        lam, n, (ta, tb) = windowed_exponents(path, t_min, t_max)
        x = (np.arange(len(lam)) + 0.5) / (2 * n)
        ax[0].plot(x, lam, color=color, label=f'{label} (n={n}, t {ta:.0f}-{tb:.0f})')
        ax[1].plot(x, lam, color=color)
    for a in ax:
        a.axhline(0, color=MUTED, lw=0.8)
        a.set_xlabel('i / 2n')
    ax[0].set_ylabel(r'$\lambda_i$')
    ax[0].set_title('sorted non-negative half', loc='left')
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set_xlim(0.3, 0.5)
    ax[1].set_ylim(-0.01, 0.06)
    ax[1].set_title('near-zero end', loc='left')
    fig.suptitle(title, x=0.01, ha='left')
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--t-min', type=float, default=11.0)
    a = p.parse_args()
    os.makedirs(FIGS, exist_ok=True)
    for key, (title, entries) in SCANS.items():
        # the Route B L=8 run stops at t = 16.4, so the dt/route check uses
        # a common window ending there
        t_max = 16.4 if key == 'dt_route' else np.inf
        fig = plot_scan(title, entries, a.t_min, t_max)
        fig.savefig(os.path.join(FIGS, f'compare_{key}.png'), dpi=150)
        print(f'written figures/compare_{key}.png')
