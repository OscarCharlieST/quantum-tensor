"""
Dispersion of the Lyapunov vectors: for every vector, the wavevector
content of its energy-density profile against its exponent.

Per vector: energy profile -> phase-free power spectrum over a fine q grid
(the projection onto {cos, sin} at each q, uniform removed, so a shifted
wave is not split between bins) -> centroid q_bar, width sd, peak. Then
plot q against lambda, coloured by sd, and ask whether lambda ~ q^2.

The width matters: a vector whose power is spread over the whole zone has
no meaningful wavevector, so the sd both colours the scatter and selects
the subset worth fitting.

    python lyapunov/tdvp_lyapunov/mode_dispersion.py \
        C:/Users/charl/lyapunov_runs/L16_D4_beta1_k2n.h5 [--blocks 3] [--copy phys]

writes figures/<run>_dispersion.png (centroid) and _dispersion_peak.png.
"""

import argparse
import os
import sys
import time as clock

import h5py
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from lyapunov.tdvp_lyapunov.benettin import load_frame
from lyapunov.tdvp_lyapunov.plots import (
    load_run, exponents_from, FIGS, BLUE, ORANGE, MUTED,
)
from lyapunov.tdvp_lyapunov.hlm import profile_map, phase_free_power, spectral_moments


def collect(path, n_blocks, copy, n_q, discard):
    """q_bar, sd, peak and lambda for every vector, pooled over blocks."""
    run = load_run(path)
    lam = exponents_from(run, discard)
    stored = [j for j in run['Q_blocks'] if j < run['done']]
    use = stored[len(stored) // 2:][:n_blocks] or stored[-n_blocks:]
    q_grid = np.linspace(0, np.pi, n_q)

    out = {k: [] for k in ('q_bar', 'sd', 'peak', 'lam', 'block')}
    for j in use:
        t0 = clock.time()
        with h5py.File(path, 'r') as f:
            Q = f['Q'][str(j)][()]
            frame = load_frame(f['frame'][str(j)])
        prof = profile_map(frame, Q, copy).T           # (k, n_bonds)
        power = phase_free_power(prof, q_grid)
        mean, sd, peak = spectral_moments(q_grid, power)
        out['q_bar'].append(mean)
        out['sd'].append(sd)
        out['peak'].append(peak)
        out['lam'].append(lam)
        out['block'].append(np.full(len(lam), j))
        print(f"    block {j}: {len(lam)} vectors in {clock.time() - t0:.0f} s")
    return {k: np.concatenate(v) for k, v in out.items()}, run, q_grid


def binned(x, y, n_bins=13, min_count=8):
    """Bin y against x. Returns centres, mean, sem, median|y|, q25|y|, q75|y|."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    edges = np.linspace(x.min(), x.max(), n_bins + 1)
    c, mu, se, md, lo, hi = [], [], [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= min_count:
            c.append(0.5 * (a + b))
            mu.append(y[m].mean())
            se.append(y[m].std() / np.sqrt(m.sum()))
            md.append(np.median(np.abs(y[m])))
            lo.append(np.percentile(np.abs(y[m]), 25))
            hi.append(np.percentile(np.abs(y[m]), 75))
    return [np.array(v) for v in (c, mu, se, md, lo, hi)]


def plot_dispersion(d, run, name, xkey='q_bar'):
    q, sd, lam = d[xkey], d['sd'], d['lam']
    peaked = sd < np.nanmedian(sd)
    xlabel = r'$\bar q$' if xkey == 'q_bar' else r'$q_{\rm peak}$'
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

    sc = ax[0].scatter(q, lam, c=sd, s=7, cmap='viridis_r', linewidths=0)
    ax[0].axhline(0, color=MUTED, lw=0.8)
    ax[0].set_xlabel(xlabel + ' of the vector profile')
    ax[0].set_ylabel(r'$\lambda$')
    ax[0].set_title(f'all {len(lam)} vectors', loc='left')
    fig.colorbar(sc, ax=ax[0], label='width of q distribution')

    # the signed relation: mean lambda against q
    c, mu, se, *_ = binned(q[peaked], lam[peaked])
    c2, mu2, se2, *_ = binned(q[~peaked], lam[~peaked])
    ax[1].axhline(0, color=MUTED, lw=0.8)
    ax[1].errorbar(c, mu, yerr=se, fmt='o-', ms=5, color=BLUE, capsize=2,
                   label='peaked half')
    ax[1].errorbar(c2, mu2, yerr=se2, fmt='s--', ms=4, color=ORANGE, capsize=2,
                   alpha=0.8, label='broad half')
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(r'mean $\lambda$')
    ax[1].set_title('signed exponent vs wavevector', loc='left')
    ax[1].legend(frameon=False, fontsize=8)

    # magnitudes, split by sign: is either branch diffusive?
    for sign, color, lab in [(-1, BLUE, r'$\lambda < 0$'), (+1, ORANGE, r'$\lambda > 0$')]:
        m = peaked & (np.sign(lam) == sign)
        if m.sum() < 20:
            continue
        c3, _, _, md, lo, hi = binned(q[m], lam[m])
        if len(c3) < 3:
            continue
        ax[2].errorbar(c3, md, yerr=[md - lo, hi - md], fmt='o-', ms=5,
                       color=color, capsize=2, label=lab)
        slope = np.polyfit(np.log(c3), np.log(md), 1)[0]
        mid = len(c3) // 2
        ax[2].plot(c3, md[mid] * (c3 / c3[mid]) ** slope, ':', color=color, lw=1,
                   label=f'slope {slope:+.2f}')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel(xlabel)
    ax[2].set_ylabel(r'median $|\lambda|$')
    ax[2].set_title('slope 2 would be diffusive', loc='left')
    ax[2].legend(frameon=False, fontsize=8)

    fig.suptitle(f'{name}  n={run["n"]}, k={run["k"]}', x=0.01, ha='left')
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--blocks', type=int, default=3, help='stored blocks to pool')
    p.add_argument('--copy', default='phys', choices=['phys', 'aux'])
    p.add_argument('--n-q', type=int, default=201, help='points on the q grid')
    p.add_argument('--discard', type=int, default=40)
    a = p.parse_args()

    name = os.path.splitext(os.path.basename(a.path))[0]
    print(f"{name}: computing profiles and phase-free power spectra")
    d, run, q_grid = collect(a.path, a.blocks, a.copy, a.n_q, a.discard)

    sd_med = np.nanmedian(d['sd'])
    peaked = d['sd'] < sd_med
    print(f"\n  vectors: {len(d['lam'])} pooled;  q_bar in "
          f"[{np.nanmin(d['q_bar']):.2f}, {np.nanmax(d['q_bar']):.2f}];  "
          f"sd median {sd_med:.2f} (zone is 0..{np.pi:.2f})")
    for key in ('q_bar', 'peak'):
        print(f"  binned by {key} (peaked half only):")
        for lo, hi in [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0),
                       (2.0, 2.5), (2.5, np.pi + 1e-9)]:
            m = peaked & (d[key] >= lo) & (d[key] < hi)
            if m.sum():
                print(f"    [{lo:.1f},{hi:.1f}): {m.sum():5d} vectors, "
                      f"median |lambda| = {np.median(np.abs(d['lam'][m])):.4f}, "
                      f"mean lambda = {np.mean(d['lam'][m]):+.4f} "
                      f"+- {d['lam'][m].std() / np.sqrt(m.sum()):.4f}")

    os.makedirs(FIGS, exist_ok=True)
    for xkey in ('q_bar', 'peak'):
        fig = plot_dispersion(d, run, name, xkey)
        tag = '' if xkey == 'q_bar' else '_peak'
        out = os.path.join(FIGS, f'{name}_dispersion{tag}.png')
        fig.savefig(out, dpi=150)
        print(f"  written {os.path.relpath(out)}")


if __name__ == '__main__':
    main()
