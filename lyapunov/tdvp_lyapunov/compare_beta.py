"""
Temperature comparison for full-spectrum (k = 2n) runs: does the machinery
survive as beta -> 0, where the thermofield double becomes nearly rank-1,
and does the hydrodynamic signal survive with it?

    python lyapunov/tdvp_lyapunov/compare_beta.py [--L 8 --D 8] [--m 150]

Prints a diagnostics table (conditioning, pairing, spectrum) and writes
  figures/compare_beta_spectra.png    sorted spectra, and the near-zero end
  figures/compare_beta_enrichment.png template enrichment vs q, per beta
  figures/compare_beta_modes.png      the k=1 candidate mode at each beta
"""

import argparse
import os
import sys

import h5py
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from lyapunov.tdvp_lyapunov.benettin import load_frame, ginelli_backward
from lyapunov.tdvp_lyapunov.plots import (
    load_run, exponents_from, FIGS, BLUE, ORANGE, AQUA, MUTED,
)
from lyapunov.tdvp_lyapunov.hlm import (
    template_vectors, template_spectral_weights, band_enrichment, band_indices,
    subspace_projector_fraction,
)
from lyapunov.tdvp_lyapunov.run_hlm import describe_mode

RUNS = 'C:/Users/charl/lyapunov_runs'
COLORS = [BLUE, ORANGE, AQUA, '#4a3aa7']


def collect(path, m, discard, kmax, copy='phys'):
    """Everything the figures need for one run."""
    run = load_run(path)
    lam = exponents_from(run, discard)
    blocks = [j for j in run['Q_blocks'] if j < run['done']]
    block = min(blocks, key=lambda j: abs(j - 0.6 * run['done']))
    with h5py.File(path, 'r') as f:
        frame = load_frame(f['frame'][str(block)])
        s_min = f['s_min'][()][:run['done']]
    n_bonds = len(frame.sites) - 1
    q, _ = template_vectors(frame, n_bonds - 1, copy)

    bands = {key: band_indices(lam, key, m=m) for key in ('zero+', 'zero-', 'top', 'bottom')}
    per_block = {key: [] for key in bands}
    for j in blocks:
        with h5py.File(path, 'r') as f:
            Qj = f['Q'][str(j)][()]
            frame_j = load_frame(f['frame'][str(j)])
        _, templ = template_vectors(frame_j, n_bonds - 1, copy)
        rows = {key: [] for key in bands}
        for ak in templ:
            w, _ = template_spectral_weights(Qj, ak)
            for key, idx in bands.items():
                rows[key].append(band_enrichment(w, idx))
        for key in bands:
            per_block[key].append(rows[key])
    enrich = {key: np.mean(per_block[key], axis=0) for key in bands}
    err = {key: np.std(per_block[key], axis=0) / np.sqrt(len(blocks)) for key in bands}

    # candidate mode from the contracting near-zero band, which gave the
    # cleaner modes at beta = 1
    V = ginelli_backward(path, [block], discard_last=run['done'] - 1 - block)['clv'][block]
    _, templates = template_vectors(frame, kmax, copy)
    _, y, proj = subspace_projector_fraction(V[:, bands['zero-']], templates[1])
    mode = describe_mode(frame, proj / la.norm(proj), copy)
    mode['lambda_eff'] = float(lam[bands['zero-']] @ (y ** 2) / np.sum(y ** 2))
    mode['purity'] = mode['dct'][1] ** 2 / np.sum(mode['dct'] ** 2)

    srt = np.sort(lam)[::-1]
    return {'run': run, 'lam': lam, 'q': q, 'enrich': enrich, 'err': err,
            'mode': mode, 'block': block, 'n_blocks': len(blocks),
            'frame': frame, 's_min': s_min,
            'pairing': np.abs(srt + srt[::-1]).max(), 'sum': srt.sum(),
            'schmidt': min(s.min() for s in frame.schmidt_values().values())}


def plot_spectra(data, labels):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for color, lab, d in zip(COLORS, labels, data):
        lam = np.sort(d['lam'])[::-1]
        x = (np.arange(len(lam)) + 0.5) / len(lam)
        ax[0].plot(x, lam, color=color, label=lab)
        ax[1].plot(x, lam, color=color)
    for a in ax:
        a.axhline(0, color=MUTED, lw=0.8)
        a.set_xlabel('i / 2n')
    ax[0].set_ylabel(r'$\lambda_i$')
    ax[0].set_title('full spectrum', loc='left')
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set_xlim(0.4, 0.6)
    ax[1].set_ylim(-0.05, 0.05)
    ax[1].set_title('near-zero end', loc='left')
    fig.tight_layout()
    return fig


def plot_enrichment(data, labels):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for color, lab, d in zip(COLORS, labels, data):
        for a, key, style in [(ax[0], 'bottom', '-'), (ax[1], 'top', '-')]:
            a.errorbar(d['q'], d['enrich'][key], yerr=d['err'][key], fmt='o' + style,
                       ms=4, lw=1.5, capsize=2, color=color, label=lab)
    for a, title in [(ax[0], 'contracting band (bottom)'), (ax[1], 'expanding band (top)')]:
        a.axhline(1.0, color=MUTED, ls='--', lw=1)
        a.set_xlabel('q of template')
        a.set_xticks([0, np.pi / 2, np.pi])
        a.set_xticklabels(['0', r'$\pi/2$', r'$\pi$'])
        a.set_title(title, loc='left')
    ax[0].set_ylabel('enrichment')
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_modes(data, labels):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    for color, lab, d in zip(COLORS, labels, data):
        m = d['mode']
        bonds = np.array(d['frame'].sites[:-1]) + 0.5
        p = m['profile'] / la.norm(m['profile'])
        ax[0].plot(bonds, p, 'o-', ms=4, color=color,
                   label=f"{lab}  purity {m['purity']:.2f}")
        ax[1].plot(m['q'], np.abs(m['dct']) / la.norm(m['dct']), 'o-', ms=4, color=color)
        ax[2].plot(np.arange(len(m['site_weights'])), m['site_weights'], 'o-', ms=4,
                   color=color, label=f"extent {m['extent']:.1f}")
    x = np.arange(len(data[0]['mode']['profile']))
    ref = np.cos(np.pi * (x + 0.5) / len(x))
    ax[0].plot(np.array(data[0]['frame'].sites[:-1]) + 0.5, ref / la.norm(ref), '--',
               color=MUTED, lw=1, label='cos(q1 j)')
    ax[0].axhline(0, color=MUTED, lw=0.8)
    ax[0].set_xlabel('bond')
    ax[0].set_ylabel(r'$\delta\langle h_j\rangle$ (normalized)')
    ax[0].set_title('k=1 candidate, contracting side', loc='left')
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].set_xlabel('q')
    ax[1].set_ylabel('|DCT| (normalized)')
    ax[1].set_xticks([0, np.pi / 2, np.pi])
    ax[1].set_xticklabels(['0', r'$\pi/2$', r'$\pi$'])
    ax[1].set_title('cosine transform', loc='left')
    ax[2].set_xlabel('site')
    ax[2].set_ylabel(r'$|X^n|^2$ (norm.)')
    ax[2].set_title('site weight', loc='left')
    ax[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--L', type=int, default=8)
    p.add_argument('--D', type=int, default=8)
    p.add_argument('--betas', nargs='*', default=['1', '0.1', '0.01'])
    p.add_argument('--m', type=int, default=150)
    p.add_argument('--discard', type=int, default=40)
    p.add_argument('--kmax', type=int, default=3)
    p.add_argument('--copy', default='phys', choices=['phys', 'aux'])
    a = p.parse_args()

    paths = [f'{RUNS}/L{a.L}_D{a.D}_beta{b}_k2n.h5' for b in a.betas]
    labels = [rf'$\beta$ = {b}' for b in a.betas]
    data, kept = [], []
    for b, lab, path in zip(a.betas, labels, paths):
        if not os.path.exists(path):
            print(f"  missing: {path}")
            continue
        data.append(collect(path, a.m, a.discard, a.kmax, a.copy))
        kept.append(lab)
        print(f"  loaded beta={b}")

    print(f"\n{'beta':>6s} {'n':>6s} {'lam_max':>8s} {'sum lam':>9s} {'pairing':>8s} "
          f"{'s_min':>9s} {'enrich q1':>10s} {'purity':>7s} {'lam_eff':>9s} {'extent':>7s}")
    for b, d in zip(a.betas, data):
        print(f"{b:>6s} {d['run']['n']:6d} {np.max(d['lam']):8.4f} {d['sum']:+9.4f} "
              f"{d['pairing']:8.4f} {d['schmidt']:9.2e} "
              f"{d['enrich']['bottom'][1]:10.2f} {d['mode']['purity']:7.2f} "
              f"{d['mode']['lambda_eff']:+9.4f} {d['mode']['extent']:7.1f}")

    os.makedirs(FIGS, exist_ok=True)
    for name, fig in [('spectra', plot_spectra(data, kept)),
                      ('enrichment', plot_enrichment(data, kept)),
                      ('modes', plot_modes(data, kept))]:
        fig.suptitle(f'L = {a.L}, D = {a.D}, full spectrum', x=0.01, ha='left')
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f'compare_beta_{name}.png'), dpi=150)
        print(f"  written figures/compare_beta_{name}.png")


if __name__ == '__main__':
    main()
