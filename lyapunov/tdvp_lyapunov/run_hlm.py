"""
Candidate hydrodynamic Lyapunov modes from a stored run.

Builds the exact local-temperature template a_k at every wavevector
(hlm.template_vectors), measures where its weight sits in the Lyapunov
spectrum, and plots the best candidate mode the near-zero cluster of
covariant vectors can make at the first few wavevectors.

The statistic is the *enrichment*: the share of template weight landing in
a band of m vectors, divided by m/k, the share a uniform spread would give.
1 is chance. Wavevector-resolved, it is a direct test of the hydrodynamic
picture: long-wavelength templates should be enriched in the near-zero
band, short-wavelength ones should not.

    python lyapunov/tdvp_lyapunov/run_hlm.py C:/Users/charl/lyapunov_runs/L16_D4_beta1.h5 \
        --m 120 --kmax 3 [--copy phys] [--show]

writes figures/<run>_hlm_{candidates,enrichment}.png.
"""

import argparse
import os
import sys
import time as clock

import h5py
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from lyapunov.tdvp_lyapunov.benettin import ginelli_backward, load_frame
from lyapunov.tdvp_lyapunov.plots import (
    load_run, exponents_from, FIGS, BLUE, ORANGE, AQUA, MUTED,
)
from lyapunov.tdvp_lyapunov.analysis import cosine_transform
from lyapunov.tdvp_lyapunov.frame import complexify
from lyapunov.tdvp_lyapunov.hlm import (
    template_vectors, template_spectral_weights, band_enrichment,
    subspace_projector_fraction, band_indices, profile_map,
)

BANDS = [('near-zero', 'zero', BLUE), ('mid-spectrum', 'mid', ORANGE), ('top', 'top', AQUA)]


def describe_mode(frame, vec, which='phys'):
    prof = profile_map(frame, vec[:, None], which)[:, 0]
    q, dct = cosine_transform(prof)
    w = np.array([frame.site_weights(complexify(vec)).get(s, 0.0) for s in frame.sites])
    w = w / w.sum()
    return {'profile': prof, 'q': q, 'dct': dct, 'site_weights': w,
            'extent': float(1.0 / np.sum(w ** 2))}


def plot_candidates(cands, bonds, label=''):
    rows = len(cands)
    fig, axes = plt.subplots(rows, 3, figsize=(11, 2.6 * rows), squeeze=False)
    for r, c in enumerate(cands):
        a0, a1, a2 = axes[r]
        p, q, k = c['profile'], c['q'], c['k']
        x = np.arange(len(p))
        basis = np.cos(np.pi * k * (x + 0.5) / len(p))
        basis *= np.dot(p, basis) / np.dot(basis, basis)
        a0.plot(bonds + 0.5, p, 'o-', ms=4, color=ORANGE, label='mode profile')
        a0.plot(bonds + 0.5, basis, '--', color=MUTED, lw=1, label=f'cos(q{k} j)')
        a0.axhline(0, color=MUTED, lw=0.8)
        a0.set_xlabel('bond')
        a0.set_ylabel(r'$\delta\langle h_j\rangle$')
        a0.legend(frameon=False, fontsize=8)
        a0.set_title(f'k={k}  q={q[k]:.2f}   enrichment {c["enrichment"]:.2f}x', loc='left')

        a1.bar(q, np.abs(c['dct']), width=q[1] * 0.8, color=AQUA)
        a1.set_xlabel('q')
        a1.set_ylabel('|DCT|')
        a1.set_xticks([0, np.pi / 2, np.pi])
        a1.set_xticklabels(['0', r'$\pi/2$', r'$\pi$'])
        a1.set_title(f'purity {c["purity"]:.2f},  '
                     rf'$\lambda_{{\rm eff}}$ {c["lambda_eff"]:+.4f}', loc='left')

        a2.bar(np.arange(len(c['site_weights'])), c['site_weights'], color=BLUE, width=0.8)
        a2.set_xlabel('site')
        a2.set_ylabel(r'$|X^n|^2$ (norm.)')
        a2.set_title(f'extent {c["extent"]:.1f} of {len(c["site_weights"])} sites', loc='left')
    fig.suptitle(label, x=0.01, ha='left')
    fig.tight_layout()
    return fig


def plot_enrichment(q, enrich, err, in_half, cum, lam, m, n_blocks, label=''):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for (name, key, color) in BANDS:
        ax[0].errorbar(q, enrich[key], yerr=err[key], fmt='o-', ms=4, lw=1.5,
                       capsize=2, color=color, label=f'{name} (m={m})')
    ax[0].axhline(1.0, color=MUTED, ls='--', lw=1)
    ax[0].annotate('chance', (q[-1], 1.0), xytext=(0, 4), textcoords='offset points',
                   ha='right', color=MUTED, fontsize=8)
    ax[0].set_xlabel('q of template')
    ax[0].set_ylabel('enrichment')
    ax[0].set_xticks([0, np.pi / 2, np.pi])
    ax[0].set_xticklabels(['0', r'$\pi/2$', r'$\pi$'])
    ax[0].set_title(f'template vs band ({n_blocks} blocks)', loc='left')
    ax[0].legend(frameon=False, fontsize=8)

    ax[1].plot(q, in_half, 'o-', ms=4, color=BLUE)
    ax[1].set_xlabel('q of template')
    ax[1].set_ylabel('fraction in computed half')
    ax[1].set_ylim(0, 1)
    ax[1].set_xticks([0, np.pi / 2, np.pi])
    ax[1].set_xticklabels(['0', r'$\pi/2$', r'$\pi$'])
    ax[1].set_title('weight in non-negative half', loc='left')

    lam_sorted = np.sort(lam)[::-1]
    for color, (k, c) in zip([BLUE, ORANGE, AQUA], cum.items()):
        ax[2].plot(lam_sorted, c, color=color, label=f'k={k}, q={q[k]:.2f}')
    ax[2].plot(lam_sorted, np.linspace(0, 1, len(lam)), '--', color=MUTED, lw=1,
               label='uniform')
    ax[2].set_xlabel(r'$\lambda$ (descending)')
    ax[2].set_ylabel('cumulative weight')
    ax[2].invert_xaxis()
    ax[2].set_title('cumulative template weight', loc='left')
    ax[2].legend(frameon=False, fontsize=8)
    fig.suptitle(label, x=0.01, ha='left')
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--m', type=int, default=120, help='vectors per band')
    p.add_argument('--kmax', type=int, default=3, help='candidate modes to plot')
    p.add_argument('--copy', default='phys', choices=['phys', 'aux'])
    p.add_argument('--discard', type=int, default=20)
    p.add_argument('--show', action='store_true')
    a = p.parse_args()

    name = os.path.splitext(os.path.basename(a.path))[0]
    run = load_run(a.path)
    lam = exponents_from(run, a.discard)
    target = 0.6 * run['done']
    block = min((j for j in run['Q_blocks'] if j < run['done']), key=lambda j: abs(j - target))
    print(f"{name}: n={run['n']}, {run['done']} blocks, vectors at block {block}")

    with h5py.File(a.path, 'r') as f:
        Q = f['Q'][str(block)][()]
        frame = load_frame(f['frame'][str(block)])

    n_bonds = len(frame.sites) - 1
    q, templates = template_vectors(frame, n_bonds - 1, a.copy)
    bands = {key: band_indices(lam, key, m=a.m) for _, key, _ in BANDS}
    for nm, key, _ in BANDS:
        print(f"  {nm:12s} band: lambda in "
              f"[{lam[bands[key]].min():+.4f}, {lam[bands[key]].max():+.4f}]")

    # Enrichment at every stored block: the spread over blocks (i.e. over
    # points on the trajectory) is the error bar on a single-block number.
    blocks = [j for j in run['Q_blocks'] if j < run['done']]
    order = np.argsort(lam)[::-1]
    per_block = {key: [] for key in bands}
    half_pb, cum = [], {}
    for j in blocks:
        with h5py.File(a.path, 'r') as f:
            Qj = f['Q'][str(j)][()]
            frame_j = load_frame(f['frame'][str(j)])
        _, templ_j = template_vectors(frame_j, n_bonds - 1, a.copy)
        e = {key: [] for key in bands}
        hh = []
        for k, ak in enumerate(templ_j):
            w, frac = template_spectral_weights(Qj, ak)
            hh.append(frac)
            for key, idx in bands.items():
                e[key].append(band_enrichment(w, idx))
            if j == block and 1 <= k <= a.kmax:
                cum[k] = np.cumsum(w[order]) / w.sum()
        for key in bands:
            per_block[key].append(e[key])
        half_pb.append(hh)
    enrich = {key: np.mean(per_block[key], axis=0) for key in bands}
    err = {key: np.std(per_block[key], axis=0) / np.sqrt(len(blocks)) for key in bands}
    in_half = np.mean(half_pb, axis=0)
    print(f"  enrichment averaged over {len(blocks)} stored blocks "
          f"(error = s.e.m. over blocks)")

    print(f"\n  {'k':>2s} {'q':>6s} {'in half':>8s} " +
          "".join(f"{nm:>16s}" for nm, _, _ in BANDS))
    for k in range(len(templates)):
        print(f"  {k:2d} {q[k]:6.2f} {in_half[k]:8.3f} " +
              "".join(f"{enrich[key][k]:11.2f}+-{err[key][k]:.2f}" for _, key, _ in BANDS))

    # candidate modes: built in the covariant span of the near-zero cluster
    t0 = clock.time()
    out = ginelli_backward(a.path, [block], discard_last=run['done'] - 1 - block)
    V = out['clv'][block]
    print(f"\n  Ginelli backward pass: {clock.time() - t0:.0f} s")
    cands = []
    for k in range(1, a.kmax + 1):
        _, y, proj = subspace_projector_fraction(V[:, bands['zero']], templates[k])
        d = describe_mode(frame, proj / la.norm(proj), a.copy)
        d.update({'k': k, 'enrichment': enrich['zero'][k],
                  'purity': d['dct'][k] ** 2 / np.sum(d['dct'] ** 2),
                  'lambda_eff': float(lam[bands['zero']] @ (y ** 2) / np.sum(y ** 2))})
        cands.append(d)

    os.makedirs(FIGS, exist_ok=True)
    lab = f'{name}  m={a.m} per band, {a.copy} copy, block {block}'
    fig = plot_enrichment(q, enrich, err, np.array(in_half), cum, lam, a.m,
                          len(blocks), lab)
    fig.savefig(os.path.join(FIGS, f'{name}_hlm_enrichment.png'), dpi=150)
    fig = plot_candidates(cands, np.array(frame.sites[:-1]),
                          lab + '  — best near-zero mode per wavevector')
    fig.savefig(os.path.join(FIGS, f'{name}_hlm_candidates.png'), dpi=150)
    print(f"  written figures/{name}_hlm_{{enrichment,candidates}}.png")
    if a.show:
        plt.show()


if __name__ == '__main__':
    main()
