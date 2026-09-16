"""
Eyeball diagnostics for a Lyapunov run. Pure consumers of a benettin()
result / h5 file and of analysis.mode_report.

    python lyapunov/tdvp_lyapunov/plots.py runs/L8_D4_beta1.h5 [--modes 0 5 -1] [--show]

writes figures/<run>_{spectrum,convergence}.png and one
figures/<run>_mode{i}.png per requested Gram-Schmidt vector (from the last
stored Q). Negative mode indices count from the bottom of the stored half.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from lyapunov.tdvp_lyapunov.benettin import running_exponents, load_frame, ginelli_backward
from lyapunov.tdvp_lyapunov.analysis import mode_report, pairing_residual, q_weight_by_exponent

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'figures')

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e6e5e1'

matplotlib.rcParams.update({
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'axes.grid': True,
    'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'lines.linewidth': 1.5, 'font.size': 10,
})


# ----------------------------------------------------------------- spectrum

def plot_spectrum(exponents, n=None, dt=None, title='', ax=None):
    """
    Left: sorted lambda_i against the normalized index i/2n (collapse across
    L at fixed D is the extensive-chaos test). Right: histogram of the
    exponents, with the near-zero region where hydrodynamic modes would sit.
    """
    lam = np.sort(exponents)[::-1]
    k = len(lam)
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(k) / (2 * n if n else k)
    ax[0].plot(x, lam, '.', ms=3, color=BLUE)
    ax[0].axhline(0, color=MUTED, lw=0.8)
    ax[0].set_xlabel('i / 2n' if n else 'i / k')
    ax[0].set_ylabel(r'$\lambda_i$')
    ax[0].set_title(f'sorted spectrum, k={k}' + (f', n={n}' if n else ''), loc='left')

    bins = min(60, max(10, k // 8))
    ax[1].hist(lam, bins=bins, color=BLUE, edgecolor='white', linewidth=0.5)
    ax[1].axvline(0, color=MUTED, lw=0.8)
    ax[1].set_xlabel(r'$\lambda$')
    ax[1].set_ylabel('count')
    ax[1].set_title('histogram', loc='left')
    if title:
        ax[0].figure.suptitle(title, x=0.01, ha='left')
    ax[0].figure.tight_layout()
    return ax[0].figure


def plot_convergence(log_diag, dt, tau, which=(0, 1, -2, -1), t=None, ax=None):
    """Running estimate of a few exponents against time. A converged
    exponent is a plateau; the bottom of the half spectrum is the slowest."""
    run = running_exponents(log_diag, dt, tau)
    if t is None:
        t = np.arange(1, len(run) + 1) * tau * dt
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 3.4))
    k = run.shape[1]
    colors = [BLUE, AQUA, ORANGE, '#4a3aa7']
    for c, i in zip(colors, which):
        idx = i if i >= 0 else k + i
        ax.plot(t, run[:, idx], color=c, label=f'i={idx}')
        ax.annotate(f'i={idx}', (t[-1], run[-1, idx]), xytext=(4, 0),
                    textcoords='offset points', color=c, va='center', fontsize=8)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel('t')
    ax.set_ylabel(r'running $\lambda_i$')
    ax.set_title('convergence', loc='left')
    ax.figure.tight_layout()
    return ax.figure


def plot_pairing(exponents, ax=None):
    """lambda_i + lambda_{2n+1-i}: zero for a Hamiltonian flow (full k=2n only)."""
    r = pairing_residual(exponents)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    ax.plot(r, '.', ms=3, color=BLUE)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel('i')
    ax.set_ylabel(r'$\lambda_i + \lambda_{2n+1-i}$')
    ax.set_title('symplectic pairing', loc='left')
    ax.figure.tight_layout()
    return ax.figure


# --------------------------------------------------------------------- modes

def plot_mode(report, label='', exponent=None, axes=None):
    """
    One tangent vector, three views: site weights |X^n|^2; the
    physical-copy energy-density profile delta<h_j> over bonds; its cosine
    transform against q = pi k / N_bonds. A hydrodynamic mode is one whose
    DCT weight sits at small q.
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    a0, a1, a2 = axes
    a0.bar(report['sites'], report['site_weights'], color=BLUE, width=0.8)
    a0.set_xlabel('site')
    a0.set_ylabel(r'$|X^n|^2$')
    a0.set_title('site weight', loc='left')

    a1.plot(report['bonds'] + 0.5, report['energy_profile'], 'o-', ms=4, color=ORANGE)
    a1.axhline(0, color=MUTED, lw=0.8)
    a1.set_xlabel('bond')
    a1.set_ylabel(r'$\delta\langle h_j\rangle$')
    a1.set_title('energy-density profile', loc='left')

    a2.bar(report['q'], np.abs(report['dct']), width=np.pi / len(report['q']) * 0.8, color=AQUA)
    a2.set_xlabel('q')
    a2.set_ylabel('|DCT|')
    a2.set_title('cosine transform', loc='left')
    a2.set_xticks([0, np.pi / 2, np.pi])
    a2.set_xticklabels(['0', r'$\pi/2$', r'$\pi$'])

    head = label
    if exponent is not None:
        head += f'   $\\lambda$ = {exponent:+.4f}'
    axes[0].figure.suptitle(head, x=0.01, ha='left')
    axes[0].figure.tight_layout()
    return axes[0].figure


def plot_q_weight(edges, q, W, counts, label='', n_low=2, axes=None):
    """
    Left: mean normalized DCT power of the energy profile, exponent bin vs
    wavevector (single-hue sequential map). Right: the fraction of that
    power in the n_low longest wavelengths, per bin, with bin counts. A
    hydrodynamic band is a rise of the right-hand curve as lambda -> 0.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={'width_ratios': [1.3, 1]})
    a0, a1 = axes
    centers = 0.5 * (edges[:-1] + edges[1:])
    im = a0.imshow(W, aspect='auto', origin='lower', cmap='Blues', vmin=0,
                   extent=[q[0] - 0.5 * (q[1] - q[0]), q[-1] + 0.5 * (q[1] - q[0]), edges[0], edges[-1]])
    a0.set_xlabel('q')
    a0.set_ylabel(r'$\lambda$')
    a0.set_xticks([0, np.pi / 2, np.pi])
    a0.set_xticklabels(['0', r'$\pi/2$', r'$\pi$'])
    a0.set_title('energy-profile power by wavevector', loc='left')
    a0.grid(False)
    a0.figure.colorbar(im, ax=a0, label='mean fraction')

    low = W[:, :n_low].sum(1)
    ok = counts > 0
    a1.plot(centers[ok], low[ok], 'o-', ms=4, color=BLUE)
    a1.axhline(n_low / len(q), color=MUTED, lw=0.8, ls='--')
    a1.annotate('uniform', (centers[ok][-1], n_low / len(q)), xytext=(0, 4),
                textcoords='offset points', color=MUTED, ha='right', fontsize=8)
    for c, l, k in zip(centers[ok], low[ok], counts[ok]):
        a1.annotate(str(k), (c, l), xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=7, color=MUTED)
    a1.set_xlabel(r'$\lambda$')
    a1.set_ylabel(f'fraction in {n_low} longest wavelengths')
    a1.set_title('long-wavelength fraction (bin counts)', loc='left')
    if label:
        a0.figure.suptitle(label, x=0.01, ha='left')
    a0.figure.tight_layout()
    return a0.figure


# ----------------------------------------------------------------- from h5

def load_run(path):
    """Exponent data from an h5 run, rebuilding from R if the run was cut short."""
    with h5py.File(path, 'r') as f:
        out = {key: f.attrs[key] for key in ['dt', 'tau', 'n', 'k', 'transient_steps']}
        R = f['R']
        if 'blocks_done' in f.attrs:
            done = int(f.attrs['blocks_done'])
        else:   # older files: first all-zero R block marks where a crash cut the run
            done = next((j for j in range(R.shape[0]) if not np.any(R[j])), R.shape[0])
        if 'log_diag_R' in f and 'exponents' in f:
            out['log_diag'] = f['log_diag_R'][()][:done]
            out['t'] = f['t'][()][:done]
        else:
            out['log_diag'] = np.array([np.log(np.abs(np.diag(R[j]))) for j in range(done)])
            out['t'] = (out['transient_steps'] + np.arange(1, done + 1) * out['tau']) * out['dt']
        out['done'], out['total'] = done, R.shape[0]
        out['Q_blocks'] = sorted(int(j) for j in f['Q'])
    return out


def exponents_from(run, discard=0):
    """Time-averaged exponents from blocks discard..done."""
    ld = run['log_diag'][discard:]
    return ld.sum(0) / (len(ld) * run['tau'] * run['dt'])


def plot_run(path, modes=(0, -1), which='phys', show=False, discard=0, clv=False):
    name = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(FIGS, exist_ok=True)
    run = load_run(path)
    dt, tau, n, k = run['dt'], run['tau'], run['n'], run['k']
    if run['done'] < run['total']:
        print(f"partial run: {run['done']} of {run['total']} blocks")
    lam = exponents_from(run, discard)
    title = name + (f'  (blocks {discard}-{run["done"]})' if discard else '')

    fig = plot_spectrum(lam, n=n, title=title)
    fig.savefig(os.path.join(FIGS, f'{name}_spectrum.png'), dpi=150)
    fig = plot_convergence(run['log_diag'], dt, tau, t=run['t'])
    fig.savefig(os.path.join(FIGS, f'{name}_convergence.png'), dpi=150)
    if k == 2 * n:
        fig = plot_pairing(lam)
        fig.savefig(os.path.join(FIGS, f'{name}_pairing.png'), dpi=150)

    if clv:
        # covariant vectors at the stored block nearest 60% of the run, so
        # that the remaining 40% serves as the backward transient
        target = 0.6 * run['done']
        block = min((j for j in run['Q_blocks'] if j < run['done']), key=lambda j: abs(j - target))
        out = ginelli_backward(path, [block], discard_last=run['done'] - 1 - block)
        V, frame, kind = out['clv'][block], out['frame'][block], 'CLV'
    else:
        block = max(j for j in run['Q_blocks'] if j < run['done'])
        with h5py.File(path, 'r') as f:
            V = f['Q'][str(block)][()]
            frame = load_frame(f['frame'][str(block)])
        kind = 'GS vector'

    edges, q, W, counts = q_weight_by_exponent(frame, V, lam, which=which)
    fig = plot_q_weight(edges, q, W, counts, label=f'{name}  {kind}s at block {block}')
    fig.savefig(os.path.join(FIGS, f'{name}_qweight{"_clv" if clv else ""}.png'), dpi=150)

    # columns are in QR order, i.e. sorted by exponent
    for i in modes:
        idx = i if i >= 0 else k + i
        rep = mode_report(frame, V[:, idx], which)
        fig = plot_mode(rep, label=f'{name}  {kind} {idx} at block {block}', exponent=lam[idx])
        fig.savefig(os.path.join(FIGS, f'{name}_{"clv" if clv else "mode"}{idx}.png'), dpi=150)
    if show:
        plt.show()
    return lam


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--modes', type=int, nargs='*', default=[0, -1])
    p.add_argument('--copy', default='phys', choices=['phys', 'aux'])
    p.add_argument('--show', action='store_true')
    p.add_argument('--discard', type=int, default=0, help='blocks dropped from the average')
    p.add_argument('--clv', action='store_true', help='mode figures from covariant vectors')
    a = p.parse_args()
    plot_run(a.path, a.modes, a.copy, a.show, a.discard, a.clv)
