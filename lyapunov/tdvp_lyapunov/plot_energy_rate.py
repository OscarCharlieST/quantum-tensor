"""
Figures for the energy-rate study (steps 0 and 1).

    python lyapunov/tdvp_lyapunov/plot_energy_rate.py

  figures/energy_rate_green_kubo.png   ED: G(t)/G(0) and D(t) = I(t)/c per L
  figures/energy_rate_bracket_L{L}.png sum_k k^2 B against D, TFD and trajectory
"""

import glob
import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

sys.path.insert(0, os.getcwd())

from lyapunov.tdvp_lyapunov.plots import FIGS, BLUE, ORANGE, AQUA, INK, MUTED
from lyapunov.tdvp_lyapunov.energy_rate import sum_k2, EXACT_SUM_K2B

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')


def ramp(n, base=BLUE):
    """n steps of one hue, light (small L) to full (large L)."""
    b = np.array(to_rgb(base))
    return [tuple(1 - (1 - b) * f) for f in np.linspace(0.35, 1.0, n)]


def green_kubo():
    d = np.load(os.path.join(RUNS, 'energy_rate_ed.npz'))
    Ls = sorted(int(m.group(1)) for k in d.files if (m := re.match(r'L(\d+)_t$', k)))
    c, G0 = float(d['c']), float(d['G0'])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for L, col in zip(Ls, ramp(len(Ls))):
        t, G = d[f'L{L}_t'], d[f'L{L}_G']
        I = np.concatenate([[0], np.cumsum(0.5 * (G[1:] + G[:-1]) * np.diff(t))])
        ax[0].plot(t, G / G0, color=col, lw=1.5, label=f'L = {L}')
        ax[1].plot(t, I / c, color=col, lw=1.5, label=f'L = {L}')
    ax[0].axhline(0, color=MUTED, lw=0.8)
    ax[0].set_xlabel('t')
    ax[0].set_ylabel(r'$G(t)/G(0)$')
    ax[0].set_title('total-current autocorrelation, infinite T', loc='left')
    ax[0].set_xlim(0, 10)
    ax[1].set_xlabel('t')
    ax[1].set_ylabel(r'$D(t) = \int_0^t G\,/\,c$')
    ax[1].set_title(r'running Green–Kubo $D$ (periodic chains)', loc='left')
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIGS, 'energy_rate_green_kubo.png')
    fig.savefig(out, dpi=150)
    print(f"written {os.path.relpath(out)}")


SHOW = [('B1', 'exact single ordering  B1', INK, 'o'),
        ('BP', r'literal $\langle h^+ P X^+ P H^+\rangle$  BP', BLUE, 's'),
        ('Csym', r'exact full $\ddot E$, symmetric  $-4C_{sym}$', ORANGE, '^'),
        ('Cphys', r'exact full $\ddot E$, physical  $-4C_{phys}$', AQUA, 'D')]
SCALE = {'Csym': -4.0, 'Cphys': -4.0}


def brackets_vs_D(L):
    files = glob.glob(os.path.join(RUNS, f'energy_rate_L{L}_D*.npz'))
    if not files:
        return
    runs = sorted((np.load(f) for f in files), key=lambda r: int(r['D']))
    Ds = np.array([int(r['D']) for r in runs])
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for j, (key, label, col, mk) in enumerate(SHOW):
        S = np.array([[SCALE.get(key, 1) * sum_k2(B, r['k']) for B in r[key]] for r in runs])
        x = Ds * (1 + 0.03 * (j - 1.5))            # small dodge
        ax[0].plot(x, S[:, 0].real, mk + '-', color=col, ms=6, lw=1.5, label=label)
        tr = S[:, 1:].real
        ax[1].errorbar(x, tr.mean(1), yerr=tr.std(1), fmt=mk + '-', color=col,
                       ms=6, lw=1.5, capsize=3, label=label)
    for a in ax[:2]:
        a.axhline(EXACT_SUM_K2B, color=MUTED, ls='--', lw=1)
        a.set_xscale('log', base=2)
        a.set_xticks(Ds, [str(D) for D in Ds])
        a.set_xlabel('bond dimension D')
    ax[0].set_ylabel(r'$\sum_k k^2 B_k$')
    ax[0].set_title(r'at the $\beta=0.1$ thermofield double', loc='left')
    ax[1].set_title(r'along the $H_{sym}$ trajectory (mean $\pm$ sd)', loc='left')
    ax[1].legend(frameon=False, fontsize=7.5)
    capH = np.array([r['capture_H'][1:].mean() for r in runs])
    caph = np.array([r['capture_h0'][1:].mean() for r in runs])
    ax[2].plot(Ds, capH, 'o-', color=BLUE, ms=6, label=r'$|PH\psi|^2/\mathrm{Var}\,H$')
    ax[2].plot(Ds, caph, 's-', color=ORANGE, ms=6, label=r'$|Ph_0^+\psi|^2/\mathrm{Var}\,h_0^+$')
    ax[2].set_xscale('log', base=2)
    ax[2].set_xticks(Ds, [str(D) for D in Ds])
    ax[2].set_ylim(0, 1.02)
    ax[2].set_xlabel('bond dimension D')
    ax[2].set_title('tangent-space capture on the trajectory', loc='left')
    ax[2].legend(frameon=False, fontsize=8)
    fig.suptitle(f'L = {L}', x=0.01, ha='left')
    fig.tight_layout()
    out = os.path.join(FIGS, f'energy_rate_bracket_L{L}.png')
    fig.savefig(out, dpi=150)
    print(f"written {os.path.relpath(out)}")


if __name__ == '__main__':
    os.makedirs(FIGS, exist_ok=True)
    green_kubo()
    for L in (8, 12, 16):
        brackets_vs_D(L)
