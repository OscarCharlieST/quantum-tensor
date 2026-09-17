"""
The longest-wavelength candidate Lyapunov modes of a full-spectrum run,
with a free phase at each wavevector.

At each of the first few non-uniform wavevectors q_k = pi k / N_bonds, the
available profile shapes are the 2D family {cos(q(j+1/2)), sin(q(j+1/2))}
with the uniform (conserved-energy) component projected out. The mode
reported is the vector of the chosen band that best represents that
family -- a free-phase generalization of projecting a single cosine.

    python lyapunov/tdvp_lyapunov/long_wavelength_modes.py \
        C:/Users/charl/lyapunov_runs/L16_D4_beta1_k2n.h5 [--band zero-] [--m 120] [--kmax 3]

writes figures/<run>_longwave_<band>.png and prints, per wavevector, the
captured fraction with and without the phase freedom.
"""

import argparse
import os
import sys

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
    bond_tangent_vectors, profile_template, sinusoid_family, best_mode_in_subspace,
    fit_sinusoid, band_indices, profile_map, subspace_projector_fraction,
)


def plot_modes(modes, bonds, label=''):
    rows = len(modes)
    fig, axes = plt.subplots(rows, 3, figsize=(11.5, 2.7 * rows), squeeze=False)
    for r, mo in enumerate(modes):
        a0, a1, a2 = axes[r]
        p, q = mo['profile'], mo['q']
        j = np.arange(len(p)) + 0.5
        fit = mo['amp'] * np.cos(q * j + mo['phase'])
        a0.plot(bonds + 0.5, p, 'o-', ms=4, color=ORANGE, label='mode profile')
        a0.plot(bonds + 0.5, fit, '--', color=MUTED, lw=1.2,
                label=rf'$\cos(qj {mo["phase"]:+.2f})$')
        a0.plot(bonds + 0.5, mo['amp'] * np.cos(q * j), ':', color=AQUA, lw=1.2,
                label='pure cosine')
        a0.axhline(0, color=MUTED, lw=0.8)
        a0.set_xlabel('bond')
        a0.set_ylabel(r'$\delta\langle h_j\rangle$')
        a0.legend(frameon=False, fontsize=7.5)
        a0.set_title(f'k={mo["k"]}  q={q:.2f}  '
                     rf'$\lambda_{{\rm eff}}$ {mo["lambda_eff"]:+.4f}', loc='left')

        a1.bar(mo['dct_q'], np.abs(mo['dct']), width=mo['dct_q'][1] * 0.8, color=AQUA)
        a1.set_xlabel('q')
        a1.set_ylabel('|DCT|')
        a1.set_xticks([0, np.pi / 2, np.pi])
        a1.set_xticklabels(['0', r'$\pi/2$', r'$\pi$'])
        a1.set_title(f'sinusoid fit explains {mo["fit_frac"]:.2f} of power', loc='left')

        a2.bar(np.arange(len(mo['site_weights'])), mo['site_weights'], color=BLUE, width=0.8)
        a2.set_xlabel('site')
        a2.set_ylabel(r'$|X^n|^2$ (norm.)')
        a2.set_title(f'extent {mo["extent"]:.1f} of {len(mo["site_weights"])} sites',
                     loc='left')
    fig.suptitle(label, x=0.01, ha='left')
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('path')
    p.add_argument('--band', default='zero-',
                   help="band to build modes from: zero-, zero+, bottom, top, ...")
    p.add_argument('--m', type=int, default=120)
    p.add_argument('--kmax', type=int, default=3, help='how many wavevectors')
    p.add_argument('--copy', default='phys', choices=['phys', 'aux'])
    p.add_argument('--discard', type=int, default=40)
    a = p.parse_args()

    name = os.path.splitext(os.path.basename(a.path))[0]
    run = load_run(a.path)
    lam = exponents_from(run, a.discard)
    blocks = [j for j in run['Q_blocks'] if j < run['done']]
    block = min(blocks, key=lambda j: abs(j - 0.6 * run['done']))
    with h5py.File(a.path, 'r') as f:
        frame = load_frame(f['frame'][str(block)])
    idx = band_indices(lam, a.band, m=a.m)
    print(f"{name}: n={run['n']}, k={run['k']}, block {block}, band '{a.band}' "
          f"lambda in [{lam[idx].min():+.4f}, {lam[idx].max():+.4f}]")

    V = ginelli_backward(a.path, [block], discard_last=run['done'] - 1 - block)['clv'][block]
    V_band = V[:, idx]
    w = bond_tangent_vectors(frame, a.copy)
    n_bonds = len(w)

    print(f"\n  {'k':>2s} {'q':>6s} {'captured':>9s} {'cos only':>9s} {'gain':>6s} "
          f"{'phase':>7s} {'fit':>5s} {'lam_eff':>8s} {'extent':>7s}")
    modes = []
    for k in range(1, a.kmax + 1):
        q = np.pi * k / n_bonds
        # free-phase family, uniform projected out
        dirs = sinusoid_family(n_bonds, q)
        A = np.column_stack([profile_template(frame, dirs[:, i], a.copy, w)
                             for i in range(dirs.shape[1])])
        A_on = la.qr(A, mode='economic')[0]
        x, captured, proj = best_mode_in_subspace(V_band, A_on)
        # cosine-only comparison: the DCT-II cosine at the same q
        j = np.arange(n_bonds) + 0.5
        c = np.cos(q * j)
        c -= c.mean()
        c /= la.norm(c)
        a_cos = profile_template(frame, c, a.copy, w)
        cos_only, _, _ = subspace_projector_fraction(V_band, a_cos)

        vec = proj / la.norm(proj)
        prof = profile_map(frame, vec[:, None], a.copy)[:, 0]
        amp, phase, frac = fit_sinusoid(prof, q)
        dq, dct = cosine_transform(prof)
        sw = np.array([frame.site_weights(complexify(vec)).get(s, 0.0)
                       for s in frame.sites])
        sw = sw / sw.sum()
        # where in the spectrum this mode sits: weight of the band coefficients
        y = la.solve(V_band.T @ V_band, V_band.T @ proj, assume_a='pos')
        lam_eff = float(lam[idx] @ y ** 2 / np.sum(y ** 2))
        modes.append({'k': k, 'q': q, 'profile': prof, 'amp': amp, 'phase': phase,
                      'fit_frac': frac, 'dct': dct, 'dct_q': dq, 'site_weights': sw,
                      'extent': float(1.0 / np.sum(sw ** 2)), 'lambda_eff': lam_eff})
        print(f"  {k:2d} {q:6.2f} {captured:9.3f} {cos_only:9.3f} "
              f"{captured / max(cos_only, 1e-12):6.2f} {phase:+7.2f} {frac:5.2f} "
              f"{lam_eff:+8.4f} {modes[-1]['extent']:7.1f}")

    os.makedirs(FIGS, exist_ok=True)
    fig = plot_modes(modes, np.array(frame.sites[:-1]),
                     f"{name}  band '{a.band}' (m={a.m}), {a.copy} copy, block {block} "
                     f"— longest-wavelength modes, free phase")
    out = os.path.join(FIGS, f"{name}_longwave_{a.band.replace('-', 'neg').replace('+', 'pos')}.png")
    fig.savefig(out, dpi=150)
    print(f"\n  written {os.path.relpath(out)}")


if __name__ == '__main__':
    main()
