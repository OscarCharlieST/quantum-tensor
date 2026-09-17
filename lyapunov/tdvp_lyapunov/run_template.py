"""
Seed the tangent flow with a *single* long-wavelength local-temperature
template and watch it, instead of computing a spectrum.

This is the linearized analogue of the nonlinear transport measurement in
`qtensor.visualise` (`near_thermal` -> energy profiles -> Gaussian widths
-> D). Here the perturbation is a tangent vector a_q, its profile is
delta<h_j> rather than <h_j>, and diffusion would show as a rate
lambda(q) ~ -D q^2 rather than as sigma^2 ~ 2Dt.

The whole point is cheapness: at k = 1 the exponential action is free and
the cost is the generator plus the transport, which are k-independent, so
this is a few minutes where a spectrum is hours. The cost is that there is
no QR to hold the vector away from the leading direction -- it aligns with
the top Lyapunov direction at rate lambda_max - lambda, so this is a
*short-window* measurement and the point of the diagnostics below is to
find how short the window is.

    python lyapunov/tdvp_lyapunov/run_template.py --L 16 --D 3 --beta 0.1 \
        --blocks 90 --kmode 1 --out-dir C:/Users/charl/lyapunov_runs

writes the h5 and figures/<run>_template.png:
  1. the running and per-block growth rate, against lambda_max if known
  2. the profile delta<h_j> at a few times, against cos(q j)
  3. leakage: how much of the vector is still the q-mode, and where its
     spectral weight has moved to
"""

import argparse
import os
import sys

import h5py
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

import qtensor.operators as ops
import qtensor.thermofield as tf

from lyapunov.tdvp_lyapunov.frame import Frame
from lyapunov.tdvp_lyapunov.stepper import tdvp_step, exact_method, lanczos_method
from lyapunov.tdvp_lyapunov.benettin import benettin, load_frame
from lyapunov.tdvp_lyapunov.plots import FIGS, BLUE, ORANGE, AQUA, MUTED
from lyapunov.tdvp_lyapunov.hlm import (
    template_vectors, profile_map, profile_template, sinusoid_family,
    bond_tangent_vectors, phase_free_power, spectral_moments,
)
from lyapunov.relaxation.run_relaxation_scan import build_uniform_thermofield

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')


def seed_vector(frame, kmode, copy):
    """Normalized local-temperature template at wavevector index kmode."""
    q, templates = template_vectors(frame, kmode, copy)
    a = templates[kmode]
    return q[kmode], a / la.norm(a)


def mode_subspace(frame, q, copy):
    """
    Orthonormal basis for the free-phase family at q, as tangent vectors:
    the images of {cos(q(j+1/2)), sin(q(j+1/2))} with the uniform
    (conserved-energy) direction projected out. A vector's captured
    fraction in this subspace is the phase-blind 'is it still the q-mode'.
    """
    w = bond_tangent_vectors(frame, copy)
    dirs = sinusoid_family(len(w), q)
    A = np.column_stack([profile_template(frame, dirs[:, i], copy, w)
                         for i in range(dirs.shape[1])])
    return la.qr(A, mode='economic')[0]


def analyse(path, kmode, copy, n_q=201):
    """Per-block profile, wavevector content and overlap with the q-family."""
    q_grid = np.linspace(0, np.pi, n_q)
    out = {k: [] for k in ('t', 'prof', 'q_bar', 'sd', 'peak',
                           'captured', 'overlap')}
    with h5py.File(path, 'r') as f:
        blocks = sorted(int(s) for s in f['Q'])
        t_all = f['t'][()]
        log_diag = f['log_diag_R'][()]
        done = int(f.attrs['blocks_done'])
        for j in blocks:
            if j >= done:
                continue
            v = f['Q'][str(j)][()]                    # (2n, 1)
            frame = load_frame(f['frame'][str(j)])
            prof = profile_map(frame, v, copy)[:, 0]
            mean, sd, peak = spectral_moments(q_grid, phase_free_power(prof, q_grid))
            q, a_q = seed_vector(frame, kmode, copy)
            A_on = mode_subspace(frame, q, copy)
            out['t'].append(t_all[j])
            out['prof'].append(prof)
            out['q_bar'].append(mean[0])
            out['sd'].append(sd[0])
            out['peak'].append(peak[0])
            out['captured'].append(float(la.norm(A_on.T @ v[:, 0]) ** 2))
            out['overlap'].append(float(abs(a_q @ v[:, 0]) ** 2))
    d = {k: np.asarray(v) for k, v in out.items()}
    d['t_all'] = t_all[:done]
    d['log_diag'] = log_diag[:done, 0]
    return d


def plot(d, q, dt, tau, name, lam_max=None):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # 1. growth rate: per block, and the running average
    t, ld = d['t_all'], d['log_diag']
    inst = ld / (tau * dt)
    run = np.cumsum(ld) / (np.arange(1, len(ld) + 1) * tau * dt)
    ax[0].plot(t, inst, '-', lw=0.8, color=AQUA, alpha=0.7, label='per block')
    ax[0].plot(t, run, '-', lw=2, color=BLUE, label='running mean')
    ax[0].axhline(0, color=MUTED, lw=0.8)
    if lam_max is not None:
        ax[0].axhline(lam_max, color=ORANGE, ls='--', lw=1.2,
                      label=rf'$\lambda_{{\max}}$ = {lam_max:.3f}')
    ax[0].set_xlabel('t')
    ax[0].set_ylabel(r'growth rate of $\|\delta\psi\|$')
    ax[0].set_title(f'seeded at q = {q:.3f}', loc='left')
    ax[0].legend(frameon=False, fontsize=8)

    # 2. the profile, early to late
    n_bonds = d['prof'].shape[1]
    bonds = np.arange(n_bonds) + 0.5
    ref = np.cos(q * bonds)
    ref /= la.norm(ref)
    picks = np.unique(np.linspace(0, len(d['t']) - 1, 4).astype(int))
    for i, idx in enumerate(picks):
        p = d['prof'][idx]
        p = p / max(la.norm(p), 1e-30)
        if p @ ref < 0:
            p = -p
        ax[1].plot(bonds, p, 'o-', ms=3, alpha=0.4 + 0.6 * i / max(len(picks) - 1, 1),
                   color=BLUE, label=f"t = {d['t'][idx]:.2f}")
    ax[1].plot(bonds, ref, '--', color=ORANGE, lw=1.5, label=rf'$\cos(q j)$')
    ax[1].axhline(0, color=MUTED, lw=0.8)
    ax[1].set_xlabel('bond')
    ax[1].set_ylabel(r'$\delta\langle h_j\rangle$ (normalized)')
    ax[1].set_title('does the seeded wave survive?', loc='left')
    ax[1].legend(frameon=False, fontsize=7.5)

    # 3. leakage
    ax[2].plot(d['t'], d['captured'], 'o-', ms=4, color=BLUE,
               label=rf'in the $q$ family (free phase)')
    ax[2].plot(d['t'], d['overlap'], 's--', ms=3.5, color=AQUA, alpha=0.8,
               label=r'on the template $a_q$')
    ax[2].set_xlabel('t')
    ax[2].set_ylabel('fraction of the vector')
    ax[2].set_ylim(-0.02, 1.02)
    ax[2].set_title('leakage out of the seeded mode', loc='left')
    a2 = ax[2].twinx()
    a2.grid(False)
    a2.fill_between(d['t'], d['q_bar'] - d['sd'], d['q_bar'] + d['sd'],
                    color=ORANGE, alpha=0.15)
    a2.plot(d['t'], d['q_bar'], '-', color=ORANGE, lw=1.5)
    a2.axhline(q, color=ORANGE, ls=':', lw=1)
    a2.set_ylabel(r'$\bar q$ of the profile', color=ORANGE)
    a2.set_ylim(0, np.pi)
    ax[2].legend(frameon=False, fontsize=8, loc='center left')

    fig.suptitle(name, x=0.01, ha='left')
    fig.tight_layout()
    return fig


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--L', type=int, default=16)
    p.add_argument('--D', type=int, default=3)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--imag-steps', type=int, default=60)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--blocks', type=int, default=90)
    p.add_argument('--tau', type=int, default=1)
    p.add_argument('--transient', type=int, default=160,
                   help='TDVP steps before the template is built and released')
    p.add_argument('--kmode', type=int, default=1,
                   help='wavevector index; 1 is the longest non-uniform wave')
    p.add_argument('--copy', default='phys', choices=['phys', 'aux'])
    p.add_argument('--method', default='lanczos', choices=['lanczos', 'exact'])
    p.add_argument('--lam-max', type=float, default=None,
                   help='reference line from a spectrum run, if you have one')
    p.add_argument('--tag', default='template')
    p.add_argument('--out-dir', default=RUNS)
    p.add_argument('--analyse-only', action='store_true')
    return p.parse_args()


def main():
    a = parse()
    np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    name = f"L{a.L}_D{a.D}_beta{a.beta:g}_{a.tag}{a.kmode}"
    path = os.path.join(a.out_dir, f"{name}.h5")

    psi, energy = build_uniform_thermofield(a.L, a.D, a.beta, a.imag_steps)
    H_sym = tf.thermofield_hamiltonian(ops.tilted_ising(N=a.L), asym=False)
    method = lanczos_method() if a.method == 'lanczos' else exact_method()

    # the transient happens here, not inside benettin, because the template
    # must be built in the frame at the point the tangent flow starts
    for _ in range(a.transient):
        psi = tdvp_step(psi, H_sym, a.dt, method)
    frame = Frame(psi, a.D)
    q, a_q = seed_vector(frame, a.kmode, a.copy)
    print(f"L={a.L} D={a.D} beta={a.beta}: E/L = {energy / a.L:.4f}, n = {frame.n}, "
          f"seeded at q = {q:.4f} (k = {a.kmode} of {len(frame.sites) - 1} bonds)")

    if not a.analyse_only:
        benettin(psi, H_sym, a.dt, a.blocks, 1, tau=a.tau, method=method,
                 max_bond_dim=a.D, transient_steps=0, store_path=path,
                 store_Q_blocks=set(range(a.blocks)), Q0=a_q[:, None])
        print(f"written {path}")

    d = analyse(path, a.kmode, a.copy)
    rate = np.cumsum(d['log_diag']) / (np.arange(1, len(d['log_diag']) + 1) * a.tau * a.dt)
    print(f"\n  {'t':>6s} {'rate':>8s} {'in q family':>12s} {'on a_q':>8s} "
          f"{'q_bar':>7s} {'sd':>6s}")
    for i in range(len(d['t'])):
        j = int(round(d['t'][i] / (a.dt * a.tau))) - 1
        print(f"  {d['t'][i]:6.2f} {rate[min(j, len(rate) - 1)]:+8.4f} "
              f"{d['captured'][i]:12.3f} {d['overlap'][i]:8.3f} "
              f"{d['q_bar'][i]:7.3f} {d['sd'][i]:6.3f}")

    os.makedirs(FIGS, exist_ok=True)
    fig = plot(d, q, a.dt, a.tau, f"{name}  n={frame.n}, single seeded tangent vector",
               a.lam_max)
    out = os.path.join(FIGS, f"{name}.png")
    fig.savefig(out, dpi=150)
    print(f"\n  written {os.path.relpath(out)}")


if __name__ == '__main__':
    main()
