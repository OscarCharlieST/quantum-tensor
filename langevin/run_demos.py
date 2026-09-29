"""
Simple-case behaviour of the Langevin TDVP, run from the repo root:

    python langevin/run_demos.py [saturation|noise|friction|schemes|all]
    python langevin/run_demos.py friction_scan --N 16 --D 32 --traj 16 --tf 40 --dt 0.04
    (--state random|plus, --state-seed k; 'all' does not include the scan)

Each demo writes langevin/figures/<name>.png and langevin/runs/<name>.npz and
prints the numbers it is judged on.

  saturation  gamma = gammaT = 0: mid-chain entropy grows ballistically and
              saturates at a D-dependent ceiling; energy is conserved.
  noise       gamma = 0, gammaT > 0 (the reference code's 'dephasing' limit):
              trajectory-averaged entropy growth against gammaT.
  friction    gamma > 0, gammaT = 0: energy falls monotonically, at the rate
              dE/dt = -gamma A^T S A the rearranged equation predicts.
  schemes     gamma > 0, gammaT > 0: 'paper' vs 'euler' vs 'heun' ensemble
              energies -- where the Ito/Stratonovich question shows up.

  friction_scan  Fig. 1 of arXiv:2111.06408 at reduced size: T fixed, gamma
              scanned; ensemble entropy, S0 tanh(t/t0) fits, collapse, and
              S0, t0 against gammaT.

Initial state (--state): 'random' (default) is a seeded product of randomly
oriented spins (models.random_product_state, --state-seed); 'plus' is |+>^N,
the reference code's state (the README's first four demos, 2026-09-28, used it).
The state is the same for every trajectory of an ensemble: the ensemble is over
noise. H is the reference code's H = sum -Z Z - 1.05 X + 0.5 Z; bond dimension
is grown with two-site TDVP first (grow_to), as the reference code does.
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())

from langevin.models import plus_state, random_product_state
from langevin.integrator import (LangevinModel, run_trajectory, run_ensemble,
                                 run_many, friction_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'figures')
RUNS = os.path.join(HERE, 'runs')


def save(name, fig, **arrays):
    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(RUNS, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name + '.png'), dpi=130)
    plt.close(fig)
    np.savez(os.path.join(RUNS, name + '.npz'), **arrays)
    print(f"  -> figures/{name}.png, runs/{name}.npz")


STATE = {'kind': 'random', 'seed': 0}


def initial_state(N):
    """The demos' initial state, unpadded (D = 1), per --state / --state-seed."""
    if STATE['kind'] == 'plus':
        return plus_state(N, 1)
    return random_product_state(N, seed=STATE['seed'])


def stack(results, key):
    return np.array([r[key] for r in results])


# ------------------------------------------------------------------ demos

def demo_saturation(N=12, Ds=(4, 8, 16), dt=0.05, t_f=15.0):
    print(f"[saturation] N={N}, D in {Ds}, gamma = gammaT = 0")
    steps = int(round(t_f / dt))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    out = {}
    for D in Ds:
        res = run_ensemble(initial_state(N), dict(N=N), seeds=[0], processes=1,
                           dt=dt, steps=steps, grow_to=D)[0]
        t, S, E = res['t'], res['S_mid'], res['energy']
        late = S[t > t_f / 2]
        print(f"  D={D:3d}: S_mid(late) = {late.mean():.3f} +- {late.std():.3f}  "
              f"(ln D = {np.log(D):.3f}), max|E - E0| = {np.max(abs(E - E[0])):.1e}, "
              f"two-site until t = {res['t_switch']}")
        ax[0].plot(t, S, label=f'D = {D}')
        ax[1].plot(t, E - E[0], label=f'D = {D}')
        out[f'S_D{D}'], out[f'E_D{D}'], out['t'] = S, E, t
    ax[0].set(xlabel='t', ylabel='mid-chain S (nats)', title=f'Noiseless, N = {N}')
    ax[1].set(xlabel='t', ylabel='E(t) - E(0)', title='Energy conservation')
    ax[0].legend(), ax[1].legend()
    save('saturation', fig, **out)


def demo_noise(N=12, D=16, dt=0.02, t_f=10.0, gammaTs=(0.0, 0.05, 0.2, 0.5),
               n_traj=10):
    print(f"[noise] N={N}, D={D}, gamma = 0, gammaT in {gammaTs}, {n_traj} trajectories each")
    steps = int(round(t_f / dt))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    out = {}
    for gT in gammaTs:
        seeds = [0] if gT == 0 else list(range(n_traj))
        t0 = time.time()
        res = run_ensemble(initial_state(N), dict(N=N), seeds, dt=dt, steps=steps,
                           gamma=0.0, gammaT=gT, grow_to=D, measure_every=5)
        t = res[0]['t']
        S, E = stack(res, 'S_mid'), stack(res, 'energy')
        Sm, Se = S.mean(0), S.std(0) / np.sqrt(len(seeds))
        i1, i3 = np.argmin(abs(t - 1)), np.argmin(abs(t - 3))
        print(f"  gammaT={gT:5.2f}: <S>(t=1) = {Sm[i1]:.3f}, <S>(t=3) = {Sm[i3]:.3f}, "
              f"<S>(t_f) = {Sm[-1]:.3f} +- {Se[-1]:.3f}, <E>(t_f) - E0 = "
              f"{E[:, -1].mean() - E[0, 0]:.2f}   ({time.time()-t0:.0f}s)")
        ax[0].plot(t, Sm, label=f'$\\gamma T$ = {gT}')
        ax[0].fill_between(t, Sm - Se, Sm + Se, alpha=0.25)
        ax[1].plot(t, E.mean(0), label=f'$\\gamma T$ = {gT}')
        out[f'S_gT{gT}'], out[f'E_gT{gT}'], out['t'] = S, E, t
    ax[0].set(xlabel='t', ylabel='<S_mid> (nats)', title=f'Noise only ($\\gamma$ = 0), N = {N}, D = {D}')
    ax[1].set(xlabel='t', ylabel='<E>', title='Energy (noise heats)')
    ax[0].legend(), ax[1].legend()
    save('noise', fig, **out)


def demo_friction(N=10, D=8, dt=0.02, t_f=10.0, gamma=0.2):
    print(f"[friction] N={N}, D={D}, gamma={gamma}, gammaT = 0")
    steps = int(round(t_f / dt))
    model = LangevinModel(N)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    out = {}
    for scheme in ('euler', 'heun'):
        res = run_trajectory(initial_state(N), model, dt, steps, gamma=gamma,
                             scheme=scheme, grow_to=D)
        t, E = res['t'], res['energy']
        dE = np.diff(E)
        print(f"  {scheme:5s}: E {E[0]:.3f} -> {E[-1]:.3f}; steps with dE > 1e-10: "
              f"{np.sum(dE > 1e-10)} of {len(dE)} (largest {dE.max():.1e})")
        ax[0].plot(t, E, label=scheme)
        out[f'E_{scheme}'], out['t'] = E, t
    # predicted instantaneous rate along the heun trajectory, re-evaluated at a few points
    psi = initial_state(N)
    pred, meas, tt = [], [], []
    for k in range(5):
        res = run_trajectory(psi, model, dt, 50, gamma=gamma, scheme='heun', grow_to=D)
        psi = res['final_state']
        A, K, f, E0, _ = model.brackets(psi)
        S = friction_matrix(K, gamma)
        pred.append(-gamma * A @ S @ A)
        nxt = run_trajectory(psi, model, dt / 10, 1, gamma=gamma, scheme='heun', grow_to=D)
        meas.append((nxt['energy'][1] - nxt['energy'][0]) / (dt / 10))
        tt.append((k + 1) * 50 * dt)
    for a, b, c in zip(tt, pred, meas):
        print(f"  t={a:4.1f}: dE/dt predicted -gamma A.S.A = {b:.4f}, measured {c:.4f}")
    ax[1].plot(tt, pred, 'o-', label='$-\\gamma A^T S A$')
    ax[1].plot(tt, meas, 'x--', label='measured (one step, dt/10)')
    ax[0].set(xlabel='t', ylabel='E', title=f'Friction only, $\\gamma$ = {gamma}')
    ax[1].set(xlabel='t', ylabel='dE/dt')
    ax[0].legend(), ax[1].legend()
    save('friction', fig, pred=np.array(pred), meas=np.array(meas), tt=np.array(tt), **out)


def demo_schemes(N=8, D=8, dts=(0.04, 0.02), t_f=6.0, gamma=0.5, gammaT=0.5,
                 n_traj=40):
    print(f"[schemes] N={N}, D={D}, gamma={gamma}, gammaT={gammaT} (T={gammaT/gamma}), "
          f"{n_traj} trajectories per scheme")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    out = {}
    for dt in dts:
        steps = int(round(t_f / dt))
        for scheme, ls in (('paper', ':'), ('euler', '--'), ('heun', '-')):
            t0 = time.time()
            res = run_ensemble(initial_state(N), dict(N=N), list(range(n_traj)),
                               dt=dt, steps=steps, gamma=gamma, gammaT=gammaT,
                               scheme=scheme, grow_to=D, measure_every=max(1, int(0.1 / dt)))
            t = res[0]['t']
            E, S = stack(res, 'energy'), stack(res, 'S_mid')
            late = t > t_f / 2
            Em = E[:, late].mean()
            Ee = E[:, late].mean(1).std() / np.sqrt(n_traj)
            print(f"  dt={dt:5.3f} {scheme:5s}: <E>(t > {t_f/2}) = {Em:.3f} +- {Ee:.3f}, "
                  f"<S_mid>(t_f) = {S[:, -1].mean():.3f}   ({time.time()-t0:.0f}s)")
            ax[0].plot(t, E.mean(0), ls, label=f'{scheme}, dt={dt}')
            ax[1].plot(t, S.mean(0), ls, label=f'{scheme}, dt={dt}')
            out[f'E_{scheme}_{dt}'], out[f'S_{scheme}_{dt}'], out[f't_{dt}'] = E, S, t
    ax[0].set(xlabel='t', ylabel='<E>', title=f'$\\gamma$ = {gamma}, $\\gamma T$ = {gammaT}')
    ax[1].set(xlabel='t', ylabel='<S_mid>')
    ax[0].legend(fontsize=7), ax[1].legend(fontsize=7)
    save('schemes', fig, **out)


def tanh_fit(t, S, err):
    """S0, t0 (and their standard errors) of S(t) = S0 tanh(t/t0), weighted by err."""
    from scipy.optimize import curve_fit
    f = lambda t, S0, t0: S0 * np.tanh(t / t0)
    sigma = np.maximum(err, 1e-3)
    p, cov = curve_fit(f, t, S, p0=(max(S[-1], 1e-2), t[-1] / 4), sigma=sigma,
                       bounds=([0, 1e-3], [np.inf, np.inf]), maxfev=20000)
    return p[0], p[1], np.sqrt(np.diag(cov))


def demo_friction_scan(N=16, D=32, T=0.2, gammaTs=(0.02, 0.05, 0.08, 0.11, 0.15, 0.25),
                       n_traj=16, dt=0.04, t_f=40.0, scheme='heun', processes=21,
                       noiseless=True, tag='', gamma=None, prefix='friction_scan'):
    """
    Fig. 1 of arXiv:2111.06408 (there: N = 16, D = 128, T = 0.2, 70 trajectories,
    t = 60) at reduced D and trajectory count. By default T is fixed and
    gamma = gammaT / T; pass gamma to fix the friction instead (T = gammaT / gamma).
    gammaT = 0 is the noiseless trajectory (one run, deterministic).

    dt and t_f may be scalars or one value per gammaT: strong noise needs a
    small dt (the per-step kick is sqrt(2 gammaT dt) per Pauli), slow
    saturation a long t_f. The noiseless run uses the first entry of each.

    Saves the raw ensembles to runs/{prefix}_N{N}_D{D}{tag}.npz, then replots
    from *every* runs/{prefix}_N{N}_D{D}*.npz, so extra gammaT values can be
    added later with a different tag (and noiseless=False).
    """
    n = len(gammaTs)
    dts = list(np.broadcast_to(dt, n))
    t_fs = list(np.broadcast_to(t_f, n))
    gammas = [gamma if gamma is not None else gT / T for gT in gammaTs]
    fixed = f'gamma={gamma}' if gamma is not None else f'T={T}'
    print(f"[{prefix}] N={N}, D={D}, {fixed}, gammaT in {tuple(gammaTs)}, dt={dts}, t_f={t_fs}, "
          f"{n_traj} trajectories, scheme={scheme}, "
          f"state={STATE['kind']}(seed {STATE['seed']}), noiseless={noiseless}", flush=True)

    def common(dt, t_f):
        return dict(dt=dt, steps=int(round(t_f / dt)), scheme=scheme, grow_to=D,
                    measure_every=max(1, int(round(0.2 / dt))))
    jobs = [dict(common(dts[0], t_fs[0]), gamma=0.0, gammaT=0.0, seed=0)] if noiseless else []
    for g, gT, d, tf in zip(gammas, gammaTs, dts, t_fs):
        jobs += [dict(common(d, tf), gamma=g, gammaT=gT, seed=s) for s in range(n_traj)]
    # longest jobs first, so they are not left running alone at the end
    order = sorted(range(len(jobs)), key=lambda j: -jobs[j]['steps'])
    t0 = time.time()
    res_sorted = run_many(initial_state(N), dict(N=N), [jobs[j] for j in order], processes)
    res = [None] * len(jobs)
    for j, r in zip(order, res_sorted):
        res[j] = r
    print(f"  {len(jobs)} trajectories in {time.time() - t0:.0f}s", flush=True)

    out = {'T': T} if gamma is None else {'gamma': gamma}
    i = 0
    if noiseless:
        out['S_gT0.0'], out['E_gT0.0'] = stack(res[:1], 'S_mid'), stack(res[:1], 'energy')
        out['t_gT0.0'] = res[0]['t']
        i = 1
    for gT in gammaTs:
        rs = res[i:i + n_traj]
        out[f'S_gT{gT}'], out[f'E_gT{gT}'] = stack(rs, 'S_mid'), stack(rs, 'energy')
        out[f't_gT{gT}'] = rs[0]['t']
        i += n_traj
    out['scheme'] = scheme
    os.makedirs(RUNS, exist_ok=True)
    raw = os.path.join(RUNS, f'{prefix}_N{N}_D{D}{tag}.npz')
    np.savez(raw, **out)
    print(f"  -> {os.path.relpath(raw, HERE)}")
    plot_friction_scan(N, D, T=T, gamma=gamma, prefix=prefix)


def plot_friction_scan(N=16, D=32, T=0.2, gamma=None, prefix='friction_scan'):
    """
    Fits and the four-panel figure from all runs/{prefix}_N{N}_D{D}*.npz (raw
    ensembles S_gT<x>, E_gT<x>, and t_gT<x> or a shared t), merged by gammaT.
    Writes figures/{prefix}_N{N}_D{D}.png and runs/{prefix}_N{N}_D{D}_fits.npz.
    gamma: the fixed friction of a fixed-gamma scan; None means T is fixed.

    Windows are relative to each curve's own t_f: late averages over the last
    quarter, and 'saturated' means <S> rose by < 1% over the last eighth
    (for t_f = 40: the last 10 and last 5 time units).
    """
    import glob
    files = sorted(f for f in glob.glob(os.path.join(RUNS, f'{prefix}_N{N}_D{D}*.npz'))
                   if not f.endswith('_fits.npz'))
    data, schemes = {}, set()
    for f in files:
        with np.load(f) as z:
            # runs saved before the scheme was recorded were all heun
            schemes.add(str(z['scheme']) if 'scheme' in z.files else 'heun')
            for k in z.files:
                if k.startswith('S_gT'):
                    x = k[4:]
                    t = z['t_gT' + x] if 't_gT' + x in z.files else z['t']
                    data[float(x)] = (t, z[k], z['E_gT' + x])
    print(f"  plotting gammaT = {sorted(data)} from {[os.path.basename(f) for f in files]}")
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    cmap = plt.get_cmap('viridis')
    rows = []
    for k, gT in enumerate(sorted(data)):
        t, S, E = data[gT]
        t_f = t[-1]
        late = t >= 0.75 * t_f
        Sm = S.mean(0)
        Se = S.std(0) / np.sqrt(len(S)) if len(S) > 1 else np.full_like(Sm, 0.02)
        S0, t0_, perr = tanh_fit(t, Sm, Se)
        # saturated, judged from the data rather than the fit
        rise = (Sm[-1] - Sm[t >= 0.875 * t_f][0]) / Sm[-1]
        g = 0.0 if gT == 0 else (gamma if gamma is not None else gT / T)
        rows.append((gT, g, S0, perr[0], t0_, perr[1], Sm[late].mean(), E[:, late].mean(),
                     len(S), rise < 0.01, t_f))
        c = 'k' if gT == 0 else cmap(k / len(data))
        lab = 'noiseless' if gT == 0 else f'$\\gamma T$ = {gT:g}'
        ax[0, 0].plot(t, Sm, color=c, label=lab)
        if len(S) > 1:
            ax[0, 0].fill_between(t, Sm - Se, Sm + Se, color=c, alpha=0.2)
        ax[0, 0].plot(t, S0 * np.tanh(t / t0_), '--', color=c, lw=0.8)
        ax[0, 1].plot(t / t0_, Sm / S0, color=c, label=lab)
        print(f"  gammaT={gT:6.3f} gamma={g:5.3f} ({len(S)} traj, t_f={t_f:g}): "
              f"S0 = {S0:.3f} +- {perr[0]:.3f}, t0 = {t0_:.2f} +- {perr[1]:.2f}, "
              f"<S>(last quarter) = {Sm[late].mean():.3f}, <E>(last quarter) = "
              f"{E[:, late].mean():.2f}, rise over last eighth = {100*rise:.1f}%", flush=True)
    x = np.linspace(0, max(ax[0, 1].get_xlim()[1], 1), 200)
    ax[0, 1].plot(x, np.tanh(x), 'r:', lw=1.5, label='tanh')
    ax[0, 1].set_xlim(0, 10)
    rows = np.array(rows)
    # A curve not yet saturated at its t_f has no measured S0: the fit extrapolates
    # (it can even land above ln D), and its t0 is correspondingly soft. Those
    # are drawn open in (c), (d); curves still linear (t0 > t_f) are left out.
    noisy = (rows[:, 0] > 0) & (rows[:, 4] < rows[:, 10])
    sat = noisy & (rows[:, 9] > 0)
    unsat = noisy & ~sat
    for gT in rows[(rows[:, 0] > 0) & ~noisy, 0]:
        print(f"  gammaT={gT:g}: still linear at t_f (t0 > t_f), left out of (c), (d)")
    for gT in rows[unsat, 0]:
        print(f"  gammaT={gT:g}: not saturated by t_f (<S> still rising > 1% over the last "
              f"eighth), S0 and t0 are extrapolated -- open markers")
    for j, col in ((0, 2), (1, 4)):
        ax[1, j].errorbar(rows[noisy, 0], rows[noisy, col], rows[noisy, col + 1],
                          fmt='-', color='C0', alpha=0.4)
        ax[1, j].plot(rows[sat, 0], rows[sat, col], 'o', color='C0', label='saturated')
        ax[1, j].plot(rows[unsat, 0], rows[unsat, col], 'o', mfc='white', color='C0',
                      label='not saturated by t_f (extrapolated)')
        if gamma is not None:
            ax[1, j].set_xscale('log')
    if rows[0, 0] == 0:
        ax[1, 0].axhline(rows[0, 2], color='k', ls=':', label='noiseless')
        ax[1, 1].axhline(rows[0, 4], color='k', ls=':', label='noiseless')
    ax[1, 0].axhline(np.log(D), color='r', ls='--', lw=0.8, label='ln D')
    n_traj = int(rows[rows[:, 0] > 0, 8].max())
    fixed = f'$\\gamma$ = {gamma:g}' if gamma is not None else f'T = {T}'
    ax[0, 0].set(xlabel='t', ylabel='half-chain S (nats)',
                 title=f'(a) N = {N}, D = {D}, {fixed}, {n_traj} traj., '
                       f'{"/".join(sorted(schemes))}')
    ax[0, 1].set(xlabel='t / t0', ylabel='S / S0', title='(b) collapse')
    ax[1, 0].set(xlabel='$\\gamma T$', ylabel='S0', title='(c) saturation S0')
    ax[1, 1].set(xlabel='$\\gamma T$', ylabel='t0', title='(d) timescale t0')
    ax[0, 0].legend(fontsize=7), ax[0, 1].legend(fontsize=7)
    ax[1, 0].legend(fontsize=7), ax[1, 1].legend(fontsize=7)
    # columns: gammaT, gamma, S0, dS0, t0, dt0, <S>late, <E>late, n_traj, saturated, t_f
    save(f'{prefix}_N{N}_D{D}_fits', fig, fits=rows)
    os.replace(os.path.join(FIGS, f'{prefix}_N{N}_D{D}_fits.png'),
               os.path.join(FIGS, f'{prefix}_N{N}_D{D}.png'))


def plot_D_comparison(prefix='fig1_point', N=16, Ds=(32, 64)):
    """
    One gammaT point at several D (runs/{prefix}_N{N}_D{D}.npz from friction_scan):
    noisy ensemble mean against the noiseless run at each D. A plateau set by the
    bath would show as noisy curves that stop short of their own noiseless
    ceiling and agree across D; a variational plateau tracks each ceiling.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for D, c in zip(Ds, ('C0', 'C1', 'C2', 'C3')):
        with np.load(os.path.join(RUNS, f'{prefix}_N{N}_D{D}.npz')) as z:
            keys = [k for k in z.files if k.startswith('S_gT') and k != 'S_gT0.0']
            for k in keys:
                t, S = z['t_gT' + k[4:]], z[k]
                m, e = S.mean(0), S.std(0) / np.sqrt(len(S))
                ax.plot(t, m, color=c, label=f'D = {D}, $\\gamma T$ = {k[4:]} ({len(S)} traj.)')
                ax.fill_between(t, m - e, m + e, color=c, alpha=0.25)
            ax.plot(z['t_gT0.0'], z['S_gT0.0'][0], ':', color=c, label=f'D = {D}, noiseless')
            scheme = str(z['scheme']) if 'scheme' in z.files else 'heun'
    ax.set(xlabel='t', ylabel='half-chain S (nats)', title=f'N = {N}, {scheme} scheme')
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIGS, f'{prefix}_N{N}_D_comparison.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  -> {os.path.relpath(out, HERE)}")


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('demos', nargs='*', default=['all'])
    ap.add_argument('--state', choices=('random', 'plus'), default='random')
    ap.add_argument('--state-seed', type=int, default=0)
    ap.add_argument('--N', type=int, default=16, help='friction_scan only')
    ap.add_argument('--D', type=int, default=32, help='friction_scan only')
    ap.add_argument('--traj', type=int, default=16, help='friction_scan only')
    ap.add_argument('--tf', type=float, nargs='+', default=[40.0],
                    help='friction_scan only: one value, or one per gammaT')
    ap.add_argument('--dt', type=float, nargs='+', default=[0.04],
                    help='friction_scan only: one value, or one per gammaT')
    ap.add_argument('--T', type=float, default=0.2, help='friction_scan only: fixed T')
    ap.add_argument('--gamma', type=float, default=None,
                    help='friction_scan only: fix gamma instead of T')
    ap.add_argument('--prefix', default='friction_scan',
                    help='friction_scan only: output name (keep scans that should not merge apart)')
    ap.add_argument('--processes', type=int, default=21, help='friction_scan only')
    ap.add_argument('--gammaTs', type=float, nargs='+',
                    default=[0.02, 0.05, 0.08, 0.11, 0.15, 0.25], help='friction_scan only')
    ap.add_argument('--no-noiseless', action='store_true', help='friction_scan only')
    ap.add_argument('--tag', default='', help='friction_scan only: suffix of the raw npz')
    ap.add_argument('--scheme', choices=('paper', 'euler', 'heun'), default='heun',
                    help='friction_scan only: integration scheme (see integrator.py)')
    args = ap.parse_args()
    STATE.update(kind=args.state, seed=args.state_seed)
    demos = dict(saturation=demo_saturation, noise=demo_noise,
                 friction=demo_friction, schemes=demo_schemes,
                 friction_scan=lambda: demo_friction_scan(
                     N=args.N, D=args.D, T=args.T, gamma=args.gamma, prefix=args.prefix,
                     n_traj=args.traj, t_f=args.tf if len(args.tf) > 1 else args.tf[0],
                     dt=args.dt if len(args.dt) > 1 else args.dt[0],
                     processes=args.processes, gammaTs=args.gammaTs,
                     noiseless=not args.no_noiseless, tag=args.tag, scheme=args.scheme),
                 friction_scan_compare_D=lambda: plot_D_comparison(prefix=args.prefix, N=args.N),
                 friction_scan_plot=lambda: plot_friction_scan(
                     N=args.N, D=args.D, T=args.T, gamma=args.gamma, prefix=args.prefix))
    for name, fn in demos.items():
        # 'all' means the four quick demos; the scan is run by name
        if ('all' in args.demos and not name.startswith('friction_scan')) or name in args.demos:
            t0 = time.time()
            fn()
            print(f"  ({time.time() - t0:.0f}s)\n", flush=True)
