"""
Eyeball diagnostics for the tangent-space relaxation calculation: where the
spectral weight of P O|psi*> sits on the spectrum of P H P, and what the
response function it produces does as a function of time.

Both functions are standalone consumers of what `run_relaxation_scan.run_one`
already computes -- nothing here re-runs the physics. The thin
`*_from_result` wrappers just unpack a result dict:

    import pickle, lyapunov.relaxation.plots as plots
    results = pickle.load(open('lyapunov/relaxation/scan_results.pkl','rb'))['results']
    plots.spectral_weights_from_result(results[-1], 'z_mid')
    plots.response_from_result(results[-1], 'z_mid')

Or, straight out of a live run:

    r = run_one(8)
    plots.plot_spectral_weights(r['omega'], r['observables']['z_mid']['weights'])

`plot_timescale_scan` is the one function here that wants the whole scan
rather than a single result: how tau behaves across L, whether it sits
inside the window where a rate means anything, and the ratio between two
observables. From the CLI:

    python lyapunov/relaxation/plots.py --pickle lyapunov/relaxation/scan_results.pkl \
        --L 16 --obs current_mid --scan
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Same repo-root-on-the-path convention as the notebooks and
# run_relaxation_scan, but derived from __file__ rather than the cwd, so that
# this file also works when run directly as a script.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import lyapunov.relaxation.response as resp


# Fixed slots, assigned by role rather than cycled, so the same thing is the
# same colour in both figures: measured data, fitted model, a secondary
# series, and recessive reference lines.
C_DATA = '#2a78d6'
C_MODEL = '#eb6834'
C_AUX = '#1baf7a'
C_REF = '#8a8a85'


def _finite(x):
    return x is not None and np.isfinite(x)


def _annotate(ax, lines):
    ax.text(0.98, 0.95, '\n'.join(lines), transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color='#444444',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#d8d8d4'))


def _legend(ax, **kwargs):
    # Faint white backing rather than no frame: several of these panels have
    # a curve running through the corner where the legend has to sit.
    kwargs.setdefault('loc', 'upper right')
    return ax.legend(fontsize=8, facecolor='white', edgecolor='none',
                     framealpha=0.85, **kwargs)


def _recede(ax):
    ax.grid(True, lw=0.5, color='#e6e6e2')
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#c8c8c4')


# ------------------------------------------------------------- spectral side

def plot_spectral_weights(omega, weights, scales=None, tau=None, bins=61,
                          omega_range=None, axes=None, title=None):
    """
    The distribution of |<k|O|psi*>|^2 over the eigenvalues of the projected
    Hamiltonian -- i.e. the spectral function A_O(omega) that decides the
    decay law.

    Three panels sharing the omega axis:

    1. A_O(omega) as a weight density, against the bare density of states
       rho(omega) on the same axis. Both are normalized to unit area, so
       they are directly comparable: if the weight tracks the DOS the
       observable is featureless, and a peak means it picks out a band of
       modes. If `tau` is given, the Lorentzian of half-width 1/tau implied
       by that relaxation time is overlaid -- exponential decay *is* a
       Lorentzian lineshape, so the overlay is the direct test of whether
       the fitted tau means anything.
    2. Per-mode weight on a log scale. This is where the discreteness shows:
       a smooth cloud is the ETH-like smoothness the continuum limit needs,
       while a few isolated spikes mean the response is a handful of phases
       and will never decay.
    3. Cumulative weight fraction, which makes the concentration
       quantitative -- compare the width of its rise against the n_eff in
       the annotation.

    Parameters
    ----------
    omega, weights : arrays
        `result['omega']` and `result['observables'][name]['weights']`.
    scales : dict, optional
        `result['observables'][name]['scales']`; recomputed from the
        spectrum via response.timescales if omitted.
    tau : float, optional
        Relaxation time for the Lorentzian overlay, e.g. `obs['tau_fit']`.
    bins : int
    omega_range : (lo, hi), optional
        Restrict the plotted window. Weights outside it are still counted in
        the normalization and reported in the annotation, so the fraction
        shown is honest.
    axes : sequence of 3 matplotlib axes, optional
    title : str, optional

    Returns
    -------
    (fig, axes)
    """
    omega = np.asarray(omega, dtype=float)
    weights = np.asarray(np.real(weights), dtype=float)
    if scales is None:
        scales = resp.timescales(omega, weights)

    p = weights / weights.sum()
    order = np.argsort(omega)
    omega_s, p_s = omega[order], p[order]

    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(7.5, 8), sharex=True,
                                 height_ratios=[2, 2, 1])
    else:
        fig = np.asarray(axes).ravel()[0].figure
    ax_d, ax_w, ax_c = np.asarray(axes).ravel()[:3]

    lo, hi = omega_range if omega_range else (omega.min(), omega.max())
    edges = np.linspace(lo, hi, bins + 1)
    width = edges[1] - edges[0]
    centres = 0.5 * (edges[:-1] + edges[1:])

    # Both normalized to unit area over the *whole* spectrum, so the two
    # curves can share one y axis and one meaning: probability density in
    # omega.
    a_omega = np.histogram(omega, bins=edges, weights=p)[0] / width
    dos = np.histogram(omega, bins=edges)[0] / (omega.size * width)

    ax_d.bar(centres, a_omega, width=width * 0.92, color=C_DATA,
             linewidth=0, label=r'$A_O(\omega)$ (weight density)')
    ax_d.step(centres, dos, where='mid', color=C_AUX, lw=2,
              label=r'$\rho(\omega)$ (density of states)')
    if _finite(tau) and tau > 0:
        gamma = 1.0 / tau
        w_fine = np.linspace(lo, hi, 800)
        lorentz = (gamma / np.pi) / ((w_fine - scales['mean']) ** 2 + gamma ** 2)
        ax_d.plot(w_fine, lorentz, color=C_MODEL, lw=2, ls='--',
                  label=rf'Lorentzian, HWHM $1/\tau = {gamma:.3g}$')
    # Scale to the measured densities; a narrow Lorentzian is allowed to run
    # off the top rather than flattening the data it is being compared to.
    ax_d.set_ylim(0, 1.15 * max(a_omega.max(), dos.max()))
    ax_d.set_ylabel('density (unit area)')
    _legend(ax_d, loc='upper left')
    _recede(ax_d)

    inside = (omega >= lo) & (omega <= hi)
    _annotate(ax_d, [
        f"modes = {omega.size}",
        f"$n_{{eff}}$ = {scales['n_eff']:.1f}",
        f"$\\sigma_\\omega$ = {scales['std']:.3g}",
        f"spacing = {scales['spacing']:.3g}",
        f"weight in view = {100 * p[inside].sum():.1f}%",
    ])

    nonzero = p_s > 0
    ax_w.semilogy(omega_s[nonzero], p_s[nonzero], ls='none', marker='o',
                  ms=3.5, mfc=C_DATA, mec='white', mew=0.4, alpha=0.85)
    uniform = 1.0 / max(omega.size, 1)
    ax_w.axhline(uniform, color=C_REF, lw=1, ls=':')
    ax_w.text(0.01, uniform, ' uniform', color=C_REF, fontsize=8,
              va='bottom', transform=ax_w.get_yaxis_transform())
    ax_w.set_ylabel(r'weight fraction $w_k / \sum w$')
    _recede(ax_w)

    ax_c.plot(omega_s, np.cumsum(p_s), color=C_DATA, lw=2,
              drawstyle='steps-post')
    ax_c.set_ylim(0, 1.02)
    ax_c.set_ylabel('cumulative')
    ax_c.set_xlabel(r'$\omega$')
    _recede(ax_c)

    ax_d.set_xlim(lo, hi)
    if title:
        ax_d.set_title(title, fontsize=10, loc='left')
    fig.tight_layout()
    return fig, (ax_d, ax_w, ax_c)


# ------------------------------------------------------------- response side

def plot_response(times, response, scales=None, tau_fit=None, t_fit_end=None,
                  tau_cross=None, axes=None, title=None, tau_max_factor=5.0,
                  t_zoom=None, floor=0.05):
    """
    The response function C(t), and whether it settles into anything.

    Three panels:

    1. C(t) linear over the *full* trace, with the 1/e level, the window
       actually used by the exponential fit shaded, and the fitted
       exponential extended past it. The Zeno and Heisenberg times are
       marked: the trace is only expected to look exponential between them,
       and past t_heis the wobble is recurrence, not physics.
    2. |C(t)| on a log axis, where an exponential is a straight line. The
       downward spikes are sign changes -- once they start, C is oscillating
       about zero rather than decaying.
    3. The running relaxation time tau(t) = -1/(d ln C/dt). This is the "has
       it settled down" panel: a genuine exponential regime is a *plateau*
       here and the fitted tau (dashed) should sit on it, while a tau(t)
       that drifts monotonically means the single-exponential fit is
       averaging over a decay that has no one rate.

    Panels 2 and 3 are zoomed to the decaying part of the trace rather than
    sharing panel 1's axis. run_one integrates out to 3*t_heis, which for
    these spectra is a couple of hundred times the decay time, so on the
    full axis the decay is a single pixel at the origin.

    Parameters
    ----------
    times, response : arrays
        `obs['times']` and `obs['response']` from run_one.
    scales : dict, optional
        `obs['scales']`; the t_zeno/t_heis markers are skipped if omitted.
    tau_fit, t_fit_end, tau_cross : float, optional
        `obs['tau_fit']`, `obs['t_fit_end']`, `obs['tau_cross']`. Each is
        allowed to be nan -- that is what run_one stores when there was no
        usable fit window -- and is then simply not drawn.
    axes : sequence of 3 matplotlib axes, optional
    title : str, optional
    tau_max_factor : float
        y limit of the running-tau panel, in units of tau_fit.
    t_zoom : float, optional
        End of the zoomed window for panels 2 and 3. Default: a few
        relaxation times, from whichever of t_fit_end / tau_cross / t_zeno /
        t_heis are available.
    floor : float
        Below this |C|, the running tau in panel 3 is masked out: past the
        floor C oscillates about zero and d ln C/dt is measuring the
        oscillation, not a decay. Matches the default floor of
        response.fit_relaxation_time.

    Returns
    -------
    (fig, axes)
    """
    times = np.asarray(times, dtype=float)
    C = np.asarray(np.real(response), dtype=float)

    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(7.5, 8),
                                 height_ratios=[2, 2, 1.5])
    else:
        fig = np.asarray(axes).ravel()[0].figure
    ax_lin, ax_log, ax_tau = np.asarray(axes).ravel()[:3]

    t_zeno = scales['t_zeno'] if scales else None
    t_heis = scales['t_heis'] if scales else None
    shade = _finite(t_zeno) and _finite(t_fit_end)

    if t_zoom is None:
        candidates = [t for t in (t_fit_end, 5 * tau_cross if _finite(tau_cross)
                                  else None, 20 * t_zeno if _finite(t_zeno)
                                  else None) if _finite(t)]
        t_zoom = max(candidates) if candidates else times[-1]
    t_zoom = float(np.clip(t_zoom, times[min(1, times.size - 1)], times[-1]))
    zoom = times <= t_zoom

    # --- panel 1: the trace itself
    ax_lin.axhline(0, color=C_REF, lw=0.8)
    ax_lin.axhline(1 / np.e, color=C_REF, lw=1, ls=':')
    ax_lin.text(0.99, 1 / np.e, '$1/e$ ', color=C_REF, fontsize=8,
                va='bottom', ha='right',
                transform=ax_lin.get_yaxis_transform())
    if shade:
        ax_lin.axvspan(t_zeno, t_fit_end, color=C_MODEL, alpha=0.10, lw=0,
                       label='fit window')
    ax_lin.plot(times, C, color=C_DATA, lw=1.8, label='$C(t)$')
    if _finite(tau_fit) and tau_fit > 0:
        ax_lin.plot(times, np.exp(-times / tau_fit), color=C_MODEL, lw=1.8,
                    ls='--', label=rf'$e^{{-t/\tau}}$, $\tau = {tau_fit:.3g}$')
    if _finite(tau_cross):
        ax_lin.plot([tau_cross], [1 / np.e], ls='none', marker='o', ms=7,
                    mfc='none', mec=C_AUX, mew=2,
                    label=rf'$\tau_{{1/e}} = {tau_cross:.3g}$')
    span = times[-1] - times[0]
    for t_mark, name in ((t_zeno, '$t_{Zeno}$'), (t_heis, '$t_{Heis}$')):
        if _finite(t_mark) and times[0] <= t_mark <= times[-1]:
            frac = (t_mark - times[0]) / span if span else 0.0
            ha = 'left' if frac < 0.08 else 'right' if frac > 0.92 else 'center'
            ax_lin.axvline(t_mark, color=C_REF, lw=1, ls='-.')
            ax_lin.text(t_mark, 0.97, ' ' + name + ' ', color=C_REF,
                        fontsize=8, ha=ha, va='top',
                        transform=ax_lin.get_xaxis_transform())
    ax_lin.set_ylabel('$C(t)$')
    ax_lin.set_xlabel('$t$   (full trace)')
    _legend(ax_lin)
    _recede(ax_lin)

    # --- panel 2: log |C| over the decay, where an exponential is a line
    nonzero = zoom & (np.abs(C) > 0)
    ax_log.semilogy(times[nonzero], np.abs(C[nonzero]), color=C_DATA, lw=1.4)
    if _finite(tau_fit) and tau_fit > 0:
        ax_log.semilogy(times[zoom], np.exp(-times[zoom] / tau_fit),
                        color=C_MODEL, lw=1.8, ls='--',
                        label=rf'$e^{{-t/\tau}}$, $\tau = {tau_fit:.3g}$')
        _legend(ax_log, loc='lower left')
    if shade:
        ax_log.axvspan(t_zeno, t_fit_end, color=C_MODEL, alpha=0.10, lw=0)
    bottom = np.abs(C[nonzero]).min() if nonzero.any() else 1e-4
    ax_log.set_ylim(max(bottom, 1e-4), 2.0)
    # A trace that only falls by a factor of a few spans less than a decade,
    # where matplotlib labels no ticks at all by default.
    ax_log.yaxis.set_minor_locator(ticker.LogLocator(subs=(2., 3., 5.)))
    ax_log.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%.3g'))
    ax_log.tick_params(axis='y', which='minor', labelsize=7)
    ax_log.set_ylabel('$|C(t)|$')
    ax_log.tick_params(labelbottom=False)
    ax_log.text(0.99, 0.95, f'zoom: $t \\leq {t_zoom:.3g}$', fontsize=8,
                color=C_REF, ha='right', va='top', transform=ax_log.transAxes)
    _recede(ax_log)

    # --- panel 3: does the decay rate settle?
    # Only meaningful while C is positive and still above the floor: below
    # it C oscillates about zero, and d ln C/dt then measures the
    # oscillation rather than any decay.
    good = C > floor
    tau_run = np.full_like(times, np.nan)
    if good.sum() > 3:
        dlogC = np.gradient(np.log(C[good]), times[good])
        with np.errstate(divide='ignore', invalid='ignore'):
            tau_run[good] = np.where(dlogC < 0, -1.0 / dlogC, np.nan)
    ax_tau.plot(times, tau_run, color=C_DATA, lw=1.6, label=r'$\tau(t)$')
    if _finite(tau_fit) and tau_fit > 0:
        ax_tau.axhline(tau_fit, color=C_MODEL, lw=1.8, ls='--',
                       label=rf'fitted $\tau = {tau_fit:.3g}$')
        ax_tau.set_ylim(0, tau_max_factor * tau_fit)
    elif np.isfinite(tau_run[zoom]).any():
        ax_tau.set_ylim(0, np.nanpercentile(tau_run[zoom], 95))
    if shade:
        ax_tau.axvspan(t_zeno, t_fit_end, color=C_MODEL, alpha=0.10, lw=0)
    ax_tau.set_ylabel(r'$-1 / (d\ln C/dt)$')
    ax_tau.set_xlabel('$t$   (zoom)')
    _legend(ax_tau)
    _recede(ax_tau)

    ax_lin.set_xlim(times[0], times[-1])
    ax_log.set_xlim(times[0], t_zoom)
    ax_tau.set_xlim(times[0], t_zoom)
    if title:
        ax_lin.set_title(title, fontsize=10, loc='left')
    fig.tight_layout()
    return fig, (ax_lin, ax_log, ax_tau)


# --------------------------------------------------------- across the L scan

# One colour per observable, fixed so an observable keeps its colour between
# the scan figure and the per-observable ones.
OBS_COLORS = {
    'current_mid': C_MODEL,
    'energy_mid': C_DATA,
    'z_mid': C_AUX,
    'x_mid': '#7b5cd6',
}
OBS_LABELS = {
    'current_mid': r'current $j_{\rm mid}$',
    'energy_mid': r'energy $h_{\rm mid}$',
    'z_mid': r'$z_{\rm mid}$',
    'x_mid': r'$x_{\rm mid}$',
}


def _obs_style(name, i=0):
    fallback = ['#7b5cd6', '#c2456e', '#2f8f8f'][i % 3]
    return OBS_COLORS.get(name, fallback), OBS_LABELS.get(name, name)


def plot_timescale_scan(results, names=None, axes=None, title=None,
                        reference=('energy_mid', 'current_mid'), r2_min=0.9):
    """
    How the relaxation times behave across the L scan, and whether they are
    trustworthy.

    The question this answers is not "what is tau" but "is tau a property of
    the system or of the box". Three panels:

    1. tau against L. The `1/e` crossing is drawn as the primary series and
       the exponential fit only where it is actually good (R^2 >= r2_min).
       That asymmetry is deliberate, not cosmetic: several of these
       responses fall fast and then crawl through a slow tail, which is not
       one exponential, and `fit_relaxation_time` then returns an R^2 of a
       few thousandths and a tau an order of magnitude off the crossing.
       The crossing needs no model and is stable across L where the fit is
       not, so it is the number to read. Rejected fits are counted in the
       annotation rather than hidden.
    2. Validity: tau against its own [t_zeno, t_heis] window. Exponential
       decay is only expected inside that window, so a tau sitting on or
       outside either edge is not a relaxation time -- it is the box.
    3. The ratio of the two `reference` observables, tau(energy)/tau(current).
       Both are *local* dephasing times at the fixed point, so this is the
       separation between the operator that carries energy and the density
       it carries -- not a hydrodynamic time, which would grow with L.
    """
    if names is None:
        names = list(results[0]['observables'])
    L = np.array([r['L'] for r in results], dtype=float)

    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    else:
        fig = axes[0].figure
    ax_tau, ax_win, ax_ratio = axes

    def series(name, key):
        return np.array([r['observables'][name].get(key, np.nan)
                         for r in results], dtype=float)

    # --- panel 1: the robust estimator, with good fits overlaid
    n_rejected = 0
    for i, name in enumerate(names):
        color, label = _obs_style(name, i)
        ax_tau.plot(L, series(name, 'tau_cross'), 'o-', ms=5, color=color,
                    label=label)
        fit, r2 = series(name, 'tau_fit'), series(name, 'r_squared')
        ok = np.isfinite(fit) & np.isfinite(r2) & (r2 >= r2_min)
        n_rejected += int((~ok).sum())
        if ok.any():
            ax_tau.plot(L[ok], fit[ok], 'x', ms=7, mew=1.6, color=color)
    ax_tau.set_xlabel('$L$')
    ax_tau.set_ylabel(r'$\tau$')
    ax_tau.set_title(rf'$\bullet$ $1/e$ crossing,  $\times$ fit '
                     rf'(only $R^2 \geq {r2_min:g}$)', fontsize=9, loc='left')
    if n_rejected:
        _annotate(ax_tau, [f'{n_rejected} of {len(names) * len(L)} fits',
                           f'rejected at $R^2 < {r2_min:g}$'])
    _legend(ax_tau, loc='upper left')
    _recede(ax_tau)

    # --- panel 2: is tau inside the window where a rate means anything?
    for i, name in enumerate(names):
        color, label = _obs_style(name, i)
        zeno = np.array([r['observables'][name]['scales']['t_zeno']
                         for r in results])
        heis = np.array([r['observables'][name]['scales']['t_heis']
                         for r in results])
        off = (i - (len(names) - 1) / 2) * 0.22
        ax_win.vlines(L + off, zeno, heis, color=color, lw=5, alpha=0.28)
        ax_win.plot(L + off, series(name, 'tau_cross'), 'o', ms=5, color=color,
                    label=label)
    ax_win.set_yscale('log')
    ax_win.set_xlabel('$L$')
    ax_win.set_ylabel(r'$t$')
    ax_win.set_title(r'bars: $[t_{\rm zeno}, t_{\rm heis}]$;  dots: $\tau$',
                     fontsize=9, loc='left')
    _legend(ax_win, loc='upper left')
    _recede(ax_win)

    # --- panel 3: the separation of scales that gates a diffusion fit
    slow, fast = reference
    if slow in names and fast in names:
        ratio = series(slow, 'tau_cross') / series(fast, 'tau_cross')
        ax_ratio.plot(L, ratio, 'o-', ms=6, color=C_MODEL,
                      label=r'from $1/e$ crossings')
        ax_ratio.axhline(1.0, color=C_REF, lw=1.2, ls=':')
        ax_ratio.text(0.98, 0.04, 'no separation of scales below this line',
                      fontsize=7.5, color=C_REF, ha='right', va='bottom',
                      transform=ax_ratio.transAxes)
        finite = np.isfinite(ratio)
        if finite.any():
            ax_ratio.set_ylim(0, max(1.6, 1.25 * np.nanmax(ratio[finite])))
    ax_ratio.set_xlabel('$L$')
    ax_ratio.set_ylabel(rf"$\tau$({_obs_style(slow)[1]}) / $\tau$({_obs_style(fast)[1]})")
    ax_ratio.set_title('separation of scales', fontsize=9, loc='left')
    _legend(ax_ratio, loc='upper left')
    _recede(ax_ratio)

    if title:
        ax_tau.figure.suptitle(title, fontsize=10, x=0.01, ha='left')
    fig.tight_layout()
    return fig, (ax_tau, ax_win, ax_ratio)


def timescale_scan_from_results(results, save_dir=None, prefix='', **kwargs):
    """plot_timescale_scan off a list of run_one results, optionally saved."""
    cfg = results[0]
    kwargs.setdefault(
        'title', f"relaxation times across the L scan   "
                 f"D={cfg['D']}, beta={cfg['beta']}")
    fig, axes = plot_timescale_scan(results, **kwargs)
    path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f'{prefix}timescale_scan.png')
        fig.savefig(path, dpi=140)
    return fig, axes, path


# ------------------------------------------------ unpackers for run_one dicts

def _result_title(result, name):
    return (f"{name}   L={result['L']}, D={result['D']}, "
            f"beta={result['beta']}, tangent dim={result.get('dim', '?')}")


def spectral_weights_from_result(result, name='z_mid', **kwargs):
    """plot_spectral_weights straight off a run_one result dict."""
    obs = result['observables'][name]
    kwargs.setdefault('title', _result_title(result, name))
    return plot_spectral_weights(result['omega'], obs['weights'],
                                 scales=obs['scales'], tau=obs['tau_fit'],
                                 **kwargs)


def response_from_result(result, name='z_mid', **kwargs):
    """plot_response straight off a run_one result dict."""
    obs = result['observables'][name]
    kwargs.setdefault('title', _result_title(result, name))
    return plot_response(obs['times'], obs['response'], scales=obs['scales'],
                         tau_fit=obs['tau_fit'], t_fit_end=obs['t_fit_end'],
                         tau_cross=obs['tau_cross'], **kwargs)


def plot_result(result, name='z_mid', save_dir=None, prefix='', **kwargs):
    """
    Both figures for one observable. With `save_dir`, writes
    <prefix><name>_spectrum.png and <prefix><name>_response.png there and
    returns the two paths alongside the figures.
    """
    fig_s, _ = spectral_weights_from_result(result, name, **kwargs)
    fig_r, _ = response_from_result(result, name, **kwargs)
    paths = []
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for fig, kind in ((fig_s, 'spectrum'), (fig_r, 'response')):
            path = os.path.join(save_dir, f'{prefix}{name}_{kind}.png')
            fig.savefig(path, dpi=140)
            paths.append(path)
    return (fig_s, fig_r), paths


# ------------------------------------------------------------------- CLI
#
# Run from the repo root:
#
#   python lyapunov/relaxation/plots.py --L 8 --D 8 --obs energy_mid --seed 0
#
# With no --pickle it re-runs run_relaxation_scan.run_one for that L (a few
# seconds at L=8, longer at 16); with --pickle it plots a scan already saved
# by run_relaxation_scan.main, no physics re-run. The seed matters because
# inf_T_thermofield's rank-seeding noise comes from the global numpy RNG, so
# runs are otherwise not reproducible.

def _cli(argv=None):
    import argparse
    import pickle

    parser = argparse.ArgumentParser(
        description='Spectral-weight and response diagnostics for one '
                    'observable of the tangent-space relaxation calculation.')
    parser.add_argument('--L', type=int, default=8,
                        help='system size (default 8)')
    parser.add_argument('--D', type=int, default=None,
                        help='bond dimension (default: run_relaxation_scan.D '
                             'when running; any D when reading --pickle)')
    parser.add_argument('--obs', default='energy_mid',
                        help="observable name from run_one: energy_mid (the "
                             "Hamiltonian term at the chain centre), z_mid, "
                             "x_mid, or 'all'")
    parser.add_argument('--pickle', default=None,
                        help='plot from a saved scan_results.pkl instead of '
                             'running; picks the entry matching --L (and --D '
                             'if given)')
    parser.add_argument('--scan', action='store_true',
                        help='with --pickle, also write the across-L '
                             'timescale comparison (all entries, not just --L)')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed the global numpy RNG so the run repeats')
    parser.add_argument('--save-dir', default=None,
                        help='directory for the PNGs (default: figures/ next '
                             'to this file)')
    parser.add_argument('--show', action='store_true',
                        help='open the figures instead of only saving them')
    args = parser.parse_args(argv)

    all_results = None
    if args.pickle:
        with open(args.pickle, 'rb') as f:
            all_results = pickle.load(f)['results']
        results = all_results
        matching = [r for r in results if r['L'] == args.L
                    and (args.D is None or r['D'] == args.D)]
        if not matching:
            raise SystemExit(f"no L={args.L}, D={args.D} in {args.pickle}; "
                             f"have (L, D) = "
                             f"{[(r['L'], r['D']) for r in results]}")
        result = matching[-1]
    else:
        import lyapunov.relaxation.run_relaxation_scan as scan
        if args.seed is not None:
            np.random.seed(args.seed)
        D = scan.D if args.D is None else args.D
        result = scan.run_one(args.L, D=D)
        scan.summarize(result)

    save_dir = args.save_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'figures')
    names = (list(result['observables']) if args.obs == 'all' else [args.obs])
    for name in names:
        _, paths = plot_result(result, name, save_dir=save_dir,
                               prefix=f"L{result['L']}_D{result['D']}_")
        for path in paths:
            print('wrote', path)

    if args.scan:
        if all_results is None or len(all_results) < 2:
            raise SystemExit('--scan needs a --pickle holding at least two L')
        _, _, path = timescale_scan_from_results(
            all_results, save_dir=save_dir,
            prefix=f"D{all_results[0]['D']}_")
        print('wrote', path)
    if args.show:
        plt.show()


if __name__ == '__main__':
    _cli()
