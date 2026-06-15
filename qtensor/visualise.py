import qtensor.states as states
import qtensor.operators as ops
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.optimize import curve_fit


def plot_energy_density(state, H_terms, ax=None, **plot_kwargs):
    if not ax:            
        _, ax = plt.subplots(1,1)
    energy_profile = [np.real(ops.local_expect(state, H_terms[i]))
                      for i in sorted(H_terms.keys())]
    ax.plot(list(state.sites)[:-1], energy_profile, **plot_kwargs)
    ax.set_ylabel(r'$E$')
    ax.set_xlabel('Bond')

    return energy_profile, ax

def plot_energy_density_evolution(state_history, H_terms,
                                  t_f=None, block_len=1, ax=None, bond=None,
                                  expectation_func=None):
    """
    Plot the time evolution of the state and the expectations.
    """
    times = sorted(np.abs(list(state_history.keys())))
    if t_f:
        times = [t for t in times if t <= t_f and t]
        times = [times[i] for i in range(len(times)) if i % block_len == 0]
    
    if not ax:
        fig, ax = plt.subplots(3, 1, figsize=(8, 6), height_ratios=[2,1,1])

    sites = range(len(H_terms))
    if not bond:
        bond = len(sites)//2

    if not expectation_func:
        expectation_func = lambda state, term: np.real(ops.local_expect(state, term))

    bond_energy=[]
    
    cmap = mpl.colormaps['magma']

    fig.suptitle("Energy Density Evolution")

    ax[0,].set_xlabel("Bond")
    ax[0,].set_ylabel(r'$E$')
    E_profiles = []
    t_colors = []
        
    for t in times:
        local_energy = [expectation_func(state_history[t], H_terms[site])
                        for site in H_terms]
        E_profiles.append(np.row_stack([sites, local_energy]).T)
        t_colors.append(cmap(t/max(times)))
        bond_energy.append(local_energy[bond])

    line_collection = LineCollection(E_profiles,
                                     array=times,
                                     cmap='viridis')
    
    ax[0,].add_collection(line_collection)
    ax[0,].set_xlim(1, len(H_terms)-2)
    ax[0,].set_xticks(sites)
    ax[0,].set_ylim(np.min(np.array(E_profiles)[:,:,1]),
                    np.max(np.array(E_profiles)[:,:,1]))

    ax[1,].set_ylabel(fr'$E$ at site {bond}')    
    ax[1,].plot(times, bond_energy)

    ax[2,].set_xlabel("Time")
    ax[2,].set_ylabel(r'$dE/dt$')
    dE_dt = [(bond_energy[i+1] - bond_energy[i])/
            (times[i+1] - times[i])
            for i in range(len(bond_energy)-1)]
    
    ax[2,].plot(times[:-1], dE_dt)
           
    plt.colorbar(line_collection, ax=ax[0,], label='Time')
    
    plt.show()

    return times, E_profiles

def plot_spin_components_spatial(state):
    fig, ax = plt.subplots(1,1)
    sites = sorted(state.sites)
    for p in ['x', 'y', 'z']:
        paulis = ops.pauli_at_sites(sites, p)
        expects = [np.real(ops.local_expect(state, paulis[site]))
                   for site in sites]
        ax.plot(sites, expects, label=rf'$\sigma^{{{p}}}$')
    ax.set_ylabel(r'$\langle \sigma^{i} \rangle$')
    ax.set_xlabel('Site')
    ax.legend()

def plot_entropy_evolution(state_history, 
                           site=None, t_f=None, block_len=None, ax=None,
                           show_max=False, show_est=False, entropy='Renyi2'):
    """
    Finds and plots part-chain entropy as a function of time

    Parameters:
    state_history: dict of (time: state) pairs
    site: site to calculate entropy to the right of. If None, uses centre of chain.

    Returns:
    entropies: list of entropies
    times: list of times for which entropies were calculated

    """
    if not ax: 
        _, ax = plt.subplots(1,1)
    times = np.array(sorted(np.abs(list(state_history.keys()))))
    if block_len:
        times = times[::block_len]
    if t_f:
        times = times[times <= t_f]
    if not site:
        site = max(state_history[times[0]].sites)//2    
    if entropy == 'Renyi2':
        entropies = [states.entropy(state_history[t], site) for t in times]
    elif entropy == 'VonNeumann':
        entropies = [states.vn_entropy(state_history[t], site) for t in times]
    ax.plot(times, entropies, label=f'Site {site}')
    ax.set_title(f'Entanglement Entropy at site {site}')
    ax.set_ylabel(r'$S_2$')
    ax.set_xlabel('Time')
    if show_max:
        max_ent = np.log(np.max(state_history[max(times)][site].shape))
        ax.axhline(max_ent, color='red', linestyle='--', label='Max Entropy')
    # if show_est:
    #     # Entropy of random state from 
    #     est_ent = np.log2(times*2+1)
    #     ax.plot(times, est_ent, color='green', linestyle='--', label='Estimated Entropy')
    
    ax.legend()
    return entropies, times, ax

def plot_bond_dimension(state, ax=None, c=(1,0,0)):
    """
    Plots the bond dimension of the state as a function of bond index.
    """
    if not ax:
        fig, ax = plt.subplots(1,1)
    sites = sorted(state.sites)
    bond_dims = [state[site].shape[1] for site in sites[1:]]
    ax.plot(sites[1:], bond_dims, color=c)
    ax.set_xlabel('Bond')
    ax.set_ylabel(r'Bond Dimension')
    return fig, ax

def plot_bond_dimension_history(state_history, t_f=None, block_len=None, ax=None):
    if not ax:
        fig, ax = plt.subplots(1,1)
    times = sorted(np.abs(list(state_history.keys())))
    if block_len:
        times = [times[i] for i in range(len(times)) if i % block_len == 0]
    if t_f:
        times = [t for t in times if t <= t_f and t]
    bond_dims = [np.column_stack([[site for site in sorted(state_history[t].sites)[1:]],
                                  [state_history[t][site].shape[1] for site in sorted(state_history[t].sites)[1:]]
                                  ])
                 for t in times]
    ax.set_xlim(1, len(bond_dims[0][:,0]))
    ax.set_ylim(0, max([max(bond_dim[:,1]) for bond_dim in bond_dims])*1.1)
    line_collection = LineCollection(bond_dims,
                                    #  array=times,
                                     cmap='viridis')
    ax.add_collection(line_collection)
    ax.set_xlabel('Bond')
    ax.set_ylabel(r'Bond Dimension')
    plt.colorbar(line_collection, ax=ax, label='Time')
    plt.show()

def overlap_evolution(state_hist_1, state_hist_2):
    """
    Compare the evolution of two states via the overlap |<ψ(t)|φ(t)>|^2 

    Useful for comparing different evolution methods.
    """
    t_1 = state_hist_1.keys()
    t_2 = state_hist_2.keys()
    times = sorted(list(set(t_1).intersection(set(t_2))))
    assert len(times)!= 0 , "No shared times in state histories"
    overlaps = [np.abs(states.overlap(state_hist_1[t], state_hist_2[t])) for t in times]
    fig, ax = plt.subplots(1,1)
    ax.plot(times, overlaps)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$ | \langle \psi | \phi \rangle | $')
    return fig, ax

def observable_evolution(history, obs,
                         t_f=None, block_len=1,
                         axs=None, color=None, label=None):
    """
    Plot the evolution of an observable for a single history.
    """
    times = np.array(sorted(history.keys()))[::block_len]

    # Create axes if not provided
    if axs is None:
        fig, axs = plt.subplots(2, 1)
        axs[0].set_ylabel(r'$\langle \hat O \rangle$')
        axs[1].set_ylabel(r'$\frac{d}{dt} \langle \hat O \rangle$')
        axs[1].set_xlabel(r'time')

    # Apply time cutoff
    if t_f is not None:
        times = times[times < t_f]

    # Compute expectation values
    expects = np.array([ops.local_expect(history[t], obs) for t in times])

    # Compute finite difference derivative
    dt = times[1] - times[0]
    rates = (expects[1:] - expects[:-1]) / dt

    # Plot
    axs[0].plot(times, expects, color=color, label=label)
    axs[1].plot(times[:-1], rates, color=color)

def compare_obs_evolution(histories, obs, labels=None, **kwargs):
    """
    Plot observable evolution for multiple histories on shared axes.
    Any keyword arguments are forwarded to observable_evolution.
    """
    cmap = mpl.colormaps['magma']

    # Create shared axes once
    fig, axs = plt.subplots(2, 1, figsize=(8,8))
    axs[0].set_ylabel(r'$\langle \hat O \rangle$')
    axs[1].set_ylabel(r'$\frac{d}{dt} \langle \hat O \rangle$')
    axs[1].set_xlabel(r'time')
    if 'title' in kwargs:
        title = kwargs.pop('title')
        fig.suptitle(title)


    # Forward axes to inner function
    kwargs = dict(kwargs)  # copy so we can modify safely
    kwargs['axs'] = axs

    if not labels:
        labels = [None] * len(histories)

    # Plot each history
    for i, hist in enumerate(histories):
        label = labels[i]
        color = cmap(i / len(histories))
        observable_evolution(hist, obs, color=color, label=label, **kwargs)
    axs[0].legend()


def estimate_D(times, E_profiles, site=-1, t_f=None, avg_from=False):
    """
    Takes the outputs of visualise.plot_energy_density_evolution
    Returns an estimate of the diffusion constant by considering the curvature at (site)
    if site==-1, uses site at centre of chain
    """
    if site==-1:
        L = E_profiles[0].shape[0]
        site = L//2

    if t_f:
        times = np.array(times)
        times = times[times<t_f]
    curvature = {}
    rate = {}
    D = {}
    for i in range(len(times)-1):
        t1, t2 = times[i], times[i+1]
        El, Ec, Er = [E_profiles[i][site+j, 1] for j in [-1, 0, 1]]
        El2, Ec2, Er2 = [E_profiles[i+1][site+j, 1] for j in [-1, 0, 1]]
        grad_sq_E = El + Er - 2*Ec
        grad_sq_E2 = El2 + Er2 - 2*Ec2
        E_bar = (El+Ec+Er)/3
        E2_bar = (El2+Ec2+Er2)/3
        curvature[t1] = (grad_sq_E + grad_sq_E2) / 2
        rate[t1] = (E2_bar - E_bar) / (t2 - t1)
        D[t1] = rate[t1] / curvature[t1]

    fig, ax = plt.subplots(1,1)
    ax.set_xlabel('time')
    ax.set_ylabel('D')
    ax.set_title(f"Estimated diffusion constant at site {site}")
    ax.plot(D.keys(), D.values())
    if avg_from:
        t_from = times[times>avg_from]
        D_avg = [D[t_from]]
        for t in t_from[1:]:
            D_avg_new = (D_avg[-1]*len(D_avg) + D[t]) / (len(D_avg)+1)
            D_avg.append(D_avg_new)
        ax.plot(t_from, D_avg, linestyle='dashed')


    return D


def gaussian(x, amplitude, mu, sigma, offset):
    """
    Gaussian function: offset + amplitude * exp(- (x - mu)^2 / (2 * sigma^2))
    """
    return offset + amplitude * np.exp(- (x - mu)**2 / (2 * sigma**2))


def fit_gaussian(E_profile, fixed_mu=None, fixed_offset=None):
    """
    Fits a Gaussian to a single energy profile.

    Parameters:
    E_profile: 2D array with shape (n_sites, 2), where [:,0] are sites, [:,1] are energies.
    fixed_mu: optional fixed value for the mean (center) of the Gaussian.
    fixed_offset: optional fixed value for the baseline offset.

    Returns:
    dict with keys: 'amplitude', 'mu', 'sigma', 'offset', 'success' (bool), and optionally 'pcov' if successful.
    """
    x = E_profile[:, 0]
    y = E_profile[:, 1]

    # Initial guesses
    offset_guess = np.mean(y) if fixed_offset is None else fixed_offset
    amplitude_guess = np.max(y) - np.min(y)
    mu_guess = len(x)/2 if fixed_mu is None else fixed_mu
    sigma_guess = (x[-1] - x[0]) / 4  # rough guess

    # Build parameter list and function based on what's fixed
    if fixed_mu is not None and fixed_offset is not None:
        # Only fit amplitude and sigma
        def gaussian_func(x, amp, sig):
            return gaussian(x, amp, fixed_mu, sig, fixed_offset)
        p0 = [amplitude_guess, sigma_guess]
        
    elif fixed_mu is not None:
        # Fit amplitude, sigma, and offset
        def gaussian_func(x, amp, sig, offset):
            return gaussian(x, amp, fixed_mu, sig, offset)
        p0 = [amplitude_guess, sigma_guess, offset_guess]
        
    elif fixed_offset is not None:
        # Fit amplitude, mu, and sigma
        def gaussian_func(x, amp, mu, sig):
            return gaussian(x, amp, mu, sig, fixed_offset)
        p0 = [amplitude_guess, mu_guess, sigma_guess]
        
    else:
        # Fit all parameters
        gaussian_func = gaussian
        p0 = [amplitude_guess, mu_guess, sigma_guess, offset_guess]

    try:
        popt, pcov = curve_fit(gaussian_func, x, y, p0=p0, maxfev=10000)
        
        # Reconstruct full parameter vector
        if fixed_mu is not None and fixed_offset is not None:
            amplitude, sigma = popt
            mu, offset = fixed_mu, fixed_offset
        elif fixed_mu is not None:
            amplitude, sigma, offset = popt
            mu = fixed_mu
        elif fixed_offset is not None:
            amplitude, mu, sigma = popt
            offset = fixed_offset
        else:
            amplitude, mu, sigma, offset = popt
            
        return {
            'amplitude': amplitude,
            'mu': mu,
            'sigma': abs(sigma),  # sigma should be positive
            'offset': offset,
            'success': True,
            'pcov': pcov
        }
    except Exception as e:
        print(f"Fit failed: {e}")
        return {
            'amplitude': None,
            'mu': fixed_mu if fixed_mu is not None else None,
            'sigma': None,
            'offset': fixed_offset if fixed_offset is not None else None,
            'success': False,
            'error': str(e)
        }


def fit_gaussians(times, E_profiles, fixed_mu=None, fixed_offset=None):
    """
    Fits Gaussians to energy profiles over time.

    Parameters:
    times: list of times corresponding to E_profiles
    E_profiles: list of 2D arrays, each with shape (n_sites, 2)
    fixed_mu: optional fixed value for the mean (center) of the Gaussian (applied to all fits)
    fixed_offset: optional fixed value for the baseline offset (applied to all fits)

    Returns:
    list of dicts, each containing fit parameters for each time.
    Each dict has keys: 'time', 'amplitude', 'mu', 'sigma', 'offset', 'success'
    """
    fits = []
    for t, profile in zip(times, E_profiles):
        fit_result = fit_gaussian(profile, fixed_mu=fixed_mu, fixed_offset=fixed_offset)
        fit_result['time'] = t
        fits.append(fit_result)
    return fits


def plot_gaussian_fit(E_profile, fit_result, ax=None, label=None, n_points=200, plot_data=True, color=None):
    """
    Plot a single energy profile together with its Gaussian fit.

    Parameters:
    E_profile: 2D array with shape (n_sites, 2).
    fit_result: fit dictionary returned by fit_gaussian.
    ax: optional matplotlib axis.
    label: optional label for the curve.
    n_points: number of points used to draw the fitted Gaussian.
    plot_data: whether to plot the raw energy profile points.
    color: optional color for the fit line (for use in colored plotting sequences).
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)

    x = E_profile[:, 0]
    y = E_profile[:, 1]

    if plot_data:
        ax.plot(x, y, 'o',
                 label=f'Profile{f": {label}"}' if label else None,
                 color=color, alpha=0.5)

    if fit_result is None or not fit_result.get('success', False):
        return ax

    x_fit = np.linspace(np.min(x), np.max(x), n_points)
    y_fit = gaussian(x_fit,
                     fit_result['amplitude'],
                     fit_result['mu'],
                     fit_result['sigma'],
                     fit_result['offset'])
    
    ax.plot(x_fit, y_fit, '-', label=f'Gaussian fit: {label}' if label else None, 
            color=color, linewidth=2)
    ax.set_xlabel('Site')
    ax.set_ylabel(r'$E$')
    if label is not None:
        ax.set_title(f'Gaussian fit at time {label}')
    return ax


def plot_gaussian_fits(times, E_profiles, fits=None, ax=None, max_plots=None, plot_data=True, cmap='viridis', fixed_mu=None, fixed_offset=None):
    """
    Plot multiple energy profiles with their Gaussian fits, colored by time.

    Parameters:
    times: iterable of times.
    E_profiles: iterable of 2D arrays (sites, energies).
    fits: optional list of fit dictionaries from fit_gaussians. If None, fit_gaussians is called.
    ax: optional matplotlib axis.
    max_plots: maximum number of curves to plot for clarity.
    plot_data: whether to plot the raw profile points.
    cmap: colormap name to use for time-based coloring.
    fixed_mu: optional fixed value for the mean (center) of the Gaussian (passed to fit_gaussians if fits=None)
    fixed_offset: optional fixed value for the baseline offset (passed to fit_gaussians if fits=None)

    Returns:
    fig, ax, fits
    """
    if fits is None:
        fits = fit_gaussians(times, E_profiles, fixed_mu=fixed_mu, fixed_offset=fixed_offset)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.figure

    # Convert times to numpy array for normalization
    times_arr = np.asarray(times)
    
    if max_plots is None:
        max_plots = len(times)

    # Get colormap
    colormap = mpl.colormaps[cmap]
    norm = mpl.colors.Normalize(vmin=np.min(times_arr), vmax=np.max(times_arr))
    
    plotted = 0
    for t, profile, fit_result in zip(times_arr, E_profiles, fits):
        if plotted >= max_plots:
            break
        # Color based on normalized time
        color = colormap(norm(t))
        plot_gaussian_fit(profile, fit_result, ax=ax, 
                         plot_data=plot_data, color=color)
        plotted += 1

    ax.set_xlabel('Site')
    ax.set_ylabel(r'$E$')
    ax.set_title('Gaussian fits to energy density profiles')
    
    # Add colorbar
    sm = mpl.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Time')
    
    ax.legend(fontsize=8, loc='best')

    return fig, ax, fits


def estimate_diffusion_constant_from_widths(times, sigmas, fit_intercept=True):
    """
    Estimate diffusion constant from the time evolution of Gaussian widths.

    For one-dimensional diffusion, a Gaussian width squared grows as:
      sigma^2(t) = sigma0^2 + 2 D t

    Parameters:
    times: iterable of times.
    sigmas: iterable of Gaussian standard deviations.
    fit_intercept: whether to fit an intercept term.

    Returns:
    dict with keys 'D', 'slope', 'intercept', 'times', 'sigma2'.
    """
    times = np.asarray(times, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    if times.shape != sigmas.shape:
        raise ValueError('times and sigmas must have the same shape')

    sigma2 = sigmas**2
    if fit_intercept:
        slope, intercept = np.polyfit(times, sigma2, 1)
    else:
        slope = np.dot(times, sigma2) / np.dot(times, times)
        intercept = 0.0

    return {
        'D': slope / 2.0,
        'slope': slope,
        'intercept': intercept,
        'times': times,
        'sigma2': sigma2
    }


def estimate_diffusion_constant_from_gaussian_fits(fits, fit_intercept=True):
    """
    Estimate diffusion constant directly from gaussian fit results.
    """
    times = []
    sigmas = []
    for fit_result in fits:
        if fit_result.get('success', False) and fit_result.get('sigma') is not None:
            times.append(fit_result['time'])
            sigmas.append(fit_result['sigma'])

    if len(times) == 0:
        raise ValueError('No successful gaussian fits available for diffusion estimation')

    return estimate_diffusion_constant_from_widths(times, sigmas, fit_intercept=fit_intercept)

