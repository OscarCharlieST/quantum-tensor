import qtensor.operators as ops
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def plot_energy_density(state, H_terms):
    fig, ax = plt.subplots(1,1)
    energy_density = [ops.local_expect(state, term) for term in H_terms]
    ax.plot(list(state.sites)[:-1], energy_density)
    ax.set_ylabel(r'$E$')
    ax.set_xlabel('Bond')

def plot_energy_density_evolution(state_history, H_terms, t_f=None):
    """
    Plot the time evolution of the state and the expectations.
    """
    times = np.real(sorted(list(state_history.keys())))
    if t_f:
        times = [t for t in times if t <= t_f]
    sites = range(len(H_terms))
    middle_energy=[]
    
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), height_ratios=[2,1,1])
    cmap = mpl.colormaps['magma']

    fig.suptitle("Energy Density Evolution")

    ax[0,].set_xlabel("Bond")
    ax[0,].set_ylabel(r'$E$')
    E_profiles = []
    t_colors = []
        
    for t in times:
        local_energy = [np.real(ops.local_expect(state_history[t], term))
                        for term in H_terms]
        E_profiles.append(np.row_stack([sites, local_energy]).T)
        t_colors.append(cmap(t/max(times)))
        middle_energy.append(local_energy[len(local_energy)//2])

    line_collection = LineCollection(E_profiles,
                                     array=times,
                                     cmap='viridis')
    
    ax[0,].add_collection(line_collection)
    ax[0,].set_xlim(1, len(H_terms)-2)
    ax[0,].set_xticks(sites)
    ax[0,].set_ylim(np.min(np.array(E_profiles)[:,:,1]),
                    np.max(np.array(E_profiles)[:,:,1]))

    ax[1,].set_ylabel(r'central $E$')    
    ax[1,].plot(times, middle_energy)

    ax[2,].set_xlabel("Time")
    ax[2,].set_ylabel(r'$dE/dt$')
    dE_dt= [(middle_energy[i+1] - middle_energy[i])/
            (times[i+1] - times[i])
            for i in range(len(middle_energy)-1)]
    
    ax[2,].plot(times[:-1], dE_dt)
           
    plt.colorbar(line_collection, ax=ax[0,], label='Time')
    
    plt.show()

