import qtensor.states as states
import qtensor.operators as ops
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def plot_energy_density(state, H_terms, ax=None):
    if not ax:            
        fig, ax = plt.subplots(1,1)
    energy_density = [ops.local_expect(state, H_terms[i]) 
                      for i in sorted(H_terms.keys())]
    ax.plot(list(state.sites)[:-1], energy_density)
    ax.set_ylabel(r'$E$')
    ax.set_xlabel('Bond')

    return fig, ax

def plot_energy_density_evolution(state_history, H_terms, t_f=None, block_len=1, ax=None, bond=None):
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

    bond_energy=[]
    
    cmap = mpl.colormaps['magma']

    fig.suptitle("Energy Density Evolution")

    ax[0,].set_xlabel("Bond")
    ax[0,].set_ylabel(r'$E$')
    E_profiles = []
    t_colors = []
        
    for t in times:
        local_energy = [np.real(ops.local_expect(state_history[t], H_terms[term]))
                        for term in H_terms]
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
    dE_dt= [(bond_energy[i+1] - bond_energy[i])/
            (times[i+1] - times[i])
            for i in range(len(bond_energy)-1)]
    
    ax[2,].plot(times[:-1], dE_dt)
           
    plt.colorbar(line_collection, ax=ax[0,], label='Time')
    
    plt.show()

    return fig, ax

def plot_spin_components_spatial(state):
    fig, ax = plt.subplots(1,1)
    sites = sorted(state.sites)
    for p in ['x', 'y', 'z']:
        paulis = ops.pauli_at_sites(sites, p)
        expects = [np.real(ops.local_expect(state, paulis[site]))
                   for site in sites]
        ax.plot(sites, expects, label=f'$\sigma^{p}$')
    ax.set_ylabel(r'$\langle \sigma^{i} \rangle$')
    ax.set_xlabel('Site')
    ax.legend()

def plot_entropy_evolution(state_history, 
                           site=None, t_f=None, block_len=None, ax=None, show_max=False):
    if not ax: 
        fig, ax = plt.subplots(1,1)
    times = sorted(np.abs(list(state_history.keys())))
    if block_len:
        times = [times[i] for i in range(len(times)) if i % block_len == 0]
    if t_f:
        times = [t for t in times if t <= t_f and t]
    if not site:
        site = max(state_history[times[0]].sites)//2
    entropies = [states.entropy(state_history[t], site) for t in times]
    ax.plot(times, entropies, label=f'Site {site}')
    ax.set_title(f'Entanglement Entropy at site {site}')
    ax.set_ylabel(r'$S_2$')
    ax.set_xlabel('Time')
    if show_max:
        max_ent = np.log(np.max(state_history[max(times)][site].shape))
        ax.axhline(max_ent, color='red', linestyle='--', label='Max Entropy')
    ax.legend()
    

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
    overlaps = [np.abs(states.overlap(state_hist_1[t], state_hist_2[t]))**2 for t in times]
    fig, ax = plt.subplots(1,1)
    ax.plot(times, overlaps)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$ | \langle \psi | \phi \rangle | ^2 $')
    return fig, ax
