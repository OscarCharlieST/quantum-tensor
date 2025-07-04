import qtensor.operators as ops
import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_energy_density_evolution(state_history, H_terms):
    """
    Plot the time evolution of the state and the expectations.
    """
    times = sorted(list(state_history.keys()))
    sites = range(len(H_terms))
    middle_energy=[]
    
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), height_ratios=[2,1,1])
    cmap = mpl.colormaps['magma']

    ax[0,].set_xlabel("Bond")
    ax[0,].set_ylabel(r'$E$')

    ax[1,].set_ylabel(r'central $E$')

    ax[2,].set_xlabel("Time")
    ax[2,].set_ylabel(r'$dE/dt$')
        
    for t in times:
        local_energy = [ops.local_expect(state_history[t], term) 
                        for term in H_terms]
        ax[0,].plot(sites, local_energy, 
                color=cmap(t/max(times)))
        middle_energy.append(local_energy[len(local_energy)//2])

    ax[1,].plot(times, middle_energy)

    dE_dt= [(middle_energy[i+1] - middle_energy[i])/
            (times[i+1] - times[i])
            for i in range(len(middle_energy)-1)]
    
    ax[2,].plot(times[:-1], dE_dt)
           
    plt.colorbar(ax=ax[0,], label='Time')
    
    plt.show()