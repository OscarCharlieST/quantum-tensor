"""

Calculate response functions between two operators A and B using the tangent Hamiltonian method.

C(A, B) = <beta| A P B |beta> + h.c.

In particular we want the current-current response function, where A = J and B = J(t), the (manifold time-evolved) current operator.

In the eigenbasis |w_m> of the tangent Hamiltonian, we have

C(J_q, J_q(t)) = sum_m 2 cos(w_m t) |<beta| J_q |w_m>|^2

and the inner product reduces to 

<beta| J_q |w_m> = -i w_m <beta| h_q |w_m> / q

"""

import numpy as np
import time
from ncon import ncon
import os
import h5py
import matplotlib.pyplot as plt

import qtensor.states as states
import qtensor.operators as ops
import qtensor.thermofield as thf
import qtensor.simulation.updatemethod as methods
import lyapunov.relaxation.tangent_hamiltonian_me  as tangent

RUN_DIR = r"C:\Users\charl\lyapunov_runs"
FIG_DIR = r"C:\Users\charl\lyapunov_runs\figures"

# Change this as appropriate to your system. This is where the results will be saved.

def current_response_weights(psi_beta, H_sym, H_asym, q):
    """
    Compute the weights of the current-current response function in the eigenbasis of the tangent Hamiltonian by exactly diagonalizing it.

    Parameters
    ----------
    psi_beta : qtensor.states.State
        The state |beta> around which we are expanding.
    H_sym, H_asym : qtensor.operators.Operator
        The symmetric and antisymmetric parts of the Hamiltonian.
    q : qtensor.operators.Operator
        The current operator J.

    Returns
    -------
    weights : np.ndarray
        The weights |<beta| J |w_m>|^2 for each eigenvector |w_m> of the tangent Hamiltonian.
    eigenvalues : np.ndarray
        The eigenvalues w_m of the tangent Hamiltonian.
    """
    # Get consistent canonical forms
    psi_l, psi_r = tangent.canonical_states(psi_beta.tensors, max_bond_dim=np.inf)
    V = tangent.tangent_maps(psi_l)

    # Compute the tangent Hamiltonian
    t_start = time.time()
    H_tangent = tangent.tangent_hamiltonian_me(psi_l, psi_r, V, H_asym)
    print(f"Tangent Hamiltonian computation took {time.time() - t_start:.4f} seconds.")

    # Diagonalize the tangent Hamiltonian to get eigenvalues and eigenvectors
    t_start = time.time()
    eigenvalues, eigenvectors = np.linalg.eigh(H_tangent)
    print(f"Tangent Hamiltonian diagonalization took {time.time() - t_start:.4f} seconds.")

    # Compute the tangent components of J |beta>
    t_start = time.time()
    h_q = build_fourier_energy_density(H_sym, q)
    X_current = tangent_map_fourier_components(psi_l, psi_r, V, h_q)
    print(f"(Not q-normalised) current build took took {time.time() - t_start:.4f} seconds.")

    # Compute overlap and mode weights
    overlaps = X_current @ eigenvectors
    basis_weights = np.abs(overlaps) ** 2
        
    return eigenvalues, basis_weights

def current_response_weights_lanczos(psi_beta, H_sym, H_asym, q, loops=100):
    """
    Compute the weights of overlap between the local equillibrium tangent vector and lanczos estimates for the eigenvalues and eigenvectors of the tangent Hamiltonian.

    Parameters
    ----------
    psi_beta, H_sym, H_asym, q: (same as current_response_weights)
    loops: int
        Number of lanczos iterations to perform, minus one.
    Returns
    -------
    eigenvalues : np.ndarray
        The eigenvalues of the tangent Hamiltonian.
    weights : np.ndarray
        The weights |<beta| J |w_m>|^2 for each eigenvector |w_m> of the tangent Hamiltonian.
    """
    # Get consistent canonical forms
    psi_l, psi_r = tangent.canonical_states(psi_beta.tensors, max_bond_dim=np.inf)
    V = tangent.tangent_maps(psi_l)


    # Compute the tangent components of J |beta>
    t_start = time.time()
    h_q = build_fourier_energy_density(H_sym, q)
    X_current = tangent_map_fourier_components(psi_l, psi_r, V, h_q) # This is the vector to seed the lanczos algorithm with
    print(f"(Not q-normalised) current build took took {time.time() - t_start:.4f} seconds.")

    # Compute overlap and mode weights
    x = X_current.conj() 
    omega, weights, a, b = tangent.lanczos_tangent_vector(psi_l, psi_r, V, H_asym, x, loops=loops)
        
    return omega, weights, a, b

def current_response(t, omega, weight, q):
    """
    Evaluates correlations at array of times t according to formula
    2 *sum_omega cos(omega*t) * weight(omega)
    t: 1d array or floats
    omega: 1d array of frequencies
    weight: 1d array of weights (same length as omega) 
    """
    mode_weight = ((omega**2) * weight) / ((2 * np.sin(q/2))**2)
    return 2 * (np.cos(np.outer(t, omega)) @ mode_weight)

def integrated_response(t, omega, weight, q):
    """
    Explicit integral of current response
    t: 1d array or floats
    omega: 1d array of frequencies
    weight: 1d array of weights (same length as omega) 
    """
    mode_weight = ((omega) * weight) / ((2 * np.sin(q/2))**2)
    return 2 * (np.sin(np.outer(t, omega)) @ mode_weight)

def D_estimate(t, omega, weight, q):
    current_integral = integrated_response(t, omega, weight, q)  
    # Use projected (h_q, h_q) as heat capacity estimate
    # Could also use full space unprojected version
    h_q_projected_variance = 2*np.cos(np.outer(t, omega)) @ (weight)
    return current_integral / h_q_projected_variance

def build_fourier_energy_density(H_sym, q):
    """
    Build the q fourier component of the energy densitry operator h_q = sum_i e^{i q r_i} h_i, where h_i is the local energy density operator at site i.
    """

    H_fourier = {}

    for i in sorted(H_sym.sites):
        h_i = H_sym[i].astype(complex)
        H_fourier[i] = h_i.copy()
        # Calculate fourier weights and apply
        # Only cosine transform implemented, but can be extended if needed
        phase_site = np.cos(q * i) 
        H_fourier[i][:, :, 0, 3] = phase_site * h_i[:, :, 0, 3]
        # Only apply phase to bond term once (to left of site in this case)
        phase_bond = np.cos(q * (i+0.5))
        H_fourier[i][:, :, 0, 1:3] = phase_bond * h_i[:, :, 0, 1:3]

    return ops.mpo(sorted(H_fourier.items()), l=H_sym.l, r=H_sym.r)

def build_environment_contractions(psi_left, psi_right, O):
    """
    Take an extensive mpo O and state psi, and create the left and right environment contractions for each site.
    """
    LO = {min(psi_left.sites): ncon((np.eye(1), O.l), ((-1, -2), (-3,)))}

    for i in sorted(psi_left.sites)[1:]:
        A_ket = psi_left[i-1] # ket centred at defect
        A_bra = psi_right[i-1] # bra right canconical
        LO[i] = ncon((LO[i-1], A_ket, O[i-1], A_bra.conj()),
                     ((1, 2, 3), (4, 1, -1), (4, 5, 3, -3), (5, 2, -2)))

        
    RO = {max(psi_right.sites): ncon((np.eye(1), O.r), ((-1, -2), (-3,)))}

    for i in sorted(psi_right.sites, reverse=True)[1:]:
        A_ket = psi_right[i+1] # ket centred at defect - we're to the right here
        A_bra = psi_right[i+1] # bra right canconical
        RO[i] = ncon((RO[i+1], A_ket, O[i+1], A_bra.conj()),
                     ((1, 2, 3), (4, -1, 1), (4, 5, -3, 3), (5, -2, 2)))

    return LO, RO

def tangent_map_fourier_components(psi_left, psi_right, V, h_q):
    """
    Compute tagent space vector componenets with the same basis as in tangent_hamiltonian_me. Must ue same states as there.
    """
    sites = sorted(psi_left.sites)
    Lh, Rh = build_environment_contractions(psi_left, psi_right, h_q)

    # compound index (site, a, r) -> offsets[site] + a*Dr + r   (C order)
    dims = {s: V[s].shape[2] * psi_left[s].shape[2] for s in sites}
    starts = np.cumsum([0] + [dims[s] for s in sites])
    offsets = dict(zip(sites, starts[:-1]))
    dim = starts[-1]
    
    t_vec = np.zeros((dim), complex)
    
    for i in sites:
        comps_i = ncon((Lh[i], V[i], psi_right[i].conj(), h_q[i], Rh[i]), 
                        ((1, 2, 3), (4, 1, -1), (5, 2, 6),
                         (4, 5, 3, 7), (-2, 6, 7)))
        t_vec[offsets[i]:offsets[i]+dims[i]]=comps_i.ravel()

    return t_vec


def build_initials(L=101, 
                   J=0.25, h=0.125, g=-0.2625, 
                   beta=0.1,  chi=32, 
                   steps=100, noise=1e-14, seed=42, 
                   save=False):
    """
    Build Hamiltonian and thermal state for tangent space calculations.
    """
    np.random.seed(seed)

    # Build Hamiltonian
    H = ops.tilted_ising(J, h, g, L)
    H_sym = ops.thermofield_hamiltonian(H)
    H_asym = ops.thermofield_hamiltonian(H, asym=True)

    # Build initial state
    init_state= thf.inf_T_thermofield(L, chi, noise=noise)
    beta_state, _, _ = thf.finite_T_thermofield(beta,  H_sym, steps=steps,
                                                initial_state=init_state,
                                                plot=False)
    if save:
        params = dict(model="TFI", J=J, h=h, g=g,
                      L=L, chi=chi, beta=beta,
                      steps=steps, noise=noise, seed=seed)
        fname = f"TFI_L_{L}_D_{chi}_beta_{beta:g}_seed_{seed}.h5"
        beta_state.save(os.path.join(RUN_DIR, fname), params=params)
    return H_sym, H_asym, beta_state

def build_hamiltonians(L=101,  J=0.25, h=0.125, g=-0.2625):
    # Build Hamiltonian
    H = ops.tilted_ising(J, h, g, L)
    H_sym = ops.thermofield_hamiltonian(H)
    H_asym = ops.thermofield_hamiltonian(H, asym=True)
    return H_sym, H_asym

def get_state(L=101, J=0.25, h=0.125, g=-0.2625, beta=0.1,  chi=32, 
                   steps=100, noise=1e-14, seed=42):

    np.random.seed(seed)
    
    fname = state_filename(L=L, J=J, h=h, g=g, beta=beta, chi=chi,
                           steps=steps, noise=noise, seed=seed)

    path = os.path.join(RUN_DIR, fname)
    if os.path.exists(path):
        print("State already exists, no need to build. Loading...")
        state, _ = states.load_mps(path, verbose=False)
        return state 
    else:
        # Get Hamiltonian
        H_sym, _ = build_hamiltonians(L, J, h, g)

        # Build initial state
        init_state= thf.inf_T_thermofield(L, chi, noise=noise)
        beta_state, _, _ = thf.finite_T_thermofield(beta,  H_sym, steps=steps,
                                                    initial_state=init_state,
                                                    plot=False)
        params = dict(model="TFI", J=J, h=h, g=g,
                      L=L, chi=chi, beta=beta,
                      steps=steps, noise=noise, seed=seed)
        beta_state.save(path, params=params)
        return beta_state


def state_filename(**params):
    return f"TFI_J_{params['J']}_h_{params['h']}_g_{params['g']}_L_{params['L']}_D_{params['chi']}_beta_{params['beta']:g}_steps_{params['steps']}_noise_{params['noise']:g}_seed_{params['seed']}_state.h5"


def run_response(L, chi, beta, n, J, h, g, 
                 times, 
                 loops, steps, noise, seed, save=False):
    """
    Run the current-current response function calculation for given parameters.
    """
    assert n>0, "n must be positive for q = 2 pi n / L"
    q = 2*np.pi*n / L

    # Check if initial state exists on disk

    t = time.time()
    # H_sym, H_asym, beta_state = build_initials(L=L, J=J, h=h, g=g,
    #                                            beta=beta, chi=chi,
    #                                            steps=steps, noise=noise,
    #                                            seed=seed, save=False)
    H_sym, H_asym = build_hamiltonians(L, J, h, g)
    beta_state = get_state(L=L, J=J, h=h, g=g, beta=beta, chi=chi,
                           steps=steps, noise=noise, seed=seed)
    print(f"Initial state build took {time.time() - t:.4f} seconds.")

    t = time.time()
    omega, weights, a, b = current_response_weights_lanczos(
        beta_state, H_sym, H_asym, q, loops=loops)
    print(f"Current response weights calculation took {time.time() - t:.4f} seconds.")

    C_t = current_response(times, omega, weights, q)
    D = D_estimate(times, omega, weights, q)

    if save:
        fname = response_filename(L=L, chi=chi, beta=beta, n=n, J=J, h=h, g=g,
                                  loops=loops, steps=steps, noise=noise, seed=seed)
        save_response(os.path.join(RUN_DIR, fname),
            {"D": D, "C_t": C_t, "times": times, "omega": omega, "weights": weights, "a": a, "b": b},
            {"L": L, "chi": chi, "beta": beta, "n": n, "J": J, "h": h, "g": g,"loops": loops, "steps": steps, "noise": noise, "seed": seed}
            )

    return D, C_t, times, omega, weights, a, b

def plot_response(times, C_t, D, omega, weights, q, title=None, save_plot=False):
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    ax[0].plot(times, C_t)
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Current-Current Response C(t)")
    ax[0].set_title("Current-Current Response Function")

    ax[1].plot(times, D)
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Diffusion Coefficient D(t)")
    ax[1].set_title("Diffusion Coefficient Estimate")

    ax[2].scatter(omega, weights,
                  color='blue', s=5,
                  label=r'$|\langle \omega_k | P h_q | \beta \rangle|^2$')
    ax[2].scatter(omega, weights*(omega**2) / (2*np.sin(q/2))**2, 
                  color='red', s=5,
                  label=r'$|\langle \omega_k | J_q |\beta \rangle|^2$')
    ax[2].set_yscale('log') 
    ax[2].set_xlabel("Ritz frequencies")
    ax[2].set_ylabel("Weight")
    ax[2].set_title("Response Function Weights")
    ax[2].legend()

    if title is not None:
        fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_plot:    
        plt.savefig(os.path.join(FIG_DIR, response_filename(**run_params).replace('.h5', '.png')))
    
    plt.show()

# Stuff for comparing multiple parameter sets: 
# get_response takes params and loads it if it exists, otherwise runs the calculation and saves it, returning it either way.
# compare_response takes lists of full parameter sets, calculates which parameters are changing, and plots the results with labels based on changing parameters
# param lists can be generated with a list comprehension, e.g. for varying chi:
#   base = run_params
#   param_list = [base | {"chi": c} for c in (16, 24, 32)] 

def get_response(**params):
    """
    Given a set of parameters, load the response function data if it exists, otherwise run the calculation and save the results.
    """
    fname = response_filename(**params)
    path = os.path.join(RUN_DIR, fname)
    if os.path.exists(path):
        return load_response(path)
    else:
        D, C_t, times, omega, weights, a, b = run_response(**params, save=False)
        save_response(path, {"D": D, "C_t": C_t, "omega": omega, "times": params['times'], "weights": weights, "a": a, "b": b}, params)
        return {"D": D, "C_t": C_t, "times": params['times'], "omega": omega, "weights": weights, "a": a, "b": b}, params

def compare_response(param_list, compute_missing=False):
    missing_runs = [p for p in param_list if not os.path.exists(os.path.join(RUN_DIR, response_filename(**p)))]
    print(f"Missing runs: {len(missing_runs)}.")
    if not compute_missing:
        print(f"Running comparison only for existing runs")
        param_list = [p for p in param_list if  os.path.exists(os.path.join(RUN_DIR, response_filename(**p)))]
    runs = [get_response(**params) for params in param_list]
    var_params = [k for k in param_list[0] if k != "times"
                  and len({p[k] for p in param_list}) > 1]

    fig, ax = plt.subplots(1, 2, figsize=(12,5))
    for r in runs:
        param_label = ", ".join(f'{k} = {r[1][k]}' for k in var_params)
        ax[0].plot(r[0]['times'], r[0]['C_t'], label=param_label)
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Current-Current Response C(t)")
        ax[0].set_title("Current-Current Response Function")

        ax[1].plot(r[0]['times'], r[0]['D'], label=param_label)
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Diffusion Coefficient D(t)")
        ax[1].set_title("Diffusion Coefficient Estimate")

    ax[0].legend()

    tag = "_".join(f"{k}-" + "-".join(f"{p[k]:g}" for p in param_list) for k in var_params)
    fname = os.path.join(FIG_DIR, f"scan_{tag}_" + response_filename(**param_list[0]).replace('.h5', '.png'))
    fig.savefig(fname)
    plt.show()

    pass

def response_filename(**params):
    return f"TFI_J_{params['J']}_h_{params['h']}_g_{params['g']}_L_{params['L']}_D_{params['chi']}_beta_{params['beta']:g}_noise_{params['noise']:g}_seed_{params['seed']}_response_n_{params['n']}_m_{params['loops']}.h5"

def save_response(path, results, params):
    with h5py.File(path, "w") as f:
        f.attrs.update(params)                  # dict of scalars/strings
        for key, value in results.items():
            f.create_dataset(key, data=value)

def load_response(path):
    with h5py.File(path, "r") as f:
        params = dict(f.attrs)                  # copy out before the file closes
        results = {key: f[key][()] for key in f.keys()}   # [()] reads into a numpy array
    return results, params

###

run_params = {'L': 8, 'chi': 8, 'beta': 0.01, 'n': 1,
                'J': 0.25, 'h': 0.125, 'g': -0.2625, 
                'times': np.linspace(0, 100, 1000), 
                'loops': 100, 'steps': 100, 'noise': 1e-14, 'seed': 42}

if __name__ == "__main__" :
    # Check run parameters don't already exist in run directory
    base = run_params
    scan_params = [base | {"loops": l} for l in (100, 200,)]
    compare_response(scan_params, compute_missing=True)



    
