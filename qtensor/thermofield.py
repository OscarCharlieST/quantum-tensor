import numpy as np
import matplotlib.pyplot as plt
import copy
import qtensor.states as states 
import qtensor.operators as ops
from qtensor.simulation.finiteTDVP import tdvp, right_mpo_contractions, inf_T_thermofield_variational
import qtensor.simulation.finiteTDVP as sim

def infinite_T_thermofield(N, D, noise=0):
    """
    Builds an infinite temperature thermofield with N sites and bond dimension D
    """
    M = np.zeros((4, D, D))
    M[:, 0, 0] = np.array([1, 0, 0, 1])
    Ms = [M for _ in range(N)]
    if noise:
        Ms = [M + noise * np.random.rand(4, D, D) for M in Ms]
    return states.mps(Ms)


def th_onesite(A, site):
    """
    Takes a single-site spin 1/2 operator (as amatrix) and returns the thermofield mpo reprsentation 
    """
    W = np.zeros((4, 4, 1, 1))
    W[:, :, 0, 0] = np.kron(A, np.eye(2)) + np.kron(np.eye(2), A)
    return ops.mpo([site, W], np.array([1,]), np.array([1,]))

def thermofield_hamiltonian(H):
    """
    Takes a specific form of2 site hamiltonian H
    where Hl, Hr are the left and right of the two site term
    and h the 1-site term

    I   Hl  h
    0   0   Hr
    0   0   I

    and makes it into a thermofield HxI + IxH via

    I   Hla Hlb ha+hb
    0   0   0   Hrb
    0   0   0   Hra
    0   0   0   I

    There may be a more general perscription, but this is efficient for our hamiltonian

    Note that it only works where the two local terms can be written as tensor product over the two sites. 
    """
    H_th = []
    for i in H.sites:
        W = H[i]
        Hl = W[:, :, 0, 1]
        Hr = W[:, :, 1, 2]
        h = W[:, :, 0, 2]

        W_th = np.zeros((4, 4, 4, 4), dtype=np.complex64)
        W_th[:, :, 0, 0] = np.kron(np.eye(2), np.eye(2))
        W_th[:, :, 0, 1] = np.kron(Hl, np.eye(2))
        W_th[:, :, 0, 2] = np.kron(np.eye(2), Hl)
        W_th[:, :, 0, 3] = np.kron(h, np.eye(2)) + np.kron(np.eye(2), h)
        W_th[:, :, 1, 3] = np.kron(Hr, np.eye(2))
        W_th[:, :, 2, 3] = np.kron(np.eye(2), Hr)
        W_th[:, :, 3, 3] = np.kron(np.eye(2), np.eye(2))
        H_th.append((i, W_th))
    l = np.array([1, 0, 0, 0])
    r = np.array([0, 0, 0, 1])
    return ops.mpo(H_th, l, r)

def finite_T_thermofield(beta, N, D, H, steps=100, initial_state=None, plot=True, method=None):
    if not initial_state:    
        state = inf_T_thermofield_variational(N, D)
    else:
        state = copy.deepcopy(initial_state)
        # initial state must be infinite temperature
        pass
    if not method:
        method = sim.method_fast
    _, expectations = tdvp(state, H, -1j*beta*1/4, steps, method, history=True, operators=[H])
    time = np.abs(list(expectations.keys()))*4
    energy = np.real([opexp[0] for opexp in expectations.values()])/2
    if plot:
        _, ax = plt.subplots(1,1)
        ax.plot(time, energy)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$E$")
        print("Energy at finite temperature:", energy[-1])
    return state, time, energy

def near_thermal(H, profile, D, steps=100, initial_state=None):
    assert len(H.sites) == len(profile), "temp profile incorrect length"
    H_new = []
    for site, beta in zip(H.sites, profile):
        W = copy.copy(H[site]) # don't actually edit the hamiltonian
        W[:, :, :-1, 1:] = W[:, :, :-1, 1:] * np.sqrt(beta) # twosite terms get a factor from each site
        W[:, :, 0, -1] = W[:, :, 0, -1] * np.sqrt(beta) # onesite term gets both sqrts at once
        H_new.append((site, W))
    H_eff = ops.mpo(H_new, H.l, H.r)
    state, _, _ = finite_T_thermofield(1, len(profile), D, H_eff, steps=steps, initial_state=initial_state,
                                       plot=False)
    return state

def near_thermal_first_order_deformed(D, beta_profile, J=1, h=0.25, g=-0.525, t=1.0, steps=100):
    """
    Produces a near-thermal state with an additional deformation corresponding to the first order change in energy 
    For low temperatures, the near-thermal density matrix is rho = 1/z (1 + beta_i h_i)
    The time derivative of the density matrix is i[rho, H]
    So to first order in t, rho(t) = rho + i t [rho, H] = 1/z (1 + beta_i(h_i + i t [h_i, H]))
    So we modify the generator of the thermal state h_i -> h_i + i t [h_i, H]
    the larger t is, the further the deviation from the near thermal ansatz
    """
    fodh = ops.first_order_deformation_generator(beta_profile, J, h, g, t)
    tf_fodg = thermofield_hamiltonian(fodh)
    state, _, _ = finite_T_thermofield(1, len(beta_profile), D, tf_fodg, steps, plot=False, method=sim.method_exact)
    return state