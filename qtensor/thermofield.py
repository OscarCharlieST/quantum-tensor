import numpy as np
import qtensor.states as states 
import qtensor.operators as ops
from qtensor.simulation.finiteTDVP import tdvp, right_mpo_contractions

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

def finite_T_thermofield(beta, N, D, H, noise=0.0, steps=100):
    state = infinite_T_thermofield(N, D, noise)
    state.right_canonical()
    state_hist, _, _ = tdvp(state, H, beta*1/4, (beta*1/4)/steps, history=True, verbose=True)
    return state
    