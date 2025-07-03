import copy
import scipy.linalg as la
import matplotlib.pyplot as plt
import numpy as np
from ncon import ncon
from numba import jit
from numba import njit
from numba.typed import List
import time

from qtensor.states import *
from qtensor.operators import *


""" 
wrapppppppppppppppppppppppppppppp
Finite TDVP for MPS

References:
https://arxiv.org/pdf/1901.05824 algorithm 3, 5 

Indexing:

    1                           1             1
    ¦        2 -- A -- 3        ¦             ¦
    L -- 3  ,     ¦      , 3 -- W -- 4 , 3 -- R
    ¦             1             ¦             ¦
    2                           2             2 
"""

def tdvp(state, operator, t_f, steps, method,
         history=False, verbose=False, 
         **kwargs):
    """
    Perform tdvp on state under a hamiltonian operator
    Inputs:
        state: mps object
        operator: mpo object
        t_f: complex float
        history: bool, default is False
        verbose: bool, default is False
        method: callable, default is method_fast
        **operators: list of mpo objects
    Returns:
        state_history: dict
            {time: state} at each point in the evolution
        expectations: dict
            {time: [expectation of each operator provided]}
    """
    print('Initiating TDVP')
    # t = 0
    times = np.linspace(0, t_f, steps+1)
    dt = t_f/steps
    R_con = right_mpo_contractions(state, operator)
    state_history = {}
    expectations = {}
    # while np.abs(t)<np.abs(t_f): # abs in case of imaginary time
    for t in times:
        if verbose:
            print(f't: {t:.3f}')
        if history:
            now_state = copy.copy(state)
            state_history[t] = now_state   
        if 'operators' in kwargs:
            # operators = [list of mpo]
            expectations[t] = [local_expect(state, op) for op in kwargs['operators']]
            
        L_con = {min(state.sites)-1: 
                ncon((state.L.conj().T @ state.L , operator.l), ((-1, -2), (-3,)))}
        state, L_con, _ = tdvp_sweep_r(state, operator, dt, L_con, R_con, method)
        R_con = {max(state.sites)+1: 
                ncon((state.R @ state.R.conj().T , operator.r), ((-1, -2), (-3,)))}
        state, _, R_con = tdvp_sweep_l(state, operator, dt, L_con, R_con, method)

        # t += dt
    print('TDVP finished!')
    state_history[t_f] = copy.copy(state)
    if 'operators' in kwargs:
        expectations[t_f] = [local_expect(state, op) for op in kwargs['operators']]
    return state_history, expectations

def right_mpo_contractions(state, operator):
    """
    Compute the right contractions of the MPO with respect to the MPS Psi.
    Needed for intializing the TDVP algorithm.
    Inputs:
        state: mps object
        operator: mpo object

    Outputs:
        R_con: dictionary of right contractions, indexed by site
    """

    sites = sorted(state.sites)
    R = state.R @ state.R.conj().T
    r = operator.r
    R_con = {}
    R_con[max(sites)+1] = ncon((R, r), ((-1, -2), (-3,)))
    for site in reversed(sites):
        A = state[site]
        d, Dl, Dr = A.shape
        R_right = R_con[site+1]
        if site not in operator.sites:
            W = ncon((np.eye(d), np.eye(R_right.shape[2])), ((-1, -2), (-3, -4)))
        else:
            W = operator[site]
        R_con[site] = contract_right(R_right, A, W)
    return R_con

def tdvp_step_r(state, operator, dt, L_con, R_con, method):
    c_site = state.c_site
    M = state[c_site]
    d, Dl, Dr = M.shape
    
    H_eff = ncon((L_con[c_site-1], operator[c_site], R_con[c_site+1]),
                 ((-2, -5, 1), (-1, -4, 1, 2), (-3, -6, 2)))
    # M_new = M - 1j * (dt/2) * ncon((M, H_eff), ((1, 2, 3), (1, 2, 3, -1, -2, -3)))
    # exp_H_eff = method(H_eff, dt)
    # M_new = ncon((M, exp_H_eff), ((1, 2, 3), (1, 2, 3, -1, -2, -3)))
    M_new = method(M, H_eff, dt)

    M_new = M_new / la.norm(M_new)  # normalize the new tensor
    A_new, C_new = left_orthogonal(M_new)
    state[c_site] = A_new  # update the centre tensor, now left-orthogonal
    L_con[c_site] = contract_left(L_con[c_site-1], A_new, operator[c_site])
    if not c_site == max(state.sites):
        H_eff_bond = ncon((L_con[c_site], R_con[c_site+1]), ((-1, -3, 1), (-2, -4, 1)))
        # C_new = C_new + 1j * (dt/2) * ncon((C_new, H_eff_bond), ((1, 2), (1, 2, -1, -2)))
        # exp_H_eff_bond = method(H_eff_bond, -dt)
        # C_new = ncon((C_new, exp_H_eff_bond), ((1, 2), (1, 2, -1, -2)))
        C_new = method(C_new, H_eff_bond, -dt)

        C_new = C_new / la.norm(C_new)  # normalize the new centre tensor
        state[c_site+1] = C_new @ state[c_site+1]  # update the next site tensor
        state.c_site += 1  # shift the centre to the right
    else:
        state.R = C_new @ state.R 
        state.form = 'left'
    return state, L_con, R_con # not sure about best implementation for returning things...

def tdvp_step_l(state, operator, dt, L_con, R_con, method):
    c_site = state.c_site
    M = state[c_site]
    d, Dl, Dr = M.shape
    H_eff = ncon((L_con[c_site-1], operator[c_site], R_con[c_site+1]),
                 ((-2, -5, 1), (-1, -4, 1, 2), (-3, -6, 2)))
    # M_new = M - 1j * (dt/2) * ncon((M, H_eff), ((1, 2, 3), (1, 2, 3, -1, -2, -3)))
    # exp_H_eff = method(H_eff, dt)
    # M_new = ncon((M, exp_H_eff), ((1, 2, 3), (1, 2, 3, -1, -2, -3)))
    M_new = method(M, H_eff, dt)

    M_new = M_new / la.norm(M_new)  # normalize the new tensor
    B_new, C_new = right_orthogonal(M_new)
    state[c_site] = B_new
    R_con[c_site] = contract_right(R_con[c_site+1], B_new, operator[c_site])
    if not c_site == min(state.sites):
        H_eff_bond = ncon((L_con[c_site-1], R_con[c_site]), ((-1, -3, 1), (-2, -4, 1)))
        # C_new = C_new + 1j * (dt/2) * ncon((C_new, H_eff_bond), ((1, 2), (1, 2, -1, -2)))
        # exp_H_eff_bond = method(H_eff_bond, -dt)
        # C_new = ncon((C_new, exp_H_eff_bond), ((1, 2), (1, 2, -1, -2)))
        C_new = method(C_new, H_eff_bond, -dt)

        C_new = C_new / la.norm(C_new)  # normalize the new centre tensor
        state[c_site-1] = state[c_site-1] @ C_new  # update the previous site tensor
        state.c_site -= 1  # shift the centre to the left
    else:
        state.L = state.L @ C_new
        state.form = 'right'
    return state, L_con, R_con # not sure about best implementation for returning things...


def tdvp_sweep_r(state, operator, dt, L_con, R_con, method):
    """
    Perform a TDVP sweep to the right.
    Inputs:
        state: mps object
        operator: mpo object
        dt: time step
        L_con: left contractions dictionary
        R_con: right contractions dictionary
    Outputs:
        state: updated mps object
        L_con: updated left contractions dictionary
        R_con: updated right contractions dictionary
    """
    assert state.form == 'right', "MPS needs to be right canonicalized before TDVP sweep."
    state[state.c_site] = state.L @ state[state.c_site]  # absorb the left environment into the first site tensor
    state.L = np.eye(state[state.c_site].shape[1])  # reset the left environment to identity
    state.form = 'centre'  # set the form to centre after absorbing the left environment
    while state.form == 'centre':
        state, L_con, R_con = tdvp_step_r(state, operator, dt, L_con, R_con, method)
    return state, L_con, R_con


def tdvp_sweep_l(state, operator, dt, L_con, R_con, method):
    """
    Perform a TDVP sweep to the left.
    Inputs:
        state: mps object
        operator: mpo object
        dt: time step
        L_con: left contractions dictionary
        R_con: right contractions dictionary
    Outputs:
        state: updated mps object
        L_con: updated left contractions dictionary
        R_con: updated right contractions dictionary
    """
    assert state.form == 'left', "MPS needs to be left canonicalized before sweep left."
    state[state.c_site] = state[state.c_site] @ state.R  # absorb the right environment into the last site tensor
    state.R = np.eye(state.R.shape[0])  # reset the right environment to identity
    state.form = 'centre'  # set the form to centre after absorbing the right environment
    while state.form == 'centre':
        state, L_con, R_con = tdvp_step_l(state, operator, dt, L_con, R_con, method)
    return state, L_con, R_con

def gs_evolve(psi, H, t_f=1000, steps=100):
    """
    Given an intial state and a hamiltonian, approximate the ground state
    by imaginary time tdvp
    """
    print("Intial energy:", expect(psi, H))
    _, _ = tdvp(psi, H, -1j*t_f, steps, method_fast)
    print("Final energy:", expect(psi, H))
    return psi

def inf_T_thermofield_variational(N, D, t_f=1000, steps=100, state=None, seed=0):
    """
    Build the infinite t thermofield using tdvp and 
    """
    # build hamiltonian
    W = np.zeros((4, 4, 2, 2))
    v = [1, 0, 0, 1]
    W[:, :, 0, 0] = np.eye(4)
    W[:, :, 0, 1] = np.eye(4) - 2*np.outer(v, v)  # positive energy cost for all states, negative for chosen state
    W[:, :, 1, 1] = np.eye(4)
    l = np.array([1, 0])
    r = np.array([0, 1])
    H_gs = uniform_MPO(W, l, r, N)
    if not state:
        state = random_mps(N, 4, D, seed=seed)
    state = gs_evolve(state, H_gs, t_f, steps)
    return state

def method_exact(tensor, H_eff, dt, **kwargs):
    """
    Compute the matrix exponential via exact diagonalisation
    """
    full_dim = np.prod(H_eff.shape)
    sq_dim = int(np.sqrt(full_dim))
    H_eff_mat = H_eff.reshape(sq_dim, sq_dim)
    mat_exp = la.expm(-0.5*1j*dt*H_eff_mat)
    exp_H_eff = mat_exp.reshape(H_eff.shape)
    if len(tensor.shape) == 3:
        return ncon((tensor, exp_H_eff),
                    ((1, 2, 3), (1, 2, 3, -1, -2, -3)))
    elif len(tensor.shape) == 2:
        return ncon((tensor, exp_H_eff),
                    ((1, 2), (1, 2, -1, -2)))
    else:
        raise ValueError("Tensor shape not compatible with 1site tdvp")

def method_fast(tensor, H_eff, dt, **kwargs):
    """
    First order approximation to matrix exponential
    """
    # full_dim = np.prod(H_eff.shape)
    # sq_dim = int(np.sqrt(full_dim))
    # id_mat = np.eye(sq_dim)
    # id_full = id_mat.reshape(H_eff.shape)*(1+0j)

    # assert id_full.shape==H_eff.shape, 'identity built wrong'
    
    if len(tensor.shape) == 3:
        return tensor - 1j * (dt/2) * ncon((tensor, H_eff),
                                            ((1, 2, 3), (1, 2, 3, -1, -2, -3)))
    elif len(tensor.shape) == 2:
        return tensor - 1j * (dt/2) * ncon((tensor, H_eff),
                                            ((1, 2), (1, 2, -1, -2)))
    else:
        raise ValueError("Tensor shape not compatible with 1site tdvp")
    # return id_full - 0.5*1j*dt*H_eff

def method_lanczos(tensor, H_eff, dt, epsilon=1e-4, iter_limit=8):
    """
    Lanczos method for computing matrix exponential
    """
    full_dim = np.prod(H_eff.shape)
    sq_dim = int(np.sqrt(full_dim))
    H_eff_mat = H_eff.reshape(sq_dim, sq_dim)
    tensor_vec = tensor.reshape(sq_dim, 1)
    # build the lanczos vectors
    v0 = tensor_vec / la.norm(tensor_vec)
    start = time.perf_counter()
    vm = lanczos_loop(v0, H_eff_mat, epsilon=epsilon, iter_limit=iter_limit)
    end = time.perf_counter()
    print(f"time taken for loop: {end-start:.3f}")
    Vm = np.column_stack(vm)
    # if not Vm.shape == H_eff_mat.shape:
    #     print(f'Lanczos method saved time! Required {Vm.shape[1]} vectors')
    H_eff_lanczos = Vm.conj().T @ H_eff_mat @ Vm
    # compute the exponential exactly
    mat_exp = la.expm(-0.5*1j*dt*H_eff_lanczos)
    updated_tensor = (Vm @ mat_exp)[:, 0]
    updated_tensor = updated_tensor.reshape(tensor.shape)
    return updated_tensor

@jit
def lanczos_loop(v0, H_eff_mat, epsilon=1e-4, iter_limit=8):
    v0 = v0[:, 0]
    vm = [v0]
    converged = False
    iter_limit = 0
    while not converged and iter_limit > len(vm):
        v = vm[-1]
        w = H_eff_mat @ v
        for v_i in vm:
            # subtract the projection of w onto each previous vector
            w -= np.dot(v_i.conjugate(), w) * v_i 
        norm_w = np.real(np.sqrt(np.dot(w.conjugate(), w)))
        if norm_w < epsilon:
            converged = True
            break
        vm.append(w / norm_w)
        iter_limit += 1
    return vm
