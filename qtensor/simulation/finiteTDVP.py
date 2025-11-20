import copy
import scipy.linalg as la
import matplotlib.pyplot as plt
import numpy as np
from ncon import ncon
from numba import jit
from numba import njit
from numba.typed import List
import time

import qtensor.states as states
import qtensor.operators as ops
import qtensor.simulation.updatemethod as method


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

def tdvp_new(state, operator, t_f, steps, 
             method=method.exact,
             history=False, verbose=False, **kwargs):
    times = np.linspace(0, t_f, steps+1)
    dt = t_f/steps
    state_history = {}
    expectations = {}

    state.right_orthogonal()
    R_con = right_mpo_contractions_new(state, operator)

    for t in times:
        if verbose:
            print(f't: {t:.3f}')
        if history:
            now_state = copy.copy(state)
            state_history[t] = now_state   
        if 'operators' in kwargs:
            expectations[t] = [ops.local_expect(state, op) for op in kwargs['operators']]
            
        L_con = {}
        state, L_con, _ = tdvp_sweep_r_new(state, operator, dt, L_con, R_con, method)
        R_con = {}
        state, _, R_con = tdvp_sweep_l_new(state, operator, dt, L_con, R_con, method)    

    if verbose:
        print('TDVP finished!')
    return state_history, expectations 

def right_mpo_contractions_new(state, operator):
    sites = sorted(state.sites, reverse=True)
    R_con = {}
    site = sites[0]
    R_con[site] = ncon((state[site], state[site].conj(), operator[site], operator.r),
                       ((1, -1), (2, -2), (1, 2, -3, 3), (3,)))
    for site in sites[1:-1]:
        R_con[site] = ncon((state[site], state[site].conj(), operator[site], R_con[site+1]),
                           ((3, -1, 1), (4, -2, 2), (3, 4, -3, 5), (1, 2, 5)))
    return R_con

def tdvp_step_r_new(state, operator, dt, L_con, R_con, method):
    c_site = state.c_site
    M = state[c_site]
    H_eff = ncon((L_con[c_site-1], operator[c_site], R_con[c_site+1]),
                 ((-2, -5, 1), (-1, -4, 1, 2), (-3, -6, 2)))
    M_new = method(M, H_eff, dt)
    M_new = M_new / la.norm(M_new)  # normalize the new tensor
    A_new, C_new = states.left_orthogonal_tensor(M_new)
    state[c_site] = A_new  # update the centre tensor, now left-orthogonal
    L_con[c_site] = ops.contract_left(L_con[c_site-1], A_new, operator[c_site])
    
    H_eff_bond = ncon((L_con[c_site], R_con[c_site+1]), ((-1, -3, 1), (-2, -4, 1)))
    C_new = method(C_new, H_eff_bond, -dt)

    C_new = C_new / la.norm(C_new)  # normalize the new centre tensor
    state[c_site+1] = C_new @ state[c_site+1]  # update the next site tensor
    state.c_site += 1  # shift the centre to the right

    return state, L_con, R_con

def tdvp_step_l_new(state, operator, dt, L_con, R_con, method):
    c_site = state.c_site
    M = state[c_site]
    H_eff = ncon((L_con[c_site-1], operator[c_site], R_con[c_site+1]),
                 ((-2, -5, 1), (-1, -4, 1, 2), (-3, -6, 2)))
    M_new = method(M, H_eff, dt)
    M_new = M_new / la.norm(M_new)  # normalize the new tensor
    C_new, B_new = states.right_orthogonal_tensor(M_new)
    state[c_site] = B_new
    R_con[c_site] = ops.contract_right(R_con[c_site+1], B_new, operator[c_site])
    
    H_eff_bond = ncon((L_con[c_site-1], R_con[c_site]), ((-1, -3, 1), (-2, -4, 1)))
    C_new = method(C_new, H_eff_bond, -dt)

    C_new = C_new / la.norm(C_new)  # normalize the new centre tensor
    state[c_site-1] = state[c_site-1] @ C_new  # update the next site tensor
    state.c_site -= 1  # shift the centre to the right

    return state, L_con, R_con

def tdvp_sweep_r_new(state, operator, dt, L_con, R_con, method):
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
    sites = sorted(state.sites)

    # Update leftmost tensor
    current_site = sites[0]
    current_op = ncon((operator.l, operator[current_site]),
                      ((1,), (-1, -2, 1, -3)))
    H_eff = ncon((current_op, R_con[current_site+1]),
                 ((-1, -3, 2), (-2, -4, 2))) # effective hamiltonian for first site
    M = method(state[current_site], H_eff, dt)
    M = M / la.norm(M)
    A_new, s, V = la.svd(M, full_matrices=False)
    C_new = np.diag(s) @ V
    state[current_site] = A_new
    L_con[current_site] = ncon((A_new, A_new.conj(), current_op), 
                               ((1, -1), (2, -2), (1, 2, -3)))
    
    # Update C tensor
    H_eff_bond = ncon((L_con[current_site], R_con[current_site+1]), 
                      ((-1, -3, 1), (-2, -4, 1)))
    C_new = method(C_new, H_eff_bond, -dt)
    C_new = C_new / la.norm(C_new)
    state[current_site+1] = C_new @ state[current_site+1]
    state.c_site += 1  # shift the centre to the right
    
    # Update bulk
    for site in sites[1:-1]:
        state, L_con, R_con = tdvp_step_r_new(state, operator, dt, L_con, R_con, method)
    
    # Update rightmost tensor
    assert state.c_site == sites[-1], "Centre isn't at right of chain somehow"
    current_site = sites[-1]
    H_eff = ncon((L_con[state.c_site-1], operator[state.c_site], operator.r),
                 ((-2, -4, 1), (-1, -3, 1, 2), (2,)))
    M = method(state[current_site], H_eff, dt)
    M = M / la.norm(M)
    state[current_site] = M.T

    return state, L_con, R_con

def tdvp_sweep_l_new(state, operator, dt, L_con, R_con, method):
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
    assert state.c_site == np.max(state.sites), "Centre must be at right of chain."
    sites = sorted(state.sites, reverse=True)

    # Update rightmost tensor
    current_site = sites[0]
    current_op = ncon((operator[current_site], operator.r),
                      ((-1, -2, -3, 1), (1,)))
    H_eff = ncon((L_con[current_site-1], current_op),
                 ((-2, -4, 1), (-1, -3, 1))) # effective hamiltonian for last site
    M = method(state[current_site], H_eff, dt)
    M = M / la.norm(M)
    U, s, V = la.svd(M, full_matrices=False)
    B_new = V.T
    C_new = U @ np.diag(s)
    state[current_site] = B_new
    R_con[current_site] = ncon((B_new, B_new.conj(), current_op), 
                               ((1, -1), (2, -2), (1, 2, -3)))
    
    # Update C tensor
    H_eff_bond = ncon((L_con[current_site-1], R_con[current_site]), 
                      ((-1, -3, 1), (-2, -4, 1)))
    C_new = method(C_new, H_eff_bond, -dt)
    C_new = C_new / la.norm(C_new)
    state[current_site-1] = state[current_site-1] @ C_new
    state.c_site -= 1  # shift the centre to the left
    
    # Update bulk
    for site in sites[1:-1]:
        state, L_con, R_con = tdvp_step_l_new(state, operator, dt, L_con, R_con, method)
    
    # Update leftmost tensor
    assert state.c_site == state.sites[0], "Centre isn't at left of chain somehow"
    current_site = sites[-1]
    H_eff = ncon((operator.l, operator[state.c_site], R_con[state.c_site+1]),
                 ((1,), (-1, -3, 1, 2), (-2, -4, 2)))
    M = method(state[current_site], H_eff, dt)
    M = M / la.norm(M)
    state[current_site] = M

    return state, L_con, R_con

def gs_evolve(psi, H, t_f=1000, steps=100, method=method.exact):
    """
    Given an intial state and a hamiltonian, approximate the ground state
    by imaginary time tdvp
    """
    print("Intial energy:", ops.expect(psi, H))
    _, _ = tdvp_new(psi, H, -1j*t_f, steps, method_exact_new)
    print("Final energy:", ops.expect(psi, H))
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
    H_gs = ops.uniform_MPO(W, l, r, N)
    if not state:
        state = states.random_mps(N, 4, D, seed=seed)
    state.right_orthogonal()
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
    
def method_exact_new(tensor, H_eff, dt):
    # Calculate dimension of the space the vectorized tensor lives in
    vector_dim = np.product(tensor.shape)
    tensor_vec = tensor.flatten()
    # Reshape H_eff to be square matrix in vectorised space
    H_eff_mat = H_eff.reshape((vector_dim, vector_dim))
    mat_exp = la.expm(-0.5*1j*dt*H_eff_mat)
    tensor_evolved = tensor_vec @ mat_exp
    return tensor_evolved.reshape(tensor.shape)

def method_fast_new(tensor, H_eff, dt):
    # Calculate dimension of the space the vectorized tensor lives in
    vector_dim = np.product(tensor.shape)
    tensor_vec = tensor.flatten()
    # Reshape H_eff to be square matrix in vectorised space
    H_eff_mat = H_eff.reshape((vector_dim, vector_dim))
    # mat_exp   = la.expm(-0.5*1j*dt*H_eff_mat)
    #           = 1 - 0.5*1j*dt*H_eff_mat + O(dt^2)
    mat_exp_approx = np.eye(vector_dim) - 0.5*1j*dt*H_eff_mat
    tensor_evolved = tensor_vec @ mat_exp_approx
    return tensor_evolved.reshape(tensor.shape)


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
