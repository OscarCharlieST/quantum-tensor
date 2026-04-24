import copy
import scipy.linalg as la
import matplotlib.pyplot as plt
import numpy as np
import progressbar
from ncon import ncon
from numba import jit
from numba import njit
from numba.typed import List
import time

import qtensor.states as states
import qtensor.operators as ops
import qtensor.simulation.updatemethod as methods


""" 
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

def tdvp(state, operator, t_f, steps, 
         method=methods.lanczos_method(epsilon=1e-5, max_iters=16),
         history=False, 
         history_interval=1,
         verbose=False, 
         **kwargs):
    """
    method: tuple of methods
    """
    
    times = np.linspace(0, t_f, steps+1)
    dt = t_f/steps
    state_history = {}
    expectations = {}

    state.right_orthogonal()
    sites = sorted(state.sites)
    R_con = right_mpo_contractions(state, operator)
    L_con = {min(sites)-1 : ncon((np.eye(1), operator.l), ((-1, -2), (-3,)))}

    b = progressbar.ProgressBar(maxval=steps+1)
    b.start()
    step=0

    for t in times:
        if verbose:
            print(f't: {t:.3f}')
        if history:
            if step % history_interval == 0:
                now_state = copy.copy(state)
                state_history[t] = now_state   
        if 'operators' in kwargs:
            expectations[t] = [ops.local_expect(state, op) for op in kwargs['operators']]
        if 'extensive_operators' in kwargs:
            expectations[t] = [ops.expect(state, op) for op in kwargs['extensive_operators']]
        state, L_con, _ = tdvp_sweep_r(state, operator, dt, L_con, R_con, method)
        state, _, R_con = tdvp_sweep_l(state, operator, dt, L_con, R_con, method) 
        if 'termination_func' in kwargs:
            terminate = kwargs['termination_func']
            # signature should be f(state, t) = bool
            # where True means simulation should terminate
            if terminate(state, t):
                print(f"Simulation terminated at t = {t}")
                break
        step+=1
        b.update(step)

    if verbose:
        print('TDVP finished!')
    
    return state_history, expectations 

def right_mpo_contractions(state, operator):
    sites = sorted(state.sites, reverse=True)
    R_con = {}
    site = sites[0]
    R_con[site+1] = ncon((np.eye(1), operator.r),
                         ((-1, -2), (-3,)))
    for site in sites[:-1]:
        R_con[site] = ncon((state[site], state[site].conj(), operator[site], R_con[site+1]),
                           ((3, -1, 1), (4, -2, 2), (3, 4, -3, 5), (1, 2, 5)))
    return R_con

def tdvp_step_r(state, operator, dt, L_con, R_con, method):
    c_site = state.c_site
    M = state[c_site]
    M_new = method.c((M, operator[c_site], L_con[c_site-1], R_con[c_site+1]),
                      dt=dt/2)
    M_new = M_new / la.norm(M_new)  # normalize the new tensor
    A_new, C_new = states.left_orthogonal_tensor(M_new)
    state[c_site] = A_new  # update the centre tensor, now left-orthogonal
    L_con[c_site] = ops.contract_left(L_con[c_site-1], A_new, operator[c_site])
    C_new = method.b((C_new, L_con[c_site], R_con[c_site+1]),
                      dt=-dt/2)
    C_new = C_new / la.norm(C_new)  # normalize the new centre tensor
    state[c_site+1] = C_new @ state[c_site+1]
    state.c_site += 1  # shift the centre to the right

    return state, L_con, R_con

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
    sites = sorted(state.sites)

    # Update bulk
    for site in sites[:-1]:
        state, L_con, R_con = tdvp_step_r(state, operator, dt, L_con, R_con, method)
    
    # Update rightmost tensor
    assert state.c_site == sites[-1], "Centre isn't at right of chain somehow"
    c_site = sites[-1]
    M = method.c((state[c_site], operator[c_site], L_con[c_site-1], R_con[c_site+1]),
                  dt=dt/2)
    M = M / la.norm(M)
    state[c_site] = M

    return state, L_con, R_con

def tdvp_step_l(state, operator, dt, L_con, R_con, method):
    c_site = state.c_site
    M = state[c_site]
    M_new = method.c((M, operator[c_site], L_con[c_site-1], R_con[c_site+1]),
                      dt=dt/2)
    M_new = M_new / la.norm(M_new)  # normalize the new tensor
    C_new, B_new = states.right_orthogonal_tensor(M_new)
    state[c_site] = B_new
    R_con[c_site] = ops.contract_right(R_con[c_site+1], B_new, operator[c_site])
    
    C_new = method.b((C_new, L_con[c_site-1], R_con[c_site]),
                      dt=-dt/2)

    C_new = C_new / la.norm(C_new)  # normalize the new centre tensor
    state[c_site-1] = state[c_site-1] @ C_new  # update the next site tensor
    state.c_site -= 1  # shift the centre to the right

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
    assert state.c_site == np.max(state.sites), "Centre must be at right of chain."
    sites = sorted(state.sites, reverse=True)
   
    # Update bulk
    for site in sites[:-1]:
        state, L_con, R_con = tdvp_step_l(state, operator, dt, L_con, R_con, method)
    
    # Update leftmost tensor
    assert state.c_site == state.sites[0], "Centre isn't at left of chain somehow"
    c_site = sites[-1]
    M = method.c((state[c_site], operator[c_site], L_con[c_site-1], R_con[c_site+1]),
                 dt=dt/2)
    M = M / la.norm(M)
    state[c_site] = M

    return state, L_con, R_con

def gs_evolve(psi, H, t_f=1000, steps=100, method=methods.exact_method()):
    """
    Given an intial state and a hamiltonian, approximate the ground state
    by imaginary time tdvp
    """
    print("Intial energy:", ops.expect(psi, H))
    _, _ = tdvp(psi, H, -1j*t_f, steps, method=method)
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


