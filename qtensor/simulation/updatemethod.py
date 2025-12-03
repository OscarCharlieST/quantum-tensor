import numpy as np
from scipy import linalg as la
from ncon import ncon
import time

from numba import jit
from numba import njit
from numba.typed import List

    
def exact(tensor, H_eff, dt):
    # Calculate dimension of the space the vectorized tensor lives in
    vector_dim = np.product(tensor.shape)
    tensor_vec = tensor.flatten()
    # Reshape H_eff to be square matrix in vectorised space
    H_eff_mat = H_eff.reshape((vector_dim, vector_dim))
    mat_exp = la.expm(-0.5*1j*dt*H_eff_mat)
    tensor_evolved = tensor_vec @ mat_exp
    return tensor_evolved.reshape(tensor.shape)

def diagonal(tensor, H_eff, dt):
    vector_dim = np.product(tensor.shape)
    tensor_vec = tensor.flatten()
    # Reshape H_eff to be square matrix in vectorised space
    H_eff_mat = H_eff.reshape((vector_dim, vector_dim))
    eigvals, eigvecs = la.eigh(H_eff_mat)
    expvals = np.exp(-0.5*1j*dt*eigvals)
    mat_exp = eigvecs @ np.diag(expvals) @ eigvecs.conj().T
    tensor_evolved = tensor_vec @ mat_exp
    return tensor_evolved.reshape(tensor.shape)

def fast(tensor, H_eff, dt):
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

def fast_old(tensor, H_eff, dt, **kwargs):
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

def lanczos(tensor, H_eff, dt, epsilon=1e-4, iter_limit=8):
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