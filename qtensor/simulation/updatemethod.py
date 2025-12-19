import numpy as np
from scipy import linalg as la
from ncon import ncon
import time

from numba import jit
from numba import njit
from numba.typed import List

class TDVPMethod:
    def __init__(self, c_method, b_method, **opts):
        # opts decides params for the method to be used at every point of evolution; 
        # only needs to be specified when class is created
        self.c_method = c_method
        self.b_method = b_method
        self.opts = opts
    def c(self, tensors, dt):
        return self.c_method(tensors, dt, **self.opts)
    def b(self, tensors, dt):
        return self.b_method(tensors, dt, **self.opts)
    
def lanczos_method(**lanczos_params):
    return TDVPMethod(lanczos_centre, lanczos_bond, **lanczos_params)

def lanczos_centre(tensors, dt=0.01, epsilon=1e-5, max_iters=16):
    """
    Method for evolving a centre gauge tensor as exp(-i H_eff dt)|C>
    
    Parameters:
        C, W, L, R: numpy array
            Centre tensor, local mpo tensor, left and right effective environments
        dt: float, default is 0.01
            Time step (this is halved outside the function for tdvp sweeping)
        epsilon: float, default is 1e-5
            Cutoff amplitude. Stops you from amplifying division-by small errors. 
            This can happen if your initial vector is in a small eigensubspace.
            Note that this is *not* the precision control.
        max_iters: int or NoneType, default is 100
            Maximum dimension of Krylov space built. The calculation is precise on
            all vectors up to {x, Hx, ..., H^(max_iters)x}. As such, errors should be 
            roughly of the order ~(dt*|H|)^(max_iters+1) 
            For reference, for random input tensors with d=4, D=32, max_iters=16 
            errors are ~1e-10, with a speed-up ~4000x

    """
    C, W, L, R = tensors
    if not max_iters:
        max_iters = np.prod(np.shape(C))
        print(f"Iterations unlimited; full space has dimension {np.prod(np.shape(C))}")
    basis, H_mat = lanczos_parts(C, W, L, R, epsilon, max_iters)
    exp_H_mat = la.expm(-1j*dt*H_mat)
    first_col = exp_H_mat[:,0]
    evolved_C = la.norm(C)*sum([C_i*H_i0 for C_i, H_i0 in zip(basis, first_col)]) 
    return evolved_C

    
def lanczos_bond(tensors, dt=0.01, epsilon=1e-5, max_iters=16):
    """
    Apply exp(-i H_eff dt) to a bond-centred tensor via Lanczos in the Krylov subspace.
    Parameters:
        M : array
            Bond-centred tensor (shape (D_left, D_right)).
        L, R : arrays
            Left/right effective environments for apply_Heff_bond.
        dt : float
            Time step (halved outside here if using TDVP sweeps; this applies the full step).
        epsilon : float
            Cutoff for Lanczos norm convergence.
        max_iters : int or None
            Maximum Krylov dimension. If None or 0, use full flattened dimension.
    Returns:
        M_evolved : array
            The evolved bond tensor: exp(-i H_eff dt) |M>.
    """
    M, L, R = tensors

    if not max_iters:
        max_iters = np.prod(np.shape(M))
        print(f"Iterations unlimited; full space has dimension {np.prod(np.shape(M))}")

    basis, H_mat = lanczos_parts_bond(M, L, R, epsilon=epsilon, max_iters=max_iters)
    U = la.expm(-1j * dt * H_mat)
    first_col = U[:, 0]
    M_evolved = la.norm(M) * sum(M_i * U_i0 for M_i, U_i0 in zip(basis, first_col))
    return M_evolved

def exact_method():
    return TDVPMethod(exact_centre, exact_bond)

def exact_centre(tensors, dt, **kwargs):
    C, W, L, R = tensors
    H_eff = ncon((W, L, R),
                 ((-1, -4, 1, 2), (-2, -5, 1), (-3, -6, 2)))
    return exact(C, H_eff, dt)

def exact_bond(tensors, dt, **kwargs):
    M, L, R = tensors
    H_eff = ncon((L, R), 
                 ((-1, -3, 1), (-2, -4, 1)))
    return exact(M, H_eff, dt)

def exact(tensor, H_eff, dt, **options):
    # Calculate dimension of the space the vectorized tensor lives in
    vector_dim = np.product(tensor.shape)
    tensor_vec = tensor.flatten()
    # Reshape H_eff to be square matrix in vectorised space
    H_eff_mat = H_eff.reshape((vector_dim, vector_dim))
    mat_exp = la.expm(-1j*dt*H_eff_mat)
    tensor_evolved = tensor_vec @ mat_exp
    return tensor_evolved.reshape(tensor.shape)


def diagonal(tensor, H_eff, dt):
    vector_dim = np.product(tensor.shape)
    tensor_vec = tensor.flatten()
    # Reshape H_eff to be square matrix in vectorised space
    H_eff_mat = H_eff.reshape((vector_dim, vector_dim))
    eigvals, eigvecs = la.eigh(H_eff_mat)
    expvals = np.exp(-1j*dt*eigvals)
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
    mat_exp_approx = np.eye(vector_dim) - 1j*dt*H_eff_mat
    tensor_evolved = tensor_vec @ mat_exp_approx
    return tensor_evolved.reshape(tensor.shape)



# Helper functions for new and improved Lanczos method

# Split methods for centred tensor and bond-centred tensor

def apply_Heff_parts(C, W, L, R,):
    con1 = ncon((C, L),
                ((-1, 1,-3), (1, -2, -4)))
    con2 = ncon((con1, W), 
                ((1, -2, -3, 4), (1, -1, 4, -4))) # Could be faster if these legs are vectorised, but isn't bond-dim-critical.
    d1, d2, d3, d4 = con2.shape # d1=d, d2=Dl, d3=Dr, d4=chi

    con2_vec = con2.reshape((d1, d2, d3*d4))
    R_vec = np.transpose(R, [1, 0, 2]).reshape((d3, d3*d4)) # Group chi leg and bond to C
    con3 = con2_vec @ (R_vec.T) # Critical step. D^3 d chi scaling

    return con3

def contract_C(C1, C2):
    """
    Contract two centre gauge tensors <C1, C2>
    Parameters:
        C1, C2 : array of identical shape
            Tensors to contract, both given in ket form (C1 will be conjugated)
    """
    C1_vec = C1.flatten()
    C2_vec = C2.flatten()
    return C1_vec.conj() @ C2_vec
    
def lanczos_loop(basis, W, L, R):
    C_last = basis[-1]
    HC = apply_Heff_parts(C_last, W, L, R)
    H_cons = [contract_C(C_k, HC) for C_k in basis]
    C_next = HC - sum([H_cons[i]*basis[i] for i in range(len(basis))])
    norm = np.sqrt(contract_C(C_next, C_next))
    return C_next/norm, norm, H_cons

def H_subspace_matrix(H_cons):
    """
    Build the matrix <Ci, H_eff C_j> using intermediate Lanczos weights.
    Leveraging the fact that it should be hermitian.
    Should work for both centre and bond center versions
    """
    n = len(H_cons)
    H_mat = np.zeros((n,n), dtype=type(1+1j))
    for i, col in enumerate(H_cons):
        H_mat[:i+1, i] = col # These are the upper right triangular entries
    H_mat += np.triu(H_mat, 1).conj().T # Copy c.c. values to below diagonal.
    return H_mat

def lanczos_parts(C, W, L, R,
                  epsilon=1e-6,
                  max_iters=100):
    """
    Builds the orthogonal basis for the Krylov subspace, 
    and the representation of the Hamiltonian in this basis.
    """
    W=W
    C_normed = C / la.norm(C)
    basis = [C_normed]
    H_cons = []
    for i in range(max_iters):
        # print('iter ', i)
        C_next, norm, H_cons_i = lanczos_loop(basis, W, L, R)
        H_cons.append(H_cons_i)
        if norm < epsilon:
            # print("Norm converged; terminating loop")
            # print("Norm / epsilon: ", round(np.real(norm/epsilon), 4))
            return basis, H_subspace_matrix(H_cons)
        basis.append(C_next)
    # print("Hit iteration limit before convergence")
    # print("Norm / epsilon: ", round(np.real(norm/epsilon), 4))
    return basis, H_subspace_matrix(H_cons)

# Lanczos for bond centred

def apply_Heff_bond(M, L, R,):
    con1 = ncon((M, L),
                ((1,-2), (1, -1, -3))) # D^3 chi
    d1, d2, d3 = con1.shape # d1=Dl, d2=Dr, d3=chi

    con1_vec = con1.reshape((d1, d2*d3))
    R_vec = np.transpose(R, [1, 0, 2]).reshape((d2, d2*d3)) # Group chi leg and bond to C
    con2 = con1_vec @ (R_vec.T) # Critical step. D^3 chi scaling

    return con2


def contract_M(M1, M2):
    """
    Contract two bond-centred tensors <M1, M2>.
    Parameters:
        M1, M2 : arrays of identical shape
            Tensors to contract, both in ket form (M1 will be conjugated).
    Returns:
        scalar (complex): <M1 | M2>
    """
    M1_vec = M1.flatten()
    M2_vec = M2.flatten()
    return M1_vec.conj() @ M2_vec

def lanczos_loop_bond(basis, L, R):
    """
    Single Lanczos step for a bond-centred tensor.
    Parameters:
        basis : list of tensors (each same shape as the bond tensor)
        L, R  : environment tensors used in apply_Heff_bond
    Returns:
        M_next_normed : next basis vector (normalized)
        norm          : its pre-normalization norm
        H_cons        : list of projections <M_k | H_eff M_last> over current basis
    """
    M_last = basis[-1]
    HM = apply_Heff_bond(M_last, L, R)
    H_cons = [contract_M(M_k, HM) for M_k in basis]
    # Orthogonalize against existing basis
    M_next = HM - sum(H_cons[i] * basis[i] for i in range(len(basis)))
    norm = np.sqrt(contract_M(M_next, M_next))
    return M_next / norm, norm, H_cons

def lanczos_parts_bond(M, L, R, epsilon=1e-6, max_iters=100):
    """
    Build an orthonormal Krylov basis and the projected Hamiltonian for a bond tensor.
    Parameters:
        M : array
            Initial bond-centred tensor (shape (D_left, D_right)).
        L, R : arrays
            Left/right effective environments for apply_Heff_bond.
        epsilon : float
            Convergence cutoff for the new-vector norm.
        max_iters : int
            Maximum Krylov dimension.
    Returns:
        basis : list of arrays
            Orthonormal basis vectors spanning {M, H M, H^2 M, ...}.
        H_mat : (k x k) Hermitian matrix
            Projected Hamiltonian in the constructed basis.
    """
    M_norm = la.norm(M)
    if M_norm == 0:
        raise ValueError("Initial bond tensor has zero norm.")
    basis = [M / M_norm]
    H_cols = []
    norm = None

    for i in range(max_iters):
        M_next, norm, H_col = lanczos_loop_bond(basis, L, R)
        H_cols.append(H_col)
        if np.real(norm) < epsilon:
            # print("Norm converged; terminating loop")
            # print("Norm / epsilon:", round(np.real(norm / epsilon), 4))
            return basis, H_subspace_matrix(H_cols)
        basis.append(M_next)

    # print("Hit iteration limit before convergence")
    # if norm is not None:
        # print("Norm / epsilon:", round(np.real(norm / epsilon), 4))
    return basis, H_subspace_matrix(H_cols)





