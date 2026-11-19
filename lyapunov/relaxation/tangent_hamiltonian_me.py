from matplotlib.pylab import beta
import numpy as np
import scipy.linalg as la
from ncon import ncon

import qtensor.states as states
import qtensor.operators as ops
import qtensor.simulation.updatemethod as methods

def tangent_hamiltonian_me(psi_l, psi_r, V, H):
    """
    Compute the tangent Hamiltonian for a given state and asymmetric Hamiltonian.
    
    Parameters
    ----------
    psi : qtensor.states.State
        The state being expanded around.
    H : qtensor.operators.Operator
        The Hamiltonian being projected.
    
    Returns
    -------
    qtensor.operators.Operator
        The tangent Hamiltonian a tangent space map.
    """
    # Build state in left and right forms, as well as V tangent map at each site (Use left canonical form)
    # Build left and right hamiltonain contraction environments (called LH, RH)
    #   Each will be (chi, chi, w) where w is mpo bond dim. 
    # Only need to compute upper triangular part; hamiltonian is hermitian.
    # loop through sites i and j>=i
    # loop through tangent elements (zeroes exepcet for one entry)
    # zip up chain from i to j
    #   Possible to keep track of left evironment "chasing" j down the chain.

    LH, RH = hamiltonian_environments(psi_l, psi_r, H)

    sites = sorted(psi_l.sites)

    # compound index (site, a, r) -> offsets[site] + a*Dr + r   (C order)
    dims = {s: V[s].shape[2] * psi_l[s].shape[2] for s in sites}
    starts = np.cumsum([0] + [dims[s] for s in sites])
    offsets = dict(zip(sites, starts[:-1]))
    dim = starts[-1]

    H_out = np.zeros((dim, dim), dtype=complex)

    for i in sites:
        A_i = psi_l[i]
        V_i = V[i]
        dim_i = V_i.shape[2]*A_i.shape[2]
        for i_comp in range(dim_i):
            X_i = np.zeros(dim_i)
            X_i[i_comp] = 1.0
            V_i_comp = ncon((V_i, X_i.reshape(V_i.shape[2], A_i.shape[2])), ((-1, -2, 3), (3, -3)))
            LV_i = envs_of_bra_from_left(psi_l, psi_r, H, V_i_comp, LH, base=i)
            ket_tensors = psi_r.tensors.copy()
            ket_tensors[i] = V_i_comp
            for j in sites[sites.index(i):]:
                # Compute the contribution to the tangent Hamiltonian from sites i and j
                A_j = ket_tensors[j]
                V_j = V[j]
                open_contract = ncon((LV_i[j], A_j, H[j], V_j.conj(), RH[j]),
                                     ((2, 5, 6), (1, 2, 3), (1, 4, 6, 7),
                                      (4, 5, -1), (3, -2, 7)))
                H_out[offsets[j]:offsets[j] + dims[j], offsets[i] + i_comp] = open_contract.ravel()

    # blocks with site_row >= site_col are filled; copy the rest by Hermiticity
    site_of = np.repeat(sites, [dims[s] for s in sites])
    upper = site_of[:, None] < site_of[None, :]
    H_out[upper] = H_out.conj().T[upper]
    return H_out


def apply_tangent_hamiltonian(X, psi_l, psi_r, V, H, LH, RH):
    """
    Apply the projected Hamiltonian to a tangent vector X.

    Return a tangent vector Y

    Avoids having to build and store the full tangent Hamiltonian matrix

    How it will work:
    1. Split X into components and shape for each site
    2. Environements from tangentvector bra, state ket, and Hamiltonian
    2.a Take care to account for case of onsite defect vs defect already passed
    3. 
    """
    sites = sorted(psi_l.sites)

    # compound index (site, a, r) -> offsets[site] + a*Dr + r  
    dims = {s: V[s].shape[2] * psi_l[s].shape[2] for s in sites}
    starts = np.cumsum([0] + [dims[s] for s in sites])
    offsets = dict(zip(sites, starts[:-1]))
    dim = starts[-1]

    B = {} # V @ X, tangent projector and element combined for MPS-like tensor
    Y = np.zeros((dim), complex) # Tang vector P H |X>, vectorised like X

    for i in sites:
        X_i = X[offsets[i]:offsets[i] + dims[i]].reshape(V[i].shape[2], psi_l[i].shape[2])
        B[i] = ncon((V[i], X_i),
                     ((-1, -2, 3), (3, -3)))

    LB = {sites[0]: np.zeros_like(LH[sites[0]])} # Contraction of all terms with defect to left of site i
    for i in sites[:-1]:
        # Each site = term where defect is here + term where defect has already been passed
        LB[i+1] = (ncon((LH[i], B[i], psi_l[i].conj(), H[i]),
                        ((1, 2, 3), (4, 1, -1), (5, 2, -2), (4, 5, 3, -3))) # Defect at this site
                    + ncon((LB[i], psi_r[i], psi_l[i].conj(), H[i]),
                        ((1, 2, 3), (4, 1, -1), (5, 2, -2), (4, 5, 3, -3)))) # defect already passed

    # Same for defect to the right
    
    RB = {sites[-1]: np.zeros_like(RH[sites[-1]])} # Contraction of all terms with defect to left of site i
    for i in sorted(sites, reverse=True)[:-1]:
        # Each site = term where defect is here + term where defect has already been passed
        RB[i-1] = (ncon((RH[i], B[i], psi_r[i].conj(), H[i]),
                        ((1, 2, 3), (4, -1, 1), (5, -2, 2), (4, 5, -3, 3))) # Defect at this site
                    + ncon((RB[i], psi_l[i], psi_r[i].conj(), H[i]),
                        ((1, 2, 3), (4, -1, 1), (5, -2, 2), (4, 5, -3, 3)))) # defect already passed

    # Find tangent components of P H|X>

    for i in sites:
        # use apply_Heff_parts from updatmethods, as this is already vectorised optimally for large bond dimension
        M = (methods.apply_Heff_parts(psi_r[i], H[i], LB[i], RH[i])
             + methods.apply_Heff_parts(psi_l[i], H[i], LH[i], RB[i])
             + methods.apply_Heff_parts(B[i], H[i], LH[i], RH[i]))
        
        Y[offsets[i]:offsets[i] + dims[i]] = ncon((M, V[i].conj()),
                                                  ((1, 2, -2), (1, 2, -1))).ravel()

    return Y

def lanczos_tangent_vector(psi_l, psi_r, V, H, X, loops=10):
    """
    Use the Lanczos algorithm to compute the action of the tangent Hamiltonian on a tangent vector X.
    """
    LH, RH = hamiltonian_environments(psi_l, psi_r, H)

    a = []
    b = []

    Y_prev = X.copy() / np.linalg.norm(X)
    Y = apply_tangent_hamiltonian(Y_prev, psi_l, psi_r, V, H, LH, RH)
    a_i = np.vdot(Y_prev, Y).real
    Y = Y - a_i * Y_prev
    a.append(a_i)
    
    for _ in range(loops):
        b_new = np.linalg.norm(Y)
        if b_new > 1e-12:
            b.append(b_new)
        else:
            print("Norm dangerously small, breaking after {} loops".format(len(a)))
            break
        Y = Y.copy()/b[-1] # normalise
        Y_next = (apply_tangent_hamiltonian(Y, psi_l, psi_r, V, H, LH, RH) 
            - b[-1] * Y_prev )
        a_k = np.vdot(Y, Y_next).real
        Y_next = Y_next - a_k * Y
        a.append(a_k)
        Y, Y_prev = Y_next, Y

    # eigenvalues of the tridiagonal matrix are ritz values of the tangent Hamiltonian, and the first component of the eigenvectors give the overlap with the initial vector X

    tri_matrix = np.diag(a) + np.diag(b, k=1) + np.diag(b, k=-1)
    eigvals, eigvecs = la.eigh(tri_matrix)

    # eigvecs[0, :] is the overlaps of eignevectors with the initial vector X, so the weights of each eigenvalue in the expansion of X are given by the square of this overlap, multiplied by the norm of X squared.

    weights = np.linalg.norm(X)**2 * np.abs(eigvecs[0, :])**2

    return eigvals, weights, a, b 



    

def canonical_states(psi_tensors, max_bond_dim=np.inf):
    """
    Compute the left and right canonical forms of a state.
    
    """
    tensors, _ = states.right_orthogonal_state(psi_tensors, max_bond_dim=max_bond_dim)
    psi_left, _ = states.left_orthogonal_state(tensors, max_bond_dim=max_bond_dim)
    psi_right, _ = states.right_orthogonal_state(psi_left, max_bond_dim=max_bond_dim)

    return states.mps(psi_left), states.mps(psi_right)

def tangent_maps(psi):
    """
    Get tangent map V_i for site i in the state psi, assumed already in canonoical form.

    If there is no tangent space, just return (d, D, 0) array.
    """
    V = {}
    for site in sorted(psi.sites):
        A = psi[site]
        d, Dl, Dr = A.shape
        U, _, _ = la.svd(A.reshape(d * Dl, Dr), full_matrices=True)
        V[site] = U[:, Dr:].reshape(d, Dl, d*Dl - Dr)
    return V

def hamiltonian_environments(psi_left, psi_right, H):
    """
    Compute the left and right Hamiltonian contraction environments for a given state and Hamiltonian.
    """
    # Compute left Hamiltonian contraction environment
    LH = {min(psi_left.sites): ncon((np.eye(1), H.l), ((-1, -2), (-3,)))}
    for site in sorted(psi_left.sites)[1:]:
        A = psi_left[site-1]
        LH[site] = ops.contract_left(LH[site-1], A, H[site-1])

    # Compute right Hamiltonian contraction environment
    RH = {max(psi_right.sites): ncon((np.eye(1), H.r), ((-1, -2), (-3,)))}
    for site in sorted(psi_right.sites, reverse=True)[1:]:
        A = psi_right[site+1]
        RH[site] = ops.contract_right(RH[site+1], A, H[site+1])
    return LH, RH

def envs_of_bra_from_left(psi_L, psi_R, H, V_i_comp, L_ham_envs, base=0):
    """
     -- V-A-A-
    L - W-W-W- . . . 
     -- A-A-A-
    
     Takes a tangent space element at site `base` and computes the left Hamiltonian contraction environment for the bra of the tangent space element.
    """
    sites = sorted(psi_L.sites)
    L_base = L_ham_envs[base]
    LV = {base: L_base}
    LV[base+1] = ncon((L_base, V_i_comp, H[base], psi_L[base].conj()),
                    ((1, 2, 3), (4, 1, -1), (4, 5, 3, -3), (5, 2, -2)))
    for site in range(base+2, max(sites)+1):
        LV[site] = ncon((LV[site-1], psi_R[site-1], H[site-1], psi_L[site-1].conj()), 
                        ((1, 2, 3), (4, 1, -1), (4, 5, 3, -3), (5, 2, -2)))
    return LV

