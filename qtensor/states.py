# -*- coding: utf-8 -*-`
import os
import copy
import qtensor.operators as ops
import scipy.linalg as la
import numpy as np
from ncon import ncon


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
class mps:
    def __init__(self, Ms, **kwargs):
        # Ms is a list of mps tensor tuples objects 
        # L and R are (the square roots of) the left and right environments
        if type(Ms) == dict:
            self.tensors = Ms
        elif type(Ms) == list:
            self.tensors = {i: M for i, M in enumerate(Ms)}
        else:
            raise ValueError("Ms must be a list of tensors or a dictionary of sites:tensor.")
        self.sites = sorted(self.tensors.keys())
        self.centred = False
        self.bond_centred = False
        self.normalized = False
        self.form = 'none'
        self.c_site= None

    def __getitem__(self, position):
        return self.tensors[position]

    def __setitem__(self, position, tensor):
        if position not in self.tensors.keys():
            raise ValueError(f"Position {position} not in MPS.")
        self.tensors[position] = tensor

    def __copy__(self):
        new_instance = mps(copy.copy(self.tensors))
        new_instance.form = copy.copy(self.form)
        new_instance.c_site = copy.copy(self.c_site)
        return new_instance
    
    def __deepcopy__(self, memo):
        new_instance = mps(copy.deepcopy(self.tensors, memo))
        new_instance.form = copy.copy(self.form)
        new_instance.c_site = copy.copy(self.c_site)
        return new_instance

    def __len__(self):
        return len(self.tensors)
    
    def L(self):
        return self.tensors[min(self.sites)]
    
    def R(self):
        return self.tensors[max(self.sites)]
    
    def left_orthogonal(self, max_bond_dim=np.inf):
        PsiL = left_orthogonal_state(self.tensors, max_bond_dim)
        self.tensors = PsiL
        self.normalized = True
        self.form = 'left'
        self.centred = False
        self.bond_centred = False
        self.c_site = self.sites[-1]
    
    def right_orthogonal(self, max_bond_dim=np.inf):
        PsiR = right_orthogonal_state(self.tensors, max_bond_dim)
        self.tensors = PsiR
        self.normalized = True
        self.form = 'right'
        self.centred = False
        self.bond_centred = False
        self.c_site = self.sites[0]

    def centralize(self, c_site, max_bond_dim=np.inf):
        """
        Centralize the MPS at site c_site.
        """
        psi_centre = centralize_state(self.tensors, c_site, max_bond_dim)
        self.tensors = psi_centre
        self.c_site = c_site
        self.centred = True
        self.bond_centred = False
        self.form = 'center'

    def shapes(self, display=True):
        top_str = ''
        mid_str = ''
        bot_str = ''
        shapes = {}
        for site in self.sites:
            d, Dl, Dr = self.tensors[site].shape
            shapes[site] = (d, Dl, Dr)
            top_str += f'{Dl}---'
            mid_str += '  | '
            bot_str += f'  {d} '
        top_str += f'{Dr}'
        print(top_str)
        print(mid_str)
        print(bot_str)
    
def left_orthogonal_tensor(M, max_bond_dim=np.inf):
    """
    Left orthogonalize and compress a MPS tensor

    INPUTS:
    M: (d, Dl, Dr) array, bulk mps tensor (doesn't work for edge (rank 2) tensors)
    max_bond_dim: int, max bond dimension to truncate to if nessecary

    RETURNS:
    M_lorth: (d, Dl, chi) array, left orthogonal tensor with truncated dimension chi
    G: (chi, Dr) array, the gauge transformation to be applied to the right
    """
    d, Dl, Dr = M.shape
    M_eff_mat = M.reshape(d*Dl, Dr)
    U, s, V = la.svd(M_eff_mat, full_matrices=False)
    # Truncate
    chi = min(len(s), max_bond_dim)
    U = U[:, :chi]
    s = s[:chi]
    V = V[:chi,:]
    M_lorth = U.reshape(d, Dl, chi)
    G = np.diag(s) @ V
    return M_lorth, G
    
def left_orthogonal_state(statedict, max_bond_dim):
    """
    Left orthogonalize a full MPS

    INPUTS:
    statedict: dict of {site:mps tensor} pairs
    max_bond_dim: int, max bond dimension to truncate to if nessecary

    RETURNS:
    PsiL: dict of {site:mps tensor} pairs, left orthogonalized
    """
    sites = sorted(statedict.keys())
    PsiL = {}
    # Orthogonalise leftmost tensor first
    M = statedict[sites[0]]
    M_lorth, G = left_orthogonal_tensor(M, max_bond_dim)
    PsiL[sites[0]] = M_lorth
    for i in sites[1:-1]:
        M = statedict[i]
        M_eff = G @ M
        M_lorth, G = left_orthogonal_tensor(M_eff, max_bond_dim)
        PsiL[i] = M_lorth
    # Handle rightmost tensor - doesnt need to be orthogonalised
    M = statedict[sites[-1]]
    M_eff = G @ M
    norm = la.norm(M_eff)
    M_eff = M_eff / norm # normalize
    PsiL[sites[-1]] = M_eff
    return PsiL

def right_orthogonal_tensor(M, max_bond_dim=np.inf):
    """
    Right orthogonalize and compress a MPS tensor

    INPUTS:
    M: (d, Dl, Dr) array, bulk mps tensor (doesn't work for edge (rank 2) tensors)
    max_bond_dim: int, max bond dimension to truncate to if nessecary

    RETURNS:
    M_rorth: (d, chi, Dr) array, left orthogonal tensor with truncated dimension chi
    G: (Dl, chi) array, the gauge transformation to be applied to the left
    """
    d, Dl, Dr = M.shape
    M_trans = ncon(M, (-1, -3, -2))
    # Use Left canoncalization on transposed tensor
    M_trans_lorth, G_trans = left_orthogonal_tensor(M_trans, max_bond_dim)
    M_rorth = ncon(M_trans_lorth, (-1, -3, -2))
    G = G_trans.T
    return G, M_rorth
    
def right_orthogonal_state(statedict, max_bond_dim):
    """
    Right orthogonalize a full MPS

    INPUTS:
    statedict: dict of {site:mps tensor} pairs
    max_bond_dim: int, max bond dimension to truncate to if nessecary

    RETURNS:
    PsiL: dict of {site:mps tensor} pairs, right orthogonalized
    """
    sites = sorted(statedict.keys(), reverse=True) # Sort from largest site index to smallest
    PsiR = {}
    # Orthogonalise leftmost tensor first
    M = statedict[sites[0]]
    G, M_rorth = right_orthogonal_tensor(M, max_bond_dim)
    PsiR[sites[0]] = M_rorth
    for i in sites[1:-1]:
        M = statedict[i]
        M_eff = M @ G
        G, M_rorth = right_orthogonal_tensor(M_eff, max_bond_dim)
        PsiR[i] = M_rorth
    M = statedict[sites[-1]]
    M_eff = M @ G
    norm = la.norm(M_eff)
    M_eff = M_eff / norm
    PsiR[sites[-1]] = M_eff
    return PsiR

def centralize_state(statedict, c_site, max_bond_dim):
    # If centre at edge of chain, just orthogonalise 
    if c_site == max(statedict.keys()):
        return left_orthogonal_state(statedict, max_bond_dim)
    if c_site == min(statedict.keys()):
        return right_orthogonal_state(statedict, max_bond_dim)

    psi_centre = {}
    # Handle left side of chain
    sites_l = sorted([i for i in statedict.keys() if i < c_site])
    M = statedict[sites_l[0]]    
    M_lorth, Gl = left_orthogonal_tensor(M, max_bond_dim)
    psi_centre[sites_l[0]] = M_lorth
    for i in sites_l[1:]:
        M = statedict[i]
        M_eff = Gl @ M
        M_lorth, Gl = left_orthogonal_tensor(M_eff, max_bond_dim)
        psi_centre[i] = M_lorth
    
    # Handle right side of chain
    sites_r = sorted([i for i in statedict.keys() if i > c_site], reverse=True)
    M = statedict[sites_r[0]]
    Gr, M_rorth = right_orthogonal_tensor(M, max_bond_dim)
    psi_centre[sites_r[0]] = M_rorth
    for i in sites_r[1:]:
        M = statedict[i]
        M_eff = M @ Gr
        Gr, M_rorth = right_orthogonal_tensor(M_eff, max_bond_dim)
        psi_centre[i] = M_rorth

    # Handle centre tensor and normalize
    centre_tensor = Gl @ statedict[c_site] @ Gr
    centre_tensor = centre_tensor / np.sqrt(
        ncon((centre_tensor, centre_tensor.conj()), ((1, 2, 3), (1, 2, 3)))) # normalize
    psi_centre[c_site] = centre_tensor
    
    return psi_centre

def overlap(state_1, state_2):
    """
    Compute inner product between two states
    """
    assert sorted(state_1.sites) == sorted(state_2.sites), "States need to be on the same lattice."
    sites = sorted(state_1.sites)
    L = np.array([[1]])
    R = np.array([[1]])
    for i in sites:
        L = ncon((L, state_1[i], state_2[i].conj()),
                 ((1, 2), (3, 2, -2), (3, 1, -1)))
    return np.trace(L @ R)

def random(N, d, D, seed=0):
    """
    Unnormalized random MPS state generator
    """
    r = np.sqrt(2*D*d) # rough normalization factor
    np.random.seed(seed)
    statedict = {}
    sites = np.arange(N)
    statedict[sites[0]] = (np.random.normal(size=(d, 1, D)) + 1j*np.random.normal(size=(d, 1, D)))/r
    for i in sites[1:-1]:
        statedict[i] = (np.random.normal(size=(d, D, D)) + 1j*np.random.normal(size=(d, D, D)))/r
    statedict[sites[-1]] = (np.random.normal(size=(d, D, 1)) + 1j*np.random.normal(size=(d, D, 1)))/r
    state = mps(statedict)
    return state

def haar_random_unitary(n: int, seed=0) -> np.ndarray:
    """
    Generate an n x n Haar-random unitary matrix.

    Parameters:
        n (int): Dimension of the unitary matrix (n > 0)

    Returns:
        np.ndarray: Haar-distributed unitary matrix of shape (n, n)
    """
    np.random.seed(seed)
    
    # Step 1: Create a random complex matrix with entries from N(0,1) + i*N(0,1)
    z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2)

    # Step 2: QR decomposition
    q, r = np.linalg.qr(z)

    # Step 3: Normalize phases to ensure Haar distribution
    d = np.diag(r)
    ph = d / np.abs(d)  # Extract phases
    q = q * ph

    return q

def unitary_random(L, d, D, seed=0):
    """
    Left-orthogonal random MPS generator from Haar-random unitary
    """
    
    tensors = {}
    Dr = 1
    for site in np.arange(L-1):
        Dl = Dr
        Dr = min([Dl*d, D])
        n = d*Dl
        U = haar_random_unitary(n, seed+site)
        M = U[:, :Dr].reshape(d, Dl, Dr)
        tensors[site] = M
    # final tensor should be D x d matrix, not 3 legged, 3rd leg should have bond dim 1
    site = L-1
    Dl = Dr # use previous Dr
    Dr = 1
    U = haar_random_unitary(d*Dl, seed+site)
    M = U[:, :Dr].reshape(d, Dl, Dr)
    tensors[site] = M
    
    state = mps(tensors)
    state.left_orthogonal()
    return state



def spin_up(N, D, noise=0.0):
    """
    MPS representation of all spin up state
    """
    statedict = {}
    sites = np.arange(N)
    statedict[sites[0]] = np.zeros((2, 1, D))*(1+1j)
    statedict[sites[0]][0,0,0] = 1.0
    for i in sites[1:-1]:
        statedict[i] = np.zeros((2, D, D))*(1+1j)
        statedict[i][0, :, :] = np.eye(D)
    statedict[sites[-1]] = np.zeros((2, D, 1))*(1+1j)
    statedict[sites[-1]][0,0,0] = 1.0
    state = mps(statedict)
    if not noise:
        return state
    else:
        random_state = random(N, 2, D, seed=42)
        for i in sites:
            state[i] += noise * random_state[i]
        state.left_orthogonal()
        return state 
    
def entropy(state, site=0):
    """
    Compute the entanglement entropy across the bond to the right of site
    """
    sites = sorted(state.sites)
    assert site in sites, "Site not in state."
    # Centralize state at site+1 and compute entropy from purity
    # Purity is trace of square of right environment to site.
    psi_centre = centralize_state(state.tensors, site+1, max_bond_dim=np.inf)
    centre_tensor = psi_centre[site+1]
    R = ncon((centre_tensor, centre_tensor.conj()), ((1, -1, 2), (1, -2, 2) ))
    P = np.real(ncon((R, R), ((1, 2), (2, 1))))
    entropy = -np.log2(P)
    return entropy

def entropy_diagonal(state, site=None):
    sites = sorted(state.sites)
    if not site:
        site = max(sites)//2 + 1
    else:
        assert site in sites, "Site not in state."
    working_state = copy.deepcopy(state)
    working_state.left_orthogonal()

    R = np.eye(1)
    for i in sorted(sites[site+1:], reverse=True):
        A = working_state[i]
        R = ncon((A, A.conj(), R),
                 ((1, -1, 2), (1, -2, 3), (2, 3)))
    P = np.real(ncon((R, R), ((1, 2), (2, 1))))
    entropy = -np.log2(P)
    return entropy

def right_environments(state):
    sites = sorted(state.sites)
    psi_left = left_orthogonal_state(state.tensors, max_bond_dim=np.inf)
    R = {}
    R_site = np.eye(1)
    for site in sorted(sites, reverse=True):
        R_site = ncon((psi_left[site], psi_left[site].conj(), R_site),
                      ((1, -1, 2), (1, -2, 3), (2, 3)))
        R[site] = R_site
    return R

def purities(state):
    R = right_environments(state)
    P = {i: np.real(np.trace(R[i]@R[i])) for i in R}
    return P

def entropies(state):
    P = purities(state)
    entropies = {i: np.log2(P[i]) for i in P}
    return entropies

# def entropy_left(state, site=0):
#     """
#     Compute the entanglement entropy across the bond to the right of site
#     Use left-orthogonal form of state, contract from the right
#     """
#     sites = sorted(state.sites)
#     assert site in sites, "Site not in state."
#     # Centralize state at site+1 and compute entropy from purity
#     # Purity is trace of square of right environment to site.
#     psi_left = left_orthogonal_state(state.tensors, max_bond_dim=np.inf)
#     R = ncon((centre_tensor, centre_tensor.conj()), ((1, -1, 2), (1, -2, 2) ))
#     P = np.real(ncon((R, R), ((1, 2), (2, 1))))
#     entropy = -np.log2(P)
#     return entropy
