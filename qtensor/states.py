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
        assert len(np.shape(self.tensors[self.sites[0]]))==2, "Leftmost tensor must be a matrix."
        assert len(np.shape(self.tensors[self.sites[-1]]))==2, "Rightmost tensor must be a matrix."
        self.centred = False
        self.bond_centred = False
        self.normalized = False
        self.form = 'none'

    def __getitem__(self, position):
        return self.tensors[position]

    def __setitem__(self, position, tensor):
        if position not in self.tensors.keys():
            raise ValueError(f"Position {position} not in MPS.")
        self.tensors[position] = tensor

    def __copy__(self):
        new_instance = mps(copy.copy(self.tensors), L=copy.copy(self.L), R=copy.copy(self.R))
        new_instance.form = copy.copy(self.form)
        new_instance.c_site = copy.copy(self.c_site)
        return new_instance
    
    def __deepcopy__(self, memo):
        new_instance = mps(copy.deepcopy(self.tensors, memo), 
                           L=copy.deepcopy(self.L, memo), 
                           R=copy.deepcopy(self.R, memo))
        new_instance.form = copy.copy(self.form)
        new_instance.c_site = copy.copy(self.c_site)
        return new_instance

    def __len__(self):
        return len(self.tensors)
    
    def left_orthogonal(self, max_bond_dim=np.inf):
        PsiL = left_orthogonal_state(self.tensors, max_bond_dim)
        self.tensors = PsiL
        self.normalized = True
        self.form = 'left'
        self.centred = False
        self.bond_centred = False
    
    def right_orthogonal(self, max_bond_dim=np.inf):
        PsiR = right_orthogonal_state(self.tensors, max_bond_dim)
        self.tensors = PsiR
        self.normalized = True
        self.form = 'right'
        self.centred = False
        self.bond_centred = False

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
    
    def bond_centralize(self, side='right'):
        """
        Given a centralized MPS, shifts the 'centre' to the bond between c_site and c_site+1.
        """
        if not self.form == 'center':
            raise ValueError("MPS is not centered.")
        if self.bond_centred:
            raise ValueError("MPS is already bond centered.")
        sites = sorted(self.sites)
        c_site = self.c_site
        if side == 'left':
            centre_tensor = self.tensors[c_site]
            B, s, Ul, Ur = bond_centre_l(centre_tensor)
            self.tensors[c_site] = B
            for site in sites:
                if site < c_site:
                    self.tensors[site] = Ul.conj().T @ self.tensors[site] @ Ul
                elif site >= c_site:
                    self.tensors[site] = Ur @ self.tensors[site] @ Ur.conj().T
            self.R = Ur @ self.R
            self.L = self.L @ Ul
            self.form = 'bond'
            self.schmidt = s # store the singular values across the bond
            self.c_site -= 1 # shift the centre to the left

        elif side == 'right':
            centre_tensor = self.tensors[c_site]
            A, s, Ul, Ur = bond_centre_l(centre_tensor)
            self.schmidt = s # store the singular values across the bond
            self.tensors[c_site] = A
            for site in sites:
                if site <= c_site:
                    self.tensors[site] = Ul.conj().T @ self.tensors[site] @ Ul
                elif site > c_site:
                    self.tensors[site] = Ur @ self.tensors[site] @ Ur.conj().T
            self.form = 'bond'
            self.c_site += 0 # By convention, the bond is labeled according the site to it's left
            self.R = Ur @ self.R
            self.L = self.L @ Ul
    
def left_orthogonal_tensor(M, max_bond_dim):
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
    assert len(M.shape) == 2, "Leftmost tensor must be a matrix."
    U, s, V = la.svd(M, full_matrices=False)
    PsiL[sites[0]] = U
    G = np.diag(s) @ V
    for i in sites[1:-1]:
        M = statedict[i]
        M_eff = G @ M
        M_lorth, G = left_orthogonal_tensor(M_eff, max_bond_dim)
        PsiL[i] = M_lorth
    # Handle rightmost tensor
    M = statedict[sites[-1]]
    assert len(M.shape) == 2, "rightmost tensor must be a matrix."
    M_eff = (G @ M.T).T
    norm = np.trace(M_eff @ M_eff.conj().T)
    M_eff = M_eff / np.sqrt(norm) # normalize
    PsiL[sites[-1]] = M_eff
    return PsiL

def right_orthogonal_tensor(M, max_bond_dim):
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
    return M_rorth, G
    
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
    assert len(M.shape) == 2, "Rightmost tensor must be a matrix."
    U, s, V = la.svd(M.T, full_matrices=False)
    PsiR[sites[0]] = V.T
    G = U @ np.diag(s)
    for i in sites[1:-1]:
        M = statedict[i]
        M_eff = M @ G
        M_rorth, G = right_orthogonal_tensor(M_eff, max_bond_dim)
        PsiR[i] = M_rorth
    M = statedict[sites[-1]]
    assert len(M.shape) == 2, "leftmost tensor must be a matrix."
    M_eff = M @ G
    norm = np.trace(M_eff.conj().T @ M_eff)
    M_eff = M_eff / np.sqrt(norm)
    PsiR[sites[-1]] = M_eff
    return PsiR

def centralize_state(statedict, c_site, max_bond_dim):
    psi_centre = {}

    # Handle left side of chain
    sites_l = sorted([i for i in statedict.keys() if i < c_site])
    M = statedict[sites_l[0]]
    U, s, V = la.svd(M, full_matrices=False)
    psi_centre[sites_l[0]] = U
    Gl = np.diag(s) @ V
    for i in sites_l[1:]:
        M = statedict[i]
        M_eff = Gl @ M
        M_lorth, Gl = left_orthogonal_tensor(M_eff, max_bond_dim)
        psi_centre[i] = M_lorth
    
    # Handle right side of chain
    sites_r = sorted([i for i in statedict.keys() if i > c_site], reverse=True)
    M = statedict[sites_r[0]]
    U, s, V = la.svd(M.T, full_matrices=False)
    psi_centre[sites_r[0]] = V.T
    Gr = U @ np.diag(s)
    for i in sites_r[1:]:
        M = statedict[i]
        M_eff = M @ Gr
        M_rorth, Gr = right_orthogonal_tensor(M_eff, max_bond_dim)
        psi_centre[i] = M_rorth

    # Handle centre tensor and normalize
    centre_tensor = Gl @ statedict[c_site] @ Gr
    centre_tensor = centre_tensor / np.sqrt(
        ncon((centre_tensor, centre_tensor.conj()), ((1, 2, 3), (1, 2, 3)))) # normalize
    psi_centre[c_site] = centre_tensor
    
    return psi_centre

def shift_centre_r(C, B):
    """
    Given a centre tensor C and a right canonical tensor B,
    shift the centre to the right by one site.
    """
    A, T = left_orthogonal(C)
    C_new = T @ B
    return A, C_new

def shift_centre_l(C, A):
    """
    Given a centre tensor C and a left canonical tensor A,
    shift the centre to the left by one site.
    """
    B, T = right_orthogonal(C)
    C_new = A @ T
    return C_new, B

def bond_centre_r(C):
    """
    Given a centre tensor, decompose into a left-orthogonal tensor
    and the SVD of the centre term. This svd gives a diagonal matrix, 
    and two unitaries which are the left and right gauge transformations.
    """
    A, T = left_orthogonal(C)
    UL, S, UR = la.svd(T, full_matrices=False)#
    return A, S, UL, UR

def bond_centre_l(C):
    """
    Given a centre tensor, decompose into a right-orthogonal tensor
    and the SVD of the centre term. This svd gives a diagonal matrix, 
    and two unitaries which are the left and right gauge transformations.
    """
    B, T = right_orthogonal(C)
    UL, S, UR = la.svd(T, full_matrices=False)#
    return B, S, UL, UR

def overlap(state_1, state_2):
    """
    Compute inner product between two states
    """
    assert sorted(state_1.sites) == sorted(state_2.sites), "States need to be on the same lattice."
    sites = sorted(state_1.sites)
    L = state_2[sites[0]].conj().T @ state_1[sites[0]]
    R = state_1[sites[-1]].T @ state_2[sites[-1]].conj()
    for i in sites[1:-1]:
        L = ncon((L, state_1[i], state_2[i].conj()),
                 ((1, 2), (3, 2, -2), (3, 1, -1)))
    return np.trace(L @ R)

def partite_entropy(state, site):
    """
    Compute the 2nd renyi entropy of the state partitioned across site and site+1
    Inputs:
        state: mps object
        site: int
    Returns:
    """
    tool_state = copy.copy(state)
    tool_state.centralize(site)
    _, s, _, _ = bond_centre_r(tool_state[site])
    purity = np.sum([val**4 for val in s])
    return -np.log(purity)

def random_mps(N, d, D, seed=0):
    """
    Unnormalized random MPS state generator
    """
    r = np.sqrt(2*D*d) # rough normalization factor
    np.random.seed(seed)
    statedict = {}
    sites = np.arange(N)
    statedict[sites[0]] = (np.random.normal(size=(d, D)) + 1j*np.random.normal(size=(d, D)))/r
    for i in sites[1:-1]:
        statedict[i] = (np.random.normal(size=(d, D, D)) + 1j*np.random.normal(size=(d, D, D)))/r
    statedict[sites[-1]] = (np.random.normal(size=(d, D)) + 1j*np.random.normal(size=(d, D)))/r
    state = mps(statedict)
    return state


    for i in range(N):
        A = np.random.rand(d, D, D) + 1j*np.random.rand(d, D, D)
        As.append(A)
    state = mps(As)
    state.right_canonical()
    return state
