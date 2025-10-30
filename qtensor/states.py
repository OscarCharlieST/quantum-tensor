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
        self.sites = self.tensors.keys()
        self.L = kwargs['L'] if 'L' in kwargs else np.eye(Ms[0].shape[1])
        self.R = kwargs['R'] if 'R' in kwargs else np.eye(Ms[-1].shape[2])
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
    
    def left_canonical(self):
        PsiL, L_new, R_new = left_canonicalize(self.tensors, self.L, self.R)
        self.tensors = PsiL
        self.normalized = True
        self.L = L_new  
        self.R = R_new
        self.bond_centred = True
        self.centred = False
        self.form = 'left'
        self.c_site = max(self.sites)
        
    def right_canonical(self):
        PsiR, L_new, R_new = right_canonicalize(self.tensors, self.L, self.R)
        self.tensors = PsiR
        self.normalized = True
        self.L = L_new  
        self.R = R_new
        self.bond_centred = True
        self.centred = False
        self.form = 'right'
        self.c_site = min(self.sites)

    def centralize(self, c_site):
        """
        Centralize the MPS about the given site.
        Fully right canonicalise.
        Fully left canonicalise the MPS to the left of c_site.
        Absorb remaining right environments into the center.
        """
        if c_site not in self.sites:
            raise ValueError(f"Site {c_site} not in MPS.")
        sites = sorted(self.sites)
        left_state = {site: self.tensors[site] for site in sites if site < c_site}
        right_state = {site: self.tensors[site] for site in sites if site > c_site}
        centre_tensor = self.tensors[c_site]
        d, Dl, Dr = centre_tensor.shape
        if not self.form == 'left':
            PsiL, L_lcan, R_lcan = left_canonicalize(left_state, self.L, np.eye(Dl), diagonal=False)
        else:
            PsiL, L_lcan, R_lcan = left_state, self.L, np.eye(Dl)
        if not self.form == 'right':
            PsiR, L_rcan, R_rcan = right_canonicalize(right_state, np.eye(Dr), self.R, diagonal=False)
        else:
            PsiR, L_rcan, R_rcan = right_state, np.eye(Dr), self.R
        centre_tensor = R_lcan @ centre_tensor @ L_rcan
        centre_tensor = centre_tensor / np.sqrt(
            ncon((centre_tensor, centre_tensor.conj()), ((1, 2, 3), (1, 2, 3)))) # normalize
        PsiC = {c_site: centre_tensor}
        self.tensors = PsiL|PsiC|PsiR
        self.L = L_lcan 
        self.R = R_rcan 
        self.centred = True
        self.normalized = True
        self.form = 'centre'
        self.c_site = c_site
    
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
            
def left_orthogonal(M):
    """
    Left orthogonalize an MPS tensor M using QR decomposition.
    """
    d, Dl, Dr = M.shape
    M_mat = M.reshape(d*Dl, Dr)
    Q, R = la.qr(M_mat, mode='economic')
    A_new = Q.reshape(d, Dl, Dr)
    return A_new, R

def right_orthogonal(M): 
    """
    Right orthogonalize an MPS tensor M using RQ decomposition.
    """
    d, Dl, Dr = M.shape
    M = ncon(M, (-2, -1, -3)) # swap the first two indices for reshaping
    M_mat = M.reshape(Dl, d*Dr)
    R, Q = la.rq(M_mat, mode='economic')
    B_new = Q.reshape(Dl, d, Dr)
    B_new = ncon(B_new, (-2, -1, -3)) # swap the first two indices back
    return B_new, R

def left_canonicalize(statedict, rootL, rootR, diagonal=True):
    """
    Left canonicalize an MPS Psi
    Psi is assumed to have identity left and right environments.
    If this isn't the case, just absorb the environments into the MPS.
    """
    sites = sorted(statedict.keys()) # sorted from left to right
    PsiL = {}
    for i in sites:
        M = statedict[i]
        if i == min(sites):
            # If this is the first site, we can use the left environment directly:
            A, T = left_orthogonal(rootL @ M)
            PsiL[i] = A
            rootL_new = np.eye(rootL.shape[0])
        else:
            A, T = left_orthogonal(T @ M)
            PsiL[i] = A
    rootR = T @ np.sqrt(rootR) # incorporate the right environment
    rootR_new = rootR / la.norm(rootR) # normalize
    if not diagonal:
        return PsiL, rootL_new, rootR_new
    else:
        # diagonalize the right environment
        R = rootR_new @ rootR_new.conj().T
        Rdiag, v = la.eig(R)
        assert np.allclose(R, v @ np.diag(Rdiag) @ v.conj().T), "Eigen decomposition failed"
        rootR_new = np.diag(np.sqrt(Rdiag))
        # Absorb gauge transformations into the MPS tensors
        for i in sites:
            PsiL[i] = v.conj().T @ PsiL[i] @ v
        # Apply the gauge transformations to the left environment
        rootL_new = rootL_new @ v
        return PsiL, rootL_new, rootR_new
    
def left_canonicalize_compress(statedict, rootL, rootR, diagonal=True):
    """
    Left canonicalize an MPS Psi
    Psi is assumed to have identity left and right environments.
    If this isn't the case, just absorb the environments into the MPS.
    """
    sites = sorted(statedict.keys()) # sorted from left to right
    PsiL = {}


    first_site = sites[0]
    first_tensor = statedict[first_site]
    left_vec = np.ones(first_tensor.shape[1])
    first_tensor = ncon((left_vec, first_tensor), ((1), (-1, 1, -2)))
    U, S, Vh = la.svd(first_tensor, full_matrices=False)
    first_tensor = U
    PsiL[first_site] = first_tensor
    statedict[sites[1]] = np.diag(S) @ Vh @ statedict[sites[1]]

    for i in range(1, len(sites)):
        site = sites[i]
        M = statedict[site]
        A, T = left_orthogonal(T @ M)
        PsiL[i] = A
    rootR = T @ np.sqrt(rootR) # incorporate the right environment
    rootR_new = rootR / la.norm(rootR) # normalize
    if not diagonal:
        return PsiL, rootL_new, rootR_new
    else:
        # diagonalize the right environment
        R = rootR_new @ rootR_new.conj().T
        Rdiag, v = la.eig(R)
        assert np.allclose(R, v @ np.diag(Rdiag) @ v.conj().T), "Eigen decomposition failed"
        rootR_new = np.diag(np.sqrt(Rdiag))
        # Absorb gauge transformations into the MPS tensors
        for i in sites:
            PsiL[i] = v.conj().T @ PsiL[i] @ v
        # Apply the gauge transformations to the left environment
        rootL_new = rootL_new @ v
        return PsiL, rootL_new, rootR_new

def right_canonicalize(statedict, rootL, rootR, diagonal=True):
    """
    Right canonicalize an MPS dictionary with {site: tensor} structure.
    rootL and rootR are the (roots of) left and right environments, respectively.
    """
    sites = sorted(statedict.keys(), reverse=True) # sorted from right to left
    PsiR = {}
    for i in sites:
        M = statedict[i]
        if i == max(sites):
            # If this is the last site, we can use the right environment directly:
            B, T = right_orthogonal(M @ rootR)
            PsiR[i] = B
            rootR_new = np.eye(rootR.shape[0])
        else:
            B, T = right_orthogonal(M @ T)
            PsiR[i] = B
    rootL = rootL @ T # incorporate the left environment
    rootL_new = rootL / la.norm(rootL) # normalize
    if not diagonal:
        return PsiR, rootL_new, rootR_new
    else:
        # diagonalize the left environment
        L = rootL_new.conj().T @ rootL_new
        Ldiag, v = la.eig(L)
        assert np.allclose(L, v @ np.diag(Ldiag) @ v.conj().T), "Eigen decomposition failed"
        assert np.allclose(v.conj().T @ v, np.eye(v.shape[0])), "Eigenvectors are not orthonormal"
        rootL_new = np.diag(np.sqrt(Ldiag))
        # Absorb gauge transformations into the MPS tensors
        for i in sites:
            PsiR[i] = v.conj().T @ PsiR[i] @ v
        # Apply the gauge transformations to the right environment
        rootR_new = v.conj().T @ rootR_new
        return PsiR, rootL_new, rootR_new

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
    L = state_2.L.conj().T @ state_1.L
    R = state_1.R @ state_2.R.conj().T
    for site in sorted(state_1.sites):
        L = ncon((L, state_1[site], state_2[site].conj()),
                 ((1, 2), (3, 1, -1), (3, 2, -2)))
    return np.sqrt(np.trace(L @ R))

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
    np.random.seed(seed)
    As = []
    for i in range(N):
        A = np.random.rand(d, D, D) + 1j*np.random.rand(d, D, D)
        As.append(A)
    state = mps(As)
    state.right_canonical()
    return state
