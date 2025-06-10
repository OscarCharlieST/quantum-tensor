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

class mpo:
    def __init__(self, Ws, l, r):
        # W is the list of (position, tensor) objects
        # l and r are the left and right indices
        self.tensors = {}
        for W in Ws:
            self.tensors[W[0]] = W[1]
        self.sites = self.tensors.keys()
        self.l = l
        self.r = r
        self.d = Ws[0][1].shape[0]
        self.D = Ws[0][1].shape[2]
    
    def __getitem__(self, position):
        if position in self.tensors.keys():
            return self.tensors[position]
        else:
            return None
        
    def __setitem__(self, position, tensor):
        self.tensors[position] = tensor


def uniform_MPO(W, l, r, N):
    """
    Construct a uniform MPO from a single tensor W.
    W is the tensor at position 0.
    l and r are the left and right indices of the MPO.
    N is the number of sites in the chain.
    Sites are set to be 0, 1, 2, ..., N-1.
    """
    Ws = []
    for i in range(N):
        Ws.append((i, W))
    return mpo(Ws, l, r)

def contract_left(L, A, W):
    return ncon((L, A, A.conj(), W),
                ((1, 2, 3), (4, 1, -1), (5, 2, -2), (4, 5, 3, -3)))
def contract_right(R, A, W):
    return ncon((R, A, A.conj(), W),
                ((1, 2, 3), (4, -1, 1), (5, -2, 2), (4, 5, -3, 3)))

def tilted_ising(J=1, h=0.25, g=-0.525, N=1):
    """
    Default parameters taken from 1702.08894
    Construct the tilted Ising Hamiltonian for N spins as an MPO.
    H = -J z_i z_i+1 + h z_i + g x_i
    where z_i and x_i are the Pauli Z and X operators, respectively.
    """
    x, z = [pauli('x'), pauli('z')]
    W = np.zeros((2, 2, 3, 3), dtype=np.complex64)
    W[:, :, 0, 0] = np.eye(2)
    W[:, :, 2, 2] = np.eye(2)
    W[:, :, 0, 1] = -J * z
    W[:, :, 1, 2] = z
    W[:, :, 0, 2] = h * z + g * x
    l = np.array([1, 0, 0])
    r = np.array([0, 0, 1]) # contract with these left and right of the MPO chain
    return uniform_MPO(W, l, r, N)

def norm_operator(N, d):
    W = np.zeros((d, d, 2, 2), dtype=np.complex64)
    W[:, :, 0, 0] = np.eye(d)
    l = np.array([1, 0])
    r = np.array([1, 0])
    return uniform_MPO(W, l, r, N)

def total_z(N):
    W = np.zeros((2, 2, 2, 2), dtype=np.complex64)
    W[:, :, 0, 0] = np.eye(2)
    W[:, :, 0, 1] = pauli('z')
    W[:, :, 1, 1] = np.eye(2)
    l = np.array([1, 0])
    r = np.array([0, 1])
    return uniform_MPO(W, l, r, N)

def single_site_pauli(N, site, pauli_type='z'):
    W = np.zeros((2, 2, 2, 2), dtype=np.complex64)
    W[:, :, 0, 0] = np.eye(2)
    W[:, :, 1, 1] = np.eye(2)
    l = np.array([1, 0])
    r = np.array([1, 0])
    pauli_mpo = uniform_MPO(W, l, r, N)
    W_pauli = W.copy()
    W_pauli[:, :, 0, 1] = pauli(pauli_type)
    pauli_mpo[site] = W_pauli
    return pauli_mpo

def mpo_expect(state, operator):
    """
    Compute the expectation value of an operator O with respect to the MPS Psi.
    L and R are the left and right environments, respectively.
    For a left(right) canconical MPS, L(R) should be the identity.
    O is an MPO with left and right indices l and r.
    """
    assert sorted(state.sites) == sorted(operator.sites), "MPS and MPO sites do not match"

    L = state.L.conj().T @ state.L
    R = state.R @ state.R.conj().T

    l = operator.l
    r = operator.r 

    L = ncon((L, l), ((-1,-2), (-3,)))
    for i in sorted(state.sites):
        L = contract_left(L, state[i], operator[i])
    Wexpect = ncon((L, R, r), ((1, 2, 3), (1, 2), (3,)))
    return Wexpect