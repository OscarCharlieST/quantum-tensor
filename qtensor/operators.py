import copy
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

    # basic methods
    
    def __getitem__(self, position):
        if position in self.tensors.keys():
            return self.tensors[position]
        else:
            return None
        
    def __setitem__(self, position, tensor):
        self.tensors[position] = tensor

    def __deepcopy__(self, memo):
        new_tensors = {k: copy.deepcopy(v, memo) for k, v in self.tensors.items()}
        new_mpo = mpo(list(new_tensors.items()), 
                      copy.deepcopy(self.l, memo), copy.deepcopy(self.r, memo))
        return new_mpo

    # more complex functions
    def combine(self, mpo2, after=True):
        """
        Turn sequentially applied MPOs into a single MPO.
        mpo2 is the operator applied to the ket *after* the current mpo.
        """
        assert self.d == mpo2.d, "MPOs must have the same local dimension"
        site_set_1 = set(self.sites)
        l1 = self.l
        r1 = self.r
        site_set_2 = set(mpo2.sites)
        l2 = mpo2.l
        r2 = mpo2.r
        for site in site_set_1.union(site_set_2):
            if site in site_set_1.intersection(site_set_2):
                W1 = self.tensors[site]
                W2 = mpo2.tensors[site]
            elif site in site_set_1:
                W1 = self.tensors[site]
                W2 = ncon((np.eye(self.d), np.eye(mpo2.D)),
                          ((-1, -2), (-3, -4)))
            else:
                W1 = ncon((np.eye(self.d), np.eye(self.D)),
                          ((-1, -2), (-3, -4)))
                W2 = mpo2.tensors[site]
            if after:
                Wnew = ncon((W1, W2), ((-1, 1, -3, -5), (1, -2, -4, -6)))
                self.l = np.concatenate((l1, l2))
                self.r = np.concatenate((r1, r2))
            else:
                Wnew = ncon((W2, W1), ((-1, 1, -3, -5), (1, -2, -4, -6)))
                self.l = np.concatenate((l2, l1))
                self.r = np.concatenate((r2, r1))
            sp = Wnew.shape
            Wnew.reshape(sp[0], sp[1], sp[2]*sp[3], sp[4]*sp[5])
            self.tensors[site] = Wnew
        self.D = self.l.shape[0]
        


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
    W[:, :, 0, 1] = J * z
    W[:, :, 1, 2] = z
    W[:, :, 0, 2] = h * z + g * x
    l = np.array([1, 0, 0])
    r = np.array([0, 0, 1]) # contract with these left and right of the MPO chain
    return uniform_MPO(W, l, r, N)

def thermofield_hamiltonian(H):
    """
    Takes a specific form of2 site hamiltonian H
    where Hl, Hr are the left and right of the two site term
    and h the 1-site term

    I   Hl  h
    0   0   Hr
    0   0   I

    and makes it into a thermofield HxI + IxH via

    I   Hla Hlb ha+hb
    0   0   0   Hrb
    0   0   0   Hra
    0   0   0   I

    There may be a more general perscription, but this is efficient for our hamiltonian

    Note that it only works where the two local terms can be written as tensor product over the two sites. 
    """
    H_th = []
    for i in H.sites:
        W = H[i]
        Hl = W[:, :, 0, 1]
        Hr = W[:, :, 1, 2]
        h = W[:, :, 0, 2]

        W_th = np.zeros((4, 4, 4, 4), dtype=np.complex64)
        W_th[:, :, 0, 0] = np.kron(np.eye(2), np.eye(2))
        W_th[:, :, 0, 1] = np.kron(Hl, np.eye(2))
        W_th[:, :, 0, 2] = np.kron(np.eye(2), Hl)
        W_th[:, :, 0, 3] = np.kron(h, np.eye(2)) + np.kron(np.eye(2), h)
        W_th[:, :, 1, 3] = np.kron(Hr, np.eye(2))
        W_th[:, :, 2, 3] = np.kron(np.eye(2), Hr)
        W_th[:, :, 3, 3] = np.kron(np.eye(2), np.eye(2))
        H_th.append((i, W_th))
    l = np.array([1, 0, 0, 0])
    r = np.array([0, 0, 0, 1])
    return mpo(H_th, l, r)

def symmetric_thermofield(H):
    """
    Improved version of thermofield_hamiltonian that works for arbitrarily many 2-site terms.
    Takes in 2(n+2) x 2(n+2) hamiltonian of the form
    I  Hl1 Hl2 ... Hln  h
    0   0   0  ...  0  Hrn
    ¦   ¦   ¦  `.,  ¦   ¦
    0   0   0  ...  0  Hr2
    0   0   0  ...  0  Hr1
    0   0   0  ...  0   I

    and makes it into a symetric thermofield HxI + IxH
    """
    D = H.D
    H_th = []
    for i in H.sites:
        W = H[i]
        Hls = [W[:, :, 0, k] for k in range(1, D-1)]
        Hrs = [W[:, :, k, D-1] for k in range(1, D-1)]
        h = W[:, :, 0, D-1]

        W_th = np.zeros((4, 4, 2*D-2, 2*D-2), dtype=np.complex64)
        W_th[:, :, 0, 0] = np.kron(np.eye(2), np.eye(2))
        W_th[:, :, 2*D-3, 2*D-3] = np.kron(np.eye(2), np.eye(2))

        # Two-site terms
        for k, Hl, Hr in zip(range(1, D-1), Hls, Hrs):
            W_th[:, :, 0, 2*k-1] = np.kron(Hl, np.eye(2)) # real copy
            W_th[:, :, 0, 2*k] = np.kron(np.eye(2), Hl) # auxilliary
            W_th[:, :, 2*k-1, 2*D-3] = np.kron(Hr, np.eye(2)) 
            W_th[:, :, 2*k, 2*D-3] = np.kron(np.eye(2), Hr)

        # Local terms
        W_th[:, :, 0, 2*D-3] = np.kron(h, np.eye(2)) + np.kron(np.eye(2), h)
        H_th.append((i, W_th))

    l = np.zeros(2*D-2)
    l[0] = 1
    r = np.zeros(2*D-2)
    r[-1] = 1
    return mpo(H_th, l, r)

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

def single_site_pauli(site, pauli_type='z'):
    W = np.zeros((2, 2, 1, 1), dtype=np.complex64)
    W[:, :, 0, 0] = pauli(pauli_type)
    l = np.array([1,])
    r = np.array([1,])
    return mpo([(site, W)], l, r)

def two_site_pauli(site_l, pauli_l='z', pauli_r='z'):
    W_l = np.zeros((2, 2, 1, 1), dtype=np.complex64)
    W_r = copy.copy(W_l)
    W_l[:, :, 0, 0] = pauli(pauli_l)
    W_r[:, :, 0, 0] = pauli(pauli_r)
    l = np.array([1,])
    r = np.array([1,])
    return mpo([(site_l, W_l), (site_l+1, W_r)], l, r)

def extensive_twosite_local_term(H, site):
    """
    Construct a local (2site) energy term between (site, site+1)
    Convention is that term takes the full 2-site term
    and half of the local term at each end.
    """
    Wl = copy.deepcopy(H[site])
    Wr = copy.deepcopy(H[site+1])
    i_one = Wl.shape[-1] - 1 # works with normal or thermofield
    Wl[:, :, 0, i_one] = Wl[:, :, 0, i_one] / 2 # only need half the single site term at each end
    Wr[:, :, 0, i_one] = Wr[:, :, 0, i_one] / 2
    l = np.zeros(Wl.shape[-2])
    l[0] = 1
    r = np.zeros(Wr.shape[-1]) # contract with these left and right of the MPO chain
    r[-1] = 1
    return mpo([(site, Wl), (site+1, Wr)], l, r)

def extensive_as_terms(H):
    """
    Takes extensive local mpo, returns list of local summands
    ### SHOULD CHANGE THIS TO BE DICTIONARY!
    ## changed, but miht be some dependency issues
    """
    loc_ops = {}
    for site in sorted(H.sites):
        if not site == max(H.sites):
            loc_ops[site] = extensive_twosite_local_term(H, site)
        else:
            return loc_ops

def expect(state, operator):
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

def local_expect(state, operator):
    working_state = copy.copy(state) # don't change original state
    x_max = max(operator.sites)
    working_state.centralize(x_max) # so we can contract with identities either side of operator
    L = working_state.L.conj().T @ working_state.L # should be id
    R = working_state.R @ working_state.R.conj().T # should be id
    l = operator.l
    r = operator.r 
    L = ncon((L, l), ((-1,-2), (-3,)))
    for i in sorted(operator.sites):
        L = contract_left(L, working_state[i], operator[i])
    Wexpect = ncon((L, R, r), ((1, 2, 3), (1, 2), (3,)))
    return Wexpect

def pauli(i):
    """
    Returns the Pauli matrix corresponding to the given string.

    Parameters
    ----------
    'i' : str
        The string representing the Pauli matrix to return. Must be one of 
        'x', 'y', or 'z'.

    Returns
    -------
    (2, 2) complex array
        The Pauli matrix corresponding to the given string.

    Raises
    ------
    ValueError
        If the input string is not one of 'x', 'y', or 'z'.

    """
    if i == 'x':
        return np.array([[0, 1], [1, 0]])
    elif i == 'y':
        return np.array([[0, -1j], [1j, 0]])
    elif i == 'z':
        return np.array([[1, 0], [0, -1]])
    else:
        raise ValueError("Invalid input: must be one of 'x', 'y', or 'z'.")

def first_order_deformation_generator(beta_profile, J=1, h=0.25, g=-0.525, t=1.0):
    """
    Specifically for TFI hamiltonian and a temperature profile and a parameter t, calculate
    an mpo for h_i + i t [h_i, H]
    ! NOTE !
    This is the appropriate generator for the density matrix. 
    Whether or not you can simply plug it into the purification formulae I dont know...
    """
    y = pauli('y')
    delta_beta = list(beta_profile[1:] - beta_profile[:-1])
    delta_beta.append(delta_beta[-1])  # extend to match length of beta_profile, with gradient equal at last point
    initial_mpo = tilted_ising(J, h, g, len(beta_profile))
    Ws = initial_mpo.tensors # is this right?
    newWs = []
    for i in Ws.keys():
        W = copy.deepcopy(Ws[i])
        W[:, :, 0, 1] = W[:, :, 0, 1]
        W[:, :, 1, 2] = W[:, :, 1, 2]*beta_profile[i] + 2*g*t*delta_beta[i]*y
        W[:, :, 0, 2] = W[:, :, 0, 2]*beta_profile[i]
        newWs.append((i,W))
    return mpo(newWs, initial_mpo.l, initial_mpo.r)

def ising_commutator(site, J, g):
    """
    Specifically for TFI hamiltonian, calculate the commutator
    [h_i, H] where h_i is the local term at site i and H is the full hamiltonian.
    Uses the convention that the h_i contains local terms on site i
    and 2-local terms on i and i+1.
    h_i = J z_i z_i+1 + h z_i + g x_i
    [h_i, h_i+1] = Jg (z_i [z_i+1, x_i+1]) 
                 = 2j*J*g z_i y_i+1
    [h_i, h_i-1] = -2j*J*g z_i-1 y_i 
    [h_i, H] = [h_i, h_i+1] + [h_i, h_i-1] = 2j*J*g*(z_i y_i+1 - z_i-1 y_i)
    returns it as a single 3-site mpo.
    """
    x, y, z = [pauli(i) for i in ['x', 'y', 'z']]
    Wl, Wi, Wr = [np.zeros((2, 2, 3, 3), dtype=np.complex128)]*3
    Wl[:, :, 0, 0] = np.eye(2)
    Wl[:, :, 2, 2] = np.eye(2)
    Wl[:, :, 0, 1] = -2j*J*g * z

    Wi[:, :, 0, 0] = np.eye(2)
    Wi[:, :, 2, 2] = np.eye(2)
    Wi[:, :, 0, 1] = 2j*J*g * z
    Wi[:, :, 1, 2] = y

    Wr[:, :, 0, 0] = np.eye(2)
    Wr[:, :, 2, 2] = np.eye(2)
    Wr[:, :, 1, 2] = y

    l = np.array([1, 0, 0])
    r = np.array([0, 0, 1]) # contract with these left and right of the MPO chain
    return mpo([(site-1, Wl), (site, Wi), (site+1, Wr)], l, r)

def dhi_dt(state, site, J=1, g=-0.525):
    """
    Time derivative of local energy density at site
    """
    commutator = ising_commutator(site, J, g)
    thf_commutator = symmetric_thermofield(commutator)
    return -1j*local_expect(state, thf_commutator)
