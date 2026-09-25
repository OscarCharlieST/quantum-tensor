import numpy as np
import matplotlib.pyplot as plt
import copy
import qtensor.states as states 
import qtensor.operators as ops
import qtensor.simulation.finiteTDVP as sim
import qtensor.simulation.updatemethod as methods
from ncon import ncon

def inf_T_thermofield(N, D, noise=0):
    """
    Builds an infinite temperature thermofield with N sites and bond dimension D
    """
    M_L = np.zeros((4, 1, D))
    M_L[:,0,0] = np.array([1, 0, 0, 1])
    M_R = np.zeros((4, D, 1))
    M_R[:,0,0] = np.array([1, 0, 0, 1])
    M = np.zeros((4,D,D))
    M[:, 0, 0] = np.array([1, 0, 0, 1])
    Ms = [M for _ in range(N-2)]
    if noise:
        Ms = [M + noise * np.random.rand(4, D, D) for M in Ms]
    tensor_list = [M_L] + Ms + [M_R]
    return states.mps(tensor_list)

# The doubled physical index is i = 2 * a_phys + b_anc, so exchanging the two
# copies is the permutation 2a + b -> 2b + a. inf_T_thermofield's seed
# [1, 0, 0, 1] = |00> + |11> is a fixed point of it.
SWAP_COPIES = [0, 2, 1, 3]


def swap_copies(psi):
    """
    S|psi>: exchange the physical and ancilla copy on every site. Takes and
    returns {site: tensor}; an mps is accepted and unwrapped.

    In the vectorization psi <-> M this is transposition, S|M>> = |M^T>>. The
    thermofield double is built from rho^(1/2) and so is a fixed point of it,
    and H_sym = H x I + I x H keeps it there (M -> e^{-iHt} M e^{-iHt} for
    real symmetric H). H_asym = H x I - I x H does *not*: it sends
    M -> e^{-iHt} M e^{+iHt}, which preserves Hermiticity of M rather than
    symmetry.
    """
    tensors = psi.tensors if hasattr(psi, 'tensors') else psi
    return {site: T[SWAP_COPIES, :, :].copy() for site, T in tensors.items()}


def swap_odd_amplitude(psi):
    """
    |(1 - S) psi / 2|: how far the state is from swap-symmetric. Linear in the
    asymmetry, unlike <psi|S|psi>, which is 1 - 2 a^2 and so hides it.
    Saturates at 1/sqrt(2), where psi is orthogonal to its own swap.
    """
    norm = abs(states.overlap(psi, psi))
    ov = states.overlap(psi, states.mps(swap_copies(psi))) / norm
    return float(np.sqrt(max(0.0, (1.0 - ov.real) / 2.0)))


def symmetrize_copies(psi, max_bond_dim=None):
    """
    Project onto the swap-even sector: (psi + S psi)/||.||, recompressed to
    max_bond_dim (the state's own largest bond dimension by default, since the
    sum lives at bond 2D). Does not mutate psi; returns a normalized state.

    Use it where the dynamics is swap-even and the numerics is not: an MPS
    truncated at a rank-deficient bond picks its null directions arbitrarily
    and drifts out of the sector. See lyapunov/tdvp_lyapunov/SWAP_SYMMETRY.md.
    """
    if max_bond_dim is None:
        max_bond_dim = max(T.shape[2] for T in psi.tensors.values())
    out = copy.deepcopy(psi)
    out.add(states.mps(swap_copies(psi)), max_bond_dim=max_bond_dim)
    return out


def th_onesite(A, site):
    """
    Takes a single-site spin 1/2 operator (as amatrix) and returns the thermofield mpo reprsentation 
    """
    W = np.zeros((4, 4, 1, 1))
    W[:, :, 0, 0] = np.kron(A, np.eye(2)) + np.kron(np.eye(2), A)
    return ops.mpo([site, W], np.array([1,]), np.array([1,]))

def tf_twosite(A, B, site_l):
    """
    Taskes two 2x2 matrices and returns AB_a + AB_b on each copy of Hilbert space
    """
    Wl = np.zeros((4, 4, 1, 2))
    Wr = np.zeros((4, 4, 2, 1))
    Wl[:, :, 0, 0] = np.kron(A, np.eye(2))
    Wl[:, :, 0, 1] = np.kron(np.eye(2), A)
    Wr[:, :, 0, 0] = np.kron(B, np.eye(2))
    Wr[:, :, 1, 0] = np.kron(np.eye(2), B)
    return ops.mpo([(site_l, Wl), (site_l+1, Wr)], np.array([1, ]), np.array([1,]))


def thermofield_hamiltonian(H, asym=False):
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
    a=1
    if asym:
        a=-1

    H_th = []
    for i in H.sites:
        W = H[i]
        Hl = W[:, :, 0, 1]
        Hr = W[:, :, 1, 2]
        h = W[:, :, 0, 2]

        W_th = np.zeros((4, 4, 4, 4), dtype=np.complex64)
        W_th[:, :, 0, 0] = np.kron(np.eye(2), np.eye(2))
        W_th[:, :, 0, 1] = np.kron(Hl, np.eye(2))
        W_th[:, :, 0, 2] = a * np.kron(np.eye(2), Hl)
        W_th[:, :, 0, 3] = np.kron(h, np.eye(2)) + a * np.kron(np.eye(2), h)
        W_th[:, :, 1, 3] = np.kron(Hr, np.eye(2))
        W_th[:, :, 2, 3] = np.kron(np.eye(2), Hr)
        W_th[:, :, 3, 3] = np.kron(np.eye(2), np.eye(2))
        H_th.append((i, W_th))
    l = np.array([1, 0, 0, 0])
    r = np.array([0, 0, 0, 1])
    return ops.mpo(H_th, l, r)

def finite_T_thermofield(beta,  H, steps=100,
                         initial_state=None, N=2, D=4,
                         plot=True, method=None, sym_every=0):
    """
    Imaginary-time evolve to inverse temperature beta.

    sym_every: project onto the swap-even sector every this many steps
    (0 = never). The build is where the state is most rank-deficient, and a
    rank-deficient truncation is what knocks it out of the sector; see
    symmetrize_copies. Costs one MPS sum and a recompression per projection,
    and forces the cached environments to be rebuilt on the next step.
    """
    if not initial_state:
        state = inf_T_thermofield(N, D)
    else:
        state = copy.deepcopy(initial_state)
        # initial state must be infinite temperature
        pass
    if not method:
        method = methods.lanczos_method(max_iters=8)
    bond_dim = max(T.shape[2] for T in state.tensors.values())
    t_f = -1j * beta * 1/4
    dt = t_f / steps
    # steps + 1 evolutions, recording before each: what sim.tdvp does, kept so
    # that existing builds reproduce. Note it leaves the state one dt past
    # t_f, i.e. at beta * (1 + 1/steps).
    times, energies = [], []
    env = None
    for step in range(steps + 1):
        times.append(abs(step * dt) * 4)
        energies.append(np.real(ops.expect(state, H)) / 2)
        state, env = sim.tdvp_step(state, H, dt, method, env)
        if sym_every and (step + 1) % sym_every == 0:
            state = symmetrize_copies(state, bond_dim)
            env = None          # the environments belong to the old state
    time = np.array(times)
    energy = np.array(energies)
    if plot:
        _, ax = plt.subplots(1,1)
        ax.plot(time, energy)
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(r"$E$")
        print("Energy at finite temperature:", energy[-1])
    return state, time, energy

def near_thermal(H, profile, initial_state, steps=100, sym_every=0):
    """
    Perform imaginary time evolution with a spatially varying temperature profile.
    
    :param H: Symmetric thermofield hamiltonian
    :param profile: inverse temperature profile 
    :param D: max bond dimension
    :param steps: Description
    :param initial_state: Description
    """

    assert len(H.sites) == len(profile), "temp profile incorrect length"
    
    b_profile_r = (profile + np.roll(profile, -1)) / 2 # Bond to the left of site avg temp
    
    H_new = []
    for site, beta, br in zip(H.sites, profile, b_profile_r):
        W = copy.copy(H[site]) # don't actually edit the hamiltonian
        W[:, :, :-1, 1:-1] = W[:, :, :-1, 1:-1] * br # twosite terms get average temperature across the bond
        W[:, :, 1:-1, 1:] = W[:, :, 1:-1, 1:]
        W[:, :, 0, -1] = W[:, :, 0, -1] * beta # onesite term get local temperature
        H_new.append((site, W))

    H_eff = ops.mpo(H_new, H.l, H.r)
    state, _, _ = finite_T_thermofield(1, H_eff, initial_state=initial_state,
                                       steps=steps, plot=False,
                                       sym_every=sym_every)
    return state

def near_thermal_delta_function(H, bond, delta_beta, initial_state, steps=10):
    """
    Perform imaginary time evolution with a delta function temperature profile, i.e. a local quench in temperature
    :param H: Symmetric thermofield hamiltonian
    :param site: Bond between (site, site+1) to apply generator to
    :param delta_beta: change in inverse temperature at the bond.
    :param initial_state: state to apply generator to - should be thermal at some temperature
    """

    root_beta = np.sqrt(delta_beta + 0j) # make complex to avoid issues with negative delta beta

    H_new = []
    for site in H.sites:
        if not site==bond and not site==bond+1:
            W = copy.copy(H[site]) # don't actually edit the hamiltonian
            W[:, :, :-1, 1:] = W[:, :, :-1, 1:] * 0 # Set everything to zero
            H_new.append((site, W))
        else:
            W = copy.copy(H[site])
            W[:, :, :-1, 1:] = W[:, :, :-1, 1:] * root_beta 
            W[:, :, 0, -1] = W[:, :, 0, -1] * root_beta
            H_new.append((site, W))

    H_eff = ops.mpo(H_new, H.l, H.r)
    state, _, _ = finite_T_thermofield(1, H_eff, steps=steps,
                                       initial_state=initial_state,
                                       plot=False)
    return state

def near_thermal_first_order_deformed(D, beta_profile, J=1, h=0.25, g=-0.525, t=1.0, steps=100):
    """
    Produces a near-thermal state with an additional deformation corresponding to the first order change in energy 
    For low temperatures, the near-thermal density matrix is rho = 1/z (1 - beta_i h_i)
    The time derivative of the density matrix is i[rho, H]
    So to first order in t, rho(t) = rho + i t [rho, H] = 1/z (1 + beta_i(h_i + i t [h_i, H]))
    So we modify the generator of the thermal state h_i -> h_i + i t [h_i, H]
    the larger t is, the further the deviation from the near thermal ansatz
    """
    fodh = ops.first_order_deformation_generator(beta_profile, J, h, g, t)
    tf_fodg = thermofield_hamiltonian(fodh)
    state, _, _ = finite_T_thermofield(1, len(beta_profile), D, tf_fodg, steps, plot=False, method=sim.method_exact)
    return state

def single_copy_expectation(psi, O):
    """
    Given an operator O on the un-doubled hilbert space and a thermofield state psi,
    calculate the expectation of O as the expectation of (O x I).

    This way of taking the expectation is no symmetric;
    disentanglers can be applied to the auxilliary space. 

    Parameters:     
        psi: Thermofield MPS - doesn't have to be symmetric but does need to be on doubled
          hilbert space
        O: Local mpo on the single hilbert space      

    Returns:
        << O >>
    """
    psi = copy.deepcopy(psi)
    op_sites = O.sites
    min_site = min(op_sites)
    psi.centralize(min_site)

    l, r = O.l, O.r
    d2, Dl, Dr = psi[min_site].shape
    d = int(np.sqrt(d2))

    L_con = ncon((np.eye(Dl), l),
                 ((-1, -2), (-3,)))
    
    for site in op_sites:
        A = psi[site]
        _, Dl, Dr = A.shape
        M = psi[site].reshape(d, d, Dl, Dr)
        W = O[site]
        L_con = ncon((L_con, M, M.conj(), W),
                     ((1, 2, 3), (4, 6, 1, -1), (5, 6, 2, -2), (4, 5, 3, -3)))
    return ncon((L_con, r),
                ((1, 1, 2), (2, )))




