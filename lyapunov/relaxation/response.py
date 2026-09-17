"""
Linear response from the tangent-space spectrum: observable weights, the
response function, and the relaxation time extracted from it.

The perturbation is taken to be a local operator kick,
`|dpsi(0)> = P_tangent O|psi*>`, which makes the response the
tangent-projected autocorrelation function of O. The weights then come out
manifestly non-negative and the whole calculation reduces to one vector.
See README.md ("From oscillatory phases to an exponential rate").
"""

import numpy as np
import scipy.linalg as la

import qtensor.operators as ops
import qtensor.simulation.updatemethod as methods
import lyapunov.relaxation.tangent_hamiltonian as tangent


# ---------------------------------------------------------------- observables
#
# The physical copy of the doubled Hilbert space is the *first* factor of
# the (d, d) index, matching thermofield.single_copy_expectation. So an
# observable on the physical copy alone is kron(A, I).

def single_copy_onesite(A, site):
    """MPO for A ⊗ I_aux at `site`: A on the physical copy, auxiliary untouched."""
    W = np.zeros((4, 4, 1, 1), dtype=complex)
    W[:, :, 0, 0] = np.kron(A, np.eye(2))
    return ops.mpo([(site, W)], np.array([1.0]), np.array([1.0]))


def single_copy_energy_density(site_l, J=1, h=0.25, g=-0.525):
    """
    MPO for the tilted-Ising energy density on the bond (site_l, site_l+1),
    on the physical copy only:  J z_l z_{l+1} + a_l + a_{l+1},  with
    a = (h z + g x)/2 so that summing over bonds recovers H.
    """
    Z = np.kron(ops.pauli('z'), np.eye(2))
    X = np.kron(ops.pauli('x'), np.eye(2))
    I4 = np.eye(4)
    a = 0.5 * (h * Z + g * X)

    Wl = np.zeros((4, 4, 1, 3), dtype=complex)
    Wl[:, :, 0, 0] = J * Z
    Wl[:, :, 0, 1] = I4
    Wl[:, :, 0, 2] = a

    Wr = np.zeros((4, 4, 3, 1), dtype=complex)
    Wr[:, :, 0, 0] = Z
    Wr[:, :, 1, 0] = a
    Wr[:, :, 2, 0] = I4

    return ops.mpo([(site_l, Wl), (site_l + 1, Wr)],
                   np.array([1.0]), np.array([1.0]))

def single_copy_current(site, J=1, h=0.25, g=-0.525):
    """
    MPO for the tilted-Ising energy current through `site`, on the physical
    copy only:  J g (y_i z_{i+1} - z_{i-1} y_i),  a three-site operator on
    (site-1, site, site+1).

    This is the current that satisfies continuity with the density above,

        d<h_l>/dt = <j_l> - <j_{l+1}>,

    where `h_l` is single_copy_energy_density on the bond between sites l
    and l+1, and `j` is indexed by *site* -- energy enters bond l through
    site l and leaves through site l+1.

    The symmetrization of the density is what fixes the form. For the
    unsymmetrized convention h_l = J z_l z_{l+1} + 2 a_l, which is the one
    `operators.ising_commutator` assumes, the current is instead
    -2 J g z_{l-1} y_l; the two densities differ by a lattice derivative
    and their currents are *not* interchangeable (pairing one with the
    other breaks continuity at O(1)).

    Only g drives energy transport: the longitudinal field h commutes with
    the z z coupling and drops out. It is accepted here so that this can be
    called with the same parameters as the density, but does not enter.
    """
    Y = np.kron(ops.pauli('y'), np.eye(2))
    Z = np.kron(ops.pauli('z'), np.eye(2))
    I4 = np.eye(4)

    # bond state 0 carries the -z y term, state 1 the +y z term
    Wl = np.zeros((4, 4, 1, 2), dtype=complex)
    Wl[:, :, 0, 0] = -J * g * Z
    Wl[:, :, 0, 1] = I4

    Wm = np.zeros((4, 4, 2, 2), dtype=complex)
    Wm[:, :, 0, 0] = Y
    Wm[:, :, 1, 1] = J * g * Y

    Wr = np.zeros((4, 4, 2, 1), dtype=complex)
    Wr[:, :, 0, 0] = I4
    Wr[:, :, 1, 0] = Z

    return ops.mpo([(site - 1, Wl), (site, Wm), (site + 1, Wr)],
                   np.array([1.0]), np.array([1.0]))

def pad_with_identity(O_mpo, sites, d=4):
    """
    Extend a local MPO to a {site: tensor} dict covering the whole chain by
    padding with bond-dimension-1 identity tensors. Valid because the
    observables above have length-1 boundary vectors, so the bond dimension
    is 1 at the edges of their support.
    """
    W = {}
    for site in sites:
        if O_mpo[site] is not None:
            W[site] = O_mpo[site]
        else:
            pad = np.zeros((d, d, 1, 1), dtype=complex)
            pad[:, :, 0, 0] = np.eye(d)
            W[site] = pad
    return W


# ------------------------------------------------------------------- weights

def observable_tangent_vector(O_mpo, A_L, A_R, C, V_L, basis_index_map, sites):
    """
    v_i = <b_i| O |psi*>, the components of O|psi*> along the orthonormal
    tangent basis.

    Same contraction as project_H_onto_tangent_basis, but the ket is psi*
    itself rather than a tangent vector, so no defect is carried and the
    precomputed environments apply directly on both sides: at site n the
    ket is the centre tensor C^n, with A_L environments to its left and A_R
    environments to its right.

    Since the tangent basis is orthogonal to psi*, the component of O|psi*>
    along |psi*> is dropped automatically -- the response is connected
    without having to subtract <O> by hand.
    """
    W = pad_with_identity(O_mpo, sites)
    left_envs, right_envs = tangent.build_environments(
        A_L, A_R, W, O_mpo.l, O_mpo.r, sites
    )

    dim = sum(n_null * D_r for _, _, (n_null, D_r) in basis_index_map)
    v = np.zeros(dim, dtype=complex)
    for site, start, (n_null, D_r) in basis_index_map:
        F = methods.apply_Heff_parts(
            C[site], W[site], left_envs[site - 1], right_envs[site + 1]
        )
        d, Dl, Dr = F.shape
        block = V_L[site].conj().T @ F.reshape(d * Dl, Dr)
        v[start:start + n_null * D_r] = block.reshape(-1)
    return v


def spectral_weights(omega, U, v):
    """
    Weights of the perturbation P O|psi*> on the tangent eigenmodes, given
    the eigendecomposition (omega, U) of H_tangent.

    With the kick and the measured observable both equal to O, the weight is
    |<k|O|psi*>|^2 -- non-negative, so A_O(omega) is a genuine spectral
    density and the Lorentzian fit is well posed.
    """
    u = U.conj().T @ v
    return np.abs(u) ** 2


def response_function(omega, weights, times):
    """
    C(t) = sum_k w_k cos(omega_k t), normalized to C(0) = 1.

    This is the entire time trace, for any t, from one diagonalization --
    no time stepping. Evaluate it well past the expected relaxation time to
    see the recurrences.
    """
    return (np.cos(np.outer(times, omega)) @ weights) / weights.sum()


# --------------------------------------------------------------- timescales

def timescales(omega, weights):
    """
    The window bounds of the pole approximation, estimated from the weighted
    spectrum: exponential decay is only expected for t_zeno << t << t_heis.

    t_zeno  = 1/sqrt(Var[omega])   -- below this the decay is quadratic.
    t_heis  = 2*pi/spacing         -- above this, recurrences.

    The spacing uses the *effective* number of modes carrying weight
    (inverse participation ratio) over the weighted bandwidth, rather than
    the raw dimension, since modes with no overlap cannot dephase anything.
    """
    p = weights / weights.sum()
    mean = p @ omega
    var = p @ (omega - mean) ** 2
    n_eff = 1.0 / np.sum(p ** 2)
    bandwidth = 4.0 * np.sqrt(var)
    spacing = bandwidth / max(n_eff, 1.0)
    return {
        'mean': mean,
        'std': np.sqrt(var),
        'n_eff': n_eff,
        'bandwidth': bandwidth,
        'spacing': spacing,
        't_zeno': 1.0 / np.sqrt(var),
        't_heis': 2 * np.pi / spacing,
    }


def fit_relaxation_time(times, C, t_min, t_max, floor=0.05):
    """
    Fit C(t) ~ exp(-t/tau) by least squares on log C, over the window from
    t_min up to whichever comes first: t_max, or the time C first drops
    below `floor`.

    Cutting at the floor is essential. Past it the response is oscillating
    about zero rather than decaying, and including that tail drags the fit
    into meaninglessness -- fitting all the way to t_heis gives R^2 of a few
    percent and a tau an order of magnitude off the 1/e crossing.

    Returns (tau, r_squared, t_fit_end), all nan if there is no usable
    window -- which is itself the answer when the system is too small to
    have an exponential regime.
    """
    # The fit has to be confined to the part that actually decays. Two
    # cutoffs, whichever binds first: the floor (past it the response is
    # oscillating about zero, not decaying) and three 1/e times (responses
    # here often drop fast and then crawl through a slow tail, and mixing
    # the two into one exponential gives a tau an order of magnitude off).
    reached_e = np.flatnonzero(C < 1.0 / np.e)
    if not reached_e.size:
        return np.nan, np.nan, np.nan
    t_end = min(t_max, 3.0 * times[reached_e[0]])
    reached_floor = np.flatnonzero(C < floor)
    if reached_floor.size:
        t_end = min(t_end, times[reached_floor[0]])

    window = (times >= t_min) & (times <= t_end) & (C > 0)
    if window.sum() < 5:
        return np.nan, np.nan, np.nan
    t, y = times[window], np.log(C[window])
    slope, intercept = np.polyfit(t, y, 1)
    if slope >= 0:
        return np.nan, np.nan, np.nan
    residual = y - (slope * t + intercept)
    r_squared = 1 - np.sum(residual ** 2) / np.sum((y - y.mean()) ** 2)
    return -1.0 / slope, r_squared, t_end


def crossing_time(times, C, level=1.0 / np.e):
    """
    First time C(t) drops below `level`. Fit-free, so it is the more robust
    of the two estimates when the decay is not cleanly exponential -- but it
    is only a relaxation time if that crossing happens inside the window
    from timescales().
    """
    below = np.flatnonzero(C < level)
    return times[below[0]] if below.size else np.nan
