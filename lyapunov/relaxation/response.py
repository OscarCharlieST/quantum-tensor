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


def single_copy_total_current(sites, J=1, h=0.25, g=-0.525):
    """
    MPO for the *total* tilted-Ising energy current J_tot = sum_l j_l over
    the whole chain, on the physical copy only.

    Needed because Green-Kubo is a statement about the total current, not a
    local one. The autocorrelator of `single_copy_current` is only the
    r = 0 term of sum_r <j_r(t) j_0(0)>; the r != 0 terms are where the
    diffusive contribution lives, and they are exactly what summing the
    current into one operator supplies.

    Summing the three-site j_l over *every* site, dropping the factors that
    fall off the ends of an open chain, telescopes: the -z_{l-1} y_l piece
    reindexes onto the same bonds as the +y_l z_{l+1} piece, leaving a pure
    nearest-neighbour operator,

        J_tot = J g sum_l ( y_l z_{l+1} - z_l y_{l+1} ),

    over bonds l = 0 .. L-2. So the bond dimension is 4 regardless of L,
    rather than the ~2L an uncompressed `mpo.__add__` direct sum of L local
    currents would give. The finite-state machine is

        0 -> 0   I            (before the term)
        0 -> 1   J g y        (open the +y z term)
        0 -> 2   -J g z       (open the -z y term)
        1 -> 3   z            (close it)
        2 -> 3   y            (close it)
        3 -> 3   I            (after the term)

    with the boundary vectors selecting 0 on the left and 3 on the right,
    so the ends truncate themselves: a term opened on the last site has no
    way to close and is projected out.

    Open boundaries mean J_tot is not a translation average -- the two end
    bonds are missing the partners a periodic chain would give them -- so
    the O(1/L) boundary contribution has to be watched in the L scan rather
    than assumed away.

    As with the local current only g enters; h is accepted so this can be
    called with the same parameters as the density.
    """
    Y = np.kron(ops.pauli('y'), np.eye(2))
    Z = np.kron(ops.pauli('z'), np.eye(2))
    I4 = np.eye(4)

    W = np.zeros((4, 4, 4, 4), dtype=complex)
    W[:, :, 0, 0] = I4
    W[:, :, 0, 1] = J * g * Y
    W[:, :, 0, 2] = -J * g * Z
    W[:, :, 1, 3] = Z
    W[:, :, 2, 3] = Y
    W[:, :, 3, 3] = I4

    l = np.array([1.0, 0.0, 0.0, 0.0])
    r = np.array([0.0, 0.0, 0.0, 1.0])
    return ops.mpo([(site, W) for site in sorted(sites)], l, r)


def single_copy_hamiltonian(sites, J=1, h=0.25, g=-0.525):
    """
    MPO for H (x) I_aux: the tilted-Ising Hamiltonian acting on the physical
    copy alone.

    `thermofield.thermofield_hamiltonian` gives H(x)I +/- I(x)H, never H(x)I
    on its own, and building it as (H_sym + H_asym)/2 would go through the
    uncompressed direct sum in `mpo.__add__` and double the bond dimension
    for nothing. This is the same three-state machine as
    `operators.tilted_ising`, with every Pauli lifted by kron(A, I_2).
    """
    Z = np.kron(ops.pauli('z'), np.eye(2))
    X = np.kron(ops.pauli('x'), np.eye(2))
    I4 = np.eye(4)

    W = np.zeros((4, 4, 3, 3), dtype=complex)
    W[:, :, 0, 0] = I4
    W[:, :, 2, 2] = I4
    W[:, :, 0, 1] = J * Z
    W[:, :, 1, 2] = Z
    W[:, :, 0, 2] = h * Z + g * X

    l = np.array([1.0, 0.0, 0.0])
    r = np.array([0.0, 0.0, 1.0])
    return ops.mpo([(site, W) for site in sorted(sites)], l, r)


# --------------------------------------------------------- static response

def static_variance(psi, O_mpo, tol=1e-8):
    """
    Var(O) = <O^2> - <O>^2 in the thermofield state, evaluated *exactly*
    from MPO algebra -- no tangent projection anywhere.

    Two uses. It is the Green-Kubo denominator (see static_susceptibility),
    and it is the yardstick for how much of an operator the tangent space
    actually sees: the tangent weights satisfy sum_k w_k = <O P O>, which is
    the same thing with the tangent projector P inserted, so

        sum_k w_k / static_variance(psi, O)

    is the fraction of O's static weight the tangent space captures. At
    beta = 0.1, D = 8 that ratio is 1.0000 to machine precision for every
    observable here -- z, x, h_l, j_l, J_tot and H alike -- so the
    Green-Kubo numerator carries *no* static truncation error. The ratio is
    not vacuous: a weight-L product of random single-site rotations scores
    0.66 at L = 4, and a random product on the full doubled site 0.78. What
    the physical observables have in common is that they are sums of
    low-weight terms, and near beta = 0 the state is close to rank 1, where
    the padded tangent space is generous.

    Capture = 1 says only that C(0) is exact. The *evolution* is still
    tangent-projected, so the dynamical error is untouched by this and
    remains bounded by fixed_point_residual.

    O must span every site of psi, since `ops.expect` assumes it does; the
    single_copy_* builders above with an explicit `sites` argument all do.
    Var(O) is real for Hermitian O, and a large imaginary part means psi is
    not normalized or the MPO is not Hermitian, so it is checked.
    """
    mean = complex(ops.expect(psi, O_mpo))
    sq = complex(ops.expect(psi, O_mpo @ O_mpo))
    var = sq - mean ** 2
    if abs(var.imag) > tol * max(1.0, abs(var.real)):
        raise ValueError(
            f"Var(O) is not real: {var}. Either psi is not normalized "
            "or O is not Hermitian.")
    return float(var.real)


def static_susceptibility(psi, sites, J=1, h=0.25, g=-0.525):
    """
    The Green-Kubo denominator: Var(H) = <H^2> - <H>^2 on the physical copy.

    The energy diffusion constant is D = kappa / c with

        kappa = (beta^2 / L) int_0^inf dt <J(t) J(0)>_c,
        c     = (beta^2 / L) ( <H^2> - <H>^2 ),

    so the beta^2 and the 1/L both cancel and

        D = int_0^inf dt <J(t) J(0)>_c / Var(H).

    Nothing here needs beta or L explicitly, which is worth keeping in mind:
    the temperature enters only through psi, and the extensivity of
    numerator and denominator is what has to cancel for D to converge in L.
    Both are extensive, so the ratio is the quantity to watch in the L scan.

    Note this is the *static* susceptibility of the whole chain, computed
    exactly -- it is deliberately not tangent-projected, because it is a
    thermodynamic quantity the tangent space has no business truncating.
    The projection error then lives entirely in the numerator, where
    static_variance's capture ratio can be used to bound it.
    """
    return static_variance(psi, single_copy_hamiltonian(sites, J=J, h=h, g=g))


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


# --------------------------------------------------------------- Green-Kubo
#
# Every function here takes the *raw* weights, not the normalized ones.
# response_function divides by weights.sum() to put C(0) = 1; Green-Kubo
# needs the absolute scale, and the tangent weights already carry it --
# sum_k w_k = <O P O> is the connected variance, because the tangent basis
# is orthogonal to psi* so the <O>^2 subtraction happens for free.

def green_kubo_integral(omega, weights, times):
    """
    The running Green-Kubo integral, in closed form:

        I(t) = int_0^t C(s) ds = sum_k w_k sin(omega_k t) / omega_k.

    No quadrature and no time stepping -- the same one diagonalization that
    gives C(t) gives its integral to any t, exactly. There is nothing to
    converge and no step size to choose.

    Written with sinc rather than a division, since
    sin(omega t)/omega = t sinc(omega t / pi) and np.sinc(0) = 1, so the
    conserved omega_k = 0 modes contribute their w_k * t automatically
    instead of needing a near-zero branch. That matters here: the energy
    density has a genuine omega = 0 pole carrying 7-35% of its weight, and
    it is exactly the term that must survive to make I(t) grow without
    bound for a conserved density.
    """
    phase = np.outer(np.asarray(times, dtype=float), omega) / np.pi
    return np.asarray(times, dtype=float) * (np.sinc(phase) @ weights)


def green_kubo_broadened(omega, weights, eta):
    """
    The Lorentzian-regulated Green-Kubo integral,

        I(eta) = int_0^inf dt e^{-eta t} C(t)
               = sum_k w_k eta / (omega_k^2 + eta^2),

    also exact, and with no time grid at all.

    This is the estimator to trust, not the t -> inf limit of
    green_kubo_integral. On a finite chain the spectrum is a discrete set
    of deltas, so C(t) never actually decays -- it dephases and then
    recurs, and I(t) oscillates about its mean forever rather than
    converging. Averaging that is possible but noisy: over the nominal
    plateau window the scatter in I(t) exceeds its mean several times over
    at these sizes.

    Broadening each delta to a Lorentzian of width eta fixes the finite-L
    problem at its source. It is the same quantity -- I(eta) -> pi A(0) as
    eta -> 0 for a continuous spectrum -- but eta can be chosen large
    enough to wash out the level discreteness and still small enough not to
    cut into the physical decay. See broadening_window for that choice: the
    answer is only meaningful if D(eta) has a plateau in eta, which is the
    finite-size diagnostic that replaces convergence of the time integral.

    `eta` may be a scalar or an array; the return shape follows it.
    """
    eta = np.asarray(eta, dtype=float)
    denom = omega[None, :] ** 2 + eta.reshape(-1, 1) ** 2
    out = (eta.reshape(-1, 1) / denom) @ weights
    return out.reshape(eta.shape)


def broadening_window(scales, tau, safety=3.0):
    """
    The range of broadenings eta that are admissible at all:

        safety * spacing  <  eta  <  1 / (safety * tau).

    The lower bound is the finite-size floor -- a Lorentzian narrower than
    the level spacing resolves individual modes, and I(eta) collapses
    towards zero because no single mode sits exactly at omega = 0. The
    upper bound is the physics ceiling -- a Lorentzian broader than the
    decay rate 1/tau integrates over the correlator's own structure and
    starts measuring the broadening rather than the transport.

    The window exists only when tau * spacing < 1 / safety^2, i.e. when the
    correlator decays well before the Heisenberg time. That is the same
    separation of scales the relaxation-time analysis needs, restated in
    frequency, and `exists` in the returned dict reports it honestly rather
    than silently producing a number from an empty window.
    """
    eta_min = safety * scales['spacing']
    eta_max = 1.0 / (safety * tau)
    return {
        'eta_min': eta_min,
        'eta_max': eta_max,
        'exists': bool(eta_max > eta_min),
        'decades': (np.log10(eta_max / eta_min) if eta_max > eta_min
                    else np.nan),
    }


def diffusion_constant(omega, weights, chi, scales, tau, safety=3.0,
                       n_eta=64):
    """
    D = int_0^inf dt <J(t) J(0)>_c / Var(H), swept over the admissible
    broadenings.

    `weights` must come from the *total* current (single_copy_total_current)
    -- the local j_l autocorrelator is only the r = 0 term of
    sum_r <j_r(t) j_0(0)>, and the r != 0 terms are where the diffusive
    contribution lives. `chi` is static_susceptibility(psi, sites).

    Returns a dict with the eta grid, D(eta) on it, and a plateau summary:
    `D` is the geometric mean over the window and `log_slope` the fitted
    d log D / d log eta, which is the number that says whether there is a
    plateau at all. |log_slope| << 1 means D is eta-independent and the
    value means something; a slope near +1 means D(eta) is still tracking
    the broadening and the window has not opened yet.

    Also returned are `D_peak` and `eta_peak`, the stationary point of
    D(eta) found over a wide sweep. D(eta) is forced to zero at both ends
    -- as eta -> 0 no mode sits exactly at omega = 0 so I(eta) -> 0
    linearly, and as eta -> infinity I(eta) -> sum_k w_k / eta -- so there
    is always a maximum somewhere between. On a system with real scale
    separation the maximum sits in the middle of a flat region and
    coincides with `D`; without one it is a bare crossover between the two
    artefacts, and `D_peak` is then an upper bound rather than a
    measurement. Read it only together with `log_slope`.

    D, D_scatter and log_slope are nan when the window does not exist;
    D_peak and eta_peak are always available.
    """
    window = broadening_window(scales, tau, safety=safety)
    result = dict(window)

    # The crossover maximum, over a sweep deliberately wider than the
    # admissible window so it can be found even when that window is empty.
    wide = np.logspace(np.log10(scales['spacing'] / 10.0),
                       np.log10(10.0 / tau), 20 * n_eta)
    D_wide = green_kubo_broadened(omega, weights, wide) / chi
    peak = int(np.argmax(D_wide))
    result.update(D_peak=float(D_wide[peak]), eta_peak=float(wide[peak]))

    if not window['exists']:
        result.update(eta=np.array([]), D_eta=np.array([]),
                      D=np.nan, D_scatter=np.nan, log_slope=np.nan)
        return result

    eta = np.logspace(np.log10(window['eta_min']),
                      np.log10(window['eta_max']), n_eta)
    D_eta = green_kubo_broadened(omega, weights, eta) / chi
    log_slope, _ = np.polyfit(np.log(eta), np.log(D_eta), 1)
    result.update(
        eta=eta,
        D_eta=D_eta,
        D=float(np.exp(np.mean(np.log(D_eta)))),
        D_scatter=float(np.std(np.log(D_eta))),
        log_slope=float(log_slope),
    )
    return result


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
