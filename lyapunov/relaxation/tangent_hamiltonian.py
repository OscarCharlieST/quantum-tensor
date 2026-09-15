"""
Build the MPS-tangent-space projection of the antisymmetric thermofield
Hamiltonian H_asym at a fixed point psi_uniform.

Reuses rather than reimplements two pieces of existing qtensor
infrastructure:
  - qtensor.operators.contract_left / contract_right: the MPO-environment
    contraction primitives already used by finiteTDVP.tdvp's L_con/R_con.
    Only a bra/ket-asymmetric variant of them is new here.
  - qtensor.simulation.updatemethod.apply_Heff_parts: every matrix element
    of the tangent Hamiltonian -- diagonal *and* off-diagonal blocks alike
    -- is that same contraction, evaluated against mixed environments and
    then projected with V_L.
"""

import numpy as np
import scipy.linalg as la
from ncon import ncon

import qtensor.states as states
import qtensor.operators as ops
import qtensor.simulation.updatemethod as methods


def build_tangent_hamiltonian(psi_uniform, H_asym, max_bond_dim=np.inf):
    """
    Project H_asym onto the MPS tangent space at psi_uniform.

    Parameters
    ----------
    psi_uniform : qtensor.states.mps
        The purified, doubled-physical-dimension uniform-temperature
        thermofield state -- the fixed point being linearized around. Not
        mutated; canonicalization works on copies of its tensors.
    H_asym : qtensor.operators.mpo
        The antisymmetric thermofield Hamiltonian H⊗I - I⊗H.
    max_bond_dim : int, optional
        Passed through to canonicalization; should match whatever bond
        dimension psi_uniform was generated/truncated at.

    Returns
    -------
    H_tangent : (dim, dim) complex ndarray
        Hermitian. Diagonalize with scipy.linalg.eigh for the tangent-space
        normal-mode frequencies.
    basis_index_map : list of (site, start, (n_null, D_r))
        Which rows/columns belong to which site's X^n: the block for `site`
        occupies rows/columns start : start + n_null*D_r, and unflattens to
        shape (n_null, D_r). Sites contributing no tangent directions are
        absent from the list.
    """
    sites = sorted(psi_uniform.sites)
    A_L, A_R, left_envs, right_envs = canonicalize_and_build_environments(
        psi_uniform, H_asym, max_bond_dim
    )
    V_L = {n: build_null_space_tensor(A_L[n]) for n in sites}
    return assemble_tangent_hamiltonian(
        sites, A_L, A_R, V_L, left_envs, right_envs, H_asym
    )


def canonicalize_and_build_environments(psi_uniform, H_asym, max_bond_dim=np.inf):
    """
    Build both canonical gauges of psi_uniform and the two environment
    families every matrix element needs.

    A tangent vector at site n uses left-orthogonal tensors to its left and
    right-orthogonal tensors to its right, so both gauges are needed, and
    they must agree bond-for-bond. A single sweep in each direction does not
    guarantee that -- each sweep clips the bond dimensions to its own
    staircase -- so three sweeps are taken, after which the dimensions
    respect both staircases and the final sweep is a pure (lossless) gauge
    transformation.

    Returns
    -------
    A_L, A_R : dict {site: array}
        Left- and right-orthogonal tensors for the same normalized state.
    left_envs : dict {site: array}
        left_envs[j] contracts A_L over sites <= j against H_asym's MPO,
        with left_envs[first-1] the MPO's left boundary vector. Same object
        as finiteTDVP.tdvp's L_con.
    right_envs : dict {site: array}
        right_envs[j] contracts A_R over sites >= j, with right_envs[last+1]
        the MPO's right boundary vector. Same object as finiteTDVP's R_con.
    """
    sites = sorted(psi_uniform.sites)

    tensors, _ = states.right_orthogonal_state(psi_uniform.tensors, max_bond_dim)
    A_L, _ = states.left_orthogonal_state(tensors, max_bond_dim)
    A_R, _ = states.right_orthogonal_state(A_L, max_bond_dim)

    for site in sites:
        assert A_L[site].shape == A_R[site].shape, (
            f"Canonical forms disagree on bond dimensions at site {site}: "
            f"{A_L[site].shape} (left) vs {A_R[site].shape} (right)"
        )

    left_envs, right_envs = build_environments(
        A_L, A_R, H_asym.tensors, H_asym.l, H_asym.r, sites
    )
    return A_L, A_R, left_envs, right_envs


def build_environments(A_L, A_R, W, l, r, sites):
    """
    The L_con/R_con environment families for an arbitrary MPO, given as a
    {site: tensor} dict covering every site plus its boundary vectors.
    Split out from canonicalization because observable MPOs need the same
    environments as H_asym does.
    """
    left_envs = {sites[0] - 1: ncon((np.eye(1), l), ((-1, -2), (-3,)))}
    for site in sites:
        left_envs[site] = ops.contract_left(left_envs[site - 1], A_L[site], W[site])

    right_envs = {sites[-1] + 1: ncon((np.eye(1), r), ((-1, -2), (-3,)))}
    for site in reversed(sites):
        right_envs[site] = ops.contract_right(right_envs[site + 1], A_R[site], W[site])

    return left_envs, right_envs


def build_centre_tensors(A_L, A_R, check=True, tol=1e-8):
    """
    The centre tensors C^n, for which psi = A_L^1..A_L^{n-1} C^n
    A_R^{n+1}..A_R^N -- i.e. the state itself written in mixed canonical
    form around each site in turn.

    Built from the bond matrices via C^n = Lambda^{n-1} A_R^n and
    Lambda^n = (A_L^n)^dagger C^n, which uses only the isometry property of
    A_L. No SVD is involved, so the centres are guaranteed to be in the
    same gauge as the A_L and A_R they were built from. Re-deriving them
    with an independent canonicalization sweep does *not* work: wherever the
    Schmidt spectrum is near-degenerate or near-zero the singular vectors
    are numerically arbitrary, so the sweep silently lands in a different
    gauge and every overlap built on it is garbage. Thermofield states have
    Schmidt values down at 1e-11, so this is the normal case here, not an
    edge case.

    With check=True, verifies each centre tensor is normalized, which fails
    loudly if A_L and A_R are not gauges of the same normalized state.
    """
    sites = sorted(A_R.keys())
    C = {}
    Lam = np.eye(1)
    for site in sites:
        C[site] = Lam @ A_R[site]
        d, Dl, Dr = A_L[site].shape
        Lam = A_L[site].reshape(d * Dl, Dr).conj().T @ C[site].reshape(d * Dl, -1)

    if check:
        for site in sites:
            norm = np.linalg.norm(C[site])
            assert abs(norm - 1.0) < tol, (
                f"Centre tensor at site {site} has norm {norm:.6f}, not 1 -- "
                f"A_L and A_R are not gauges of the same normalized state"
            )
    return C


def build_null_space_tensor(A_n, check=True, tol=1e-10):
    """
    Reshape the left-orthogonal tensor A_n from (d, Dl, Dr) to a (d*Dl, Dr)
    isometry, and return an orthonormal basis for the orthogonal complement
    of its column space, as a (d*Dl, d*Dl - Dr) array. Empty (zero columns)
    when d*Dl == Dr exactly -- that site then contributes no tangent
    directions.

    If check is True, verifies the tangent gauge condition
    (A_n)^dagger @ V_L == 0 and that V_L has orthonormal columns, raising
    an AssertionError if either fails beyond tol.
    """
    d, Dl, Dr = A_n.shape
    M = A_n.reshape(d * Dl, Dr)
    V_L = la.null_space(M.conj().T)

    if check and V_L.shape[1] > 0:
        gauge_residual = np.max(np.abs(M.conj().T @ V_L))
        assert gauge_residual < tol, (
            f"V_L fails the tangent gauge condition A^dagger V_L = 0 "
            f"(max residual {gauge_residual:.2e})"
        )
        orthonormality_residual = np.max(np.abs(
            V_L.conj().T @ V_L - np.eye(V_L.shape[1])
        ))
        assert orthonormality_residual < tol, (
            f"V_L columns aren't orthonormal (max residual "
            f"{orthonormality_residual:.2e})"
        )

    return V_L


def build_tangent_vector(A_L, A_R, V_L_site, site, X):
    """
    Build the tensors of |Phi_n(X)>: left-orthogonal tensors left of `site`,
    right-orthogonal tensors right of it, and the defect V_L_site @ X
    (reshaped to (d, Dl, Dr)) at `site` itself.

    Returned as a plain {site: tensor} dict rather than an mps object: it is
    a tangent vector, not a normalized state, so the mps gauge flags would
    be meaningless.
    """
    d, Dl, Dr = A_L[site].shape
    B = (V_L_site @ X).reshape(d, Dl, Dr)
    return {n: (A_L[n] if n < site else A_R[n] if n > site else B)
            for n in A_L}


def contract_left_mixed(L, A_ket, A_bra, W):
    """
    contract_left with different tensors on the bra and ket chains. Needed
    because an off-diagonal matrix element <Phi_n|H|Phi_m> has a bra in the
    right-orthogonal gauge and a ket in the left-orthogonal gauge over the
    sites between n and m.
    """
    return ncon((L, A_ket, A_bra.conj(), W),
                ((1, 2, 3), (4, 1, -1), (5, 2, -2), (4, 5, 3, -3)))


def contract_right_mixed(R, A_ket, A_bra, W):
    """
    contract_right with different tensors on the bra and ket chains. See
    contract_left_mixed.
    """
    return ncon((R, A_ket, A_bra.conj(), W),
                ((1, 2, 3), (4, -1, 1), (5, -2, 2), (4, 5, -3, 3)))


def project_H_onto_tangent_basis(psi_ket, defect_site, A_L, A_R, V_L,
                                 left_envs, right_envs, H_asym):
    """
    Given a concrete tangent vector |psi_ket> = |Phi_m(Y)> with its defect at
    `defect_site`, return {n: G^n} where

        G^n[a, r] = <Phi_n(e_{a,r})| H_asym |psi_ket>

    i.e. one full column of the tangent Hamiltonian, resolved into its
    per-site blocks.

    Environments below the defect on the ket side are the precomputed
    all-A_L ones and above it the all-A_R ones; the rest are rebuilt with
    the mixed contractions, since the bra chain stays in the A_L/A_R gauges
    while the ket carries the defect. Contracting a site against its two
    environments and the MPO is then exactly apply_Heff_parts, reused
    unchanged -- the only new step is the final V_L projection, which picks
    out the gauge-fixed tangent directions.
    """
    sites = sorted(psi_ket.keys())

    LE = dict(left_envs)
    for site in sites:
        if site >= defect_site:
            LE[site] = contract_left_mixed(
                LE[site - 1], psi_ket[site], A_L[site], H_asym[site]
            )

    RE = dict(right_envs)
    for site in reversed(sites):
        if site <= defect_site:
            RE[site] = contract_right_mixed(
                RE[site + 1], psi_ket[site], A_R[site], H_asym[site]
            )

    G = {}
    for site in sites:
        if V_L[site].shape[1] == 0:
            continue
        F = methods.apply_Heff_parts(
            psi_ket[site], H_asym[site], LE[site - 1], RE[site + 1]
        )
        d, Dl, Dr = F.shape
        G[site] = V_L[site].conj().T @ F.reshape(d * Dl, Dr)
    return G


def assemble_tangent_hamiltonian(sites, A_L, A_R, V_L, left_envs, right_envs, H_asym):
    """
    Stack the per-site blocks into one Hermitian matrix, one column at a
    time: sweep over every basis direction (site m, index into X^m), build
    that tangent vector, and read off its overlaps with every bra direction
    via project_H_onto_tangent_basis.

    The tangent basis is orthonormal -- V_L's gauge condition kills the
    n != m overlaps and its orthonormal columns fix the n == m ones -- so
    these overlaps are the matrix elements directly, with no metric to
    invert.

    Sites with an empty V_L^n contribute no block and are skipped.

    Returns
    -------
    H_tangent : (dim, dim) complex ndarray
    basis_index_map : list of (site, start, (n_null, D_r))
    """
    basis_index_map = []
    dim = 0
    for site in sites:
        n_null = V_L[site].shape[1]
        if n_null == 0:
            continue
        D_r = A_L[site].shape[2]
        basis_index_map.append((site, dim, (n_null, D_r)))
        dim += n_null * D_r

    H_tangent = np.zeros((dim, dim), dtype=complex)
    for m, col_start, (n_null_m, D_m) in basis_index_map:
        for j in range(n_null_m * D_m):
            Y = np.zeros((n_null_m, D_m), dtype=complex)
            Y.flat[j] = 1.0
            psi_ket = build_tangent_vector(A_L, A_R, V_L[m], m, Y)
            G = project_H_onto_tangent_basis(
                psi_ket, m, A_L, A_R, V_L, left_envs, right_envs, H_asym
            )
            for n, row_start, (n_null_n, D_n) in basis_index_map:
                H_tangent[row_start:row_start + n_null_n * D_n,
                          col_start + j] = G[n].reshape(-1)

    return H_tangent, basis_index_map
