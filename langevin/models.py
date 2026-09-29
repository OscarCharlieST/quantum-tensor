"""
Hamiltonian, bath operators and initial states for the Langevin runs.

Matches the model of Azad, Hallam, Morley & Green (arXiv:2111.06408) and its
code (github.com/AndrewHallam/Langevin, Finitechain_Ising_Langevin.m):

    H = sum_i J Z_i Z_{i+1} + hx X_i + hy Y_i + hz Z_i
    J = -1, hx = -1.05, hz = 0.5, hy = 0     (the quench Hamiltonian)
    psi_0 = ground state of -sum X  =  |+>^N
    bath operators F_n = X_j, Y_j, Z_j on every site j (3N of them)

Two conventions to keep straight, both inherited from qtensor:

  * An MPO tensor W[in, out, l, r] stores the *transpose* of the local operator:
    contract_left / mps.apply / apply_Heff_parts all act as
    (O psi)[out] = sum_in W[in, out] psi[in]. For X and Z that is invisible;
    for Y it is a sign (pauli('y') placed directly in an MPO slot represents
    -Y). Everything here goes through onsite_block(), which transposes.
  * All tensors are complex128. qtensor.operators builds MPOs in complex64;
    the noise coefficients are added to the MPO every step, so single precision
    there would put a 1e-7 floor under everything.
"""

import numpy as np

import qtensor.states as states
import qtensor.operators as ops

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

# The bath species, in the order of the reference code: F_{3j + a} = PAULIS[a]_j
PAULIS = (X, Y, Z)
SPECIES = ('x', 'y', 'z')

PAPER_PARAMS = dict(J=-1.0, hx=-1.05, hy=0.0, hz=0.5)


def onsite_block(O):
    """The MPO slot content that represents the operator O (see module doc)."""
    return np.asarray(O, dtype=complex).T


def ising_mpo(N, J=-1.0, hx=-1.05, hy=0.0, hz=0.5):
    """
    H = sum J Z Z + hx X + hy Y + hz Z as a qtensor mpo, complex128, bond
    dimension 3, upper-triangular convention (l = e_0, r = e_2, on-site term
    in slot [0, 2]) -- the same layout as ops.tilted_ising, so the thermofield
    helpers in qtensor would accept it too.
    """
    W = np.zeros((2, 2, 3, 3), dtype=complex)
    W[:, :, 0, 0] = onsite_block(I2)
    W[:, :, 2, 2] = onsite_block(I2)
    W[:, :, 0, 1] = onsite_block(J * Z)
    W[:, :, 1, 2] = onsite_block(Z)
    W[:, :, 0, 2] = onsite_block(hx * X + hy * Y + hz * Z)
    l = np.array([1, 0, 0], dtype=complex)
    r = np.array([0, 0, 1], dtype=complex)
    return ops.uniform_MPO(W, l, r, N)


def with_onsite_fields(H, coeffs):
    """
    H + sum_{j,a} coeffs[j, a] * PAULIS[a]_j, as a new mpo.

    coeffs : (N, 3) array, rows in sorted-site order. Real, so the result is
             Hermitian; complex only for the 'paper_exact' typo.

    The fields go into the on-site slot [0, -1] of each tensor, so the bond
    dimension does not grow -- this is what makes the per-step Langevin
    generator exactly as cheap to evolve under as H itself. Requires the
    upper-triangular layout with l = e_0 and r = e_{-1}.
    """
    sites = sorted(H.sites)
    D = H.D
    assert np.allclose(H.l, np.eye(D)[0]) and np.allclose(H.r, np.eye(D)[-1]), \
        "with_onsite_fields needs l = e_0, r = e_{D-1}"
    coeffs = np.asarray(coeffs)
    coeffs = coeffs.astype(complex if np.iscomplexobj(coeffs) else float)
    coeffs = coeffs.reshape(len(sites), 3)
    Ws = []
    for i, site in enumerate(sites):
        W = np.array(H[site], dtype=complex)   # copy; uniform_MPO shares W
        field = sum(c * P for c, P in zip(coeffs[i], PAULIS))
        W[:, :, 0, D - 1] += onsite_block(field)
        Ws.append((site, W))
    return ops.mpo(Ws, np.array(H.l, dtype=complex), np.array(H.r, dtype=complex))


def ising_local_terms(J=-1.0, hx=-1.05, hy=0.0, hz=0.5):
    """(onsite 2x2, bond 4x4) of the same H, for brackets.local_brackets."""
    return hx * X + hy * Y + hz * Z, J * np.kron(Z, Z)


def random_product_state(N, seed=0, D=1):
    """
    Product of spins in seeded random directions -- states.rand_product (per-site
    theta ~ U(0, pi), phi ~ U(0, 2 pi); note: not uniform on the sphere).

    The default initial state for Langevin runs. D = 1 (unpadded) is intended,
    to grow with run_trajectory(..., grow_to=D); see plus_state for why padding
    is avoided.
    """
    psi = states.rand_product(N, D, seed=seed)
    for site in psi.sites:
        psi[site] = np.asarray(psi[site], dtype=complex)
    psi.right_orthogonal()
    return psi


def plus_state(N, D):
    """
    |+>^N, zero-padded to bond dimension D (the ground state the reference
    code prepares by DMRG from H(J=0, hx=-1)).

    Single-site TDVP cannot change bond dimension, so the padding is what sets
    D. The first canonicalization turns the padding into an arbitrary
    orthonormal completion (see CLAUDE.md, 'A padded rank-1 seed does fill
    its bond dimension'), which is what lets the evolution use it.
    """
    psi = states.spin_product(N, D, theta=np.pi / 2, phi=0.0)
    for site in psi.sites:
        psi[site] = np.asarray(psi[site], dtype=complex)
    psi.right_orthogonal()
    return psi


def dense_state(psi):
    """Full state vector (site 0 most significant). Small N only."""
    sites = sorted(psi.sites)
    v = psi[sites[0]][:, 0, :]                     # (d, D)
    for site in sites[1:]:
        M = psi[site]
        v = np.einsum('ab,sbc->asc', v, M).reshape(-1, M.shape[2])
    return v[:, 0]


def dense_onsite(O, j, N):
    """O acting on site j of N, as a 2^N x 2^N matrix."""
    out = np.eye(1, dtype=complex)
    for k in range(N):
        out = np.kron(out, O if k == j else I2)
    return out


def dense_ising(N, J=-1.0, hx=-1.05, hy=0.0, hz=0.5):
    H = np.zeros((2 ** N, 2 ** N), dtype=complex)
    for j in range(N):
        H += dense_onsite(hx * X + hy * Y + hz * Z, j, N)
        if j < N - 1:
            H += J * dense_onsite(Z, j, N) @ dense_onsite(Z, j + 1, N)
    return H
