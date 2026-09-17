"""
Hydrodynamic-mode search inside the near-zero Lyapunov cluster.

A bin-averaged diagnostic -- is a *typical* vector in an exponent band
long-wavelength? -- was tried first and found nothing: it dilutes a few
genuine modes among the hundreds in the near-zero cluster. (Removed
2026-09-17; see the README.) This
module asks the sharper question: **does the near-zero subspace contain a
long-wavelength mode at all, and how much of one?**

The key is that the energy-density profile is real-linear in the tangent
vector,

    p_j(y) = delta<h_j>(y) = 2 Re <psi| h_j |Phi(X(y))>,

so on a subspace spanned by vectors V (columns, real frame coordinates) the
whole map is one matrix P = [p(v_1) ... p(v_m)] of shape (n_bonds, m), and
its cosine transform A = DCT(P) of shape (n_q, m) gives every mode's
amplitude at every wavevector. For a unit vector y of subspace coefficients
the amplitude at wavevector k is the linear functional A[k] . y, so the
maximizing direction is simply A[k]/|A[k]| and the best achievable
amplitude is |A[k]| -- no optimization loop, and no arbitrariness from
rotations within a degenerate cluster.

Better still, that functional has an exact representative. Since

    amplitude_k(y) = sum_j c_kj delta<h_j>(y) = 2 Re <psi| O_k |Phi(X(y))>,
    O_k = sum_j cos(q_k (j+1/2)) h_j,

the gradient is the tangent vector a_k = 2 * realify(P_tangent O_k |psi>) --
**the local-temperature template mode** of the design plan, computed by
linearity from one projection per bond. Then for any subspace S,

    f_k(S) = |Pi_S a_k| / |a_k|

is the fraction of that template the subspace can represent, between 0 and
1, and comparable across subspaces of different dimension because a random
m-dimensional subspace of the 2n-dimensional tangent space gives exactly
E[f_k^2] = m/2n. That is the statistic to report: **is the near-zero
cluster more aligned with the long-wavelength energy mode than chance?**

Note the covariant vectors are *not* orthonormal, so Pi_S uses the Gram
matrix: Pi_S = V (V^T V)^-1 V^T.

Also reported per wavevector:
  purity     |A[k].y|^2 / |P y|^2 for the best y: how monochromatic it is
  lambda_eff y^T diag(lambda) y: where in the spectrum the mode sits
  extent     participation ratio of the mode's site weights (delocalization)
"""

import numpy as np
import scipy.linalg as la

import qtensor.operators as ops
from lyapunov.relaxation.response import pad_with_identity
from lyapunov.tdvp_lyapunov.analysis import (
    energy_density_mpo, local_profile, cosine_transform, onesite_mpo,
)
from lyapunov.tdvp_lyapunov.frame import complexify, realify, project_to_frame
from lyapunov.tdvp_lyapunov.tangent_generator import apply_mpo


def profile_map(frame, V, which='phys', observable='energy', **params):
    """
    (n_sites_of_observable, m) matrix of local profiles, one column per
    column of V. `observable` is 'energy' (bond energy density) or a Pauli
    letter for a one-site observable.
    """
    if observable == 'energy':
        positions = frame.sites[:-1]
        mpo_at = lambda j: energy_density_mpo(j, which, **params)
    else:
        positions = frame.sites
        mpo_at = lambda j: onesite_mpo(j, observable, which)
    return np.column_stack([
        local_profile(frame, V[:, i], mpo_at, positions) for i in range(V.shape[1])
    ])


def dct_map(P):
    """Cosine transform each column of a profile map. Returns q, A."""
    cols = [cosine_transform(P[:, i]) for i in range(P.shape[1])]
    q = cols[0][0]
    return q, np.column_stack([c[1] for c in cols])


def best_mode_at_q(A, k, P=None, lam=None, V=None, frame=None):
    """
    The subspace direction maximizing the profile amplitude at wavevector
    index k, and its diagnostics. Returns a dict.
    """
    row = A[k]
    norm = la.norm(row)
    y = row / norm if norm > 0 else row
    out = {'k': k, 'amplitude': norm, 'y': y}
    if P is not None:
        prof = P @ y
        total = np.sum(prof ** 2)
        out['profile'] = prof
        out['purity'] = (A[k] @ y) ** 2 / total if total > 0 else 0.0
        out['dct'] = A @ y
    if lam is not None:
        out['lambda_eff'] = float(lam @ y ** 2)
        out['lambda_spread'] = float(np.sqrt(max(lam ** 2 @ y ** 2 - (lam @ y ** 2) ** 2, 0)))
    if V is not None and frame is not None:
        vec = V @ y
        w = np.array([frame.site_weights(complexify(vec)).get(s, 0.0) for s in frame.sites])
        w = w / w.sum()
        out['site_weights'] = w
        out['extent'] = float(1.0 / np.sum(w ** 2))      # participation ratio
        out['vector'] = vec
    return out


def scan_subspace(frame, V, lam=None, which='phys', k_max=4, **params):
    """
    Run best_mode_at_q for wavevector indices 1..k_max (skipping k=0, the
    uniform component, which for the energy density is the conserved total
    energy rather than a hydrodynamic mode).
    """
    P = profile_map(frame, V, which, **params)
    q, A = dct_map(P)
    modes = [best_mode_at_q(A, k, P=P, lam=lam, V=V, frame=frame)
             for k in range(1, min(k_max, len(q) - 1) + 1)]
    return {'q': q, 'A': A, 'P': P, 'modes': modes}


def bond_tangent_vectors(frame, which='phys', **params):
    """
    w_j = P_tangent h_j |psi> in real frame coordinates, one per bond:
    the building blocks of every energy-profile functional, by linearity.
    """
    out = []
    for j in frame.sites[:-1]:
        O = energy_density_mpo(j, which, **params)
        W = pad_with_identity(O, frame.sites)          # h_j as a full-chain MPO
        full = ops.mpo(list(W.items()), O.l, O.r)
        out.append(project_to_frame(apply_mpo(frame.A_R, full), frame))
    return out


def template_vectors(frame, k_max=4, which='phys', **params):
    """
    a_k = 2 * realify(sum_j cos(q_k (j+1/2)) w_j) for k = 0..k_max: the
    exact gradient of the profile amplitude at each wavevector, i.e. the
    local-temperature template modes. Returns (q, [a_0 ... a_kmax]).
    """
    w = bond_tangent_vectors(frame, which, **params)
    nb = len(w)
    q = np.pi * np.arange(nb) / nb
    out = []
    for k in range(min(k_max, nb - 1) + 1):
        # scaled to match scipy's orthonormal DCT-II, so that a_k . y is
        # exactly the k-th coefficient of cosine_transform(profile)
        scale = np.sqrt((1.0 if k == 0 else 2.0) / nb)
        c = scale * np.cos(np.pi * k * (np.arange(nb) + 0.5) / nb)
        out.append(2.0 * realify(sum(ck * wk for ck, wk in zip(c, w))))
    return q, out


def template_spectral_weights(Q, a):
    """
    Distribution of a template over the Gram-Schmidt vectors (columns of Q,
    orthonormal, in QR order = descending exponent). Returns

        w        normalized weights, summing to 1 over the computed half
        in_half  |Q^T a|^2 / |a|^2, the fraction of the template that lies
                 in the computed half of the tangent space at all

    Q is used rather than the covariant vectors because it is exactly
    orthonormal, so these weights are a true decomposition. The CLV span is
    the right object for *building* a mode (it is covariant); the GS
    filtration is the right object for *measuring* where weight sits.
    """
    c = Q.T @ a / la.norm(a)
    in_half = float(c @ c)
    return c ** 2 / (c @ c), in_half


def band_enrichment(w, idx):
    """
    Share of template weight in a band of indices, divided by the share a
    uniform spread over the computed half would give. 1 = chance.
    """
    return float(w[idx].sum() / (len(idx) / len(w)))


def subspace_projector_fraction(V, a):
    """
    |Pi_S a| / |a| for the span of V's columns (not assumed orthonormal),
    and the coefficient vector y of that projection.
    """
    G = V.T @ V
    b = V.T @ a
    y = la.solve(G, b, assume_a='pos')
    proj = V @ y
    return la.norm(proj) / la.norm(a), y, proj


def band_indices(lam, centre='zero', m=None, band=None):
    """
    Indices of a contiguous band of the spectrum: the m smallest |lambda|
    ('zero'), m around the median ('mid'), the m largest ('top'), or
    everything with |lambda| < band.
    """
    if band is not None:
        return np.where(np.abs(lam) < band)[0]
    if centre == 'top':
        order = np.argsort(lam)[::-1]
    elif centre == 'mid':
        order = np.argsort(np.abs(lam - np.median(lam)))
    else:
        order = np.argsort(np.abs(lam))
    return np.sort(order[:m])
