"""
Step 0 of the energy-rate study: the exact anchor, by exact diagonalization.

The claim under test (see README, "Energy rate vs bond dimension") is

    dE_0/dt = -(tau/8) grad^2 beta  sum_k k^2 B_k,
    B_k     = < h_k^+  [H, h_0]^+  H^+ >,        A^+ = A (x) 1 + 1 (x) A,

for the physical-copy energy of a local-equilibrium state near infinite
temperature, with rho ~ exp(-sum_j beta_j h_j). This script checks every
piece of that with no MPS anywhere:

1. Static infinite-T coefficients on an open chain: C_k = <hdot_k hdot_0>,
   the sum rules sum_k C_k = 0 and sum_k k^2 C_k = -2 G(0), and the closed
   forms G(0) = 2 (g J)^2, c = J^2 + h^2 + g^2.
2. The thermofield bracket: B_k = -4 C_k at beta = 0, in the vectorized
   picture psi <-> M, (A (x) B) psi <-> A M B^T. Also B_k at beta = 0.1, the
   exact value step 1's MPS numbers at the thermofield double are compared to.
3. The linear-response second derivative itself, d^2<h_0>/dt^2 from
   rho = exp(-sum dbeta_k h_k)/Z, against sum_k dbeta_k C_k = -1/4 sum dbeta_k B_k.
4. The Green-Kubo reference on periodic chains by dynamical typicality:
   G(t) = <J(t) J>/L, I(t) = int_0^t G, D(t) = I(t)/c and the correlation
   time tau_GK(t) = I(t)/G(0) -- the tau the formula needs.

    python lyapunov/tdvp_lyapunov/energy_rate_ed.py [--Ls 10 12 14 16 18] [--T 25]

writes runs/energy_rate_ed.npz and figures/energy_rate_ed_green_kubo.png.
"""

import argparse
import os
import sys
import time as clock

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

sys.path.insert(0, os.getcwd())

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')

J, HF, G = 1.0, 0.25, -0.525          # tilted Ising, ops.tilted_ising defaults

I2 = np.eye(2)
PX = np.array([[0, 1], [1, 0]], dtype=complex)
PY = np.array([[0, -1j], [1j, 0]], dtype=complex)
PZ = np.array([[1, 0], [0, -1]], dtype=complex)


# ------------------------------------------------------------ dense, small L

def site_op(A, i, L):
    out = np.ones((1, 1), dtype=complex)
    for s in range(L):
        out = np.kron(out, A if s == i % L else I2)
    return out


def dense_H(L, pbc=False):
    H = np.zeros((2 ** L, 2 ** L), dtype=complex)
    for i in range(L):
        H += HF * site_op(PZ, i, L) + G * site_op(PX, i, L)
    for i in range(L if pbc else L - 1):
        H += J * site_op(PZ, i, L) @ site_op(PZ, i + 1, L)
    return H


def bond_energy(b, L):
    """h_b = J z_b z_{b+1} + a_b + a_{b+1}, a = (h z + g x)/2: the density of
    analysis.energy_density_mpo / response.single_copy_energy_density."""
    a = lambda i: 0.5 * (HF * site_op(PZ, i, L) + G * site_op(PX, i, L))
    return J * site_op(PZ, b, L) @ site_op(PZ, b + 1, L) + a(b) + a(b + 1)


def local_current(b, L):
    """j_b = J g (y_b z_{b+1} - z_{b-1} y_b): energy entering bond b through
    site b, so that hdot_b = j_b - j_{b+1} (response.single_copy_current)."""
    return J * G * (site_op(PY, b, L) @ site_op(PZ, b + 1, L)
                    - site_op(PZ, b - 1, L) @ site_op(PY, b, L))


def static_coefficients(L=10):
    """C_k = tr(hdot_{b0+k} hdot_{b0}) / 2^L on an open chain, b0 central."""
    H = dense_H(L)
    N = 2 ** L
    b0 = L // 2 - 1
    hdot = {b: 1j * (H @ bond_energy(b, L) - bond_energy(b, L) @ H)
            for b in range(L - 1)}
    ks = np.arange(-b0, L - 1 - b0)
    C = np.array([np.trace(hdot[b0 + k] @ hdot[b0]).real / N for k in ks])
    # continuity check, away from the edges
    cont = max(la.norm(hdot[b] - (local_current(b, L) - local_current(b + 1, L)))
               for b in range(1, L - 2))
    h0 = bond_energy(b0, L)
    chi = sum(np.trace(bond_energy(b, L) @ h0).real / N for b in range(L - 1))
    return ks, C, cont, chi


def thermofield_bracket(L, beta):
    """
    B_k = < dh_k^+ psi | [H, h_0]^+ | dH^+ psi > on the thermofield double at
    inverse temperature beta, connected (each operator minus its mean, which
    is what the projective tangent space keeps). psi <-> M = rho^(1/2).
    """
    H = dense_H(L)
    w, U = la.eigh(H)
    M = (U * np.exp(-beta * w / 2)) @ U.conj().T
    M /= la.norm(M)
    plus = lambda A, N: A @ N + N @ A.T              # (A (x) 1 + 1 (x) A) vec(N)
    ip = lambda A, B: np.vdot(A, B)                  # <A|B> = tr(A^dag B)

    def connected(A):
        v = plus(A, M)
        return v - ip(M, v) * M

    b0 = L // 2 - 1
    h0 = bond_energy(b0, L)
    X = H @ h0 - h0 @ H
    XH = plus(X, connected(H))
    ks = np.arange(-b0, L - 1 - b0)
    B = np.array([ip(connected(bond_energy(b0 + k, L)), XH) for k in ks])
    return ks, B


def linear_response_check(L=10, dbeta=1e-5, q=None, seed=1):
    """
    d^2<h_0>/dt^2 at t = 0 from rho = exp(-sum dbeta_k h_k)/Z, i.e. exactly
    -<[H, [H, h_0]]>_rho, divided by dbeta, against sum_k profile_k C_k.
    (Not <[[h_0, H], H]>: that is the same double commutator with the
    opposite sign, and was the first version of this check.)
    A random profile, so nothing is helped by symmetry.
    """
    H = dense_H(L)
    b0 = L // 2 - 1
    rng = np.random.default_rng(seed)
    prof = rng.normal(size=L - 1)
    K = sum(dbeta * p * bond_energy(b, L) for b, p in enumerate(prof))
    rho = la.expm(-K)
    rho /= np.trace(rho)
    h0 = bond_energy(b0, L)
    hdot = 1j * (H @ h0 - h0 @ H)                    # Heisenberg, twice
    hddot = 1j * (H @ hdot - hdot @ H)
    second = np.trace(rho @ hddot).real / dbeta
    ks, C, _, _ = static_coefficients(L)
    predicted = sum(prof[b0 + k] * c for k, c in zip(ks, C))
    return second, predicted


# --------------------------------------------------------- sparse, large L

def sparse_ops(L):
    """
    Periodic tilted Ising H (real) and the total current J_tot = i * Jr, Jr
    real antisymmetric, from bit operations. s's bit b is site b, bit 0 ->
    z = +1. With y = i x z:
        y_b z_{b+1} |s> = i z_b z_{b+1}(s) |s ^ e_b>,
        z_b y_{b+1} |s> = i z_b z_{b+1}(s) |s ^ e_{b+1}>,
    and J_tot = J g sum_b (y_b z_{b+1} - z_b y_{b+1}).
    """
    N = 2 ** L
    s = np.arange(N, dtype=np.int64)
    bit = lambda b: 1 << (L - 1 - b)       # site 0 most significant, as in site_op
    z = [1 - 2 * ((s // bit(b)) & 1) for b in range(L)]
    diag = sum(J * z[b] * z[(b + 1) % L] + HF * z[b] for b in range(L)).astype(float)
    rows, cols, vals = [s], [s], [diag]
    for b in range(L):
        rows.append(s ^ bit(b)); cols.append(s); vals.append(np.full(N, G))
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(N, N))
    rows, cols, vals = [], [], []
    for b in range(L):
        c = (b + 1) % L
        zz = (J * G * z[b] * z[c]).astype(float)
        rows += [s ^ bit(b), s ^ bit(c)]
        cols += [s, s]
        vals += [zz, -zz]
    Jr = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(N, N))
    return H, Jr


def check_sparse(L=6):
    """Sparse builders against the dense kron ones (periodic)."""
    H, Jr = sparse_ops(L)
    Hd = dense_H(L, pbc=True)
    Jd = J * G * sum(site_op(PY, b, L) @ site_op(PZ, b + 1, L)
                     - site_op(PZ, b, L) @ site_op(PY, b + 1, L) for b in range(L))
    return la.norm(H.toarray() - Hd), la.norm(1j * Jr.toarray() - Jd)


def current_autocorrelation(L, T, dt, n_samples, seed=0, chunk=40):
    """
    G(t) = tr(J(t) J) / (2^L L) by typicality: with r Haar-random,
    tr(J(t) J)/2^L ~ <r(t)| J |a(t)>, a = J r, both evolved by e^{-iHt}.
    Statistical error ~ 2^(-L/2) per sample.
    """
    H, Jr = sparse_ops(L)
    A = (-1j * H).tocsr()
    times = np.arange(0, T + dt / 2, dt)
    Gt = np.zeros((n_samples, len(times)))
    rng = np.random.default_rng(seed)
    for smp in range(n_samples):
        r = rng.normal(size=2 ** L) + 1j * rng.normal(size=2 ** L)
        r /= la.norm(r)
        B = np.column_stack([r, 1j * (Jr @ r)])
        Gt[smp, 0] = np.real(np.vdot(B[:, 0], 1j * (Jr @ B[:, 1]))) / L
        i = 0
        while i < len(times) - 1:
            m = min(chunk, len(times) - 1 - i)
            out = expm_multiply(A, B, start=0, stop=m * dt, num=m + 1, endpoint=True)
            for j in range(1, m + 1):
                Bj = out[j]
                Gt[smp, i + j] = np.real(np.vdot(Bj[:, 0], 1j * (Jr @ Bj[:, 1]))) / L
            B = out[-1]
            i += m
    return times, Gt.mean(axis=0), Gt.std(axis=0) / np.sqrt(max(n_samples - 1, 1))


def cumulative_trapezoid(y, dx):
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * dx)])


# ------------------------------------------------------------------- main

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--Ls', type=int, nargs='+', default=[10, 12, 14, 16, 18])
    p.add_argument('--T', type=float, default=25.0)
    p.add_argument('--dt', type=float, default=0.05)
    p.add_argument('--skip-dynamics', action='store_true')
    return p.parse_args()


def main():
    a = parse()
    G0, c_exact = 2 * (G * J) ** 2, J ** 2 + HF ** 2 + G ** 2

    print("== 1. static infinite-T coefficients (open L = 10, central bond)")
    ks, C, cont, chi = static_coefficients(10)
    for k, v in zip(ks, C):
        if abs(k) <= 3:
            print(f"   C_{k:+d} = {v:+.6f}")
    print(f"   sum C_k = {C.sum():+.2e}   sum k C_k = {(ks * C).sum():+.2e}")
    print(f"   sum k^2 C_k = {(ks**2 * C).sum():+.6f}   -2 G(0) = {-2 * G0:+.6f}")
    print(f"   continuity |hdot_b - (j_b - j_b+1)| = {cont:.1e}")
    print(f"   c = sum_k <h_k h_0> = {chi:.6f}   J^2+h^2+g^2 = {c_exact:.6f}")

    print("\n== 2. thermofield bracket B_k (open L = 8)")
    ks8, C8, _, _ = static_coefficients(8)
    _, B0 = thermofield_bracket(8, 0.0)
    _, B1 = thermofield_bracket(8, 0.1)
    print(f"   {'k':>3s} {'B_k(beta=0)':>14s} {'-4 C_k':>11s} {'B_k(beta=0.1)':>22s}")
    for k, b0, c, b1 in zip(ks8, B0, C8, B1):
        print(f"   {k:+3d} {b0.real:+14.6f} {-4 * c:+11.6f} {b1.real:+12.6f}{b1.imag:+.1e}j")
    print(f"   max |B_k + 4 C_k| at beta=0: {np.max(np.abs(B0 + 4 * C8)):.1e}")
    print(f"   sum k^2 B_k: beta=0 {(ks8**2 * B0).sum().real:+.6f} (8 G(0) = {8 * G0:+.6f}),"
          f"  beta=0.1 {(ks8**2 * B1).sum().real:+.6f}")

    print("\n== 3. linear response: d^2<h_0>/dt^2 / dbeta vs sum profile_k C_k (L = 10)")
    second, predicted = linear_response_check(10)
    print(f"   exact {second:+.8f}   predicted {predicted:+.8f}   "
          f"rel diff {abs(second - predicted) / abs(predicted):.1e}")

    path = os.path.join(RUNS, 'energy_rate_ed.npz')
    out = dict(np.load(path)) if os.path.exists(path) else {}   # keep other L
    out.update(ks=ks, C=C, ks8=ks8, B0=B0, B1=B1, G0=G0, c=c_exact)
    if not a.skip_dynamics:
        dH, dJ = check_sparse(6)
        print(f"\n== 4. Green-Kubo, periodic, typicality "
              f"(sparse vs dense: H {dH:.1e}, J {dJ:.1e})")
        for L in a.Ls:
            t0 = clock.time()
            n_samp = max(1, 2 ** max(0, 16 - L))
            t, Gt, err = current_autocorrelation(L, a.T, a.dt, n_samp)
            I = cumulative_trapezoid(Gt, a.dt)
            out[f'L{L}_t'], out[f'L{L}_G'], out[f'L{L}_err'] = t, Gt, err
            imax = np.argmax(I)
            print(f"   L={L:2d} ({n_samp} samples, {clock.time() - t0:6.0f} s): "
                  f"G(0)={Gt[0]:.4f}  max I={I[imax]:.4f} at t={t[imax]:.2f} "
                  f"-> D={I[imax] / c_exact:.4f}, tau_GK={I[imax] / G0:.4f};  "
                  f"I(5)={I[t.searchsorted(5)]:.4f} I(10)={I[t.searchsorted(10)]:.4f}",
                  flush=True)
            os.makedirs(RUNS, exist_ok=True)
            np.savez(os.path.join(RUNS, 'energy_rate_ed.npz'), **out)
    np.savez(os.path.join(RUNS, 'energy_rate_ed.npz'), **out)
    print(f"\nwritten {os.path.join(RUNS, 'energy_rate_ed.npz')}")


if __name__ == '__main__':
    main()
