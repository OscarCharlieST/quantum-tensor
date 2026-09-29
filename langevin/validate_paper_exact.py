"""
Checks for the 'paper_exact' scheme (integrator.py, matlab_lanczos.py).

1. lanczos_exp (the reference code's Krylov exponential) against dense expm,
   for Hermitian and non-Hermitian H; qtensor's Hermitian-only routines shown
   for comparison.
2. Conventions for a non-Hermitian generator: single-site TDVP at full bond
   dimension (where it is exact) under H + sum c F with a complex
   last-site Y coefficient, against dense evolution.
3. End to end: paper_exact and paper trajectories at full bond dimension
   against a dense trajectory with the same noise and dense A, K.
4. gamma = 0: paper_exact and paper coincide (only the Krylov routine differs),
   through the two-site growth phase.

Run from the repo root: python langevin/validate_paper_exact.py
"""

import copy
import os
import sys

import numpy as np
import scipy.linalg as la

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qtensor.simulation.finiteTDVP as sim
import qtensor.simulation.updatemethod as methods
import qtensor.simulation.twositeTDVP as tdvp2
from langevin.models import (PAULIS, ising_mpo, with_onsite_fields, random_product_state,
                             dense_state, dense_onsite, dense_ising)
from langevin.integrator import LangevinModel, run_trajectory, coefficients
from langevin.matlab_lanczos import lanczos_exp, matlab_method

FAIL = []


def check(name, err, tol):
    ok = err < tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: {err:.2e} (tol {tol:.0e})")
    if not ok:
        FAIL.append(name)


def fidelity_error(u, v):
    u, v = u / la.norm(u), v / la.norm(v)
    return 1 - abs(np.vdot(u, v))


def test_krylov():
    print("1. Krylov exponentials against dense expm (dt = 0.02)")
    rng = np.random.default_rng(1)
    dt = 0.02
    for n in (4, 16, 200):
        X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        Hh = (X + X.conj().T) / 2
        X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        Ah = (X + X.conj().T) / (2 * np.sqrt(n))
        for label, H in (('Hermitian', Hh), ('non-Hermitian', Hh - 1j * Ah)):
            b = rng.normal(size=n) + 1j * rng.normal(size=n)
            exact = la.expm(-1j * dt * H) @ b
            out = lanczos_exp(lambda x: H @ x, b, -1j * dt)
            # n = 4: m = n, non-orthogonal basis, error O((dt|H|)^n / n!)
            check(f"lanczos_exp, n = {n}, {label}", fidelity_error(out, exact),
                  1e-10 if n > 4 else 1e-6)


def dense_generator(N, c, params):
    G = dense_ising(N, **params)
    for j in range(N):
        for a, P in enumerate(PAULIS):
            G = G + c[3 * j + a] * dense_onsite(P, j, N)
    return G


def test_conventions():
    print("2. Non-Hermitian generator, single-site TDVP at full D (N = 5, D = 4)")
    N, D, dt, steps = 5, 4, 0.02, 10
    params = LangevinModel(N).params
    rng = np.random.default_rng(2)
    c = rng.normal(size=3 * N).astype(complex)
    c[-2] = 0.7 - 1.3j                       # the typo's shape: complex last-site Y
    G_mpo = with_onsite_fields(ising_mpo(N, **params), c)
    G = dense_generator(N, c, params)
    psi0 = random_product_state(N, seed=3, D=D)
    v = dense_state(psi0)
    U = la.expm(-1j * dt * G)
    for _ in range(steps):
        v = U @ v
        v = v / la.norm(v)
    for label, method, tol in (('exact_method', methods.exact_method(), 1e-10),
                               ('matlab_method', matlab_method(), 1e-10),
                               ('qtensor lanczos (Hermitian-only)',
                                methods.lanczos_method(epsilon=1e-10, max_iters=16), None)):
        psi = copy.deepcopy(psi0)
        for _ in range(steps):
            psi, _ = sim.tdvp_step(psi, G_mpo, dt, method)
        err = fidelity_error(dense_state(psi), v)
        if tol is None:
            print(f"       {label}: {err:.2e} (expected to fail)")
        else:
            check(label, err, tol)


def dense_trajectory(psi0, N, dt, steps, gamma, gammaT, seed, scheme, params):
    rng = np.random.default_rng(seed)
    H = dense_ising(N, **params)
    F = [dense_onsite(P, j, N) for j in range(N) for P in PAULIS]
    v = dense_state(psi0)
    v = v / la.norm(v)
    for _ in range(steps):
        dW = rng.normal(0.0, np.sqrt(2 * gammaT * dt), 3 * N)
        A = np.array([np.real(1j * np.vdot(v, (H @ f - f @ H) @ v)) for f in F])
        K = np.array([[np.real(1j * np.vdot(v, (f @ g - g @ f) @ v)) for g in F] for f in F])
        c = coefficients(A, K, dW, dt, gamma, scheme)
        G = H + sum(ci * f for ci, f in zip(c, F))
        v = la.expm(-1j * dt * G) @ v
        v = v / la.norm(v)
    return v


def test_end_to_end():
    print("3. Trajectories at full D against dense (N = 5, D = 4, gamma = 0.9, gammaT = 0.18)")
    N, D, dt, steps = 5, 4, 0.04, 25
    model = LangevinModel(N)
    psi0 = random_product_state(N, seed=4, D=D)
    finals = {}
    # paper_exact with its own (reference) Krylov routine is not exact on the
    # 4-dimensional edge spaces: m = n with a non-orthogonal basis leaves an
    # O((dt |H_eff|)^4 / 4!) error per update. With exact_method it must be.
    for scheme, method, label, tol in (
            ('paper', None, 'paper', 1e-12),
            ('paper_exact', methods.exact_method(), 'paper_exact, exact_method', 1e-12),
            ('paper_exact', None, 'paper_exact, reference Krylov', 1e-7)):
        res = run_trajectory(psi0, model, dt, steps, gamma=0.9, gammaT=0.18, seed=7,
                             scheme=scheme, method=method)
        ref = dense_trajectory(psi0, N, dt, steps, 0.9, 0.18, 7, scheme, model.params)
        finals[label] = dense_state(res['final_state'])
        check(f"{label} vs dense", fidelity_error(finals[label], ref), tol)
    print(f"       paper vs paper_exact (size of the typo's effect): "
          f"{fidelity_error(finals['paper'], finals['paper_exact, exact_method']):.2e}")


def test_gamma_zero():
    print("4. gamma = 0: paper_exact = paper through two-site growth (N = 8, D = 8)")
    N, D, dt, steps = 8, 8, 0.04, 40
    model = LangevinModel(N)
    psi0 = random_product_state(N, seed=0)
    S = {}
    for scheme in ('paper', 'paper_exact'):
        res = run_trajectory(psi0, model, dt, steps, gamma=0.0, gammaT=0.18, seed=3,
                             scheme=scheme, grow_to=D)
        S[scheme] = res['S_mid']
    check("max |S_mid(paper) - S_mid(paper_exact)|",
          np.max(np.abs(S['paper'] - S['paper_exact'])), 1e-8)


if __name__ == '__main__':
    test_krylov()
    test_conventions()
    test_end_to_end()
    test_gamma_zero()
    print("FAIL: " + ", ".join(FAIL) if FAIL else "PASS")
