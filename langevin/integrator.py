"""
Single-site TDVP integration of the rearranged TDVP Langevin equation

    x' = X_H + sum_m X_{F_m} c_m,     c = S (gamma A + eta),   S = (1 + gamma K)^-1
    <eta_m(t) eta_n(t')> = 2 gammaT delta_mn delta(t - t')

(ito_issues.tex, eq. tdvpl-rearr; A, K from langevin.brackets). The right-hand
side is the TDVP flow of the generator H + sum_m c_m F_m with *real* c_m, so it
is Hermitian and one step is exactly qtensor's tdvp_step under the MPO
with_onsite_fields(H, c) -- the Lanczos/exact update methods and the sweeps
are reused unchanged. Over a step the noise enters as c_m dt ~ dW_m, i.e. the
per-step coefficient carries dW / dt.

Schemes (how c is chosen for the step):

  'paper'  The reference code (tdvp_langevin_finite.m + calculate_friction.m):
           c = dW/dt + gamma S (A - K dW), everything at the start of the step.
           Note the K dW: the rearranged equation needs K dW/dt there. It is
           reproduced here as written, for comparison.
  'euler'  The rearranged equation with coefficients frozen at the start of
           the step: c = S0 (gamma A0 + dW/dt). The vector fields X_F are
           followed along the step (tdvp_step re-projects as it goes), the
           coefficients are not -- the Ito/Stratonovich mix the notes describe.
  'heun'   Stratonovich predictor-corrector: predict psi~ with c0, recompute
           c~ at psi~ with the *same* dW, take the real step from psi with
           (c0 + c~)/2. With the exact flow along X_F at frozen coefficients,
           this reproduces the Stratonovich second-order term, i.e. the
           two-pass fix of the notes.
  'paper_exact'
           'paper' plus the reference code's last-site typo
           (update_H_MPO_langevin.m adds (etak - 1i*friction) * Y on the last
           site), so the generator has an anti-Hermitian part -i f Y_N, and
           every local exponential done with the reference Krylov routine
           (matlab_lanczos.py), which, unlike qtensor's, applies a
           non-Hermitian generator correctly.

At gamma = 0 the coefficients do not depend on the state (c = dW/dt), all
schemes coincide and no brackets are computed.

Growth: while bonds are short, steps are two-site TDVP. switch='all' hands
over to single-site once every bond is at min(D, full); switch='any' once the
largest bond reaches D, as tdvp_langevin_finite_2site.m's Dmaxout does, which
leaves shorter bonds frozen where they were.
"""

import copy
import time

import numpy as np

import qtensor.simulation.finiteTDVP as sim
import qtensor.simulation.twositeTDVP as tdvp2
import qtensor.simulation.updatemethod as methods

from langevin.models import ising_mpo, ising_local_terms, with_onsite_fields, PAPER_PARAMS
from langevin.brackets import Frame, local_brackets
from langevin.matlab_lanczos import matlab_method, matlab_two_site_expm

SCHEMES = ('paper', 'euler', 'heun', 'paper_exact')
SWITCH_RULES = ('all', 'any')


def default_method(scheme=None):
    if scheme == 'paper_exact':
        return matlab_method()
    # epsilon well below the qtensor default 1e-5: see lyapunov stepper.py
    return methods.lanczos_method(epsilon=1e-10, max_iters=16)


def grown(psi, grow_to, switch='all'):
    """Whether growth is over and single-site TDVP takes over (see module doc)."""
    if switch == 'all':
        return tdvp2.bonds_saturated(psi, grow_to)
    sites = sorted(psi.sites)
    N = len(sites)
    largest_full = 2 ** (N // 2)
    return max(psi[n].shape[2] for n in sites[:-1]) >= min(grow_to, largest_full)


def friction_matrix(K, gamma):
    """S = (1 + gamma K)^-1, block by block (K is block diagonal, see brackets.py)."""
    n = K.shape[0]
    S = np.zeros_like(K)
    for j in range(0, n, 3):
        S[j:j + 3, j:j + 3] = np.linalg.inv(np.eye(3) + gamma * K[j:j + 3, j:j + 3])
    return S


def coefficients(A, K, dW, dt, gamma, scheme):
    """Real field coefficients c_m for one step (generator H + sum c_m F_m)."""
    if gamma == 0:
        return dW / dt
    S = friction_matrix(K, gamma)
    if scheme in ('paper', 'paper_exact'):
        friction = gamma * S @ (A - K @ dW)
        c = dW / dt + friction
        if scheme == 'paper_exact':
            # last site's Y: (etak(end-1) - 1i*friction(end-1)) in the reference
            c = c.astype(complex)
            c[-2] = dW[-2] / dt - 1j * friction[-2]
        return c
    return S @ (gamma * A + dW / dt)


class LangevinModel:
    """The Hamiltonian in both forms the step needs: MPO for TDVP, local terms for brackets."""

    def __init__(self, N, J=-1.0, hx=-1.05, hy=0.0, hz=0.5):
        self.N = N
        self.params = dict(J=J, hx=hx, hy=hy, hz=hz)
        self.H = ising_mpo(N, **self.params)
        self.onsite, self.bond = ising_local_terms(**self.params)

    def brackets(self, psi):
        """(A, K, f, energy, frame) at psi, from one canonicalization pass."""
        frame = Frame(psi)
        A, K, f, E = local_brackets(frame, self.onsite, self.bond)
        return A, K, f, E, frame


def langevin_step(psi, model, dt, gamma, gammaT, rng, scheme='heun',
                  method=None, at_start=None, grow_to=None, switch='all'):
    """
    One Langevin step. Mutates and returns psi (right-canonical, as tdvp_step
    leaves it), plus a dict with the noise and coefficients used.

    at_start : optional (A, K) already computed at psi, to avoid recomputing
               them when the caller also measures there.
    grow_to  : target bond dimension. Until growth is over (see `grown`), the
               step is two-site TDVP truncated to grow_to
               (twositeTDVP.tdvp2_step); after that, single-site tdvp_step.
               None: always single-site (psi must already carry its bond
               dimension).
    switch   : 'all' or 'any', the hand-over rule (module doc).
    """
    assert scheme in SCHEMES, scheme
    assert switch in SWITCH_RULES, switch
    method = method or default_method(scheme)
    if grow_to is not None and not grown(psi, grow_to, switch):
        expm = matlab_two_site_expm if scheme == 'paper_exact' else tdvp2.krylov_expm

        def evolve(state, G):
            return tdvp2.tdvp2_step(state, G, dt, grow_to, method, expm=expm)
    else:
        def evolve(state, G):
            return sim.tdvp_step(state, G, dt, method)
    dW = rng.normal(0.0, np.sqrt(2 * gammaT * dt), 3 * model.N) if gammaT > 0 \
        else np.zeros(3 * model.N)
    if gamma == 0:
        c = dW / dt
    else:
        if at_start is None:
            A0, K0 = model.brackets(psi)[:2]
        else:
            A0, K0 = at_start
        c = coefficients(A0, K0, dW, dt, gamma, scheme)
        if scheme == 'heun':
            pred, _ = evolve(copy.deepcopy(psi), with_onsite_fields(model.H, c))
            A1, K1 = model.brackets(pred)[:2]
            c = 0.5 * (c + coefficients(A1, K1, dW, dt, gamma, scheme))
    psi, _ = evolve(psi, with_onsite_fields(model.H, c))
    return psi, {'dW': dW, 'c': c}


def mid_bond_entropy(frame):
    """Von Neumann entropy (natural log, as measure_entanglement.m) on every bond."""
    out = []
    for n, s in sorted(frame.schmidt_values().items()):
        p = s ** 2
        p = p[p > 1e-300] / np.sum(p)
        out.append(-np.sum(p * np.log(p)))
    return np.array(out)


def run_trajectory(psi0, model, dt, steps, gamma=0.0, gammaT=0.0, seed=0,
                   scheme='heun', method=None, measure_every=1, verbose=False,
                   grow_to=None, switch='all'):
    """
    One trajectory. grow_to, switch: see langevin_step -- pass the target bond
    dimension with an unpadded (D = 1) product state, as the reference code does.
    Returns a dict of arrays over the measured times:
      t, energy, entropy (n_times, N-1 bonds; natural log), sigma (n_times, N, 3),
      S_mid (the bond in the middle of the chain), plus the parameters.
    psi0 is not mutated.
    """
    rng = np.random.default_rng(seed)
    method = method or default_method(scheme)
    psi = copy.deepcopy(psi0)
    N = model.N
    out = {k: [] for k in ('t', 'energy', 'entropy', 'sigma')}
    t0 = time.time()
    t_switch = None
    for step in range(steps + 1):
        A, K, f, E, frame = model.brackets(psi)
        if step % measure_every == 0:
            out['t'].append(step * dt)
            out['energy'].append(E)
            out['entropy'].append(mid_bond_entropy(frame))
            out['sigma'].append(f.reshape(N, 3))
            if verbose:
                print(f"t={step*dt:7.3f}  E={E: .5f}  S_mid={out['entropy'][-1][N//2-1]:.4f}"
                      f"  ({time.time()-t0:.1f}s)", flush=True)
        if step == steps:
            break
        # continue from the frame's own right-canonical state, so the
        # brackets and the step start from the same canonicalization pass
        psi = frame.state()
        if grow_to is not None and t_switch is None and grown(psi, grow_to, switch):
            t_switch = step * dt
        psi, _ = langevin_step(psi, model, dt, gamma, gammaT, rng, scheme, method,
                               at_start=(A, K), grow_to=grow_to, switch=switch)
    res = {k: np.array(v) for k, v in out.items()}
    # time at which two-site growth handed over to single-site TDVP
    res['t_switch'] = t_switch
    res['S_mid'] = res['entropy'][:, N // 2 - 1]
    res.update(dict(dt=dt, steps=steps, gamma=gamma, gammaT=gammaT, seed=seed,
                    scheme=scheme, switch=switch, N=N, D=grow_to, **model.params))
    res['final_state'] = psi
    return res


def _ensemble_worker(args):
    psi0, model_kwargs, run_kwargs = args
    model = LangevinModel(**model_kwargs)
    res = run_trajectory(psi0, model, **run_kwargs)
    res.pop('final_state')
    return res


def run_ensemble(psi0, model_kwargs, seeds, processes=None, **run_kwargs):
    """
    Independent trajectories, one per seed, in parallel processes.
    Call from under `if __name__ == '__main__':` (Windows spawns).
    Returns a list of trajectory dicts (without final states).
    """
    return run_many(psi0, model_kwargs, [dict(run_kwargs, seed=s) for s in seeds],
                    processes)


def run_many(psi0, model_kwargs, run_kwargs_list, processes=None):
    """
    One trajectory per entry of run_kwargs_list (each a run_trajectory kwargs
    dict, including its seed), all in one process pool -- so a parameter scan
    keeps every worker busy instead of synchronizing per parameter value.
    Results come back in input order.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(psi0, model_kwargs, kw) for kw in run_kwargs_list]
    if processes == 1:
        return [_ensemble_worker(j) for j in jobs]
    # one BLAS thread per worker: the local problems are small, and N workers
    # each spinning up a full thread pool oversubscribe the machine
    threads = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
    saved = {k: os.environ.get(k) for k in threads}
    os.environ.update({k: '1' for k in threads})
    try:
        with ProcessPoolExecutor(max_workers=processes) as ex:
            return list(ex.map(_ensemble_worker, jobs))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
