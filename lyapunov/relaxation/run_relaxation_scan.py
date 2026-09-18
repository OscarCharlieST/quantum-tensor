"""
System-size scan: build the uniform thermofield double, project H_asym onto
its tangent space, and extract a relaxation time for a few local
observables at each L.

Run from the repo root:   python lyapunov/relaxation/run_relaxation_scan.py

The question being asked is whether the relaxation time converges with L at
fixed bond dimension. A non-conserved local observable (z, x) should
converge to an intrinsic local timescale; the energy density is conserved
and should instead show diffusive L^2 growth, so it is included as a
contrast rather than as a convergence test.
"""

import os
import pickle
import sys
import time as clock

import numpy as np
import scipy.linalg as la

sys.path.insert(0, os.getcwd())

import qtensor.operators as ops
import qtensor.thermofield as tf
import qtensor.simulation.updatemethod as methods
import lyapunov.relaxation.tangent_hamiltonian as tangent
import lyapunov.relaxation.response as resp


# ------------------------------------------------------------------- config

J, H_FIELD, G_FIELD = 1, 0.25, -0.525   # tilted Ising defaults (1702.08894)
BETA = 0.1                              # inverse temperature of psi_uniform
D = 8                                   # bond dimension, same for every L
L_VALUES = [8, 12, 16]
# 240, not 60: below D = 16 the fixed-point residual is limited by the
# variational manifold, but at D >= 16 it is limited by this build instead,
# and 60 steps leaves a factor of 25 on the table (3.3e-09 -> 1.3e-10 at
# L = 16, D = 16). The build is seconds against an eigh of hours, so there
# is no reason to economize here. Past 240 it is roundoff scatter, not
# improvement. See convergence_probe.py and README, "Cost and convergence".
IMAG_STEPS = 240                        # TDVP steps for the imaginary-time build
SEED_NOISE = 0.0                        # none needed, see build_uniform_thermofield
T_MAX_FACTOR = 3.0                      # response evaluated to this * t_heis
N_TIMES = 6000                          # floor on the sample count
SAMPLES_PER_ZENO = 20                   # ceiling on dt, see response_times
MAX_TIMES = 400_000                      # cap, ~3 MB transient per observable


def response_times(scales):
    """
    Sample grid for C(t): spans T_MAX_FACTOR * t_heis, but with dt small
    enough to resolve t_zeno.

    A fixed sample count cannot do both. t_heis = 2 pi / spacing grows as
    the spectrum gets finer, so at fixed N_TIMES the step dt grows with it
    and the *better* calculation gets the *worse* time resolution -- at
    L = 24, D = 16 that had reached 1.2 samples per Zeno time, enough to
    quantise tau_1/e onto the grid. The response is no longer stored
    (see `run_one`), so a fine grid costs only transient memory.
    """
    t_max = T_MAX_FACTOR * scales['t_heis']
    n_needed = int(np.ceil(t_max / (scales['t_zeno'] / SAMPLES_PER_ZENO))) + 1
    return np.linspace(0, t_max, min(max(N_TIMES, n_needed), MAX_TIMES))

D_VALUES = [6, 8, 10, 12]               # bond-dimension scan, at L_FIXED
L_FIXED = 16

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'scan_results.pkl')
BOND_OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'bond_scan_results.pkl')


def build_uniform_thermofield(L, D, beta, steps, noise=SEED_NOISE):
    """
    Imaginary-time evolve the infinite-temperature thermofield double under
    the *symmetric* thermofield Hamiltonian to inverse temperature beta.

    No seeding noise is needed, and adding it is actively harmful.
    inf_T_thermofield returns a rank-1 state zero-padded to bond dimension
    D, and single-site TDVP is a fixed-rank method, which long looked like
    it meant the evolution would stay rank 1. It does not:
    states.left_orthogonal_tensor calls la.svd(..., full_matrices=False)
    and keeps every singular value including the exact zeros, so after one
    canonicalization the A tensors are dense isometries whose columns past
    the rank are an arbitrary orthonormal completion. The environments then
    have support on every bond index, H_eff couples the centre tensor into
    the zero-weight directions, and the evolution walks off the
    rank-deficient boundary into the interior on its own.

    Measured 2026-09-18 at L=16, D=12: the noiseless build reaches full
    rank (12 of 12 Schmidt values above 1e-10) with
    ||P H_asym psi*|| = 1.1e-7, against 7.1e-2 for the noise=1e-2 seed that
    was previously thought necessary -- five orders of magnitude, and the
    seed was the dominant error in every number this subproject produced
    before that date. The residual also resumes falling with D once the
    noise is gone (5.9e-7 at D=8 -> 1.1e-7 at D=12), which it had not done
    at all while the noise set the floor.
    """
    H_phys = ops.tilted_ising(J=J, h=H_FIELD, g=G_FIELD, N=L)
    H_sym = tf.thermofield_hamiltonian(H_phys, asym=False)
    psi_0 = tf.inf_T_thermofield(L, D, noise=noise)
    psi, _, energy = tf.finite_T_thermofield(
        beta, H_sym, steps=steps, initial_state=psi_0, plot=False
    )
    return psi, energy[-1]


def run_one(L, D=D, beta=BETA, steps=IMAG_STEPS, store_response=False):
    """
    Everything for a single system size. Returns a result dict.

    `store_response` keeps the sampled C(t) trace in the result. Off by
    default: it is 2 * N_TIMES floats per observable and is a pure function
    of `omega` and `weights`, both of which are kept, so plots.py recomputes
    it on demand. At D = 18 that is the difference between ~0.4 MB and
    ~0.7 MB per entry, and nothing in the pipeline needs it stored.

    Nothing of size dim^2 is ever kept -- not H_tangent, not the
    eigenvectors. At D = 18 those are 2.5 GB apiece.
    """
    result = {'L': L, 'D': D, 'beta': beta}
    t0 = clock.time()

    psi, energy = build_uniform_thermofield(L, D, beta, steps)
    result['energy_density'] = energy / L
    result['t_build_state'] = clock.time() - t0

    H_phys = ops.tilted_ising(J=J, h=H_FIELD, g=G_FIELD, N=L)
    H_asym = tf.thermofield_hamiltonian(H_phys, asym=True)

    t0 = clock.time()
    sites = sorted(psi.sites)
    A_L, A_R, left_envs, right_envs = tangent.canonicalize_and_build_environments(
        psi, H_asym, max_bond_dim=D
    )
    V_L = {n: tangent.build_null_space_tensor(A_L[n]) for n in sites}
    C = tangent.build_centre_tensors(A_L, A_R)
    H_tangent, basis_index_map = tangent.assemble_tangent_hamiltonian(
        sites, A_L, A_R, V_L, left_envs, right_envs, H_asym
    )
    result['t_build_tangent'] = clock.time() - t0
    result['dim'] = H_tangent.shape[0]
    # Row-blocked rather than np.max(np.abs(H - H.conj().T)): that
    # expression allocates three dim^2 temporaries (conj, subtract, abs),
    # which at dim ~ 12000 is 7 GB of scratch to compute one number.
    result['hermiticity'] = float(max(
        np.abs(H_tangent[i:i + 512] - H_tangent[:, i:i + 512].conj().T).max()
        for i in range(0, H_tangent.shape[0], 512)))
    result['bond_dims'] = {n: A_L[n].shape for n in sites}

    t0 = clock.time()
    # overwrite_a: H_tangent is not needed past this point, and at
    # dim ~ 12000 the copy eigh would otherwise take is 2.4 GB.
    omega, U = la.eigh(H_tangent, overwrite_a=True)
    del H_tangent
    result['t_eigh'] = clock.time() - t0
    result['omega'] = omega

    # How good a fixed point is psi_uniform? ||P H_asym psi*|| is exactly
    # zero for the true thermofield double; whatever we get here is the
    # finite-D and finite-imaginary-time error, and it bounds how much of
    # any measured rate could be artefact. With SEED_NOISE = 0 this now
    # falls with D instead of sitting on a noise floor.
    residual = resp.observable_tangent_vector(
        H_asym, A_L, A_R, C, V_L, basis_index_map, sites
    )
    result['fixed_point_residual'] = float(np.linalg.norm(residual))

    mid = sites[len(sites) // 2]
    # Energy and current only. The single-site z and x responses were here
    # as generic non-conserved contrasts -- something that should relax to
    # an intrinsic local rate, against a conserved density that should not
    # -- and they served that purpose, but they carry no transport
    # information and cost a diagonalization each. To put them back:
    #     'z_mid': resp.single_copy_onesite(ops.pauli('z'), mid),
    #     'x_mid': resp.single_copy_onesite(ops.pauli('x'), mid),
    # `response.single_copy_onesite` is unchanged and still builds them,
    # and plots.py still knows their colours and labels.
    observables = {
        'energy_mid': resp.single_copy_energy_density(
            mid, J=J, h=H_FIELD, g=G_FIELD
        ),
        # The current is the one that gates a diffusion measurement: energy
        # cannot flow diffusively until the current has reached its
        # constitutive value, so tau(current) is the time to wait before
        # fitting D. See README, "The current relaxation time".
        'current_mid': resp.single_copy_current(
            mid, J=J, h=H_FIELD, g=G_FIELD
        ),
        # Green-Kubo is a statement about the *total* current: the
        # autocorrelator of current_mid above is only the r = 0 term of
        # sum_r <j_r(t) j_0(0)>, and it falls off like 1/L, so it measures
        # a vanishing fraction of the transport. See README, "Green-Kubo".
        'current_total': resp.single_copy_total_current(
            sites, J=J, h=H_FIELD, g=G_FIELD
        ),
    }

    result['observables'] = {}
    for name, O in observables.items():
        v = resp.observable_tangent_vector(
            O, A_L, A_R, C, V_L, basis_index_map, sites
        )
        W = resp.pad_with_identity(O, sites)
        weights = resp.spectral_weights(omega, U, v)
        scales = resp.timescales(omega, weights)
        times = response_times(scales)
        C_t = resp.response_function(omega, weights, times)
        # An observable overlapping a conserved quantity relaxes to that
        # overlap, not to zero; both estimates below are taken on the part
        # that actually dephases. For energy_mid this floor is 0.16 at
        # L = 8 and 0.074 at L = 16.
        c_inf = resp.conserved_fraction(omega, weights)
        tau_fit, r_squared, t_fit_end = resp.fit_relaxation_time(
            times, C_t, scales['t_zeno'], scales['t_heis'], c_inf=c_inf,
            spacing=scales['spacing']
        )
        result['observables'][name] = {
            'weights': weights,
            'scales': scales,
            'tau_fit': tau_fit,
            'r_squared': r_squared,
            't_fit_end': t_fit_end,
            'c_inf': c_inf,
            # The low-frequency exponent of A(w), which is the diffusion
            # question in its cleanest form: A_J(0) finite means diffusive,
            # A_h ~ |w|^-1/2 is the same statement seen from the density.
            'spectral': resp.spectral_exponent(omega, weights, scales),
            'n_times': len(times),
            't_max': float(times[-1]),
            'tau_cross': resp.crossing_time(times, C_t, c_inf=c_inf),
            **({'times': times, 'response': C_t} if store_response else {}),
            'total_weight': float(weights.sum()),
            # What fraction of O's static weight the tangent space sees.
            # Exactly 1 for every observable here, which is worth recording
            # rather than assuming -- it says the Green-Kubo numerator has
            # no *static* truncation error, leaving fixed_point_residual to
            # bound the dynamical one on its own.
            'capture': float(weights.sum() / resp.static_variance(
                psi, ops.mpo([(n, W[n]) for n in sites], O.l, O.r)
            )),
        }

    # The Green-Kubo denominator, exact rather than tangent-projected.
    result['chi'] = resp.static_susceptibility(
        psi, sites, J=J, h=H_FIELD, g=G_FIELD
    )
    tot = result['observables']['current_total']
    result['green_kubo'] = resp.diffusion_constant(
        omega, tot['weights'], result['chi'], tot['scales'], tot['tau_cross']
    )
    return result


def summarize(result):
    L, dim = result['L'], result['dim']
    print(f"\n  L={L}  tangent dim={dim}  "
          f"|H-H*|={result['hermiticity']:.1e}  "
          f"||P H psi*||={result['fixed_point_residual']:.3e}")
    print(f"  timing: state {result['t_build_state']:.1f}s, "
          f"tangent {result['t_build_tangent']:.1f}s, "
          f"eigh {result['t_eigh']:.1f}s")
    gk = result['green_kubo']
    win = (f"[{gk['eta_min']:.3f}, {gk['eta_max']:.3f}] "
           f"({gk['decades']:.2f} dec)" if gk['exists'] else "empty")
    print(f"  Var(H)={result['chi']:.3f}   Green-Kubo eta window {win}   "
          f"D_win={gk['D']:.4f}  dlnD/dlneta={gk['log_slope']:.3f}   "
          f"D_peak={gk['D_peak']:.4f} at eta={gk['eta_peak']:.3f}")
    for name, obs in result['observables'].items():
        s = obs['scales']
        print(f"    {name:11s} tau_fit={obs['tau_fit']:8.3f} "
              f"(R^2={obs['r_squared']:.3f}, fit to t={obs['t_fit_end']:.2f})  "
              f"tau_1/e={obs['tau_cross']:8.3f}  "
              f"window=[{s['t_zeno']:.2f}, {s['t_heis']:.2f}]  "
              f"n_eff={s['n_eff']:.0f}")


def main(l_values=L_VALUES, d_value=None, out_path=None):
    out_path = out_path or OUT_PATH
    results = []
    for L in l_values:
        print(f"\n=== L = {L}, D = {d_value or D} ===", flush=True)
        t0 = clock.time()
        results.append(run_one(L, D=d_value or D))
        summarize(results[-1])
        # Saved after every L, as in main_bond_scan: the largest system is
        # the one most likely to run out of memory, and it should not take
        # the cheaper ones down with it.
        with open(out_path, 'wb') as f:
            pickle.dump({'config': {'J': J, 'h': H_FIELD, 'g': G_FIELD,
                                    'beta': BETA, 'D': d_value or D,
                                    'imag_steps': IMAG_STEPS,
                                    'seed_noise': SEED_NOISE},
                         'results': results}, f)
        print(f"  [saved {len(results)} entries to {out_path}; "
              f"L={L} took {clock.time() - t0:.0f}s]", flush=True)

    with open(out_path, 'wb') as f:
        # d_value, not D: this final write overwrites the incremental ones,
        # and the module default here silently mislabels the config of
        # every run launched with --D (it did, for l24_d16_results.pkl).
        pickle.dump({'config': {'J': J, 'h': H_FIELD, 'g': G_FIELD,
                                'beta': BETA, 'D': d_value or D,
                                'imag_steps': IMAG_STEPS,
                                'seed_noise': SEED_NOISE},
                     'results': results}, f)
    print(f"\nsaved to {out_path}")

    print("\n=== convergence check ===")
    for name in results[0]['observables']:
        taus = [f"{r['observables'][name]['tau_fit']:.3f}" for r in results]
        print(f"  {name:11s} " + "  ".join(
            f"L={r['L']}: {t}" for r, t in zip(results, taus)))

    # The number the diffusion question turns on: energy cannot relax
    # diffusively faster than the current that carries it, so tau(energy) /
    # tau(current) has to be >> 1 for a diffusive description to have a
    # window to live in.
    if {'energy_mid', 'current_mid'} <= set(results[0]['observables']):
        print("\n=== separation of scales: tau(energy) / tau(current) ===")
        for r in results:
            e, j = r['observables']['energy_mid'], r['observables']['current_mid']
            print(f"  L={r['L']:<3d} fit {e['tau_fit'] / j['tau_fit']:6.2f}   "
                  f"1/e {e['tau_cross'] / j['tau_cross']:6.2f}   "
                  f"(tau_j = {j['tau_fit']:.3f}, in window "
                  f"[{j['scales']['t_zeno']:.2f}, {j['scales']['t_heis']:.2f}])")
    return results


def main_bond_scan(L=L_FIXED, d_values=D_VALUES, out_path=None):
    """
    The same analysis at fixed L, scanning bond dimension instead.

    This is the scan the Green-Kubo question actually needs. Widening the
    admissible broadening window needs more modes carrying weight, and the
    L scan buys them slowly -- worse, the total current concentrates its
    weight on far fewer modes than a local current does (n_eff 98 vs 409 at
    L=16, D=8), so the effective level spacing is large for exactly the
    observable that needs it small. The tangent dimension goes as ~39 D^2
    at L=16, so bond dimension is the cheaper lever on the mode count. It
    is also the convergence check the variational approximation needs
    anyway: the correlator is tangent-projected, and nothing so far says
    how much of D_peak is physics and how much is the manifold.

    eigh dominates and scales as the cube of the tangent dimension, so
    D=12 costs roughly 11x D=8. Measured at L=16: eigh ~ dim^2.94 and the
    tangent build ~ dim^2.5, with dim ~ 39 D^2, giving roughly 16 / 34 / 67
    minutes at D = 14 / 16 / 18. Memory is the tighter constraint --
    H_tangent and its eigenvectors are 2 * dim^2 * 16 bytes, i.e. 1.9 / 3.2
    / 5.1 GB at those same D, before LAPACK workspace.
    """
    out_path = out_path or BOND_OUT_PATH
    results = []
    for D_val in d_values:
        print(f"\n=== L = {L}, D = {D_val} ===", flush=True)
        t0 = clock.time()
        results.append(run_one(L, D=D_val))
        summarize(results[-1])
        # Written after every D, not at the end: the largest D is the one
        # most likely to exhaust memory or run past a time budget, and
        # losing the cheap ones with it would be silly.
        with open(out_path, 'wb') as f:
            pickle.dump({'config': {'J': J, 'h': H_FIELD, 'g': G_FIELD,
                                    'beta': BETA, 'L': L,
                                    'imag_steps': IMAG_STEPS,
                                    'seed_noise': SEED_NOISE},
                         'results': results}, f)
        print(f"  [saved {len(results)} entries to {out_path}; "
              f"D={D_val} took {clock.time() - t0:.0f}s]", flush=True)

    print("\n=== Green-Kubo against bond dimension ===")
    for r in results:
        gk = r['green_kubo']
        win = (f"[{gk['eta_min']:.3f}, {gk['eta_max']:.3f}] "
               f"({gk['decades']:.2f} dec)" if gk['exists'] else 'empty')
        print(f"  D={r['D']:<3d} dim={r['dim']:<6d} "
              f"resid={r['fixed_point_residual']:.3e}  n_eff="
              f"{r['observables']['current_total']['scales']['n_eff']:6.1f}  "
              f"window {win:>24s}  D_peak={gk['D_peak']:.4f}  "
              f"slope={gk['log_slope']:.3f}")
    return results


if __name__ == '__main__':
    if '--l-scan' in sys.argv:
        kw = {}
        for i, a in enumerate(sys.argv):
            if a == '--l-values':
                kw['l_values'] = [int(x) for x in sys.argv[i + 1].split(',')]
            if a == '--D':
                kw['d_value'] = int(sys.argv[i + 1])
            if a == '--out':
                kw['out_path'] = sys.argv[i + 1]
        main(**kw)
    elif '--bond-scan' in sys.argv:
        kw = {}
        for i, a in enumerate(sys.argv):
            if a == '--d-values':
                kw['d_values'] = [int(x) for x in sys.argv[i + 1].split(',')]
            if a == '--out':
                kw['out_path'] = sys.argv[i + 1]
        main_bond_scan(**kw)
    else:
        main()
