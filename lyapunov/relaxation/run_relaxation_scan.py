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
BETA = 1.0                              # inverse temperature of psi_uniform
D = 8                                   # bond dimension, same for every L
L_VALUES = [4, 8, 12, 16]
IMAG_STEPS = 60                         # TDVP steps for the imaginary-time build
SEED_NOISE = 1e-2                       # rank-seeding noise, see note below
T_MAX_FACTOR = 3.0                      # response evaluated to this * t_heis
N_TIMES = 6000

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'scan_results.pkl')


def build_uniform_thermofield(L, D, beta, steps, noise=SEED_NOISE):
    """
    Imaginary-time evolve the infinite-temperature thermofield double under
    the *symmetric* thermofield Hamiltonian to inverse temperature beta.

    The noise is not cosmetic: inf_T_thermofield returns a rank-1 state
    zero-padded to bond dimension D, and single-site TDVP cannot grow the
    Schmidt rank, so without seeding it the evolution stays rank 1 and the
    tangent space collapses. It does mean psi_uniform is only approximately
    the thermofield double -- fixed_point_residual below measures how
    approximate.
    """
    H_phys = ops.tilted_ising(J=J, h=H_FIELD, g=G_FIELD, N=L)
    H_sym = tf.thermofield_hamiltonian(H_phys, asym=False)
    psi_0 = tf.inf_T_thermofield(L, D, noise=noise)
    psi, _, energy = tf.finite_T_thermofield(
        beta, H_sym, steps=steps, initial_state=psi_0, plot=False
    )
    return psi, energy[-1]


def run_one(L, D=D, beta=BETA, steps=IMAG_STEPS):
    """Everything for a single system size. Returns a result dict."""
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
    result['hermiticity'] = float(np.max(np.abs(H_tangent - H_tangent.conj().T)))
    result['bond_dims'] = {n: A_L[n].shape for n in sites}

    t0 = clock.time()
    omega, U = la.eigh(H_tangent)
    result['t_eigh'] = clock.time() - t0
    result['omega'] = omega

    # How good a fixed point is psi_uniform? ||P H_asym psi*|| is exactly
    # zero for the true thermofield double; whatever we get here is the
    # combined finite-D and rank-seeding error, and it bounds how much of
    # any measured rate could be artefact.
    residual = resp.observable_tangent_vector(
        H_asym, A_L, A_R, C, V_L, basis_index_map, sites
    )
    result['fixed_point_residual'] = float(np.linalg.norm(residual))

    mid = sites[len(sites) // 2]
    observables = {
        'z_mid': resp.single_copy_onesite(ops.pauli('z'), mid),
        'x_mid': resp.single_copy_onesite(ops.pauli('x'), mid),
        'energy_mid': resp.single_copy_energy_density(
            mid, J=J, h=H_FIELD, g=G_FIELD
        ),
    }

    result['observables'] = {}
    for name, O in observables.items():
        v = resp.observable_tangent_vector(
            O, A_L, A_R, C, V_L, basis_index_map, sites
        )
        weights = resp.spectral_weights(omega, U, v)
        scales = resp.timescales(omega, weights)
        times = np.linspace(0, T_MAX_FACTOR * scales['t_heis'], N_TIMES)
        C_t = resp.response_function(omega, weights, times)
        tau_fit, r_squared, t_fit_end = resp.fit_relaxation_time(
            times, C_t, scales['t_zeno'], scales['t_heis']
        )
        result['observables'][name] = {
            'weights': weights,
            'times': times,
            'response': C_t,
            'scales': scales,
            'tau_fit': tau_fit,
            'r_squared': r_squared,
            't_fit_end': t_fit_end,
            'tau_cross': resp.crossing_time(times, C_t),
            'total_weight': float(weights.sum()),
        }
    return result


def summarize(result):
    L, dim = result['L'], result['dim']
    print(f"\n  L={L}  tangent dim={dim}  "
          f"|H-H*|={result['hermiticity']:.1e}  "
          f"||P H psi*||={result['fixed_point_residual']:.3e}")
    print(f"  timing: state {result['t_build_state']:.1f}s, "
          f"tangent {result['t_build_tangent']:.1f}s, "
          f"eigh {result['t_eigh']:.1f}s")
    for name, obs in result['observables'].items():
        s = obs['scales']
        print(f"    {name:11s} tau_fit={obs['tau_fit']:8.3f} "
              f"(R^2={obs['r_squared']:.3f}, fit to t={obs['t_fit_end']:.2f})  "
              f"tau_1/e={obs['tau_cross']:8.3f}  "
              f"window=[{s['t_zeno']:.2f}, {s['t_heis']:.2f}]  "
              f"n_eff={s['n_eff']:.0f}")


def main(l_values=L_VALUES):
    results = []
    for L in l_values:
        print(f"\n=== L = {L} ===")
        results.append(run_one(L))
        summarize(results[-1])

    with open(OUT_PATH, 'wb') as f:
        pickle.dump({'config': {'J': J, 'h': H_FIELD, 'g': G_FIELD,
                                'beta': BETA, 'D': D,
                                'imag_steps': IMAG_STEPS,
                                'seed_noise': SEED_NOISE},
                     'results': results}, f)
    print(f"\nsaved to {OUT_PATH}")

    print("\n=== convergence check ===")
    for name in results[0]['observables']:
        taus = [f"{r['observables'][name]['tau_fit']:.3f}" for r in results]
        print(f"  {name:11s} " + "  ".join(
            f"L={r['L']}: {t}" for r, t in zip(results, taus)))
    return results


if __name__ == '__main__':
    main()
