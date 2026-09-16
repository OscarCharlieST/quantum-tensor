# Workflow and results

Practical record for the `lyapunov/` work: what runs, in what order, what
came out, and what still needs checking. The physics arguments live in the
subproject READMEs — this does not repeat them.

## Structure

```
lyapunov/
  README.md                     umbrella: shared background, why this split in two
  WORKFLOW.md                   this file
  relaxation/                   ACTIVE
    README.md                   theory, the dephasing argument, default parameters
    tangent_hamiltonian.py      tangent basis + projected Hamiltonian
    response.py                 observables, weights, response, rate fitting
    run_relaxation_scan.py      the L scan driver
  tdvp_lyapunov/                ACTIVE: first scan done (beta = 1)
    README.md                   design, Route B derivation, validation, cost, results, next
    frame.py                    tangent frame at a point; project_to_frame / retract / frame_change
    stepper.py                  one TDVP step as a map; FD tangent map (Route A, parallel)
    tangent_generator.py        exact generator and transport (Route B)
    benettin.py                 forward QR loop + h5 storage; Ginelli backward pass
    analysis.py                 energy-density profiles of modes, DCT, q-weight by exponent
    plots.py                    per-run figures;  compare_runs.py  cross-run overlays
    run_lyapunov.py             driver (--route, --n-jobs, --out-dir, --time-only)
    validate_frame.py, validate_benettin.py, validate_routeB.py
    runs/                       logs, queue scripts, small h5; big h5 in C:/Users/charl/lyapunov_runs
    figures/
```

## The result that forced the split

`H_asym` is exactly Hermitian, so `P H_asym P` has a real spectrum and
`‖δψ(t)‖` is exactly conserved — **every Lyapunov exponent at the fixed
point is exactly zero.** The only term that could have been non-Hermitian
(the derivative of the tangent projector) is multiplied by `H ψ* = 0`, so
it vanishes identically. The fixed-point shortcut and nonzero exponents are
mutually exclusive.

Hence: `relaxation/` computes dephasing relaxation rates of local
observables, which are genuinely nonzero. `tdvp_lyapunov/` recovers
true exponents by linearizing along a trajectory where `Hψ ≠ 0`.

## Pipeline

| # | step | function |
|---|---|---|
| 1 | imaginary-time TDVP under `H_sym` from the noise-seeded infinite-T state | `run_relaxation_scan.build_uniform_thermofield` |
| 2 | canonicalize (three sweeps) + build both environment families | `tangent_hamiltonian.canonicalize_and_build_environments` |
| 3 | null-space tensors `V_L^n` | `build_null_space_tensor` |
| 4 | centre tensors `C^n` | `build_centre_tensors` — gauge-critical, see below |
| 5 | assemble `H_tangent`, one column at a time | `assemble_tangent_hamiltonian` |
| 6 | exact diagonalization | `scipy.linalg.eigh` |
| 7 | observable vector `v_i = <b_i\|O\|psi*>` | `response.observable_tangent_vector` |
| 8 | weights `\|U†v\|²`, response, timescales, fit | `response.*` |

Step 8 is free once step 6 is done: the entire time trace at any `t` comes
from one diagonalization, with no time stepping. That is the whole payoff of
the method, and it makes the Zeno onset, the exponential window and the
recurrence all directly visible rather than assumed.

Three sweeps in step 2 rather than two: the left- and right-canonical forms
each clip bond dimensions to their own staircase, so one sweep in each
direction can leave them disagreeing bond-for-bond, which would break the
tangent basis. An assert guards it.

## How to run

```
python lyapunov/relaxation/run_relaxation_scan.py
```

About 70 s for L = 4, 8, 12, 16 at D = 8. Writes `scan_results.pkl`
alongside the script (weights, response curves, spectra, timings).

## Results so far

From setup runs at L = 4, 8, 12. **The L = 16 point has not been run and
`scan_results.pkl` has not been written** — that is the run awaiting your OK.

τ from the 1/e crossing of the response:

| observable | L=4 | L=8 | L=12 |
|---|---|---|---|
| `z_mid` | — (no decay) | 4.89 | 5.00 |
| `x_mid` | 0.342 | 0.353 | 0.379 |
| `energy_mid` | 2.30 | 1.18 | 1.19 |

Tangent dimensions 191 / 959 / 1727; `n_eff` (modes actually carrying
weight) 8–15 / 49–74 / 103–159.

L = 8 → 12 agree to ~2% for `z_mid` and `energy_mid`, so it does look
convergent. L = 4 is clearly too small to say anything.

## Validations passed

- `H_asym` annihilates the exact infinite-temperature thermofield state to
  **3.7e-16**, and the tangent-projected residual agrees at **2.1e-16** —
  checks the MPO convention and the overlap machinery together.
- `H_tangent` Hermitian to **~2e-15**, with the `n<m` and `n>m` blocks
  built by different code paths (mixed left vs mixed right environments).
- Tangent dimension matches `Σ_n (d·D_{n-1} − D_n)·D_n` computed
  independently from the bond dimensions.
- `‖P H_asym ψ*‖ ≤ ‖H_asym ψ*‖` now holds at every β tested.
- Centre tensors normalized (asserted in `build_centre_tensors`).
- `V_L` gauge condition `A†V_L = 0` and orthonormality (asserted).

## The bug worth remembering

The fixed-point residual `‖P H_asym ψ*‖` came out at ~1.0 while
`‖H_asym ψ*‖` was ~0.001 — impossible, since a projection cannot exceed the
vector it projects.

Cause: the centre tensors were being rebuilt by an independent
canonicalization sweep. Where the Schmidt spectrum has near-zero values —
thermofield states have them down at 1e-11 — the singular vectors are
numerically arbitrary, so that sweep silently landed in a **different gauge**
from the `A_L`/`A_R` used everywhere else. Every overlap built on it was
meaningless while looking entirely plausible.

Fix: derive `C^n` from the bond matrices (`C^n = Λ^{n-1} A_R^n`,
`Λ^n = (A_L^n)† C^n`), which uses only `A_L`'s isometry property and no SVD.

## To check

1. **β = 1**, against the β ≈ 1e-2 in `active.ipynb`. At your usual
   temperature the thermofield double is effectively rank 2 (Schmidt
   `[1, 5e-3, 1e-8, 7e-11]` at L=4), so most of a D=8 tangent space would sit
   on numerically null directions. β = 1 fills the bond dimension while
   keeping the residual small. This is the judgement call most worth your
   scrutiny.
2. **`x_mid` relaxes faster than its own Zeno time**, so there is no
   exponential regime and `tau_fit` correctly returns `nan`. Only the 1/e
   crossing means anything there.
3. **Seeding noise 1e-2** makes `ψ*` only approximately the thermofield
   double. `fixed_point_residual` ≈ 1e-2 against a tangent bandwidth ≈ 5.
4. **`n_eff` ≈ 110 at L=12** is a modest continuum. The dephasing argument
   wants the level spacing well below the decay rate; this is the main
   reason to expect the L=16 point to still be drifting.
5. **`tau_fit` for `z_mid` is still drifting** (9.3 → 13.7) even where
   `tau_1/e` has converged — its response drops fast then crawls through a
   slow tail, which a single exponential does not describe. Treat `tau_1/e`
   as the robust estimator and `R²` as the flag.

## Next

- Wavevector-resolved energy density → `Γ_q` vs `q²` → diffusion constant,
  cross-checked against `visualise.estimate_diffusion_constant_from_widths`
  on a direct `finiteTDVP.tdvp` run. This is the natural physics payoff and
  connects to the transport machinery already in the repo.
- β and D scans — both cheap, and D is the lever that densifies the
  spectrum without growing L.
- The auxiliary-gauge-mode caveat (unitary rotations on the purification's
  auxiliary copy are physically trivial but may appear as tangent
  directions) is still unresolved.

## tdvp_lyapunov (2026-09-16)

Full detail in `tdvp_lyapunov/README.md`; this is the practical summary.

### What runs

    python lyapunov/tdvp_lyapunov/run_lyapunov.py --L 8 --D 8 --route A --n-jobs 16 \
        --blocks 250 --transient 160 --dt 0.05 --store-Q-every 25 \
        --out-dir C:/Users/charl/lyapunov_runs
    python lyapunov/tdvp_lyapunov/plots.py C:/Users/charl/lyapunov_runs/L8_D8_beta1.h5 --discard 20 --clv
    python lyapunov/tdvp_lyapunov/compare_runs.py

Validation: `validate_frame.py`, `validate_benettin.py [2|3]`,
`validate_routeB.py`. Route choice: B (exact generator) for n ≲ 700, A
(parallel finite differences) above; `--time-only` measures before you
commit. Big h5 files go outside OneDrive.

### Pipeline

| # | step | function |
|---|---|---|
| 1 | thermofield double at β, then 160 plain TDVP steps under `H_sym` | `build_uniform_thermofield`, `stepper.tdvp_step` |
| 2 | frame at the current point (one canonicalization pass) | `frame.Frame` |
| 3 | one TDVP step; frame at the new point | `tdvp_step`, `Frame` |
| 4a | Route A: FD tangent map on k columns, parallel | `stepper.tangent_map` |
| 4b | Route B: generator `H_tan + conj∘K`, polar-factor transport, `expm` | `tangent_generator.*` |
| 5 | QR, log diag R, store R / Q / frame | `benettin.benettin` |
| 6 | Ginelli backward pass → covariant vectors | `benettin.ginelli_backward` |
| 7 | energy profiles, DCT, exponent-binned q weights; figures | `analysis.*`, `plots.py`, `compare_runs.py` |

### Results of the first scan (β = 1)

Six runs: L = 8, 12, 16 at D = 4; D = 4, 8, 12 at L = 8; plus a dt/route
check. All healthy (energy to 1e-13, `s_min` ≥ 0.04, no significantly
negative exponent in any half spectrum).

- **Route- and dt-independent**: Route B dt = 0.05 and Route A dt = 0.025
  agree across the spectrum to ~0.005.
- **Extensive per tangent dimension**: L = 8 and 16 collapse on λ_i vs
  i/2n; normalize by n, not L.
- **Top ~5% of the spectrum noisy (±15%) and slow to become stationary**;
  bulk and near-zero end reliable. Two runs (L = 12, D = 12) still drifting.
- **Strong, non-monotone D dependence** (D = 8 above D = 12), not yet
  trustworthy because the D = 12 run was short.
- **No HLM signature yet.** Fast covariant vectors are short-wavelength
  (q ≈ π); as λ → 0 the long-wavelength fraction rises only to the
  flat-spectrum value, never clearly above it. The near-zero cluster
  (100–400 vectors) needs to be looked at without averaging before this
  reads as a negative result.

### Things learned the hard way

- **`retract` must canonicalize losslessly before truncating.** The block
  sum `psi + eps*Phi(X)` is not canonical, and a single truncating sweep on
  it dropped an O(eps) piece of the tangent vector (round-trip error 0.95,
  eps-independent).
- **Lanczos in `updatemethod.py` was wrong whenever a local space had
  dimension ≤ max_iters** (all bond tensors at D = 4): single-pass
  Gram–Schmidt lost orthogonality geometrically and produced a garbage
  vector at Krylov exhaustion. Fixed (second orthogonalization pass, cap at
  the space dimension). **Any earlier D ≤ 4 Lanczos result in the repo is
  suspect.**
- **Frame transport must be the polar factor** of the projection between
  tangent spaces, not the projection itself, or Route B is only first order.
- **SVD-based polar factor crashed a run** (LAPACK non-convergence at
  block 289 of 400); Newton–Schulz iterations replaced it.
- **Comparing maps composed over two steps needs the frame change between
  the two endpoint frames** — two canonicalizations of the same state are
  different frames. Forgetting this gives a spurious O(1) disagreement.
- **Write run summaries to h5 per block and flush.** The crashed run was
  only recoverable because `R` happened to be written each block.
- **Killing a queue's bash process on Windows leaves its running Python
  child alive** (tested) — safe way to replace a queue mid-run.
