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
  tdvp_lyapunov/                ACTIVE: default beta = 0.1 (see "Defaults")
    README.md                   design, Route B derivation, validation, cost, results, next
    frame.py                    tangent frame at a point; project_to_frame / retract / frame_change
    stepper.py                  one TDVP step as a map: the trajectory
    tangent_generator.py        exact generator, transport, exponential action
    benettin.py                 forward QR loop + h5 storage; Ginelli backward pass
    analysis.py                 energy-density profiles of modes, cosine transform
    hlm.py                      local-temperature templates, spectral enrichment; run_hlm.py driver
    plots.py                    per-run figures;  compare_runs.py  cross-run overlays
    run_lyapunov.py             spectrum driver (--L --D --beta --k --blocks --time-only)
    run_template.py             k=1: seed one local-temperature template and watch it
    mode_dispersion.py          per-vector wavevector content vs exponent
    long_wavelength_modes.py    free-phase longest-wavelength candidate modes
    compare_beta.py             temperature comparison across full-spectrum runs
    validate_frame.py (rung 1), validate_benettin.py (rungs 2-3)
    runs/                       logs and queue scripts only; all h5 in C:/Users/charl/lyapunov_runs
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
| 1 | imaginary-time TDVP under `H_sym` from the padded rank-1 infinite-T state (no noise seed; the evolution fills the padding itself) | `run_relaxation_scan.build_uniform_thermofield` |
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

Full scan L = 4, 8, 12, 16 at D = 8, **β = 0.1**, seed 0, ~70 s;
`scan_results.pkl` written. τ from the 1/e crossing of the response (the
exponential fits are unreliable here — see below):

| observable | L=4 | L=8 | L=12 | L=16 |
|---|---|---|---|---|
| `z_mid` | 2.021 | 2.050 | 2.131 | 2.189 |
| `x_mid` | 0.479 | 0.479 | 0.501 | 0.480 |
| `energy_mid` | 3.024 | 1.494 | 1.531 | 1.484 |
| `current_mid` | 0.903 | 0.973 | 0.974 | 0.958 |

Tangent dimensions 191 / 959 / 1727 / 2495. L = 8 onward agree to ~2% for
everything except L = 4, which is too small.

`z_mid` and `x_mid` were dropped from the default scan on 2026-09-18 — they
were generic non-conserved contrasts and carry no transport information, so
the rows above are a record rather than something the current code
reproduces. `response.single_copy_onesite` still builds them; see the
commented lines in `run_relaxation_scan.run_one`.

- **τ(current) ≈ 0.96, independent of L** — the number the current operator
  was added for. Microscopic, as a current relaxation time must be, and two
  to three orders below the `L²/D` profile time. **The wait before
  transport can look diffusive is not the obstacle.**
- **The spectral densities separate the way hydrodynamics needs.** Weight
  within `|ω| < 0.25`: energy density 0.37 (4.7x chance, with a single ω = 0
  mode carrying ~9% on its own), current 0.019 (0.24x chance, bimodal with
  peaks at ω ≈ ±1). Conserved density has the low-frequency pole, its
  current does not — the precondition for a finite Green–Kubo `D`.
- **τ(energy)/τ(current) ≈ 1.55 is NOT a hydrodynamic separation.** Both are
  *local* dephasing times at the fixed point; both are flat in L, whereas a
  hydrodynamic time grows like L². Do not read that ratio as gating
  diffusion.
- **`tau_fit` is unreliable in this regime**: 11 of 16 fits fail at
  R² < 0.9, some NaN, some an order of magnitude off the crossing
  (energy at L = 12: 30.3 at R² = 0.003 against a crossing of 1.53). The
  responses fall fast then crawl through an oscillating tail, which is not
  one exponential. Fixing `fit_relaxation_time` is open work; the scan
  figure plots crossings and overlays fits only where R² ≥ 0.9.

Figures: `figures/D8_timescale_scan.png` (across L) and
`figures/L16_D8_{current,energy}_mid_{spectrum,response}.png`.

Next: `A_j(ω→0)` is the Green–Kubo integrand, so turning it into a `D` and
comparing with the nonlinear Gaussian-width `D` from `qtensor.visualise`
needs no new machinery. Not done.

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
3. ~~**Seeding noise 1e-2** makes `ψ*` only approximately the thermofield
   double.~~ **Resolved 2026-09-18: the noise was never needed.** Single-site
   TDVP is fixed-rank, but `left_orthogonal_tensor`'s
   `la.svd(..., full_matrices=False)` keeps the zero singular values and
   fills their columns with an arbitrary orthonormal completion, so the
   environments reach every bond index and `H_eff` drives the centre tensor
   off the rank-deficient boundary. `SEED_NOISE = 0` now; the residual drops
   from ≈ 7e-2 to ≈ 1e-7 and resumes falling with D. Numbers in this file
   predating that date carry the ≈ 6% artefact.
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

### Defaults (2026-09-17)

**New runs are at β = 0.1 unless there is a reason otherwise.** Hydrodynamics
is a high-temperature expectation, so hunting it at low temperature is the
wrong place; the temperature scan below independently shows β = 0.1 is
better conditioned than β = 1 and already saturated (β = 0.01 adds nothing).
`run_lyapunov.py` and `run_template.py` carry this default. The earlier
results tables are at β = 1 and are labelled as such — they are not
superseded, but new work should not be compared to them across temperature
without checking the scan. `relaxation/` now also defaults to β = 0.1, but
the cost there is real and measured (see its README): `s_min` falls from
4e-7 to ~1e-11, because that subproject works at the imaginary-time fixed
point with no real-time transient to fill the bond dimension. **β = 0.01 is
unusable there** (`s_min` ~ 1e-15). The `tdvp_lyapunov` finding that
conditioning improves at high temperature does *not* carry across.

Carry forward the one caveat: template enrichment was *q-selective* at
β = 1 and nearly flat at β ≤ 0.1, measured on L = 8 (7 wavevectors only).
q-resolved results at the new default want L = 16.

### What runs

    python lyapunov/tdvp_lyapunov/run_lyapunov.py --L 8 --D 8 \
        --blocks 250 --transient 160 --dt 0.05 --store-Q-every 25 \
        --out-dir C:/Users/charl/lyapunov_runs
    python lyapunov/tdvp_lyapunov/run_template.py --L 16 --D 3 --blocks 80 --kmode 1 \
        --out-dir C:/Users/charl/lyapunov_runs        # k=1, ~5 min
    python lyapunov/tdvp_lyapunov/plots.py C:/Users/charl/lyapunov_runs/L8_D8_beta1.h5 --discard 20 --clv
    python lyapunov/tdvp_lyapunov/compare_runs.py
    python lyapunov/tdvp_lyapunov/run_hlm.py C:/Users/charl/lyapunov_runs/L16_D4_beta1.h5 --m 120
    python lyapunov/tdvp_lyapunov/mode_dispersion.py C:/Users/charl/lyapunov_runs/L16_D4_beta1_k2n.h5

Validation: `validate_frame.py` (frame primitives), `validate_benettin.py
[2|3]` (full-Hilbert-space and fixed-point limits). `--time-only` measures
before you commit. Big h5 files go outside OneDrive.

### Pipeline

| # | step | function |
|---|---|---|
| 1 | thermofield double at β, then 160 plain TDVP steps under `H_sym` | `build_uniform_thermofield`, `stepper.tdvp_step` |
| 2 | frame at the current point (one canonicalization pass) | `frame.Frame` |
| 3 | one TDVP step; frame at the new point | `tdvp_step`, `Frame` |
| 4 | generator `H_tan + conj∘K`, polar-factor transport, exponential action | `tangent_generator.*` |
| 5 | QR, log diag R, store R / Q / frame | `benettin.benettin` |
| 6 | Ginelli backward pass → covariant vectors | `benettin.ginelli_backward` |
| 7 | energy profiles, cosine transforms; figures | `analysis.*`, `plots.py`, `compare_runs.py` |
| 8 | template modes, spectral enrichment, candidate HLMs | `hlm.*`, `run_hlm.py` |

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
- **Hydrodynamic signature (2026-09-17), and its correction the same day.**
  The bin-averaged long-wavelength fraction found nothing (it averages over
  the whole near-zero cluster). Asking where the long-wavelength *template*
  sits does find structure: the local-temperature mode
  a_k = 2 realify(P sum_j cos(q_k j) h_j |psi>), decomposed over the
  orthonormal Gram-Schmidt basis. On the non-negative half it looked like
  1.4-1.8x enrichment in the near-zero cluster; **the k = 2n run showed
  that was an artefact of conditioning on that half.** With both halves:
  the template sits on the *contracting* directions (1.69x at q = 0,
  decaying monotonically to ~1 at q = pi) and avoids the expanding ones
  (0.44x), with the near-zero bands at chance. Its symplectic partner
  J a_k -- the local time-shift template -- mirrors this exactly (top 1.70,
  bottom 0.37), which is what confirms the asymmetry is the symplectic
  structure rather than a bias of the forward-filtration basis.
  Run with `run_hlm.py`; figures `<run>_hlm_{enrichment,candidates,candidates_neg}.png`.

### Temperature scan (2026-09-17)

L = 8, D = 8, full spectrum, beta = 1 / 0.1 / 0.01 (`compare_beta.py`).

- **Stable down to beta = 1e-2, and better conditioned there**: s_min rises
  0.096 -> 0.21, pairing residual improves 0.048 -> 0.009. The nearly
  rank-1 thermofield double at high temperature does *not* poison the
  tangent space, because the 160-step real-time transient fills the bond
  dimension first.
- **The flow saturates by beta = 0.1**: the 0.1 and 0.01 spectra coincide.
  No point going hotter at this L and D.
- **The contracting-band enrichment strengthens** (1.37 -> 1.67 at q1) but
  **loses its q-dependence**: flat at beta <= 0.1, monotone decay at
  beta = 1. Scale separation is a beta = 1 feature. Recheck at L = 16
  before leaning on it.

### Per-vector dispersion (2026-09-17)

`mode_dispersion.py`, on L16_D4_beta1_k2n (best q resolution per unit cost:
15 bonds, cheap D, full spectrum). Per vector: energy profile -> phase-free
power spectrum (projection onto {cos, sin} at each q, uniform removed) ->
centroid, width, peak; then lambda against q, coloured by width.

- **lambda is linear in q**, crossing zero at q0 = 1.4 +- 0.1 (wavelength
  ~4.5 sites). R2 0.97 linear vs 0.91 for q^2 at L=16; the ordering holds
  for every run and for both the centroid and the peak statistic.
- **Not diffusive**: the contracting branch's magnitude *decreases* with q
  (slope -0.75), and abs(lambda) overall is flat in q. All of the structure
  is in the sign.
- Slope 0.086 at beta = 1 (same at L = 8 and 16), 0.16 at beta = 0.01.
- Cost: 9 s per stored block for all 1374 vectors.
- **FLAGGED, UNRESOLVED: the `q_bar` estimator is biased and all of the
  above may be an artefact.** Calibrated on known-q profiles, the
  phase-free family is ~30x overcomplete, so it is a smoothed scan rather
  than a decomposition. True q = 0.209 and 0.419 both return q_bar ~ 0.39-0.43
  (the long-wavelength end has no resolution), and a *random* profile
  returns q_bar ~ 1.45 — which is where q0 = 1.4 sits. Do not lean on the
  linear fit or on q0 until this is redone against an orthonormal basis.
  Full table and reasoning in the subproject README.

### Seeded single template (2026-09-17)

`run_template.py`, L = 16, D = 3, beta = 0.1, k = 1, 80 blocks, ~4.5 min.
Seeds the flow with the k = 1 local-temperature template instead of
computing a spectrum.

- **The seed is clean**: 97.8% of its DCT power at q1 = 0.209.
- **The first-step growth rate is +0.024** against a top exponent of ~0.53
  — the long-wavelength temperature mode is a near-neutral direction, and
  this costs one step to measure.
- **Alignment is fast, as expected**: overlap with the template halves by
  t ~ 0.55, is 1/10 by t ~ 1.4, 0.4% by t = 4, while the instantaneous rate
  climbs to a peak +0.87 at t ~ 0.65 and settles at ~0.53. **The usable
  window is t < 0.5, i.e. ~10 steps.** Any use of this method must be
  short-window plus re-seeding, not a long run.
- The *energy profile* keeps the seeded cos(q j) shape much longer than
  the vector keeps the template — the 15-dimensional profile is a shadow of
  a 784-dimensional vector, and the two decorrelate at different rates.

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
- **Ginelli's backward pass is ill-conditioned over a full spectrum.** At
  k = 2n = 1374 the covariant vectors come back with condition number 2e17
  (columns collapsed); inside a 120-vector band they are fine. Measure with
  the Gram-Schmidt basis, which is exactly orthonormal, and use covariant
  vectors only within narrow bands.
- **A dense spectrum defeats vector-level Oseledets statements.** 1374
  exponents span [-0.29, 0.30], so the level spacing (4e-4) is far below
  the pairing accuracy (0.015); individual directions inside a cluster are
  numerically arbitrary. Band-level statements survive, per-vector ones do
  not.
- **Never form a matrix exponential you only need to apply.** `scipy.expm`
  on the 3934x3934 generator took 176 s; the scaled Taylor *action* on the
  same matrix takes 4.6 s for the same 1e-15 accuracy, and it is a quarter
  the flops per term because Q has n columns, not 2n. This was 88% of
  Route B's cost.
- **Scale the Taylor series by the 2-norm, not the 1-norm.** For these
  generators the 1-norm is 5x larger (163 vs 33), and the substep count is
  linear in it.
- **`pip install cupy` upgraded numpy to 2.x and broke numba/contourpy**
  (and would have broken `np.product` in `updatemethod`). Pin numpy 1.26.4
  and use `cupy-cuda12x==13.6.0`; CuPy 14 requires numpy >= 2. On Windows
  its CUDA DLLs need registering before `import cupy` (see `gpu.py`).
- **A consumer GPU is useless for this**: float64 is 1/64 of float32 on a
  4060, measured 1.1x versus the CPU. CuPy and `gpu.py` were removed again
  on 2026-09-17; the README keeps the Windows DLL notes in case it returns.
- **Route A (finite differences) was removed on 2026-09-17** after it had
  served as the independent check on the generator. `git log` has it.
