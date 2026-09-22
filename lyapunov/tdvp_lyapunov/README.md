# Lyapunov exponents of the TDVP flow

Status: **first production scan done (2026-09-16); hydrodynamic-mode
signal found (2026-09-17), corrected by the `k = 2n` run the same day.**
The tangent map is the exact generator ("Route B"), cross-validated against
a since-removed finite-difference implementation. Six runs cover L = 8, 12,
16 at D = 4 and D = 4, 8, 12 at L = 8, with covariant Lyapunov vectors.

Headline: the half spectrum is extensive per tangent dimension and
route/dt-independent; the long-wavelength local-temperature template sits
on the **contracting** directions (enrichment 1.7x) and avoids the
expanding ones (0.44x), while its symplectic partner, the local time-shift
template, does the reverse. The effect dies monotonically with wavevector
at β = 1; a β scan to 1e-2 shows the machinery is better conditioned and
the split strengthens, but the wavevector selectivity is a β = 1 feature.

See [`../README.md`](../README.md) for shared background and the argument
forcing the exponents at the `H_asym` fixed point to be exactly zero. This
strand gives up the fixed point to get genuine exponents.

## Working hypothesis

The zoomed-out claim under test: the short-range correlations an MPS
manifold can hold capture the short-range correlations of generic
high-temperature thermal states efficiently, so the "artefacts" of the
manifold restriction are a pseudo-physical compression of information —
the analogue of projecting onto slow modes in Mori–Zwanzig. Lyapunov
exponents of the TDVP flow are then properties of that compression, and
hydrodynamic Lyapunov modes would be its slow, long-wavelength content.
Programme: start at β ≈ 1, then push β down and look for results sharpening
before numerical conditioning gives out.

## Goal

Compute the **full Lyapunov spectrum** of single-site TDVP on a **finite**
chain of purified MPS under `H_sym = H⊗I + I⊗H`, by Benettin QR, and
analyse the spectrum and its vectors for **hydrodynamic Lyapunov modes**:
long-wavelength structure in the vectors of the near-zero exponents.

TDVP on the MPS manifold is a classical Hamiltonian system with
`F(ψ) = −i P_ψ H ψ`, linearizing to
`δF = −i [ P_ψ H δψ + (∂P·δψ) H ψ ]`. Under `H_sym` the second term never
vanishes, even at the uniform thermofield double: the exact evolution takes
it to a complex-temperature double `e^{−(β/2 + 2it)H}`, whose entanglement
grows. So any starting state gives a non-trivial trajectory.

Reference point: Hallam, Morley & Green, *The Lyapunov spectrum of quantum
thermalisation*, Nat. Commun. 10, 2708 (2019) — translation-invariant
infinite MPS, where the tangent space block-diagonalizes and the gauge is
fixed uniformly. None of that holds here; see §6.

## The core problem: what is "the vector"?

Benettin needs to propagate perturbations, orthonormalize them, and carry
them between steps. The obvious representation — tensor perturbations
`{dA^n}` — fails on all three:

- **Gauge redundancy.** `dA^n = A^n X`, `dA^{n+1} = −X A^{n+1}` gives
  `δψ = 0` while changing the tensors; QR on raw tensors does not remove
  these null directions.
- **Wrong metric.** The Euclidean tensor norm is not the Hilbert-space
  norm, and the two are not *uniformly* equivalent along the trajectory:
  the gauge drifts at every SVD and the conversion involves inverse Schmidt
  values. Even where the exponents survive, the Gram–Schmidt *vectors*
  depend on the metric — and those are what the HLM analysis reads.
- **The gauge changes between steps**, so tensor perturbations at `t` and
  `t+dt` are not comparable. Same mistake as
  [`../WORKFLOW.md`](../WORKFLOW.md), "The bug worth remembering".

**The resolution:** never orthonormalize, compare or store tensors. Only
Hilbert-space overlaps count, each computed from a *single* canonicalization
pass at a *single* point. Everything below follows from this.

## Method

The design as written before any code, condensed. Where things turned out
differently the later sections say so.

### 1. Phase space: projective, full-rank, real

- **Projective.** `tdvp` renormalizes at every substep, and the `V_L`
  tangent basis is automatically orthogonal to both `ψ` and `iψ`, so it
  parametrizes exactly the projective tangent space.
- **Full rank.** The fixed-bond-dimension manifold is smooth only where
  every bond has full Schmidt rank. At rank-deficient states the tangent
  space depends on arbitrary SVD null directions, the flow is not smooth,
  and exponents are not defined. This is the main numerical risk (§5).
- **Real.** `P_ψ` depends on `ψ̄`, so the linearized flow is only
  real-linear. Tangent vectors are real `(Re X, Im X)` of length `2n`, with
  `n = Σ_n (d·D_{n-1} − D_n)·D_n` the complex tangent dimension (959 at
  L = 8, D = 8). In an orthonormal `V_L` frame the Fubini–Study metric is
  the identity and the symplectic form is standard, so the flow is
  Hamiltonian on a Kähler manifold: **exponents come in `±λ` pairs with at
  least two exact zeros** (the flow direction and its energy-conjugate
  partner). That is both a check and a halving of cost.

### 2. Tangent vectors: coordinates in a moving orthonormal frame

One canonicalization pass at each point gives `A_L, A_R, C, V_L` (reusing
[`../relaxation/tangent_hamiltonian.py`](../relaxation/tangent_hamiltonian.py)).
A tangent vector is its coordinate block `{X^n}`, with
`v = Σ_n Φ_n(X^n)` and `Φ_n(X) = A_L…A_L (V_L^n X) A_R…A_R`. The map
`X → v` is an isometry with mutually orthogonal site blocks, so:

- **QR on the flattened real coordinates is exactly QR in the Hilbert-space
  metric**, with gauge directions absent by construction.
- **The frame choice does not affect the exponents.** Two frames at a point
  differ by an orthogonal `O`, and `QR(O·Y) = (O·Q)·R` leaves `R`
  unchanged. The gauge drift that makes the tensor representation unusable
  is therefore harmless here, provided each frame is orthonormal and comes
  from *one* pass.

Two primitives cover everything: **`project_to_frame(φ; frame at ψ)`**
(`response.observable_tangent_vector` with `O = I`, one mixed-environment
sweep, `O(N d D³)`, gauge-invariant in `φ`), and **`retract(ψ, X, ε)`**
(`Σ_n Φ_n(X^n)` is exactly a bond-`2D` MPS with blocks
`[[A_L^n, V_L^n X^n],[0, A_R^n]]`; recompressing to `D` changes the state
only at `O(ε²)` because `v` is tangent).

### 3. Transport: linearize the integrator, not the ODE

One TDVP step `Φ_dt` is treated as a smooth map, and its tangent map is
taken *between* the frame at `ψ` and a fresh frame at `ψ'`. The rotation of
the tangent space along the trajectory — a Christoffel-like term in a
continuous-time formulation — is then absorbed automatically, since output
coordinates are always measured in the frame where the vector now lives.

### 4. Computing the action of `DΦ_dt`

**Route B, the exact generator, is what runs** (derived below). Route A
(finite differences) was the first implementation, answered the design
question "do we need `∂P` explicitly?" with *no*, served as the independent
check on Route B — agreeing on the one-step map to `O(dt³)` and on a full
spectrum to ~0.005 — and was deleted on 2026-09-17 once `expm_action` made
it slower everywhere. `git log` has it.

Forward-mode differentiation through `tdvp_step_r/l` was never attempted:
it would differentiate the SVD inside `left_orthogonal_tensor`, whose
derivative blows up as `1/(s_i² − s_j²)` — exactly at the near-degenerate
spectra thermofield states have.

### 5. Conditioning: the main numerical risk

The generator degrades as `s_min → 0`, because the curvature term `K`
carries `Λ^{-1}`; and near rank-deficiency the manifold is strongly curved,
so large exponents may be curvature artefacts. Mitigations: record `s_min`
per bond along every trajectory; discard a transient, which lets real-time
`H_sym` evolution move the state away from small Schmidt values before
accumulation starts; and watch `K`, which blows up first.

### 6. What changes at finite size (vs Hallam et al.)

| issue | infinite, translation-invariant | finite chain, here |
|---|---|---|
| tangent space | block-diagonal in momentum, dimension per unit cell | one dense space, `n ≈ (d−1)·N·D²`, grows with N |
| bond dimensions | uniform | staircase at the edges. Sites where `d·D_{n-1} = D_n` contribute no directions (already handled in the relaxation basis) |
| gauge | uniform mixed gauge, fixed once | a fresh canonicalization at every step, handled by frame invariance (§2) |
| exponent density | directly per unit cell | `λ_i` vs `i/2n`. Extensivity has to be *shown* by collapse across N |
| wavevectors | continuous `q` | open boundaries: `q = πk/N`, cosine (no-flux) modes, no `±q` degeneracy, so HLM "steps" look different from periodic-chain literature |
| boundaries | none | edge sites have their own dynamics. Check whether edge-localized Lyapunov vectors pollute the near-zero spectrum |

### 7. The Benettin loop (structure, not code)

    ψ ← initial state; run transient under Φ_dt
    Q ← random orthonormal 2n × k real matrix     (frame at ψ)
    for each renormalization block:
        frame ← canonicalize ψ once (A_L, A_R, C, V_L, envs)
        repeat τ steps:
            ψ' ← Φ_dt(ψ);  frame' ← canonicalize ψ' once
            Q  ← [DΦ_dt]_{frame→frame'} · Q        (generator + transport)
            ψ, frame ← ψ', frame'
        Q, R ← QR(Q);  accumulate log|diag R|;  store diag R (and Q periodically)
        log s_min, energy, ‖H_sym ψ‖ diagnostics

- **`k = n` gives all non-negative exponents** if `±λ` pairing holds
  (validated by the `k = 2n` run below). HLMs sit at the *bottom* of that
  half, the slowest part to converge; there is no shortcut that skips the
  faster exponents.
- **`τ`** (steps per QR) is 1 in every run so far. Larger τ saves only the
  QR, a small part of a step.

### 8. Validation ladder (cheapest first)

1. **Frame primitives** — `project_to_frame(retract(ψ, X, ε)) = εX + O(ε²)`.
2. **Full-Hilbert-space limit** (N=4, d=4, D=16): TDVP is exact linear
   Schrödinger evolution and **every exponent must vanish**, testing
   transport and QR bookkeeping without curvature.
3. **Fixed-point limit** — `H_asym` at the thermofield double: the one-step
   map should equal `expm(−i H_tangent dt)` from the relaxation code, all
   exponents zero. A cross-strand check.
4. **D = 1 (mean field)** — product-state TDVP is classical mean-field
   dynamics of N coupled 4-level systems; an independent Benettin on those
   ODEs gives reference exponents that *do* depend on curvature. Still
   undone; see "Next".
5. **Structural checks on real runs** — `±λ` pairing, `Σλ ≈ 0`, exactly two
   zeros, energy conservation, convergence in `dt`, `τ` and transient.
   **D-dependence is physics, not convergence error**: these are exponents
   of the manifold, and the quantum dynamics itself has none.

### 9. Towards hydrodynamic Lyapunov modes

The analysis runs on the Gram–Schmidt vectors Benettin returns for free:
spectrum shape (`λ_i` vs `i/2n`, collapse across N = extensive chaos), and
the spatial structure of vectors via gauge-invariant physical profiles
`δ⟨h_j⟩ = 2 Re⟨ψ|h_j|v⟩`, cosine-transformed in `j`.

**Mode templates.** HLMs are long-wavelength modulations of the zero modes,
here the time translation `−i P H_sym ψ` and its symplectic partner
`P H_sym ψ`. The modulated versions are the natural templates:

    w_q^time = -i P Σ_j cos(q j) h_j ψ      (local time shift)
    w_q^temp =    P Σ_j cos(q j) h_j ψ      (local β shift)

The second is exactly a local-temperature perturbation, tying HLMs to
energy transport. Measure each Lyapunov vector's overlap with these and see
whether `λ(q)` goes like `q` (classical HLM) or `q²` (diffusive).

**Covariant vectors.** Gram–Schmidt vectors are not covariant; storing `R`
every block and `Q` periodically lets a Ginelli backward pass be added
without rerunning (~`2n·k·16` bytes per stored `Q`, ~30 MB at n=959, k=n).

**Finite-size tension.** Resolving several `q = πk/N` needs large N, and
cost grows like `n² ∝ N²D⁴`. So **small D is the lever here, the opposite
of the relaxation strand** — matching the low-D-TDVP-as-hydrodynamics
picture of Leviatan et al. (arXiv:1702.08894, the source of the default
`tilted_ising` parameters).

## Decisions taken

1. **`H_sym`, read as a compression.** Under exact dynamics a unitary on the
   auxiliary copy cannot change the physical reduced state, so at fixed D
   the physical-observable dynamics under `H_sym` is driven by the manifold
   restriction. That is the point (see "Working hypothesis").
2. **The exact generator, and only that.** The finite-difference route was
   built first, used for the cross-check, and removed on 2026-09-17 once
   the generator was both validated against it (one-step map to `O(dt³)`,
   full spectrum to ~0.005) and faster at every size (see "Cost").
3. **β = 0.1 is the standing default** (2026-09-17; was β ≈ 1 before that).
   Hydrodynamics is a high-temperature expectation and the temperature scan
   shows β = 0.1 is *better* conditioned (`s_min` 0.21 vs 0.096,
   pairing residual five times smaller) and already saturated — β = 0.01
   buys diminished returns. `run_lyapunov.py` and `run_template.py` default to it.
   Known tension is that wavelength enrichment stronger at low temperature.  Target regime D = 4–12, L = 8–16; develop at the bottom.
4. **Lanczos integrator**, after fixing it (see below); `exact_method()`
   for the smallest validation cases only.
5. **Ginelli from the start**: every `R` and periodic `Q` + frame are stored.
6. **Calculate full spectrum, not half** While pairing implies we only need to calculate one half of spectrum, in practice convergence in small modes is hard, and without full spectrum these modes are more muddled.
7. **Large run files live outside OneDrive**, in `C:\Users\charl\lyapunov_runs`
   (`--out-dir`): several GB, rewritten every block. Logs stay in `runs/`.

## Route B: the exact tangent generator

Linearizing `F(ψ) = −i P_ψ H ψ` along a tangent vector `v`:

    DF·v = −i [ P H v + (D_v P) H ψ ]

For any embedded manifold, `D_v P` maps tangent to normal and back, and
the tangent→normal part is the second fundamental form `II(v,·)`. Since we
only need the tangent part of `DF·v`,

    P (D_v P) H ψ = II(v,·)† (1−P) H ψ

— the adjoint of `II` applied to the *normal residual* `w_N = (1−P)Hψ`. For
MPS, `II(v,u)` is the normal part of the two-defect state (second
derivative of the MPS map along `u` and `v`). In frame coordinates the
generator is therefore

    X ↦ −i ( H_tan X + conj(K X) ),    K_ij = ⟨ w_N | D²ψ[δM^(j), δM^(i)] ⟩

with `H_tan` the tangent-projected Hamiltonian from the relaxation strand
(built with the frame's own tensors) and `K` complex-symmetric; the
`conj` is the real-linear part of the flow. The two-defect state for
(site n, direction i; site m, direction j) is the basis vector `Φ_m(e_j)`
with its site-n tensor replaced by `B_n Λ_n⁻¹` (n<m) or `Λ_{n−1}⁻¹ B_n`
(n>m): the inverse bond matrices *are* the curvature, and they are why
everything degrades as `s_min → 0`. `K` costs one mixed-environment sweep
per column against `w_N` written as a bond-`6D` MPS (direct sum of `Hψ`
and `−Φ(h) − ⟨H⟩ψ`).

Realified on `y = (Re X, Im X)`:

    A = [[ Hi − Ki,    Hr − Kr ],
         [ −(Hr + Kr), Hi + Ki ]]

**One-step map.** Per time step: `ψ' = Φ_dt(ψ)` by TDVP, a fresh frame at
`ψ'`, `A' = A(ψ')`, and

    M = expm(dt/2 · A') · T · expm(dt/2 · A)

where `T` is the transport between the two frames. **`T` must be the polar factor of the projection `O = P' Φ`** (its nearest
orthogonal matrix), not `O` itself: the bare projection shrinks vectors by
`dt²/2 · II†II`, making the scheme first order per step, while the polar
factor is second-order-accurate parallel transport. Against the Route A map
at L=6, D=4: with `O`, `|M_A − M_B|` goes 1.6e-2 → 3.8e-3 for
dt = 0.02 → 0.01 (`O(dt²)`); with the polar factor, 3.3e-3 → 4.2e-4
(`O(dt³)`). The `K` term is verified separately against a finite difference
of the projector (`O(ε²)`, 6.9e-6 at ε = 1e-5), and dropping it degrades
the Route A agreement to `O(dt)`.

Cost per step: `H_tan` and `K` assembly (`n` sweeps each), one SVD and two
`expm` of a `2n × 2n` real matrix, one `(2n)² × k` product. The dense
linear algebra dominates from n ≈ 500 up.

## Code

All in this folder; run from the repo root.

- `frame.py` — `Frame(psi, D)`: one canonicalization pass → `A_L, A_R, C,
  V_L`, `index_map`, `n`. `Frame.from_tensors` rebuilds a stored frame in
  its exact gauge (re-canonicalizing a stored state would not). Primitives:
  `project_to_frame(phi, frame)` (mixed transfer matrices, `O(N d D³)`),
  `tangent_mps(frame, X, eps, include_point)` (the bond-2D block MPS),
  `retract(frame, X, eps, D)` (lossless sweep, truncating sweep, two
  staircase-clipping sweeps), `realify`/`complexify`, `schmidt_values`.
- `stepper.py` — `tdvp_step(psi, H, dt, method)` (one right+left sweep pair,
  no progress bar, returns a new mps) and the two integrator choices,
  `exact_method` / `lanczos_method(epsilon=1e-8)`. This is the trajectory.
- `tangent_generator.py` — Route B: `apply_mpo`, `normal_residual` (`w_N`),
  `assemble_K`, `assemble_H_tan`, `realify_generator`, `generator(frame, H)`,
  `parallel_transport` (polar factor by two Newton–Schulz iterations),
  `operator_norm` (power iteration), `expm_action` (scaled Taylor action of
  the exponential — the reason Route B is fast), `propagate`;
  `half_step_propagator` and `step_matrix` are kept for validation only.
- `benettin.py` — `benettin(...)` forward loop with `positive_qr`, h5 storage (`R` every block, lzf-compressed; per-block
  `log_diag_R`, `t`, `s_min`, energy and `blocks_done` written as they
  come, so a crashed run is still readable; `Q` + frame at
  `store_Q_blocks`), `running_exponents`, `ginelli_backward(path,
  want_blocks, discard_last)` → CLVs in their stored frames.
- `analysis.py` — `energy_density_mpo(site, 'phys'|'aux')`, `local_profile`
  (`2 Re⟨ψ|O_j|Φ(X)⟩` for every `j` in `O(N)`), `energy_profile`,
  `site_weight_profile`, `cosine_transform` (DCT-II, `q = πk/N_bonds`),
  `mode_report`, `pairing_residual`.
- `plots.py` — `plot_spectrum` (sorted `λ_i` vs `i/2n` + histogram),
  `plot_convergence`, `plot_pairing`, `plot_mode` (site weights, energy
  profile, DCT), `load_run`/`exponents_from` (handle partial runs). `python lyapunov/tdvp_lyapunov/plots.py <file>.h5
  --discard 20 --clv --modes 0 1 -1` writes `figures/`.
- `compare_runs.py` — overlay figures across runs over a common time
  window: `figures/compare_{L_scan,D_scan,dt_route}.png`.
- `run_lyapunov.py` — spectrum driver: `--L --D --beta --k --blocks
  --transient --out-dir`; `--time-only` measures a step and prints a cost
  estimate. `--beta` defaults to 0.1.
- `run_template.py` — the cheap alternative to a spectrum: seed the tangent
  flow with a *single* local-temperature template `a_q` (`--kmode`) and
  watch it, `k = 1`. The exponential action then costs nothing, so the
  price is the generator plus transport, which are `k`-independent —
  minutes rather than hours. With no QR holding it off the leading
  direction it aligns with the top exponent at rate `λ_max − λ`; the three
  panels of `figures/<run>.png` measure how long the usable window is. This
  is the linearized analogue of the nonlinear transport measurement in
  `qtensor.visualise` (`near_thermal` → profiles → Gaussian widths → `D`):
  same perturbation, differentiated once about the trajectory, so diffusion
  appears as `λ(q) ∝ −q²` rather than `σ² ∝ 2Dt`.
- `runs/queue2.sh` — the queue that produced the 2026-09-16 scan, with its
  timestamped log `runs/queue2.log`.
- `validate_frame.py` (rung 1: frame primitives, retraction order, gauge
  invariance of a step), `validate_benettin.py [2|3]` (rungs 2-3, on the
  Route B map), `validate_tangent_swap.py` (rung 4: the physical/ancilla
  swap involution on the tangent space), `validate_swap_flow.py`
  (rung 5: whether the sectors survive a run — and they do not at D ≥ 8,
  where the state itself leaves the symmetric sector during the transient).
  See [`SWAP_SYMMETRY.md`](SWAP_SYMMETRY.md) for both, and for what breaks
  them.

### Lanczos fix in `qtensor/simulation/updatemethod.py`

Needed here, but a genuine pre-existing bug. Whenever a local space had
dimension ≤ `max_iters` (every `(4,4)` bond tensor and the edge centres at
D = 4), single-pass Gram–Schmidt lost orthogonality geometrically along the
iteration (2e-16 → 5e-4 by iteration 15) and, as the Krylov space was
exhausted, produced a garbage basis vector of norm ~9. Local updates were
off by 1e-1 and the step map depended on the input gauge at 1e-2. Fix: a
second Gram–Schmidt pass in `lanczos_loop`/`lanczos_loop_bond`, and
`max_iters = min(max_iters, dim)`. Afterwards every local update matches
`expm` to 1e-16 and the step is gauge-invariant to 1e-15. Any earlier D ≤ 4
Lanczos run in this repo was affected.

## Validation results (2026-09-16)

| rung | check | result |
|---|---|---|
| 1 | `project(Φ(X)) = X` | 1.7e-15 |
| 1 | `project(retract(X, ε))/ε − X` | 4e-3, 4e-5, 4e-7, 5e-9 at ε = 1e-3 … 1e-6: clean `O(ε²)` |
| 1 | step gauge-invariance, exact / Lanczos | 4e-16 / 1e-15 |
| 1 | FD column ε-plateau (L=6, D=4, dt=0.05) | flat to 5e-7 over ε ∈ [1e-5, 1e-7] |
| 3 | `H_asym` fixed point (L=6, D=4, β=1, dt=0.02): FD map vs `expm(−i H_tangent dt)` from the relaxation code, in the same frame | 2.7e-4 relative (identity is 5e-2 away); generator spectra agree to 2.9e-4 on bandwidth 5.3; singular values within 1e-3 of 1. All consistent with `dt × ‖H_asym ψ‖ = 0.02 × 1.8e-2` |
| 2 | full Hilbert space (N=4, D=16, n=255): one-step map | singular values all 1.00000000, `|eigenvalues| − 1` ≤ 1.2e-10 |
| 2 | Benettin, 3 blocks, k = 2n = 510 | `max|λ|` = 1.5e-9, at the FD floor `ε²/dt` = 2e-9 |

## Cost (measured, seconds per time step, k = n)

On the development laptop (Intel Core Ultra 7 155H: 6 performance + 10
efficiency cores, 15.5 GB; RTX 4060 laptop GPU).

| L | D | n | before `expm_action` | **now** | removed Route A, 16 workers |
|---|---|---|---|---|---|
| 8 | 4 | 303 | 2.7 | **1.4** | 4.4 |
| 16 | 4 | 687 | 24 | **6.6** | 14.9 |
| 8 | 8 | 959 | 75 | **7.3** | 11.8 |
| 8 | 12 | 1967 | 609 | **28.9** | 27 |

The gain is one change: **never form the matrix exponential.** The Benettin
loop only needs `expm(dt/2 A) Q`, and
`tangent_generator.expm_action` computes that by a scaled Taylor series —
each term is one `(2n, 2n) @ (2n, k)` product, a quarter the flops of a
`(2n)^3` step at `k = n`, and with `‖A dt/2‖ ≈ 0.8` only ~9 terms are
needed. Measured at 2n = 3934: `scipy.linalg.expm` + apply takes 176 s,
`expm_action` 4.6 s (**38x**), agreeing to 8e-16 relative, and the whole
step map agrees with the old explicit-`expm` path to 5e-15.

The substep count uses a power-iteration estimate of `‖A‖_2`, not the
1-norm: for these generators the 1-norm overestimates by ~5x (163 vs 33)
and every factor there is a factor in cost.

Remaining per-step cost at L = 8, D = 12: generator 15.2 s (`H_tan` 7.5 +
`K` 9.4), transport 5.1 s, exponential action 8.6 s, QR ~1 s. The generator
assembly is now the target if this needs to get faster again.

### The GPU: tried and discarded

CuPy was wired into the exponential and removed on 2026-09-17 with
`gpu.py`. **A consumer GeForce card cripples float64 to 1/64 of its float32
throughput**, and these exponents need float64. On the RTX 4060 a 3000³
float64 matmul took 0.25 s against 0.29 s on the CPU, `expm_action` at
2n = 3934 took 4.04 s against 4.60 s, and at 2n = 1918 the GPU was *slower*
(2.0 s against 1.3 s), transfers dominating. A float32 path would be
genuinely fast but needs an error analysis first: the near-zero exponents
are ~1e-3 of the largest and accumulate over hundreds of QR steps. If ever
revisited, note CuPy 14 requires numpy >= 2, which this repo cannot take
(`np.product` in `updatemethod.exact`); pin `cupy-cuda12x==13.6.0` and
register its DLL directories before `import cupy`.

## Results: L and D scan (2026-09-21, β = 0.1, k = 2n)

Five runs under `H_sym`, all with `SEED_NOISE = 0`: 250 blocks at
dt = 0.05, a 240-step imaginary-time build, tangent vectors switched on at
t = 8 (160 transient steps), both halves of the spectrum. h5 files in
`C:\Users\charl\lyapunov_runs`; per-run figures
`figures/<run>_{spectrum,convergence,pairing}.png`, cross-run
`figures/compare_{L,D}_scan_beta0.1_k2n.png` (`compare_runs.py`, window
t > 11).

| run | n | λ_max | Σλ | max pairing | drift (last 25%) | Σλ⁺ |
|---|---|---|---|---|---|---|
| L8_D4 | 303 | +0.575 | −3.6e-3 | 1.5e-2 | 0.029 | 62.6 |
| L12_D4 | 495 | +0.710 | −2.3e-2 | 2.0e-2 | 0.048 | 113.6 |
| L16_D4 | 687 | +0.638 | −8.7e-2 | 1.5e-2 | 0.026 | 157.6 |
| L8_D8_ns | 959 | +0.615 | −1.1e-2 | 1.5e-2 | 0.029 | 218.9 |
| L8_D12 | 1967 | +0.522 | −3.1e-2 | 7.1e-3 | 0.014 | 393.6 |

Max pairing is `max |λ_i + λ_{2n+1−i}|`, drift the largest change in any
running exponent over the last quarter of the run, Σλ⁺ the sum of the
positive exponents.

`_ns` = no seeding noise. All five runs are noiseless; the tag exists only
to stop this D = 8 rerun overwriting the noise-seeded 2026-09-17 file of
the same name, which gives λ_max = 0.529 against 0.615 here. That file also
used 60 imaginary-time steps against 240, so the two builds differ in more
than the noise.

### Finding

**The D scan collapses.** At L = 8, D = 4, 8 and 12 lie on one curve of
`λ_i` against `i/2n` across the whole spectrum
(`compare_D_scan_beta0.1_k2n.png`), with D = 8 slightly *above* the other
two rather than between them — the ordering of a convergence wobble, not of
a D trend. This supersedes the earlier β = 1 reading of a strong,
non-monotone D dependence. Three things changed at once (β = 1 → 0.1, the
seeding noise, and a D = 12 run now as long as the others), so it does not
say which of them carried that reading.

## Hydrodynamic modes: the template analysis (2026-09-17, k = 2n)

`L16_D4_beta1_k2n` — L = 16, D = 4, k = 2n = 1374, 250 blocks, 39 min,
2.1 GB. Both halves of the spectrm. Trying to cheat by only solving for positive exponents lead to spurious enrichment near zero exponent.







**The statistic.** The energy-density profile is real-linear in the tangent
vector, so the amplitude at wavevector `q_k` is a linear functional whose
gradient is itself a tangent vector:

    amplitude_k(y) = sum_j c_kj delta<h_j>(y) = a_k . y,
    a_k = 2 realify( P_tangent O_k |psi> ),   O_k = sum_j cos(q_k (j+1/2)) h_j

`a_k` is the local-temperature template of §9 (`hlm.template_vectors`, built
by linearity from one projection per bond, verified to reproduce
`cosine_transform` of the profile to 1e-17). Decompose it over the
orthonormal Gram–Schmidt vectors at a stored block and measure the
**enrichment**: the share of `|a_k|²` landing in a band of m vectors,
divided by `m/k`, the share a uniform spread would give. 1 is chance, and
the comparison is valid across bands and band sizes. Covariant vectors are
used only to *build* candidate modes — they are not orthonormal, so they
give no clean decomposition — while the GS filtration is used to w information.

**Spectrum-level pairing holds.** `Σλ = −0.139`, which is 0.14% of `Σ|λ|`;
the residual `λ_i + λ_{2n+1−i}` has rms 0.0014 and max 0.015 (5% of
λ_max). The mean level spacing is 4.3e-4, so the pairing is good to a few
level spacings — as expected from a finite averaging time.

**Vector-level conjugacy is not resolvable here, by two independent
limits.** The spectrum is dense (1374 exponents in [−0.29, +0.30]), sual Oseledets directions are near-degenerate and numerically
arbitrary within a cluster; and the Ginelli backward pass loses column
independence at this size (the CLV matrix has condition number 2e17,
against a well-conditioned ~1e1 for a 120-vector band). So the symplectic
Gram concentrates only 0.067 of `|ω|²` on conjugate pairs against 0.044 for
chance. **Consequence for analysis: measure with the Gram–Schmidt basis,
which is exactly orthonormal; use covariant vectors only inside narrow
bands.**

**The corrected result.** Enrichment of the local-temperature template,
averaged over 11 blocks, m = 120 of 1374:

| q | near-zero + | near-zero − | mid + | mid − | top | bottom |
|---|---|---|---|---|---|---|
| 0 (uniform) | 0.77 | 1.01 | 0.66 | 1.31 | **0.44** | **1.69** |
| 0.21 | 1.12 | 1.04 | 0.66 | 1.11 | **0.44** | **1.40** |
| 0.42 | 0.80 | 1.10 | 0.80 | 1.17 | **0.44** | **1.63** |
| 0.63 | 0.88 | 1.13 | 0.80 | 1.15 | 0.59 | 1.57 |
| 1.88 | 0.95 | 1.08 | 0.95 | 1.09 | 0.68 | 1.14 |
| 2.93 | 1.00 | 1.09 | 1.02 | 1.02 | 0.81 | 0.97 |

- **The long-wavelength temperature template lives on the *contracting*
  directions**, not on the near-zero ones: bottom band 1.69 at q = 0
  falling monotonically to ~1.0 at q = π, top band 0.44 rising to 0.81.
  Both near-zero bands sit at chance

 does not.

**The asymmetry is the symplectic structure, not a basis artefact.** The
Gram–Schmidt basis is the forward Oseledets filtration and so is not
time-symmetric, which could in principle manufacture a top/bottom
asymmetry. The test: `J a_k`, the symplectic partner of the temperature
template, is precisely the **local time-shift** template `−i P Σ_j cos(q_k j) h_j ψ`
of the design plan. Its enrichment mirrors:

| template | top | bottom |
|---|---|---|
| local temperature `a_k` (q=0) | 0.44 | 1.69 |
| local time shift `J a_k` (q=0) | **1.70** | **0.37** |

So the two conjugate perturbations split cleanly between the two halves of
the spectrum: **a local temperature perturbation projects onto decaying
directions, its conjugate phase/time-shift perturbation onto growing ones**,
and the split is strongest at long wavelength, weakening monotonically to
nothing at q = π. That is a relaxation statement with a wavevector
dependence, which is the object to take to a diffusive-scaling test.

**Candidate modes, contracting side** (`figures/L16_D4_beta1_k2n_hlm_candidates_neg.png`,
against `_candidates.png` for the expanding side). The near-zero− band
gives *cleaner* modes than the near-zero+ band: purity 0.92–0.94 against
0.83–0.88, extent 14.1–14.4 of 16 sites, λ_eff ≈ −0.0088 at every q. Their
profiles track `cos(q_k j)` closely.

### Temperature scan (2026-09-17): β = 1, 0.1, 0.01 at L = 8, D = 8

Three full-spectrum runs, `L8_D8_beta{1,0.1,0.01}_k2n`, 250 blocks each,
~52 min each. Figures `compare_beta_{spectra,enrichment,modes}.png`
(`compare_beta.py`). The question was whether the programme survives down
to β ≈ 1e-2, where the thermofield double is nearly rank-1.

| β | λ_max | Σλ | pairing residual | s_min | bottom enrich. at q₁ | mode purity | λ_eff | extent |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.436 | −0.007 | 0.048 | 9.6e-2 | 1.37 | 0.96 | −0.0140 | 5.9 |
| 0.1 | 0.533 | −0.010 | 0.0093 | 2.1e-1 | 1.67 | 0.81 | −0.0297 | 5.8 |
| 0.01 | 0.542 | −0.013 | 0.0112 | 2.1e-1 | 1.66 | 0.90 | −0.0268 | 5.7 |

**1. It is stable, and better conditioned at high temperature — the
opposite of the worry.** The smallest Schmidt value *rises* from 0.096 at
β = 1 to 0.21 at β ≤ 0.1, and the ±λ pairing residual improves five-fold
(0.048 → 0.009). `Σλ` stays within 0.013 of zero. The concern that the
nearly-rank-1 β = 1e-2 thermofield double would leave the tangent space
built on numerically null directions does not materialize: the evolution
fills the bond dimension before the tangent vectors oingt
2 higher temperatures.

**3. The temperature/time-shift split survives and strengthens.** The
contracting-band enrichment of the local-temperature template at the
longest wavelength goes 1.37 → 1.67 → 1.66, and the expanding-band
depletion is ~0.22–0.27 at q = 0 at every β. The effect is not a
low3temperature artefact.

**4. But the *wavevector selectivity* is a β = 1 feature.** At β = 1 the
enrichment falls monotonically with q, 1.28 at q = 0 to 0.88 at q = π — long
wavelengths are singled out. At β ≤ 0.1 it is large but nearly flat,
1.85 falling only to ~1.5, i.e. **every** wavelength of energy-density
perturbation aligns with the contracting directions about equally. A
hydrodynamic mode needs long wavelengths to be *distinguished*; that scale
separation is present at β = 1 and largely gone at high temperature.
Caveat: L = 8 resolves only 7 wavevectors, so this should be rechecked at
L = 16 before being leaned on — the β = 1, L = 16 run does show the 4lean
monotone decay.

**5. Candidate modes stay clean at every temperature** (`compare_beta_modes.png`):
purity 0.81–0.96, extent 5.7–5.9 of 8 sites, profiles tracking
`cos(q₁ j)`. Their λ_eff roughly doubles with temperature (−0.014 →
−0.027), tracking the overall speed-up of the flow rather than anything
specific to the mode.

**Consequence for the diffusive-scaling test:** run it at β = 1, where the
q-dependence exists, and preferably at L = 16 for wavevector resolution.
High temperature gives a stronger but scale-free signal.

### Per-vector dispersion (2026-09-17): lambda is linear in q, not diffusive

`mode_dispersion.py`. For every Lyapunov vector: its energy-density
profile, then a **phase-free power spectrum** over a fine q grid -- at each
q the power is the projection onto the 2D family {cos(q(j+1/2)),
sin(q(j+1/2))} with the uniform component removed, so a phase-shifted wave
lands at one q instead of being split between DCT bins. From that, the
centroid `q_bar`, its width `sd`, and the peak `q_peak`. No bands, no
templates, and measured in the Gram-Schmidt basis, which is exactly
orthonormal. 1374 vectors take 9 s per block.

**The structure is entirely in the sign of lambda.** At L = 16, D = 4,
beta = 1, pooling three blocks (4122 vectors):

| q_bar bin | vectors | median abs(lambda) | mean lambda |
|---|---|---|---|
| 0.5-1.0 | 443 | 0.055 | **-0.054** |
| 1.0-1.5 | 963 | 0.053 | -0.002 |
| 1.5-2.0 | 557 | 0.051 | **+0.025** |
| 2.0-2.5 | 97 | 0.060 | **+0.070** |

`abs(lambda)` is flat in q (log-log slope -0.03), but the *signed* mean
rises monotonically and crosses zero near q_bar = 1.37. Long-wavelength
energy profiles belong to contracting vectors, short-wavelength ones to
expanding vectors -- the per-vector form of the template asymmetry, with no
band choice involved.

**The relation is linear in q, and that is not diffusive.** Weighted fits
to the binned means:

| run | lambda vs q (R², chi2/dof) | lambda vs q² (R², chi2/dof) | slope | zero at |
|---|---|---|---|---|
| L=16, D=4, beta=1 | **0.966, 1.6** | 0.907, 4.0 | +0.0867 | q = 1.37 |
| L=8, D=8, beta=1 | 0.624, 2.3 | 0.434, 3.8 | +0.0857 | q = 1.49 |
| L=8, D=8, beta=0.01 | **0.925, 1.4** | 0.896, 2.5 | +0.1608 | q = 1.44 |

A diffusive branch would need `abs(lambda) ~ q^2`; instead the contracting
branch's magnitude *decreases* with q (slope -0.75) while the expanding
branch grows roughly linearly (+1.10). The slope of the signed relation is
the same at L = 8 and L = 16 for beta = 1 (0.086) and roughly doubles at
beta = 0.01, while the crossing sits at q0 = 1.4 +- 0.1 in all three --
a wavelength of about 4.5 sites.

Robustness: using the peak instead of the centroid, which spans the full
zone (0.13-3.00) rather than the centroid's compressed 0.49-2.46, linear
still beats q² (R² 0.896 vs 0.680), and restricting to the sharpest
quarter of vectors changes nothing (0.949 vs 0.887). The L = 8, beta = 1
fit is the poor one (R² 0.62) -- 7 bonds is not much resolution.

**Caveats.** `q_bar` is the centroid of a broad distribution (median width
0.83 out of a zone of pi), so individual vectors do not have a sharp
wavevector; this is a statistical statement over thousands of them.
Individual vectors inside a near-degenerate cluster are numerically
arbitrary, which is fine for a statistic pooled over the whole spectrum but
not for reading any one vector.

> **Flagged 2026-09-17, not yet resolved: the `q_bar` estimator is biased,
> and this conclusion may be an artefact of it.** Calibrating
> `phase_free_power` + `spectral_moments` on profiles of *known* wavevector
> at 15 bonds gives:
>
> | true q | 0.209 | 0.419 | 0.628 | 1.047 | 1.885 | 2.932 | random |
> |---|---|---|---|---|---|---|---|
> | `q_bar` returned | 0.385 | 0.429 | 0.784 | 1.167 | 1.970 | 2.949 | ~1.45 |
> | `sd` returned | 0.49 | 0.40 | 0.45 | 0.39 | 0.30 | 0.13 | ~0.9 |
>
> Two consequences. **(i)** The two longest wavelengths — exactly where a
> diffusive `lambda ~ q^2` would have to be tested — are compressed into a
> band 11% wide, so the low-q end of the dispersion plot has almost no
> resolution and any curvature there would be flattened into the line.
> **(ii)** A structureless profile returns `q_bar ~ 1.45`, which is where
> the fitted crossing `q0 = 1.4 +- 0.1` sits, and the observed median `sd`
> of 0.83 is in the random range. So "q0" may be measuring where a vector
> with no wavevector lands rather than a physical scale — which would also
> explain why it is the same at every L and beta (finding 2 of "Next").
>
> Cause: the `{cos, sin}` family is evaluated on a 201-point grid inside a
> 14-dimensional profile space, so it is ~30x overcomplete — a unit profile
> has total power 32.6 summed over the grid. It is a smoothed, overlap-
> weighted scan, not a decomposition, and the centroid of a smoothed scan is
> pulled towards the middle of the zone. The DCT does not have this problem
> (it is orthonormal, and gives 0.978 purity on the same k=1 template that
> `phase_free_power` calls `q_bar = 0.478, sd = 0.65`), but it has no phase
> freedom, which is why it was replaced. A correct phase-free estimator
> needs an orthonormal basis, not an overcomplete scan.
>
> Nothing above has been rerun; the linear fit and the claim of no
> diffusive branch should be treated as unverified until it is.

## Next

0. **Housekeeping: the β ≠ 0.1 run data is to be deleted** (16.2 GB of the
   20.0 GB in `C:\Users\charl\lyapunov_runs`). Blocked on recording the
   decisions that came out of the β = 1 runs — Charlie's call how. The
   inventory is in `../WORKFLOW.md`, "Housekeeping". The temperature scan
   and the `q0 = 1.4` result below both cite β = 1 and β = 0.01 numbers and
   become unreproducible without a re-run once the files go.
1. **Why linear?** The dispersion `lambda ~ A(q - q0)` is the open
   question. The mean exponent used so far is dominated by the bulk of the
   spectrum; the sharper probe is the time-resolved weighted decay
   `C(t) = sum_i w_i(q) exp(lambda_i t)`, whose long-time behaviour is set
   by the slowest weighted modes. That is also the estimator a diffusive
   rate would actually live in, so it is worth building before concluding
   there is no diffusion anywhere in this system. **In progress via
   `run_template.py`**: seeding the flow with `a_q` and watching
   `‖δψ(t)‖` measures that decay directly, without the spectrum. The
   expected failure mode is alignment with the top direction — the run is
   designed to measure when that happens, not to avoid it.
2. **What sets q0 = 1.4?** It is the same (to ~0.1) at L = 8 and 16 and at
   beta = 1 and 0.01, so it is not a finite-size or temperature scale. A
   D scan would say whether it is set by the bond dimension -- i.e. by how
   much correlation the manifold can hold -- which is the interesting
   possibility for the compression picture.
3. **Auxiliary copy**: rerun the profile analyses with `--copy aux`. The
   physical and auxiliary energies are separately conserved by the exact
   dynamics but not by the manifold flow, so comparing them isolates what
   the purification is doing.
4. **L = 16, D = 6, beta = 0.1, k = 2n** (requested, not started): ~4 h,
   ~10 GB. Would test the high-temperature loss of q-selectivity and the
   D-dependence of q0 at the best available wavevector resolution. Retuned
   from beta = 0.01 to the new default; the scan showed 0.1 and 0.01 give
   the same spectrum, so there is no reason to pay for the colder one.
5. Rung 4 (D = 1 mean field) — still undone, low priority now that the
   generator is validated three other ways. See item 6: the by-hand tangent
   basis wanted there is the same object.
6. **Noiseless seeding, and an exact tangent basis at infinite temperature**
   (idea, not started). `relaxation/` dropped its `noise = 1e-2` seed on
   2026-09-18 and its fixed-point residual fell from 7e-2 to 1e-7; the seed
   turned out to be unnecessary, because `left_orthogonal_tensor` keeps the
   zero singular values and fills their columns with an arbitrary
   orthonormal completion, so the evolution walks off the rank-deficient
   boundary by itself. Every run in this file predating 2026-09-18 used the
   seed; the 2026-09-21 scan does not. Two versions of the idea, and they
   are not equally promising.

   *The cheap version — just set `noise = 0` — probably changes little
   here, and the reason is worth recording.* The blocker in this subproject
   is conditioning, not rank: `tangent_generator.py` inverts the bond
   matrices explicitly (`Lam_inv = {s: la.inv(L) for s, L in
   frame.Lam.items()}`), so a near-rank-deficient point is unusable.
   Measured right after the imaginary-time build at L = 16, D = 12,
   `s_min/s_max` is 8.4e-10 noiseless against 7.5e-10 with the seed — the
   same to within nothing. The noise never conditioned the state; what does
   is the real-time `H_sym` evolution before the tangent vectors are
   switched on (`s_min` reaches 0.1–0.2 by then, see the temperature scan).
   So dropping it costs nothing and is the right default, but the gain that
   `relaxation/` saw came from starting the *tangent analysis* at the built
   state, which this subproject does not do. Measure, do not assume.

   *The interesting version is to start at infinite temperature itself.*
   There the thermofield double is an exact product of Bell pairs —
   `(|00> + |11>)/sqrt(2)` per doubled site, rank 1, no imaginary-time build
   and so no build error at all. The generic null-space construction is
   ill-defined at a rank-deficient point, which looks like a blocker, but
   the state is simple enough that `V_L` can be written down **by hand**:
   each site carries one known normalized vector in C^4, and its null space
   is the orthogonal complement, a 4x3 isometry available in closed form.
   That is *easier* than the SVD route, not harder, and it gives an exactly
   known starting frame.

   Two caveats before anyone builds it. Without padding, the by-hand frame
   is the tangent space of the **D = 1 manifold** — i.e. it is exactly rung
   4 above, reached from the other direction. With padding to bond dimension
   D it is the singular case: at a rank-deficient point the manifold has a
   boundary and the tangent cone is not a vector space, so the +/-lambda
   pairing argument (which assumes a symplectic tangent *space*) may not
   hold until the state has moved into the interior. That makes the pairing
   residual the diagnostic to watch, and it may be the more interesting
   measurement of the two — an exactly-known starting point is precisely
   where a violation would be attributable.

Not worth pursuing without a fix: covariant vectors over the *full*
spectrum, where the Ginelli backward pass is ill-conditioned (see the k = 2n
section). Narrow bands are fine.

## Open questions carried over

- **Auxiliary-gauge directions.** Rotations acting only on the auxiliary
  copy change `ψ` but not the physical state, so some Lyapunov vectors may
  be physically invisible (zero physical profile). Classifying vectors by
  their physical-copy footprint may be more useful than excluding them —
  and may be what fills the near-zero cluster.
- **Relation to the relaxation strand.** At the `H_asym` fixed point the
  Jacobian is the relaxation strand's `-i H_tangent` (validation 3). Whether
  the `H_sym` exponents and those relaxation rates describe related physics
  is still open.
