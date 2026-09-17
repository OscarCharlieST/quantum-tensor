# Lyapunov exponents of the TDVP flow

Status: **first production scan done (2026-09-16); hydrodynamic-mode
signal found (2026-09-17). β = 1 only.** The tangent map is the exact
generator ("Route B"), cross-validated against a finite-difference
implementation that has since been removed. Six runs cover L = 8, 12, 16 at D = 4 and D = 4, 8, 12 at
L = 8, with covariant Lyapunov vectors. Headline: the half spectrum is
extensive per tangent dimension and route/dt-independent; the
long-wavelength energy-density template is enriched ~1.4-1.8x in the
near-zero cluster and depleted ~0.4x at the top, with a monotone
wavevector dependence. See "Results" and "Hydrodynamic modes".

See [`../README.md`](../README.md) for shared background and for the
argument that forces the exponents at the `H_asym` fixed point to be exactly
zero. This strand gives up the fixed point to get genuine exponents.

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
chain of purified MPS, evolving under the **symmetric** thermofield
Hamiltonian `H_sym = H⊗I + I⊗H`, using a Benettin (QR) algorithm. Then
analyse the spectrum and its Lyapunov vectors for **hydrodynamic Lyapunov
modes (HLMs)**: long-wavelength structure in the vectors belonging to the
near-zero exponents.

TDVP on the MPS manifold is a classical Hamiltonian system with vector field

    F(ψ) = -i P_ψ H ψ

Linearizing gives `δF = -i [ P_ψ H δψ + (∂P·δψ) H ψ ]`. Under `H_sym` the
second term never vanishes, even at the uniform thermofield double: `H_sym`
does not annihilate it, and the exact evolution takes it to a
complex-temperature thermofield double, `e^{-(β/2 + 2it)H}` acting on one
copy of the maximally entangled state, whose entanglement grows. So any
starting state gives a non-trivial trajectory.

Reference point: Hallam, Morley & Green, *The Lyapunov spectrum of quantum
thermalisation*, Nat. Commun. 10, 2708 (2019). That paper works with
translation-invariant infinite MPS (iTDVP), where the tangent space
block-diagonalizes and the gauge can be fixed uniformly. None of that holds
here; see "What changes at finite size" below.

## The core problem: what is "the vector"?

Benettin needs three things: (i) a way to propagate a perturbation, (ii) an
inner product to orthonormalize perturbations against each other, and (iii)
a way to carry them from one time step to the next. The obvious
representation, a set of tensor perturbations `{dA^n}`, fails on all three:

- **Gauge redundancy.** `dA^n = A^n X`, `dA^{n+1} = -X A^{n+1}` changes no
  tensor-level quantity to zero, yet gives `δψ = 0`. These null directions
  carry no physics but are not removed by QR on raw tensors.
- **Wrong metric.** The Euclidean norm on tensors is not the Hilbert-space
  norm. Lyapunov exponents do not depend on the norm *only if* the two norms
  stay uniformly equivalent along the trajectory. Here the gauge drifts at
  every step (each SVD picks singular vectors freely), and the conversion
  factors involve inverse Schmidt values, so the two norms are not uniformly
  equivalent. Even where the exponents survive, the Gram–Schmidt *vectors*
  depend on the metric, and those vectors are what the HLM analysis looks at.
- **The gauge changes between steps.** Tensor perturbations from step `t`
  are written in a different gauge from those at `t+dt`, so adding or
  comparing them directly means nothing. This is the same mistake as the
  one in [`../WORKFLOW.md`](../WORKFLOW.md) ("The bug worth remembering").

**The resolution:** never orthonormalize, compare or store tensors. Only
Hilbert-space overlaps count, and each one is computed from a *single*
canonicalization pass at a *single* point. Everything below follows from
this.

## Plan

The design as it was written before any code. It still describes the
implementation; where things turned out differently (Route B was built,
the Lanczos early exit is not the smoothness problem it looked like, the
transport needs a polar factor), the later sections say so.

### 1. Phase space: projective, full-rank, real

- **Projective.** Work on the manifold of normalized states modulo global
  phase. `tdvp` already renormalizes at every substep. The `V_L` tangent
  basis from the relaxation strand is automatically orthogonal to both `ψ`
  and `iψ`, so it parametrizes exactly the projective tangent space.
- **Full rank.** The fixed-bond-dimension MPS manifold is a smooth manifold
  only where every bond has full Schmidt rank. At rank-deficient states (the
  zero-padded `inf_T_thermofield`, or Schmidt values down at 1e-11), the
  tangent space depends on arbitrary SVD null directions. The flow is not
  smooth there and Lyapunov exponents are not defined. See §5.
- **Real.** `P_ψ` depends on `ψ̄` as well as `ψ`, so the linearized flow
  (the `∂P` term) is only real-linear, not complex-linear. Tangent vectors
  are therefore handled as real vectors `(Re X, Im X)` of length `2n`, with
  `n = Σ_n (d·D_{n-1} − D_n)·D_n` the complex tangent dimension (959 at L=8,
  D=8).

  A useful consequence: in an orthonormal `V_L` frame, the Fubini–Study
  metric is the identity and the symplectic form is the standard one on
  `(Re X, Im X)`. Because the flow is Hamiltonian on a Kähler manifold,
  exponents come in `±λ` pairs, and there are at least two exact zeros (the
  flow direction and its energy-conjugate partner). This gives both a check
  and a halving of cost (§7).

### 2. Tangent vectors: coordinates in a moving orthonormal frame

At a point `ψ` on the trajectory, one canonicalization pass gives
`A_L, A_R, C, V_L` (reusing `canonicalize_and_build_environments`,
`build_null_space_tensor` and `build_centre_tensors` from
[`../relaxation/tangent_hamiltonian.py`](../relaxation/tangent_hamiltonian.py)).
A tangent vector is its coordinate block `{X^n}`, and

    v = Σ_n Φ_n(X^n),   Φ_n(X) = A_L…A_L (V_L^n X) A_R…A_R

The map `X → v` is an isometry, and the site blocks are mutually orthogonal.
So:

- **QR on the flattened real coordinates is exactly QR in the Hilbert-space
  metric.** Gauge directions are absent by construction.
- **The choice of frame does not matter for the exponents.** If two frames
  at the same point are related by an orthogonal matrix `O`, then
  `QR(O·Y) = (O·Q)·R` with the same `R`. The `R` diagonals set the
  exponents, so a different gauge choice at each step rotates the
  coordinates but leaves the exponents unchanged. The gauge drift that makes
  the tensor representation unusable is harmless here, as long as each
  frame is orthonormal and self-consistent (one pass per point).

Two primitives cover everything:

- **`project_to_frame(φ; frame at ψ)`** returns `X^n = ⟨Φ_n(e_i)|φ⟩` for an
  arbitrary MPS `φ`. This is `response.observable_tangent_vector` with
  `O = I`: mixed environments between `ψ` and `φ` in one sweep, costing
  `O(N·d·D³)`. It is gauge-invariant in `φ`, so `φ` can come from any
  canonicalization pass. Only the *frame* tensors must share one pass.
- **`retract(ψ, X, ε)`** returns an MPS for `ψ + εΦ(X)`, normalized and
  compressed back to the bond dimensions of `ψ`. The sum `Σ_n Φ_n(X^n)` is
  exactly a bond-`2D` MPS with block tensors `[[A_L^n, V_L^n X^n],[0, A_R^n]]`
  (boundary rows and columns chosen accordingly). SVD-compressing `ψ + εv`
  back to `D` changes the state only at `O(ε²)`, because `v` is tangent.

### 3. Transport between time steps: linearize the integrator, not the ODE

Treat one TDVP step `Φ_dt: ψ → ψ'` as a smooth map on the manifold, and
compute the Lyapunov exponents of that map (divided by `dt`). This is what
the code actually integrates, and it removes the need for an explicit
connection or frame-transport term.

The tangent map `DΦ_dt` sends `T_ψ` to `T_ψ'`. Its matrix is taken
*between* the frame at `ψ` (input coordinates) and a fresh frame at `ψ'`
(output coordinates). The rotation of the tangent space along the
trajectory, which would appear as a Christoffel-like term in a
continuous-time formulation, is absorbed automatically: output coordinates
are always measured in the frame where the vector now lives. Per §2, the
arbitrariness of that new frame does not affect `R`.

### 4. Computing the action of `DΦ_dt`

Two routes were built. **Route B, the exact generator, is what runs**; it is
derived in full below. Route A was the first implementation and is kept
only in this history:

**Route A (finite differences, removed 2026-09-17).** Each column came from
retracting along a tangent direction, taking a real TDVP step, and
projecting the result back:
`[project(Φ_dt(retract(ψ, X, ε))) − project(Φ_dt(ψ))] / ε`. It needed no
new algebra, which is why it came first, and it answered the original
design question — "do we need `∂P` explicitly?" — with *no*. Its cost was
one TDVP step per tangent vector per time step, and even parallelized over
16 workers it lost to Route B everywhere once `expm_action` landed (see
"Cost"). It served its purpose as the independent check on Route B: the two
agreed on the one-step map to `O(dt³)` and on a full spectrum to ~0.005.
The code was deleted to keep the module small; `git log` has it.

(A third option, forward-mode differentiation through `tdvp_step_r/l`, was
never attempted. It would differentiate the SVD inside
`left_orthogonal_tensor`, whose derivative blows up as `1/(s_i² − s_j²)`,
exactly at the near-degenerate spectra thermofield states have.)

### 5. Conditioning: the main numerical risk

The generator degrades as the smallest Schmidt value `s_min` → 0, because
the curvature term `K` carries `Λ^{-1}`. More
fundamentally, near rank-deficiency the manifold is strongly curved, so
large exponents may just be curvature artefacts. Plan:

- **Record `s_min` per bond along every trajectory,** alongside the
  exponents.
- **Initial state:** the noise-seeded imaginary-time state from
  `run_relaxation_scan.build_uniform_thermofield` at β ≈ 1 (full rank at
  D=8), not the β ≈ 1e-2 regime of `active.ipynb`, which is effectively
  rank 2.
- **Discard a transient.** Standard for Benettin anyway, and here it also
  lets real-time `H_sym` evolution (which increases entanglement) move the
  state away from small Schmidt values before accumulation starts.
- **Watch `K`**: if the run ever approaches rank deficiency, it is the
  first thing to blow up.

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

- **`k = n` (half the real dimension) gives all non-negative exponents** if
  `±λ` pairing holds. That pairing needs one validation run with `k = 2n`.
  HLMs sit at the *bottom* of that half, which is the slowest part of the
  spectrum to converge. There is no shortcut that skips the faster
  exponents.
- **`τ`** (steps per QR): 1 in every run so far. Larger τ saves only the
  QR, which is a small part of a step.
- **Checkpoint** `ψ`, `Q` and accumulated sums to h5 (same convention as
  `states/`) so long runs can resume.

### 8. Validation ladder (cheapest first)

1. **Frame primitives.** `project_to_frame(retract(ψ, X, ε)) = εX + O(ε²)`
   at the same point. The input-gauge-invariance test from §4.
2. **Full-Hilbert-space limit.** N=4, d=4, D=16: the manifold is all of
   projective Hilbert space, TDVP is exact linear Schrödinger evolution, and
   **every exponent must vanish**. This tests the transport and QR
   bookkeeping without any curvature.
3. **Fixed-point limit.** `H_asym` at the thermofield double. The one-step
   tangent map should equal `expm(-i H_tangent dt)` (realified), built by the
   relaxation code, and all exponents vanish. This is a cross-strand check.
4. **D = 1 (mean field).** Product-state TDVP is the classical mean-field
   dynamics of N coupled 4-level systems. An independent Benettin on those
   explicit ODEs gives reference exponents that do depend on curvature.
   This is the first test where the `∂P` physics is non-trivial.
5. **Structural checks on real runs.** `±λ` pairing, sum of all exponents
   `≈ 0`, exactly two zeros, energy conservation, plus convergence in `ε`,
   `dt`, `τ` and transient length. **D-dependence is physics, not
   convergence error**: these are exponents of the manifold, and the quantum
   dynamics itself has none.

### 9. Towards hydrodynamic Lyapunov modes

The analysis is built on the Gram–Schmidt vectors that Benettin returns for
free:

- **Spectrum shape.** `λ_i` vs `i/2n` for several N at fixed D (collapse =
  extensive chaos). Look for plateaus and steps near `λ → 0`.
- **Spatial structure of vectors, two views:**
  - *Site weights* `‖X^n‖²`: cheap. Their split between neighbouring sites
    depends on the `V_L` convention, but long-wavelength structure should
    not.
  - *Physical profiles* `δ⟨h_j⟩ = 2 Re⟨ψ|h_j|v⟩` for the energy density of
    the physical copy (and separately the auxiliary copy). These are
    gauge-invariant and one contraction per site. Cosine-transform them
    in `j`.
- **Mode templates.** In classical systems, HLMs are long-wavelength
  modulations of the zero modes. The zero modes here are the time
  translation `-i P H_sym ψ` and its symplectic partner `P H_sym ψ`. Their
  modulated versions are natural templates:

      w_q^time = -i P Σ_j cos(q j) h_j ψ      (local time shift)
      w_q^temp =    P Σ_j cos(q j) h_j ψ      (local β shift: first order of e^{-δβ(j) h_j/2})

  The second is exactly a local-temperature perturbation, which ties the
  HLMs directly to energy transport. Measure the overlap of each Lyapunov
  vector with these templates, and see whether `λ(q)` goes like `q`
  (classical HLM) or `q²` (diffusive).
- **Covariant Lyapunov vectors.** Gram–Schmidt vectors depend on ordering
  and are not covariant. Covariant vectors (Ginelli et al. 2007) are the
  cleaner object for HLMs. Store `R` every block and `Q` periodically so
  that a backward Ginelli pass can be added later without rerunning. Storage
  is about 2n·k·16 bytes per stored `Q`, e.g. roughly 30 MB at n=959, k=n.
- **Finite-size tension.** Resolving several `q = πk/N` below the
  correlation scale needs large N. Cost grows like `n² ∝ N²D⁴`
  (`k ~ n` vectors, each costing a sweep ∝ N). So **small D is the lever
  here, the opposite of the relaxation strand.** That matches the
  low-D-TDVP-as-hydrodynamics picture of Leviatan et al.
  (arXiv:1702.08894, the source of the default `tilted_ising` parameters).

## Decisions taken

1. **`H_sym`, read as a compression.** Under exact dynamics a unitary on the
   auxiliary copy cannot change the physical reduced state, so at fixed D
   the physical-observable dynamics under `H_sym` is driven by the manifold
   restriction. That is the point (see "Working hypothesis").
2. **The exact generator, and only that.** The finite-difference route was
   built first, used for the cross-check, and removed on 2026-09-17 once
   the generator was both validated against it (one-step map to `O(dt³)`,
   full spectrum to ~0.005) and faster at every size (see "Cost").
3. **Uniform β ≈ 1 thermofield double** as the starting state; β to be
   lowered later. Target regime D = 4–12, L = 8–16; develop at the bottom.
4. **Lanczos integrator**, after fixing it (see below); `exact_method()`
   for the smallest validation cases only.
5. **Ginelli from the start**: every `R` and periodic `Q` + frame are stored.
6. **Only the non-negative half** (`k = n`): the other half follows from the
   `±λ` pairing of a Hamiltonian flow. Consistent with every run so far (no
   exponent below −0.01 in any half spectrum), but the pairing has not been
   checked directly with a `k = 2n` run.
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

where `T` is the transport between the two frames. **`T` must be the polar
factor of the projection `O = P' Φ`** (its nearest orthogonal matrix), not
`O` itself: the bare projection shrinks vectors by `dt²/2 · II†II`, which
makes the scheme only first order per step; the polar factor is the
second-order-accurate parallel transport. Measured against the Route A
map at L=6, D=4: with `O`, `|M_A − M_B|` = 1.6e-2 → 3.8e-3 for
dt = 0.02 → 0.01 (`O(dt²)`); with the polar factor, 3.3e-3 → 4.2e-4
(`O(dt³)`), and the flow direction is transported covariantly
(`|M_B f − f'|` = 4.9e-6 at dt = 0.01, also `O(dt³)`). Route A's own
self-consistency `|M(dt) − M(dt/2)²|` scales as `dt³` too. The `K` term
is verified separately against a finite difference of the projector
itself (agreement `O(ε²)`, 6.9e-6 at ε = 1e-5), and dropping it makes the
discrepancy with Route A `O(dt)`.

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
- `run_lyapunov.py` — driver: `--route A|B`, `--n-jobs`, `--out-dir`;
  `--time-only` measures a step and prints a cost estimate.
- `runs/queue2.sh` — the queue that produced the 2026-09-16 scan, with its
  timestamped log `runs/queue2.log`.
- `validate_frame.py` (rung 1: frame primitives, retraction order, gauge
  invariance of a step), `validate_benettin.py [2|3]` (rungs 2-3, on the
  Route B map).

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

CuPy was installed and wired into the exponential, then removed on
2026-09-17 along with `gpu.py`. **A consumer GeForce card cripples float64
to 1/64 of its float32 throughput**, and these exponents need float64. On
the RTX 4060: a 3000³ float64 matmul took 0.25 s against 0.29 s on the CPU
(1.1x); `expm_action` at 2n = 3934 took 4.04 s against 4.60 s (1.14x); and
at 2n = 1918 the GPU was *slower*, 2.0 s against 1.3 s, transfers
dominating. A float32 path would be genuinely fast but needs an error
analysis first: the near-zero exponents are ~1e-3 of the largest and
accumulate over hundreds of QR steps.

Windows plumbing, if this is ever revisited: CuPy's pip CUDA libraries need
their directories registered *before* `import cupy`, both via
`os.add_dll_directory` and on `PATH` (NVRTC does its own LoadLibrary). Use
`cupy-cuda12x==13.6.0`; CuPy 14 requires numpy >= 2, which this repo cannot
take (`np.product` in `updatemethod.exact`).

## Results: first scan (2026-09-16, β = 1)

### Runs

| run | route | dt | blocks | tangent vectors from | h5 |
|---|---|---|---|---|---|
| L8_D4_beta1 | B | 0.05 | 288 of 400 (crashed in the since-replaced SVD transport; salvaged) | t = 2 | `runs/` |
| L12_D4_beta1 | B | 0.05 | 400 | t = 2 | `runs/` |
| L8_D8_beta1 | A ×16 | 0.05 | 250 | t = 8 | `C:\Users\charl\lyapunov_runs` |
| L8_D4_beta1_dt025 | A ×16 | 0.025 | 600 | t = 8 | same |
| L16_D4_beta1 | A ×16 | 0.05 | 300 | t = 8 | same |
| L8_D12_beta1 | A ×16 | 0.05 | 200 | t = 8 | same |

All start from the noise-seeded imaginary-time thermofield double
(`build_uniform_thermofield`, seed 0, 60 steps) and evolve under `H_sym`
with Lanczos TDVP. Per-run figures: `figures/<run>_{spectrum,
convergence, clv*}.png` (CLVs at ~60% of each run, the rest used as the
Ginelli backward transient), and `<run>_hlm_{enrichment,candidates}.png`. Cross-run: `figures/compare_*.png`
(`compare_runs.py`, window t > 11).

### Health of every run

- Energy ⟨H_sym⟩ conserved to 2–6e-13 over the whole trajectory.
- Smallest Schmidt value anywhere: 0.09 (D = 4), 0.07 (D = 8), 0.04
  (D = 12). No conditioning trouble at β = 1.
- No exponent below −0.01 in any non-negative half. The last few sit at
  −0.003 to −0.008 — the expected slow `1/T` bias of the exact zeros.

### Statistics (window t > 8)

| run | n | λ_0 | Σλ/L | Σλ/n | frac. \|λ\| < 0.02 |
|---|---|---|---|---|---|
| L8 D4 (B) | 303 | 0.325 | 3.07 | 0.081 | 0.20 |
| L8 D4 dt 0.025 (A) | 303 | 0.356 | 3.28 | 0.087 | 0.18 |
| L12 D4 (B) | 495 | 0.324 | 3.19 | 0.077 | 0.21 |
| L16 D4 (A) | 687 | 0.424 | 3.84 | 0.089 | 0.18 |
| L8 D8 (A) | 959 | 0.428 | 15.4 | 0.128 | 0.11 |
| L8 D12 (A) | 1967 | 0.411 | 28.2 | 0.115 | 0.12 |

### Findings

1. **Route and time step do not matter.** L = 8, D = 4, Route B at
   dt = 0.05 vs Route A at dt = 0.025 over the same window t ∈ [8, 16.4]:
   deciles of the spectrum agree to 0.005–0.01, Σλ 24.5 vs 25.7, and the
   sorted spectra overlay to ~0.005 everywhere except the top few
   (`compare_dt_route.png`). The top few differ by up to 0.03, which is the
   size of their window-to-window fluctuation (next point).
2. **Stationarity: the top of the spectrum is noisy; the bulk and bottom
   are not.** Estimates over successive 3-time-unit windows: the first
   window after the tangent vectors start is 15–30% low at the top in every
   run (vector alignment, plus the trajectory still settling); after that
   the mean of the top five fluctuates by ±15% with no consistent trend in
   most runs. The median exponent and the bottom 20 are stable across all
   windows. Two runs are still drifting at the end: L = 12 (Σλ/L climbs
   2.6 → 3.7 from t = 8 to 22 while its median `s_min` falls 0.19 → 0.16)
   and D = 12 (27 → 30 over its shorter run). **So λ_0 is only good to
   ~±15% in this scan; the shape of the spectrum below the top ~5% is
   reliable.**
3. **Extensive chaos, per tangent dimension.** At D = 4, L = 8 and L = 16
   collapse onto one curve of λ_i against i/2n across the entire spectrum
   (`compare_L_scan.png`); L = 12 sits ~10% below throughout, consistent
   with it being the drifting run whose average includes the early low
   windows. Σλ/n is the right normalization (0.081–0.089 for L = 8, 16);
   Σλ/L is not, because n/L grows with L as the edge staircase becomes a
   smaller fraction (37.9, 41.3, 42.9 for L = 8, 12, 16).
4. **D dependence is strong and not monotone** (`compare_D_scan.png`).
   Σλ/n: 0.08 (D = 4), 0.128 (D = 8), 0.115 (D = 12); λ_0 saturates near
   0.41–0.43 from D = 8. The fraction of near-zero exponents halves from
   D = 4 to D ≥ 8, and at D ≥ 8 the spectrum approaches zero linearly then
   drops sharply in the last ~1%. D = 12 below D = 8 may be the short,
   still-rising D = 12 run rather than physics. D dependence is the object
   of study, not a convergence error — but one more D = 12 run of D = 8
   length is needed before reading the non-monotonicity.
5. **Lyapunov vectors: fast = short wavelength; slow = unstructured, not
   (yet) long wavelength.** In every run, the covariant vectors with the
   largest exponents put their energy-profile power near q = π (staggered
   patterns), and the fraction of power in the two longest wavelengths
   rises almost monotonically as λ → 0. **But in the near-zero
   bins it rises only to the flat-spectrum value, not above it:** the
   bottom rows of the heatmaps are roughly flat in q. Near-zero values
   against the flat reference: 0.31 vs 0.29 (L8 D4 B), 0.26 vs 0.29 (L8 D4
   A), 0.17 vs 0.18 (L12), 0.11 vs 0.13 (L16), 0.27 vs 0.29 (L8 D8), 0.29
   vs 0.29 (L8 D12). The L = 8, D = 4 excess reported earlier was within
   noise. So what is established is a depletion of long wavelengths at the
   *top* of the spectrum, not an enrichment at the bottom — no evidence yet
   for HLMs. Caveats that could still hide them: the near-zero bins average
   100–400 vectors, so a handful of genuine modes would be diluted; only
   the physical-copy energy density was examined; and the "two longest
   wavelengths" cover a q range that shrinks with L. **All three caveats
   are answered by the template analysis in the next section**, which
   supersedes this bin-averaged statistic; its code (`q_weight_by_exponent`,
   `plot_q_weight`) and figures were removed on 2026-09-17.

End-to-end validation of the pipeline is finding 1 together with
"Validation results" (rungs 1–3) and "Route B" (one-step map agreement
`O(dt³)`, `K` vs finite-differenced projector `O(ε²)`).

## Hydrodynamic modes: the template analysis (2026-09-17)

Finding 5 above (bin-averaged long-wavelength fraction) asked whether a
*typical* vector in an exponent band is long-wavelength, and found nothing
above chance. That statistic is diluted by construction: it averages over
100–400 vectors in the near-zero cluster, so a handful of genuine modes
cannot move it. The sharper question — **where in the spectrum does the
long-wavelength energy mode live?** — has a clean answer, and it is
positive.

### The statistic

The energy-density profile is real-linear in the tangent vector, so the
amplitude at wavevector q_k is a linear functional whose gradient is itself
a tangent vector:

    amplitude_k(y) = sum_j c_kj delta<h_j>(y) = a_k . y,
    a_k = 2 realify( P_tangent O_k |psi> ),   O_k = sum_j cos(q_k (j+1/2)) h_j

`a_k` is exactly the **local-temperature template mode** of the design plan
(`hlm.template_vectors`, built by linearity from one projection per bond,
and verified to reproduce `cosine_transform` of the profile to 1e-17).

Decompose it over the orthonormal Gram–Schmidt vectors at a stored block
and measure the **enrichment**: the share of |a_k|² landing in a band of m
vectors, divided by m/k, the share a uniform spread would give. 1 is
chance, and the comparison is valid across bands and band sizes. The
covariant vectors are used only to *build* candidate modes (they are not
orthonormal, so they do not give a clean decomposition); the GS filtration
is used to *measure*.

### Result: yes, with a monotone q dependence

Enrichment of each template in the near-zero, mid-spectrum and top bands,
averaged over every stored block of a run (errors are s.e.m. over blocks,
i.e. over points on the trajectory). L = 16, D = 4, m = 120 of 687:

| q | near-zero | mid | top |
|---|---|---|---|
| 0 (uniform) | 1.57 ± 0.08 | 0.99 ± 0.02 | 0.53 ± 0.09 |
| 0.21 | 1.44 ± 0.07 | 0.91 ± 0.02 | 0.42 ± 0.08 |
| 0.42 | 1.40 ± 0.07 | 1.06 ± 0.02 | 0.55 ± 0.07 |
| 0.63 | 1.27 ± 0.03 | 0.96 ± 0.03 | 0.46 ± 0.05 |
| 1.26 | 1.21 ± 0.04 | 1.00 ± 0.02 | 0.70 ± 0.05 |
| 2.09 | 1.09 ± 0.03 | 1.04 ± 0.03 | 0.77 ± 0.03 |
| 2.93 (staggered) | 1.09 ± 0.04 | 1.03 ± 0.02 | 0.84 ± 0.03 |

- **The near-zero band is enriched, and the enrichment decreases
  monotonically with q**, from 1.44 at the longest wavelength to 1.09 at
  q ≈ π (a 5σ difference). The uniform template, k = 0, is the most
  enriched of all at 1.57 — as it must be, since the total energy is
  exactly conserved and its mode sits at λ = 0. That the conserved
  quantity comes out on top is a check on the statistic, and the
  long-wavelength modulations of it inherit the enrichment.
- **The top band is depleted, most strongly at long wavelength** (0.42 at
  q = 0.21, rising to 0.84 at q = π). The fastest Lyapunov modes avoid
  long-wavelength energy content.
- **The mid-spectrum band sits at 1.00 ± 0.03 at every q** — the null
  behaves exactly as it should, which is the best evidence that the ±40%
  effects above are real.

Same pattern in every run, each block-averaged over its own stored blocks.
Near-zero band enrichment at k = 0 (uniform) and k = 1 (longest
wavelength), and top-band depletion at k = 1:

| run | m | k = 0 | k = 1 | top, k = 1 |
|---|---|---|---|---|
| L = 8, D = 4, dt = 0.025 | 60 of 303 | 1.61 ± 0.08 | 1.53 ± 0.07 | 0.49 ± 0.07 |
| L = 12, D = 4 | 90 of 495 | 1.59 ± 0.08 | 1.40 ± 0.05 | 0.43 ± 0.06 |
| L = 16, D = 4 | 120 of 687 | 1.57 ± 0.08 | 1.44 ± 0.07 | 0.42 ± 0.08 |
| L = 8, D = 8 | 150 of 959 | 1.80 ± 0.14 | 1.59 ± 0.11 | 0.40 ± 0.08 |
| L = 8, D = 12 | 120 of 1967 | 1.74 ± 0.17 | 1.44 ± 0.06 | 0.35 ± 0.08 |

The effect is if anything slightly stronger at larger D, and shows no L
dependence at fixed D. It survives the dt = 0.025 / Route A run, so it is
not an artefact of either route or time step.

### Candidate modes

`figures/<run>_hlm_candidates.png` shows the best mode the near-zero
covariant span can build at each of the first three wavevectors — the
projection of `a_k` into that span. They look like hydrodynamic modes:

- profiles that track `cos(q_k j)` with **purity 0.83–0.88 (D = 4) and
  0.92–0.95 (D = 8)** of their profile power in the intended wavevector,
- **extensive**: participation ratio 14.2–14.4 of 16 sites at L = 16,
  5.6–5.8 of 8 at L = 8, so they are not edge or few-site objects,
- **λ_eff ≈ +0.008 to +0.015**, i.e. at the bottom of the spectrum, against
  λ_max ≈ 0.3–0.4.

### Caveats

- Only 23–48% of each template lies in the computed non-negative half at
  all, and that fraction *rises* with q (0.23 at q = 0 to 0.48 at q = π for
  L = 16). The long-wavelength templates put most of their weight in the
  contracting half, which has not been computed. A `k = 2n` run would close
  this, and is the natural next step.
- Enrichment is a statement about where template weight sits, not proof
  that a single vector *is* a hydrodynamic mode. The candidate modes are
  built by projection, so their purity is an upper bound on how cleanly the
  cluster represents a pure cosine.
- No `λ(q)` dispersion yet: all candidates sit at λ_eff within a factor ~2
  of each other, and the near-zero band is 100+ vectors wide, so the
  resolution needed to see λ ∝ q or q² is not there. Longer runs, or a
  narrower band, would be needed.
- β = 1 only.

## Next

In rough order of value per effort:

1. **`k = 2n` run** at L = 8, D = 4 or L = 16, D = 4, so the contracting
   half is available: it holds most of the long-wavelength template weight
   (52-77%), and it is also the direct test of the ±λ pairing. This is now
   the most valuable single run.
2. **λ(q) dispersion**: with a narrower near-zero band, or the full
   spectrum, test whether the candidate modes' λ_eff scales as q or q²
   (the classical-HLM vs diffusive distinction). Needs better resolution at
   the bottom of the spectrum than the present runs have.
3. **Auxiliary copy**: run the template analysis with `--copy aux`, and
   compare. If the enrichment is a purification artefact it should look
   different there.
4. **Longer runs**: D = 12 to the length of D = 8 (~2.5 h), and discard
   the first 3 time units after vectors start. Needed before reading the
   D dependence (finding 4) or λ_0.
5. **β scan**, starting β = 0.5 at L = 8, D = 8 — the original programme.
6. Rung 4 (D = 1 mean field) — still undone, lower priority now that the
   two routes cross-validate.

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
