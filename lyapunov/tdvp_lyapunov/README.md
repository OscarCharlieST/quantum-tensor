# Lyapunov exponents of the TDVP flow

Status: **first production scan done (2026-09-16), β = 1 only.** Both
routes to the tangent map are implemented and validated against each other
(Route A: parallel finite differences through the TDVP step; Route B: exact
generator). Six runs cover L = 8, 12, 16 at D = 4 and D = 4, 8, 12 at
L = 8, with covariant Lyapunov vectors. Headline: the half spectrum is
extensive per tangent dimension and route/dt-independent; fast Lyapunov
vectors are short-wavelength; the near-zero cluster is *not* yet
distinguishable from spatially unstructured. See "Results" and "Next".

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

### 4. Computing the action of `DΦ_dt`: two routes

**Route A: central finite differences through unmodified `tdvp` (do this first).**
For each real direction `Y_k` (complex `X_k`):

    ψ_± = retract(ψ, ±X_k, ε)
    ψ'_± = Φ_dt(ψ_±)                          # existing tdvp, untouched
    column_k = [project_to_frame(ψ'_+) − project_to_frame(ψ'_−)] / 2ε   (frame at ψ' = Φ_dt(ψ))

Error is `O(ε²)` plus roundoff `~1e-16/ε`. Any global phase difference
between `ψ'_±` and `ψ'` drops out, because the frame is orthogonal to `ψ'`.

- For: this answers the original "do we need `∂P` explicitly?" question
  (no), requires no new algebra, and the `2n` clone steps are embarrassingly
  parallel.
- Against: nonlinearity sets an upper limit on `ε` of roughly
  `ε ≪ s_min · (stuff)`, because manifold curvature scales like the inverse
  smallest Schmidt value. Roundoff sets a lower limit, and on
  poorly-conditioned states these limits can collide (§5).
- Requirements on the integrator. Finite differencing only works if
  `Φ_dt` is smooth in `ψ`:
  - `lanczos_parts` stops early on `norm < epsilon`, which makes the map
    *discontinuous* wherever the iteration count changes. Use a fixed
    Krylov dimension (no early exit), or `exact_method()` at small sizes.
  - Check once that `Φ_dt(ψ)` is independent of the input gauge: apply a
    random gauge transformation, compare the output overlap. `tdvp`
    re-canonicalizes on entry, and the projector-splitting step should
    depend only on the subspaces, not on their representation. This is
    expected to hold at full rank but has not been checked.
  - `tdvp`'s entry overhead (full right-canonicalization, rebuilding `R_con`,
    progress bar) is wasted when called for one step at a time from `2n`
    clones. A thin single-step wrapper is a likely early addition.

**Route B: exact tangent map (only if Route A is limited by `ε` or cost).**
Assemble the real `2n × 2n` generator in the frame at `ψ`:

- The `-i P H P` block is `assemble_tangent_hamiltonian` with `H_sym` in
  place of `H_asym`, realified.
- The `P (∂_v P) H ψ` block is new: two-defect contractions (one defect from
  the basis, one from `v`) against `Hψ`, involving inverse bond matrices.
  This is the finite-chain analogue of the Hallam et al. Jacobian, without
  momentum labels. Frame transport then needs a separate tangent-to-tangent
  overlap between the frames at `t` and `t+dt`.

Route B also yields the *instantaneous* Jacobian `M(t)`. At the `H_asym`
fixed point it must reduce to the realified `-i H_tangent` from the
relaxation strand, which is a strong cross-check. Algebra to be written out
properly before any code.

(A third option, forward-mode differentiation through `tdvp_step_r/l`, is
not recommended. It would differentiate the SVD inside
`left_orthogonal_tensor`, whose derivative blows up as `1/(s_i² − s_j²)`,
exactly at the near-degenerate spectra thermofield states have. It would
also carry the gauge-drift problem back into the propagated vectors.)

### 5. Conditioning: the main numerical risk

Both routes degrade as the smallest Schmidt value `s_min` → 0. Route A needs
`ε` inside the linear regime, and Route B carries `Λ^{-1}`. More
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
- **Report the `ε`-plateau** at a few points along the trajectory, as the
  diagnostic that Route A is in its linear regime.

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
            Q  ← [DΦ_dt]_{frame→frame'} · Q        (Route A: 2k clone steps)
            ψ, frame ← ψ', frame'
        Q, R ← QR(Q);  accumulate log|diag R|;  store diag R (and Q periodically)
        log s_min, energy, ‖H_sym ψ‖ diagnostics

- **`k = n` (half the real dimension) gives all non-negative exponents** if
  `±λ` pairing holds. That pairing needs one validation run with `k = 2n`.
  HLMs sit at the *bottom* of that half, which is the slowest part of the
  spectrum to converge. There is no shortcut that skips the faster
  exponents.
- **`τ`** (steps per QR): with Route A, it is 1 if clones are rebuilt every
  step. Clones could run for longer if `ε·e^{λ_max τ dt}` stays linear, but
  start with 1.
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
2. **Both routes, chosen by size.** Route A (finite differences) was built
   first and is the reference. Route B (exact generator, next section) was
   derived to beat Route A's cost, then Route A was parallelized over
   columns, which made it faster again above n ≈ 700. Use Route B below
   that, Route A with 16 workers above (see "Cost"). The two agree on the
   one-step map to `O(dt³)` and on a full spectrum to ~0.005.
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
  no progress bar, returns a new mps), `tangent_map(frame, frame_next, Y,
  ..., n_jobs)` (finite-difference action of `DΦ_dt` on the columns of `Y`,
  forward or central; `n_jobs > 1` splits columns over loky worker
  processes, one chunk per worker, one BLAS thread each, bit-identical to
  serial), `phase_fixed_projection`, `lanczos_method(epsilon=1e-8)`.
- `tangent_generator.py` — Route B: `apply_mpo`, `normal_residual` (`w_N`),
  `assemble_K`, `assemble_H_tan`, `realify_generator`, `generator(frame, H)`,
  `parallel_transport` (polar factor by two Newton–Schulz iterations),
  `half_step_propagator`, `propagate`, `step_matrix` (validation use).
- `frame.py` also has `Lam` (bond matrices), `mps_direct_sum`, `frame_change`.
- `benettin.py` — `benettin(..., route='B'|'A', n_jobs)` forward loop with
  `positive_qr`, h5 storage (`R` every block, lzf-compressed; per-block
  `log_diag_R`, `t`, `s_min`, energy and `blocks_done` written as they
  come, so a crashed run is still readable; `Q` + frame at
  `store_Q_blocks`), `running_exponents`, `ginelli_backward(path,
  want_blocks, discard_last)` → CLVs in their stored frames.
- `analysis.py` — `energy_density_mpo(site, 'phys'|'aux')`, `local_profile`
  (`2 Re⟨ψ|O_j|Φ(X)⟩` for every `j` in `O(N)`), `energy_profile`,
  `site_weight_profile`, `cosine_transform` (DCT-II, `q = πk/N_bonds`),
  `mode_report`, `q_weight_by_exponent` (cluster-averaged DCT power of the
  energy profile, binned by exponent), `pairing_residual`.
  `template_modes` is a stub.
- `plots.py` — `plot_spectrum` (sorted `λ_i` vs `i/2n` + histogram),
  `plot_convergence`, `plot_pairing`, `plot_mode` (site weights, energy
  profile, DCT), `plot_q_weight`, `load_run`/`exponents_from` (handle
  partial runs). `python lyapunov/tdvp_lyapunov/plots.py <file>.h5
  --discard 20 --clv --modes 0 1 -1` writes `figures/`.
- `compare_runs.py` — overlay figures across runs over a common time
  window: `figures/compare_{L_scan,D_scan,dt_route}.png`.
- `run_lyapunov.py` — driver: `--route A|B`, `--n-jobs`, `--out-dir`;
  `--time-only` measures a step and prints a cost estimate.
- `runs/queue2.sh` — the queue that produced the 2026-09-16 scan, with its
  timestamped log `runs/queue2.log`.
- `validate_frame.py`, `validate_benettin.py` — rungs 1–3 below;
  `validate_routeB.py` — `w_N` orthogonality, `K` vs finite-differenced
  projector, Route B vs Route A map.

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
efficiency cores, 15.5 GB). Route A with 16 workers is 5.6× its serial
cost (4.5× at 8 workers). Route B's dense `expm` and transport on `2n × 2n`
matrices grow like n³, Route A like n·N·D³, so they cross near n ≈ 700.

| L | D | n | Route B | Route A serial | Route A, 16 workers |
|---|---|---|---|---|---|
| 8 | 4 | 303 | **2.7** | 29 | 4.4 (production: 3.2 at dt = 0.025) |
| 12 | 4 | 495 | **6.4** (production) | 78 | ~14 |
| 16 | 4 | 687 | 24 | 139 | **14.9** (production) |
| 8 | 8 | 959 | 75 | 91 | **11.8** (production) |
| 8 | 12 | 1967 | 609 | 327 | **37** (production) |

"Production" = wall time per block over the finished run, including QR,
frame construction and h5 writes. The laptop's RTX 4060 is unused: it would
not help Route A (thousands of tiny tensor operations), but would remove
Route B's large-n bottleneck (dense `expm` / matmul) if CuPy were installed.
Replacing the dense `expm` by its action on the k columns is the CPU-only
alternative. Neither has been done.

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
convergence, qweight_clv, clv*}.png` (CLVs at ~60% of each run, the rest
used as the Ginelli backward transient). Cross-run: `figures/compare_*.png`
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
   patterns; dark top-right corner of the `qweight_clv` heatmaps at D = 8,
   D = 12, L = 16), and the fraction of power in the two longest
   wavelengths rises almost monotonically as λ → 0. **But in the near-zero
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
   wavelengths" cover a q range that shrinks with L.

End-to-end validation of the pipeline is finding 1 together with
"Validation results" (rungs 1–3) and "Route B" (one-step map agreement
`O(dt³)`, `K` vs finite-differenced projector `O(ε²)`).

## Next

In rough order of value per effort:

1. **Look inside the near-zero cluster without averaging** (no new runs):
   finer exponent bins near λ = 0; per-vector long-wavelength fraction as a
   scatter against λ rather than bin means; physical vs auxiliary copy
   energy profiles; the same for the site-weight profile. This is what
   decides whether finding 5 means "no HLMs" or "diluted HLMs".
2. **`template_modes`**: overlaps of the CLVs with `P Σ_j cos(qj) h_j ψ`
   and `−i P Σ_j cos(qj) h_j ψ`. A direct test instead of an energy-profile
   proxy.
3. **Longer runs**: D = 12 to the length of D = 8 (~2.5 h), and discard
   the first 3 time units after vectors start. Needed before reading the
   D dependence (finding 4) or λ_0.
4. **`k = 2n` pairing check** at L = 8, D = 4 (Route B, ~25 min): the one
   structural assumption behind decision 6 not yet tested directly.
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
