# Physical/ancilla swap symmetry of the tangent space

Status: **in production (2026-09-24).** `symmetry.py` holds the machinery;
`run_lyapunov.py --symmetrize-every 1 --sector even` is the production form,
and the even-sector runs are the ones to quote — see
[`../WORKFLOW.md`](../WORKFLOW.md) for the scan.

What it bought is a *controlled* D scan. The 2026-09-21 `k = 2n` runs sat at
wildly different symmetry breaking (`a_odd` = 9e-4 at D = 4, 0.70 at D = 8,
0.68 before the first block at D = 12), and their scatter in λ_max was read
as a D dependence. Holding the symmetry, λ_max is flat in D to 0.012, within
each run's own convergence noise. It also costs less: 2.9 h and 6.3 GB at
L8 D12 against 6.0 h and 19.2 GB.

## The symmetry

The doubled physical index is `i = 2 a_phys + b_anc` (from
`analysis._copy_kron`, and the infinite-T seed `[1, 0, 0, 1] = |00> + |11>`),
so exchanging the copies is the on-site permutation `[0, 2, 1, 3]` and
`S = ⊗_m s_m` is a product of on-site unitaries.

Three facts make the tangent space split:

- **`H_sym` is swap-even.** `operators.symmetric_thermofield` builds
  `H⊗I + I⊗H`, so `S H_sym S = H_sym`. `H_asym = H⊗I − I⊗H` is swap-**odd** —
  every `H_asym`-type direction is in the odd sector, including the ones
  that are exactly zero at the fixed point.
- **The point is swap-even along the whole trajectory.** In the
  vectorization `psi <-> M`, swap is transposition, `S|M>> = |M^T>>`. The
  build gives `M(0) = rho^(1/2)`, symmetric; evolution under `H_sym` sends
  `M -> e^(-iHt) M e^(-iHt)` (the tilted Ising `H` is real symmetric), still
  symmetric. So `S|psi(t)> = |psi(t)>` — a **unitary** Z2, not an antiunitary
  one, which matters because a unitary `S` commutes with the complex
  structure `J`.
- **The manifold and the flow are S-covariant.** On-site unitaries map
  rank-D MPS to rank-D MPS; with the point fixed, the TDVP vector field
  commutes with `S`, and so does its linearization.

Therefore `dS` is an involution on `T_psi`, `T = T⁺ ⊕ T⁻`, both blocks are
invariant under the tangent flow, and the Lyapunov spectrum is the disjoint
union of two sub-spectra. Benettin restricted to either block is exactly
valid, and since `[S, J] = 0` each sector is symplectic in its own right —
**the `±λ` pairing residual still works as the health check inside a
restricted run.**

## What it costs (measured, L8 D12, 250 blocks)

| | k | per block | wall | R storage |
|---|---|---|---|---|
| `k = 2n` | 3934 | 78 s | 6.0 h | 19.2 GB |
| even sector | 2098 | 42 s, +102 s per projection | 2.9 h | 6.3 GB |

The QR dominates the `k`-dependent cost — 43 s of that 78 s, measured — but
the generator (16 s) and transport (6 s) are `k`-independent and floor the
block at 22 s, so halving `k` does not halve the run. Σ itself costs 102 s
at D = 12, hence `--project-every 10`. Taking `k` to half a sector (the
non-negative half, 1049) would give ~2 h and 1.4 GB, at the price of the
pairing residual as a live check — not worth it while the restricted
machinery is this new.

`run_lyapunov`'s own cost estimate is useless here: it times the probe at
the *build* point, where s_min ~ 1e-14 makes the generator's `Λ^-1` term
enormous and one exponential action takes 600 s against ~13 s in the run.

## Building the involution

**Validation route** (`swap_matrix` in `validate_tangent_swap.py`): column j
of `Sigma` is `project_to_frame(S · tangent_mps(frame, e_j), frame)` — the
same construction as `frame.frame_change`, with the swapped state in place
of the second point. No gauge matrices: the swap is applied in Hilbert space
and the projection brings it back, so nothing crosses between
canonicalization passes. `n` projections at `O(N d D³)`; 0.4 s at L = 8,
D = 4.

**Production route** (not built): `S` is on-site, so it maps a tangent
vector parameterized at site m back to site m with the telescoped `A`s and
`B`s replaced by swap-gauge matrices at the two adjacent bonds. `Sigma` is
block-diagonal in the site index, each block `X_m -> W_m X_m Ũ_m` with
`W_m`, `Ũ_m` unitary involutions, so the `±1` basis is a product of two
small eigendecompositions — `O(D³)` per site, and the sector dimensions come
out as `tr Sigma_m = tr W_m · tr Ũ_m`. Two things to get right: the bond
gauge phases must telescope so the induced action on `psi` itself is `+1`
(calibrate on the flow direction `−i P H_sym psi`, which must be even, or
the sector labels silently swap), and `W`, `Ũ` must come from the same
canonicalization pass as `V_L`.

## Validated

`validate_tangent_swap.py` at L = 8, D = 4, β = 0.1, t = 1 (n = 303):

```
<psi|S|psi>       = +1.000000000000 - 2.4e-17j
|Sigma^2 - 1|     = 1.3e-12        involution
|Sigma^T Sigma-1| = 1.3e-12        tangent space is S-invariant
|Sigma - Sigma^T| = 7.3e-15        (1 ± Sigma)/2 are orthogonal projectors
|[Sigma, J]|      = 0.0            S is complex-linear, pairing survives
eigenvalues ±1 to 1.2e-13;  330 even / 276 odd
X ± Y are ±1 eigenvectors of S to 5e-15 (S applied in Hilbert space, not via Sigma)
```

Read the errors with care: every coordinate diagnostic — `|1 - lambda|`, the
eigenvector residual, `|Sigma^2 - 1|` — is **quadratic** in the symmetry
breaking, because `Sigma` is the compression `P S P` and what leaks out is
dropped twice. The linear measure is `leak = sqrt(1 - |Sigma e_j|^2)`, and
it tracks the swap-odd amplitude of the state,
`a_odd = |(1 - S)psi / 2|`. Expect `|1 - lambda| ~ leak²`.

Sector dimensions at the start of each production run (t = 8, `a_odd` 1e-8,
Σ an involution to 2e-14): **330/276** at D = 4, **1102/816** at D = 8,
**2098/1836** at D = 12. Even always exceeds odd, as the one-site picture
requires — the swap is transposition on a 2×2 matrix, 3 symmetric against 1
antisymmetric — and the imbalance is `tr Σ = Σ_m tr W_m · tr Ũ_m`, which
nothing forces to vanish. Read the split only at a clean point: an earlier
1038/880 at D = 8 was taken at t = 1, where leak = 2e-3 means the
eigenvalues are not ±1 and the count cannot be trusted.

## What breaks it

Two mechanisms, neither of them roundoff, and one feeds the other.

**1. Rank deficiency — the frame, not the state.** At β = 0.1 the exact
thermofield double is close to rank 1, so the imaginary-time build *ends* on
the rank-deficient boundary (s_min = 2.4e-11 at D = 8) and only real time
walks the state off it. Where a Schmidt value is numerically zero the point
is a **singular point of the rank-D manifold**: `A_R` and `V_L` past the
rank are an arbitrary isometry completion, the parameterized tangent space
is a property of the representation rather than of the state, and `S` —
which makes a different arbitrary choice — maps it elsewhere. Signature:
`leak >> a_odd`, ratio ~1e5 against 2–30 with a sound frame.

**2. Chaotic amplification — the state, not the frame.** The exact flow
commutes with `S`, but the odd sector carries positive exponents, so
whatever was injected grows at the Lyapunov rate: `a_odd` grows at +0.649 at
D = 8, β = 0.1, against `λ_max` = 0.615. Signature: `leak ~ a_odd`.

Enforcement kills the first, and with nothing left to amplify the second
has nothing to do. `validate_tangent_swap.py` still reports which of the two
it is looking at, which is what the `--transient` *time* (not step count)
is for: below t ≈ 1 the frame is singular, above t ≈ 2 the drift dominates.

### Where it enters

`validate_swap_flow.py inject` traces `a_odd` through the build and the
transient (L = 8, β = 0.1, 240 imaginary steps, dt = 0.05):

| t | s_min (D=12) | D = 4 | D = 8 | D = 12 |
|---|---|---|---|---|
| 0 (build) | 4e-14 | 1.8e-8 | 1.5e-8 | 1.5e-8 |
| 0.25 | 8e-7 | 0 | 2.0e-5 | 1.1e-5 |
| 1.0 | 8e-4 | 1.1e-8 | 3.5e-5 | 8.2e-4 |
| 2.0 | 3e-2 | 0 | 5.0e-5 | 8.8e-2 |
| 8.0 | 1.5e-1 | 5.8e-7 | 1.9e-3 | 6.8e-1 |

**The build is clean at every D** — `a_odd` ~ 1e-8, the imaginary-time
integration error, and exactly 0 at D = 4 where the state is symmetric to
the last bit. All of it is injected while the state sits at s_min ≪ 1, and
what changes with D is only how long it lingers there: D = 4 is out by
t ≈ 0.5 and never picks anything up, D = 8 by t ≈ 1, D = 12 not until
t ≈ 2, by which time it has taken on 1e-1. There is one mechanism, not two:
rank deficiency injects, chaos amplifies.

Along the stored runs of the 2026-09-21 scan (`validate_swap_flow.py run`,
`a_odd` at the first, middle and last stored frame):

| run | t = 8.05 | t = 14.55 | t = 20.50 | growth |
|---|---|---|---|---|
| L8_D4 | 6.0e-7 | 3.6e-5 | 9.1e-4 | +0.59 |
| L12_D4 | 9.7e-7 | 1.3e-4 | 1.8e-2 | +0.79 |
| L16_D4 | 6.4e-7 | 4.4e-5 | 1.9e-3 | +0.64 |
| L8_D8_ns | 2.0e-3 | 1.8e-1 | 7.0e-1 | +0.47 |
| L8_D12 | 6.8e-1 | 7.0e-1 | 7.0e-1 | saturated |

`a_odd` saturates at 1/√2 = 0.707, where `psi` is orthogonal to `S psi`.
So the D = 4 runs stay symmetric to ~1e-3 over their whole length, D = 8
loses it halfway, and **the D = 12 run was never on a thermofield
trajectory at all** — it left the symmetric sector during its transient,
before the first tangent vector was propagated. Rerunning it in the even
sector settled what that cost: λ_max comes out at 0.527, statistically the
same as the plain run's 0.522. The broken symmetry was corrupting the
*comparison between* runs, not λ_max within one.

## What the Lyapunov vectors do

`validate_swap_flow.py` at L = 8, D = 4, the size where the state itself
stays symmetric, so that what is measured is the vectors and not the point.

- **A stored run carries no sector labels.** The Q columns and the CLVs are
  ~50/50 mixtures (odd weight 0.25–0.82, not one of the 606 pure to 1%),
  and no filtration subspace `span(q_1..q_m)` decomposes either:
  `max|1 − |mu||` over the eigenvalues of `Q_m^T Sigma Q_m` is ~1 for every
  m < 2n. This is **not** symmetry breaking — the numbers are the same at
  block 0 (leak 1.5e-6) as at block 249 (leak 2.5e-3). It is the QR. Q
  starts random and its columns resolve at the local spectral gap, whose
  median here is 1.3e-3, so a bulk column needs ~800 time units against a
  run of 12.5. Nothing in the bulk is converged, so nothing has a sector,
  and this is a caveat about individual bulk CLVs generally, not only about
  symmetry. **The sector has to be imposed at the start, not read off
  afterwards.**
- **A sector-seeded set stays in its sector, to exactly `a_odd`.** 200
  blocks from t = 8 to 18, k = 32 even-seeded vectors, no enforcement: the
  odd amplitude of every column tracks the state's own `a_odd` within a
  factor of 2 (5.8e-7 at the start, 2.5e-4 at the end) and grows at the
  same rate. The tangent split is preserved exactly as well as the state's
  symmetry is, with no amplification of its own.
- **Projecting the vectors is a no-op in a full run and load-bearing in a
  restricted one.** In a `k = 2n` run at D = 4 it moves the block by
  `|ΔQ|/|Q|` = 4e-8 and leaves every exponent unchanged to 9e-12. But a
  *restricted* run has nothing orthogonalizing away the other sector, so
  leaked weight grows at the gap between the sectors' top exponents until it
  takes over — 1e-7 injected per step reaches O(1) in ~6 time units at
  D = 12. Hence `--project-every 10`, which holds the contamination at
  2–5e-15 (measured across all three production runs).
- **At D = 12 the top of the spectrum is entirely odd.** With the state held
  symmetric and `k = 2n`, the leading columns have odd weight 1.0000,
  0.9997, 0.9996 … falling to the generic 0.45 by column 200, mirrored at
  the bottom; the 104 columns above the plain run's max average 0.78. Those
  exponents measure escape *from* the thermofield submanifold, not chaos
  along it, and they are not stationary: λ_0 runs 0.618 → 0.866 → 1.066 →
  1.430 over successive windows while the bulk sits still and the middle of
  the spectrum matches the plain run to four decimals. A fresh Q at t = 18
  reproduces the elevated value (0.98 against 0.54 plain), so it is the flow,
  not QR drift. At D = 4 the two sectors' tops are degenerate (0.591 odd
  against 0.535 even) and QR mixes them, which is why this never showed up
  before.

## Consequences

- **Enforce on the state, in the transient.** That is where all the
  injection happens, and after a clean start the run holds by itself:
  `a_odd` ≤ 1.7e-8 across all 250 blocks of every production run, logged per
  block rather than assumed. `enforce_swap_symmetry(psi, D)` is the
  projection `psi -> (psi + S psi)/‖·‖` recompressed to D (`mps.add`, added
  2026-09-22 for this); it loses ≤ 1.1e-14 of the projected state, leaves E
  unchanged to 8e-12, and at D = 4 does not move the trajectory at all
  (1 − |⟨plain|enforced⟩| = 1.7e-13 over 160 steps).
- **Quote the even sector above D = 4.** It is the half that preserves the
  thermofield structure; the odd half describes leaving it and carries the
  largest, non-stationary exponents at D = 12. The restricted spectrum is
  exact: at D = 4 the union of the two sectors reproduces the unrestricted
  run with rms difference 0.0089 and Σλ agreeing to four decimals.
- **The k = 2n runs on disk at D = 8 and D = 12 are outside the symmetric
  sector.** Their exponents are still exponents of the TDVP flow at the
  points they visit, so the spectra are not nonsense; but those points are
  not thermofield doubles, and any sector language applied to them is.
- **The templates are not sector-pure.** `hlm` defaults to `which='phys'`,
  the single-copy energy density, whose odd component is `[A, M(t)]`. That
  commutator vanishes at the TFD but is generically O(1) along the
  trajectory, so a restricted run quietly discards part of the diagnostic.
  A template's own even/odd weight is cheap to measure (Σ applied to its
  coordinates) and worth measuring before choosing a sector; its
  *enrichment* against a stored spectrum is not, for the convergence reason
  above.

## Left to do

- **The L scan in the even sector.** Only the D scan has been redone; the
  L = 12 and L = 16 points are still unrestricted `k = 2n`.
- **Why the odd sector runs away at D = 12 and not at D = 4.** The
  escape rate grows with D far faster than the in-sector chaos, and it is
  not stationary. Unexplained; `‖(1−P)Hψ‖` (2.0, flat) and the entanglement
  (saturated at ln D in both) rule out the obvious candidates.
- **Half-sector runs.** `k = 1049` at D = 12 — the non-negative half of the
  even sector — would give ~2 h and 1.4 GB, at the cost of the pairing
  residual as a live check.
- **The per-site Σ.** The production route in *Building the involution*
  is still unbuilt; it would make Σ `O(D³)` per site and remove the 102 s
  per projection.
