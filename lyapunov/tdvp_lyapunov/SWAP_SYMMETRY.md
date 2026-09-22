# Physical/ancilla swap symmetry of the tangent space

Status: **symmetry confirmed and its decomposition validated (2026-09-22);
not usable above D = 4 as the pipeline stands.**
`validate_tangent_swap.py` builds the involution and checks it at a point;
`validate_swap_flow.py` follows it along a run. The two findings that
govern any use of it: the split is exact for the flow but **numerically
injected into, at the rank-deficient start, in an amount that grows steeply
with D** — the L8 D12 production run is already fully asymmetric before its
first block — and a stored run **cannot be labelled after the fact**,
because its vectors are nowhere near converged.

See [`README.md`](README.md) for the pipeline this would plug into.

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

## What it would buy

Per block the cost is `k` generator applications (linear in `k`) plus a QR
of a `(2n, k)` matrix, with `k²` stored per block.

| | k | QR | R storage (L8_D12) |
|---|---|---|---|
| today, `k = 2n` | 2n | 8n³ | 19.2 GB |
| project each step, `k = n` | n | 2n³ | 4.8 GB |
| explicit sector basis | n | n³ | 4.8 GB |

2x on the generator and 4–8x on the QR, so the 101-minute, 19 GB D = 12 run
becomes roughly half an hour and 5 GB — which puts **D = 16, or L = 24,
inside today's budget**. The sectors are not 50/50 (measured 330/276 at
L = 8, D = 4 and 1038/880 at L = 8, D = 8), so keeping the even half alone
is ~6.5x on the QR rather than 8x.

Second payoff, independent of cost: every exponent and CLV gets a **sector
label**, a real quantum number, sharper than the enrichment measure. It also
predicts that near-degenerate cross-sector pairs at the top of the spectrum
are mixed by QR — a candidate explanation for the ±15% noise in the top 5%
that no amount of running has fixed.

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

## Validated (2026-09-22)

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

## What breaks it, and what that costs

Two mechanisms, neither of them roundoff. Measured at L = 8, D = 8, β = 0.1
(`t = transient * dt`; the script takes `--transient` as a **time** for this
reason):

| config | t | s_min | a_odd | leak | \|1−λ\| |
|---|---|---|---|---|---|
| D = 4 reference | 1.0 | 1.9e-01 | 1.1e-07 | 7.4e-07 | 1.4e-13 |
| D = 8 baseline | 1.0 | 2.9e-02 | 7.7e-05 | 2.1e-03 | 2.1e-07 |
| D = 8, imag_steps 40→240 | 1.0 | 2.9e-02 | 3.5e-05 | 3.2e-04 | 5.7e-08 |
| D = 8, dt 0.05→0.01 | 0.2 | 4.0e-07 | 8.4e-07 | 2.3e-01 | 2.5e-02 |
| D = 8, imag 240 at t = 0.2 | 0.2 | 2.8e-07 | 9.2e-07 | 2.9e-01 | 4.5e-02 |
| D = 8, transient → t = 8 | 8.0 | 2.0e-01 | 7.4e-03 | 1.4e-02 | 9.0e-05 |

**1. Rank deficiency — the frame, not the state.** At β = 0.1 the exact
thermofield double is close to rank 1, so the imaginary-time build *ends* on
the rank-deficient boundary (s_min = 2.4e-11 at D = 8) and only real time
walks the state off it (2.9e-2 by t = 1, 2.0e-1 by t = 8). Where a Schmidt
value is numerically zero the point is a **singular point of the rank-D
manifold**: `A_R` and `V_L` past the rank are an arbitrary isometry
completion, the parameterized tangent space is a property of the
representation rather than of the state, and `S` — which makes a different
arbitrary choice — maps it elsewhere. Signature: `leak >> a_odd` (ratio
~1e5, against 2–30 with a sound frame). This is why converging the build
*harder* hurts while t is short: a better-converged TFD is closer to rank 1.

**2. Chaotic amplification — the state, not the frame.** The exact flow
commutes with `S`, but the odd sector carries positive exponents, so any
asymmetry grows exponentially. Measured growth of `a_odd` over t = 1..8:
**+0.649** at D = 8, β = 0.1 (+0.829 at D = 4, +1.065 at β = 1), against
`λ_max = 0.615` for the same parameters. Signature: `leak ~ a_odd`.

So there is a **window**: long enough that the bond dimension has filled,
short enough that the drift has not taken over — t ≈ 1–2 at these sizes.
Outside it the check fails for a reason, and the script now says which.
There is no setting that makes the check clean at D = 8: `a_odd` is already
7.7e-5 by the time the bonds have filled, so `leak ~ 2e-3` is the floor.

### Where the asymmetry actually enters (2026-09-22)

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
before the first tangent vector was propagated. That is the same run whose
λ_max came out low (0.522 against 0.615 at D = 8); it does not prove the
D scan's collapse is an artefact, but it is a second symptom of the same
rank-deficient start and should be weighed with it.

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
- **Soft enforcement is a no-op where it works.** Projecting onto
  `(1 + Sigma)/2` after every QR moves the block by `|ΔQ|/|Q|` = 4e-8, buys
  a factor ~3 on the contamination (each step re-injects at the current
  leak), and leaves all 32 exponents unchanged to 9e-12. At D = 4 it is not
  needed; at D ≥ 8 it cannot help, because there `Sigma` is built at a point
  that is not swap-symmetric, and projecting onto a sector of the wrong
  involution is meaningless.
- **The top exponent is not resolvably odd.** Leading even 0.5395 against
  leading odd 0.5211 over the same window, the two crossing repeatedly
  along the way — a gap well inside the convergence wobble. The earlier
  inference from `a_odd`'s growth rate (+0.649 against λ_max = 0.615) was
  over-read: that rate belongs to a continuously re-injected quantity, not
  to a freely evolving perturbation, so it is not a clean sector exponent.

## Consequences

- **Fix the state, not the vectors.** The binding constraint is not the
  tangent space, which behaves; it is that the point itself leaves the
  symmetric sector, and it does so in the first two time units, where the
  bonds are still empty. Enforcement therefore belongs in the *transient*:
  re-symmetrize `psi -> (psi + S psi)/‖·‖` while s_min is small, which needs
  an MPS sum at bond 2D recompressed to D — nothing in `states.py` adds two
  MPS yet, so this is the one piece of new machinery the programme needs.
  The discarded part is exactly the injected odd component, so at D = 4 the
  operation is a no-op and at D = 12 it is the whole point.
- **The D = 8 and D = 12 runs on disk are outside the symmetric sector.**
  Their exponents are still exponents of the TDVP flow at the points they
  visit, so the spectra are not nonsense; but those points are not
  thermofield doubles, and any sector language applied to them is nonsense.
- **The templates are not sector-pure.** `hlm` defaults to `which='phys'`,
  the single-copy energy density, whose odd component is `[A, M(t)]`. That
  commutator vanishes at the TFD itself but is generically O(1) along the
  real-time trajectory, so restricting the run would quietly discard part of
  the diagnostic. A template's own even/odd weight is well defined and cheap
  to measure — `Sigma` applied to the template's coordinates — and worth
  measuring before choosing a sector; its *enrichment* against a stored
  spectrum is not, for the convergence reason above. The fallback is still a
  win: **running the two sectors separately** costs 1/4 of the current QR
  and half the storage, loses nothing, and labels every exponent.

## Staging (revised 2026-09-22 after items 2 and 3 were measured)

1. ~~Validate the involution.~~ Done — `validate_tangent_swap.py`.
2. ~~Post-hoc labels, no new runs.~~ **Dead.** The stored vectors are not
   converged enough to carry a label, for reasons that have nothing to do
   with the symmetry; see *What the Lyapunov vectors do*. The same argument
   kills the idea of measuring the templates' even/odd weight against a
   stored spectrum — the weight is well defined, the thing it would be
   weighed against is not.
3. ~~Soft enforcement on the vectors.~~ Done — and it is not the useful
   knob. Costs 4e-8 per block, changes no exponent, and cannot rescue the
   D ≥ 8 runs.
4. **Symmetrize the state through the singular region** (new, and now the
   first real step). An MPS sum plus compression, applied every step or few
   steps while s_min ≲ 1e-2, then once more at the end of the transient.
   Measure: does `a_odd` at t = 8 come down to the D = 4 level at D = 8 and
   D = 12, and what does that do to λ_max at D = 12? This is worth doing
   for the D scan alone, independently of ever restricting a run.
5. **Restricted run.** Only after 4. `k = n` in one sector, seeded from
   `(1 ± Sigma)/2`, with that sector's own pairing residual as the health
   check and `a_odd` logged per block; cross-checked at L = 8, D = 4
   against the union of the two sectors' spectra.
