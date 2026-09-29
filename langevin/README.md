# Langevin Subproject

The goal of this subproject is to use the qtensor library to integrate noisey equations as found in https://arxiv.org/pdf/2111.06408 with code at https://github.com/AndrewHallam/Langevin .

I would like to perform the same tasks with my own library while fixing the subtle issue there, namely that the equation is treated inconsistently as both Stratonovic and Ito. Notes on this can be found in  `ito_issues.tex` . 

---

## What is here (2026-09-28)

| file | role |
|---|---|
| `models.py` | the paper's H (`J=-1, hx=-1.05, hz=0.5`) as a complex128 MPO; `with_onsite_fields(H, c)` adds `sum c_m F_m` into the on-site slot (no bond-dimension growth); `random_product_state` (**default initial state**: seeded random spin directions via `states.rand_product`), `plus_state`; dense helpers for tests |
| `brackets.py` | `A_m = {H, F_m}`, `K_mn = {F_m, F_n}` two ways: the general single-site-TDVP projector route (`poisson_brackets`), and the exact local fast path (`local_brackets`) |
| `integrator.py` | `langevin_step` (one step, four schemes), `run_trajectory`, `run_ensemble` (process-parallel over seeds), `run_many` (one pool over a whole parameter scan) |
| `matlab_lanczos.py` | line-for-line port of the reference `lanczos_exp.m`, as a qtensor `TDVPMethod` and a two-site drop-in; used by `paper_exact` |
| `validate_brackets.py` | checks of the brackets against dense linear algebra and against `tdvp_step` itself |
| `validate_paper_exact.py` | checks of the reference Krylov port and of the `paper_exact` scheme against dense evolution |
| `run_demos.py` | the simple-case behaviour runs, and `friction_scan` (the paper's Fig. 1); writes `figures/*.png`, `runs/*.npz`. `--state random` (default) or `plus`, `--state-seed`, `--scheme`, `--switch all\|any` |
| `../qtensor/simulation/twositeTDVP.py` | **new** two-site TDVP step (`tdvp2_step`), same conventions as `finiteTDVP.tdvp_step`; `expm=` selects the two-site Krylov exponential |

Nothing in the existing `qtensor` files was changed. `finiteTDVP.tdvp_step` was already the
workhorse with `tdvp` a wrapper around it (commit 383412d), so the Langevin step simply calls it
with a per-step MPO.

## The equation and how a step works

In the notation of `ito_issues.tex`, the rearranged equation is

    x' = X_H + sum_m X_{F_m} c_m,    c = S (gamma A + eta),    S = (1 + gamma K)^-1,
    <eta_m(t) eta_n(t')> = 2 gammaT delta_mn delta(t - t'),

with `X_G` the TDVP flow `psi' = -i P G psi`, and `{G, B} = d<B>/dt|_G = 2 Im <B psi|P|G psi>`.
Since `c` is real, the right-hand side is the TDVP flow of the **Hermitian** generator
`H + sum_m c_m F_m`. One step is therefore `tdvp_step(psi, with_onsite_fields(H, c), dt)`: the
sweeps, the Lanczos updates and the environments are all reused unchanged. Over a step the noise
enters as `c dt ~ dW`, with `dW ~ N(0, 2 gammaT dt)`. Bath operators are `X_j, Y_j, Z_j` on every
site (3N of them), ordered `m = 3j + a` as in the reference code.

Schemes (`langevin_step(..., scheme=)`):

- `paper`: the reference code as written, `c = dW/dt + gamma S (A - K dW)`, with everything at the
  start of the step. See issue 1 below for the `K dW`.
- `euler`: the rearranged equation with coefficients frozen at the start of the step,
  `c = S0 (gamma A0 + dW/dt)`. The vector fields `X_F` are followed along the step, because
  `tdvp_step` re-projects as it goes; the coefficients `S`, `A` are not. This is the Ito /
  Stratonovich mix the notes describe.
- `heun`: the Stratonovich fix, the "two passes" of the notes. Predict `psi~` with `c0`, recompute
  `c~` at `psi~` with the **same** `dW`, then step from `psi` with `(c0 + c~)/2`. With the exact
  flow along `X_F` at frozen coefficients, this supplies the missing `(1/2)(dc . X) dW^2` term, so
  it is Stratonovich-consistent to the order that matters. It costs 2x `euler`.
- `paper_exact`: `paper` plus the reference code's last-site typo (issue 2 below), so the
  generator gains an anti-Hermitian term `-i f Y_N`, and with every local exponential done by the
  reference Krylov routine (`matlab_lanczos.py`; see "Krylov routines" below for why that matters).

At `gamma = 0` the coefficients are state-independent (`c = dW/dt`), so all schemes are
identical and no brackets are computed.

Growth hand-over (`run_trajectory(..., switch=)`): `all` (default) switches to single-site TDVP once
every bond is at `min(D, full)`; `any` switches once the largest bond reaches `D`, as the reference
`tdvp_langevin_finite_2site.m` does through `Dmaxout`, leaving shorter bonds frozen where they were.

## Finding: A and K are exact Heisenberg quantities

For a single-site `F_j`, `exp(-i theta F_j) psi` has the same bond dimensions, so `F_j psi` is
already a tangent vector: **`P F_j psi = F_j psi`**. Therefore:

    A_m  = 2 Im <F_m psi|P|H psi>   = i<[H, F_m]>        (exact, any D)
    K_mn = 2 Im <F_n psi|P|F_m psi> = i<[F_m, F_n]>      = -2 eps_abc <sigma^c_j> on site j, else 0

Consequences:

- `K` is **exactly** 3x3 block diagonal. The reference code's same-site truncation in
  `calculate_friction.m` loses nothing, and the notes' "for our model K is 3x3 block diagonal" is
  exact.
- `S` depends only on the local Bloch vectors (`|K_j| ~ |<sigma_j>|`, which grows with on-site
  purity, as the notes say).
- `A` only needs nearest-neighbour reduced density matrices, which costs `O(N D^3)`
  (`local_brackets`), instead of the `O(N^2 D^3)` environment sums.

`validate_brackets.py` checks the fast path against the general projector route, and the
projector route against three things: exact commutators at full D; a dense `P` built from an
explicit tangent basis at D = 3 (1e-15); and finite differences of `<F>` through `tdvp_step`
itself (1e-8, which fixes the sign and transpose conventions). The whole integrator has no free
convention left.

## Issues found in the reference code

1. **The noise part of the friction term is off by a factor of `dt`.** `tdvp_langevin_finite.m`
   computes the friction with `Hnoise = update_H_MPO_langevin(H, etak)`, i.e. `H + dW . F`, but
   evolves with `etak/dt`. Its friction is therefore `gamma S (A - K dW)`, where the rearranged
   equation needs `gamma S (A - K dW/dt)`. The `gamma S K eta` term (the one that turns `eta`
   into `S eta`) is suppressed by `dt` (50x at the paper's `dt = 0.02`). The `paper` scheme
   reproduces it so it can be compared.
2. **A sign typo at the last site:** `update_H_MPO_langevin.m` adds `(etak(end-1) - 1i*friction(end-1))*Y`
   where every other site has `+ friction`. For `gamma != 0` the generator is then non-Hermitian
   on the last site: `exp(-i dt (... - i f Y_N))` contains `exp(-f dt Y_N)`, an imaginary-time
   push on the edge spin at rate `f ~ gamma`, followed by renormalization. The `paper` scheme
   uses the intended real coefficient; `paper_exact` reproduces the typo.
3. **The Ito/Stratonovich mix of the notes.** Coefficients are frozen at the start of the step
   while the vector fields are exponentiated. This is `euler` here; the fix is `heun`.
4. Minor: `lanczos_exp.m` is three-term Lanczos with no reorthogonalization and a fixed
   `m = min(20, dim)`. Unlike qtensor's pre-2026-09-16 Lanczos it stops at `m = dim`, so it does
   not produce the garbage vector; see "Krylov routines" below.
5. The reference driver `Finitechain_Ising_Langevin.m` prepares its initial state by DMRG on
   `H(J=0, hx=-1)`, i.e. `|+>^N` (E = -16.8 at N = 16 in the code's sign convention), and switches
   to single-site TDVP when the *largest* bond reaches D (`switch='any'` here). Its defaults
   (N = 30, D = 250, dt = 0.02, noise only) are not the Fig. 1 settings, so the Fig. 1 script
   itself is not in the repository.

### Krylov routines: reference vs qtensor

The four routines differ as follows (`validate_paper_exact.py`, test 1 and 2):

| | reference `lanczos_exp.m` | qtensor `updatemethod.lanczos_*` | `twositeTDVP.krylov_expm` |
|---|---|---|---|
| orthogonalization | three-term recurrence only | full Gram–Schmidt, two passes | full, two passes |
| Krylov size | `min(20, dim)` fixed | `min(16, dim)`, stop at norm < 1e-10 | `min(20, dim)`, stop at 1e-12 |
| diagonal | `alpha = conj(<v\|H v>)` | `<v\|H v>` | real part only |
| projected matrix | tridiagonal, `beta` both sides | upper triangle, lower filled as its conjugate | `(T + T^dag)/2` |
| non-Hermitian `H` | **correct** to O((dt\|H\|)^m / m!) | wrong at first order in dt | anti-Hermitian part dropped |

For Hermitian generators all agree to 1e-16 per local update. For a non-Hermitian one the
reference routine is, perhaps surprisingly, right: its recurrence defines each vector through
`H v_q = beta_q v_{q+1} + alpha_q v_q + beta_{q-1} v_{q-1}`, so `H V = V T + residual` holds
exactly whatever `alpha` is, and the conjugated `alpha` only makes `V` non-orthogonal. qtensor's
routines assume Hermiticity and get it wrong by 1e-2 per update (dt = 0.02), so the typo must be
run through the port. The one inexactness of the port is on local spaces of dimension ≤ 20 with
a non-Hermitian generator, where `m = dim` and the non-orthogonal basis leaves an
O((dt |H_eff|)^dim / dim!) error. It is 1.7e-9 in infidelity after 25 steps at N = 5, and it is
in the reference runs too.

A sign slip in `ito_issues.tex` (not edited): with the notes' own convention
(`{A, B}` = rate of change of `B` along `X_A`), the energy equation starts
`d<H>/dt = sum_n {F_n, H} eta_n`, not `{H, F_n}`. Carried through, this gives
`d<H>/dt = -gamma A^T S A - A^T S eta`. Since `sym(S) = (1 + gamma^2 K^T K)^-1 > 0`, the
friction term is strictly dissipative, as it should be. The `friction` demo checks this rate
numerically.

## Initial states: grow with two-site TDVP first

Starting single-site TDVP from a zero-padded product state makes the early dynamics depend on the
arbitrary orthonormal completion that the first SVD picks for the padding. Measured at N = 12,
`|+>^N`, `dt = 0.02`: mid-bond S(t = 0.2) is correct (0.15912, the exact value) for D = 8, 16,
24 and 48, but 0.1480 for D = 32, independent of Lanczos settings. The reference code avoids
this by running two-site TDVP until the bond dimension fills. Here, `run_trajectory(...,
grow_to=D)` with an unpadded `plus_state(N, 1)` does the same: `twositeTDVP.tdvp2_step`
(truncation cutoff 1e-10) runs until every bond reaches `min(D, 2^k, 2^(N-k))`, then single-site
takes over (`t_switch` in the output). Against exact evolution at N = 10 and 12 for D = 8, 16
and 32, the mid-bond entropy agrees to < 1e-5 up to t = 1, and to 1.7e-3 at t = 2 at D = 8
(truncation error); energy is conserved to 1e-12.

## Simple-case results (`run_demos.py`, 2026-09-28, all from `|+>^N` under the paper's H)

These four were run before the default initial state became the random product state; rerun
with `--state plus` to reproduce them.

**Noiseless saturation** (`figures/saturation.png`; N = 12, dt = 0.05, t ≤ 15). Mid-chain entropy
grows ballistically (identical across D until t ≈ 1.5), then saturates at a D-dependent ceiling:
late-time S = 0.77 / 1.34 / 1.60 for D = 4 / 8 / 16 (ln D = 1.39 / 2.08 / 2.77). Energy is
conserved to ≤ 1.2e-12 over 300 steps. Even the D = 16 ceiling is far below the N = 12 Page value
(≈ 3.7), and it rises with D: this is the manifold-limited plateau, not thermal saturation.

**Noise only, γ = 0** (`figures/noise.png`; N = 12, D = 16, dt = 0.02, 10 trajectories per γT).
Noise slows entanglement growth, monotonically in γT:

| γT | ⟨S⟩(t=1) | ⟨S⟩(t=3) | ⟨S⟩(t=10) | ⟨E⟩(10) − E0 |
|---|---|---|---|---|
| 0 | 0.713 | 1.412 | 1.660 | 0 |
| 0.05 | 0.574 | 1.378 | 2.585 ± 0.005 | 12.6 |
| 0.2 | 0.427 | 1.002 | 2.334 ± 0.019 | 12.6 |
| 0.5 | 0.315 | 0.710 | 1.783 ± 0.042 | 12.6 |

With no friction the noise also heats the chain to infinite temperature (E → 0 from E0 = −12.6),
which *raises* the entropy ceiling above the noiseless 1.66. The net effect is slower growth
towards a higher plateau. At γT = 0.5, S is still climbing at t = 10.

**Friction only, γ = 0.2, γT = 0** (`figures/friction.png`; N = 10, D = 8). Energy falls
monotonically (−10.50 → −14.13 by t = 10, 0 of 500 steps with dE > 0, for both `euler` and
`heun`). The instantaneous rate matches the rearranged equation's `dE/dt = -γ Aᵀ S A` to about
0.2% at t = 1…5 (e.g. −0.4896 predicted, −0.4903 measured), which confirms the corrected sign
above.

**Schemes, γ = 0.5, γT = 0.5** (`figures/schemes.png`; N = 8, D = 8, 40 trajectories per point,
⟨E⟩ averaged over t ∈ (3, 6)):

| dt | paper | euler | heun |
|---|---|---|---|
| 0.04 | −0.92 ± 0.07 | −1.04 ± 0.06 | −1.47 ± 0.09 |
| 0.02 | −0.88 ± 0.06 | −0.99 ± 0.07 | −1.45 ± 0.10 |

**Caveat: ⟨E⟩ is still rising at t = 6**, so these are transient energies, not a steady state.
The Stratonovich scheme heats more slowly, sitting about 0.45 lower than the frozen-coefficient
one over t ∈ (3, 6) (≈ 4σ), and ⟨S_mid⟩ is correspondingly lower. The gap does not shrink when
dt halves, so the schemes are converging to *different* stochastic dynamics: the
Ito/Stratonovich question in the notes changes the physics, not just the discretization error.
`paper` heats visibly faster than `euler` for t ≲ 2, which is issue 1 (the dt-misplaced
`γ S K η` term) acting in the transient. Later the two agree within error.

For scale: the Gibbs energy of this N = 8 chain at T = γT/γ = 1 is −12.3 (−8.6 at T = 2;
E0 = −13.05). **All three schemes are heading towards infinite temperature** (E = 0), far from
Gibbs at T. So either this "T" is not a thermodynamic temperature for the Langevin manifold
dynamics, or a noise-induced drift is missing. Longer runs (t ≫ 6) are needed to see where each
scheme actually settles. Either way, this is the next thing to understand (see the Fokker–Planck
section of the notes).

## Friction scan: the paper's Fig. 1 at N = 16, D = 32 (2026-09-28)

`python langevin/run_demos.py friction_scan --N 16 --D 32 --traj 16 --tf 40 --dt 0.04`:

- T = 0.2 fixed, γ = γT / T, `heun` scheme.
- Seeded random product state (seed 0), the same for every trajectory.
- 16 trajectories per γT, plus the noiseless run.
- Wall time 2 h 24 min on 21 workers (see the cost note below).
- Output: `figures/friction_scan_N16_D32.png`, `runs/friction_scan_N16_D32.npz`.
  Raw ensembles are in `runs/friction_scan_N16_D32*.npz`, and the fits in
  `runs/friction_scan_N16_D32_fits.npz`. Per γT the `fits` columns are: γT, γ, S0, dS0, t0,
  dt0, late ⟨S⟩, late ⟨E⟩, n_traj, saturated.

| γT | γ | S0 | t0 | ⟨S⟩(t > 30) | ⟨E⟩(t > 30) |
|---|---|---|---|---|---|
| 0 (noiseless) | 0 | 3.349 | 4.08 | 3.339 | 2.05 |
| 0.02 | 0.10 | 3.341 | 4.79 ± 0.04 | 3.340 | −0.07 |
| 0.05 | 0.25 | 3.342 | 6.25 ± 0.05 | 3.341 | −0.07 |
| 0.08 | 0.40 | 3.343 | 7.48 ± 0.07 | 3.340 | −0.09 |
| 0.11 | 0.55 | 3.343 | 8.59 ± 0.09 | 3.339 | −0.06 |
| 0.15 | 0.75 | 3.348 | 10.10 ± 0.11 | 3.338 | −0.07 |
| 0.25 | 1.25 | 3.363 | 13.46 ± 0.14 | 3.326 | −0.08 |
| 0.5 * | 2.5 | (3.477) | (21.30 ± 0.27) | 3.232 | −0.16 |
| 1.0 * | 5.0 | (3.891) | (35.17 ± 0.94) | 2.966 | −0.54 |

\* Added later the same day: `--gammaTs 0.5 1.0 --no-noiseless --tag _gT0.5-1 --processes 16`,
same state, seeds, dt and t_f, 46 min wall. These two are **not saturated by t_f = 40**: ⟨S⟩
is still rising more than 1% over the last 5 time units. Their S0 and t0 come from a tanh
extrapolation, which puts S0 above ln D = 3.47 at γT = 1, a value D = 32 cannot reach. The
figure draws them as open markers. So the apparent rise of S0 at large γT in panel (c) is an
artifact of the window, not a transition. t0 keeps growing roughly linearly (21 and 35 are
lower-bound-ish estimates), and it is the one clean trend. The strongest friction (γ = 5) is
the first run to pull ⟨E⟩ visibly below zero (−0.54), still nowhere near T = 0.2 Gibbs.
Measuring S0 at γT ≥ 0.5 needs t_f ≳ 3 t0, i.e. t ≈ 100 at γT = 1.

### Fixed γ = 1, pushing γT to 10 (2026-09-28)

`run_demos.py friction_scan --gamma 1 --gammaTs 0.1 1 10 --dt 0.04 0.04 0.005 --tf 40 80 20
--traj 7 --prefix gamma1_scan`: 7 trajectories per point, 74 min wall on 21 workers. Output:
`figures/gamma1_scan_N16_D32.png`. dt is 0.005 at γT = 10 because the per-step kick
√(2γT dt) would otherwise be 0.9 rad per Pauli.

| γT (T) | t_f | S0 | t0 | late ⟨S⟩ | late ⟨E⟩ |
|---|---|---|---|---|---|
| 0.1 (0.1) | 40 | 3.367 | 13.58 ± 0.20 | 3.331 | −0.18 |
| 1 (1) | 80 | 3.340 | 21.30 ± 0.04 | 3.329 | −0.01 |
| 10 (10) | 20 | — (not saturated) | — | 0.40 | −0.06 |

- **No saturation below the D ceiling.** γT = 0.1 and 1 both reach S ≈ 3.33, the same ceiling
  as every run so far. The noise only delays it.
- **t0 depends on γ as well as γT.** γT = 0.1 at γ = 1 gives t0 = 13.6, against 8.6 at
  γT = 0.11 with γ = 0.55; γT = 1 at γ = 1 gives 21.3, against ≳ 35 at γ = 5.
- **At γT = 10 growth is strongly suppressed but not stopped.** ⟨S⟩ = 0.03 / 0.25 / 0.52 at
  t = 1 / 10 / 20, against about 3 by t = 5 noiseless. The curve is still accelerating at
  t = 20 (slope 0.028 on [10, 20], positive curvature), and all 7 trajectories lie in
  0.41–0.64. That is slow growth, not a low plateau.
- **Open question.** Whether γT = 10 levels off short of ln D needs t ≈ 100–150: 20–30k steps
  at dt = 0.005, several hours per trajectory at N = 16, D = 32. The dt = 0.005 choice at
  γT = 10 has also not been convergence-checked.

### The reference code's scheme, γ = 0.5 (2026-09-29)

Question: does the transition appear with the original `paper` integration scheme instead of
`heun`? The command was `run_demos.py friction_scan --gamma 0.5 --gammaTs 0.1 0.5 1.0 --dt 0.04
--tf 40 60 80 --traj 7 --scheme paper --prefix paper_gamma0.5_scan`. It ran in 21 min on 21
workers; `paper` has no predictor pass, so it costs half of `heun`. Output:
`figures/paper_gamma0.5_scan_N16_D32.png`. The scan command now takes `--scheme`, and the
scheme is saved in each raw npz and shown in the figure title.

| γT (T) | t_f | S0 | t0 | late ⟨S⟩ | late ⟨E⟩ |
|---|---|---|---|---|---|
| 0.1 (0.2) | 40 | 3.343 | 7.56 ± 0.07 | 3.341 | −0.06 |
| 0.5 (1) | 60 | 3.344 | 14.02 ± 0.04 | 3.338 | −0.02 |
| 1.0 (2) | 80 | 3.344 | 22.43 ± 0.07 | 3.328 | +0.01 |

**No transition with the paper's scheme either.** All three saturate within their window, at the
same D ceiling (S ≈ 3.34, ln 32 = 3.47) as every `heun` run. S0 is flat in γT to 0.001, the
curves collapse in t/t0 exactly as the `heun` ones do, and the energy again goes to infinite
temperature. The scheme changes only the rate, and not by much. The nearest `heun` point,
γT = 0.11 at γ = 0.55, has t0 = 8.6, against 7.6 here at γT = 0.1, γ = 0.5; that is consistent
with `paper` heating slightly faster (the `schemes` demo). So neither the Ito/Stratonovich
treatment nor the reference code's dt-misplaced friction term produces saturation below ln D at
N = 16, D = 32.

### Bond-dimension test at the paper's strongest Fig. 1 point (2026-09-29)

This is the paper's largest Fig. 1 parameters: T = 0.2, γ = 0.9 (γT = 0.18, above their claimed
transition at 0.11), with the `paper` scheme, dt bug included. Run to t = 20, with 16
trajectories plus a noiseless reference at each of D = 32 and D = 64:
`friction_scan --gamma 0.9 --gammaTs 0.18 --dt 0.04 --tf 20 --traj 16 --scheme paper --prefix
fig1_point --D 64` (then `--D 32`). Wall time was 22 min for D = 64 and 12 min for D = 32 on
17 workers. Figures: `figures/fig1_point_N16_D_comparison.png` (via `friction_scan_compare_D
--prefix fig1_point`), plus the per-D four-panel figures.

Benchmark (N = 16, D = 64, `paper`; the cost barely depends on γ, T):

- Growth to D = 64 takes 35 two-site steps (to t = 1.4), 17 s.
- One-site steps cost 0.55 s alone with 1 BLAS thread. With multi-threaded BLAS it is slower,
  0.83 s. Per worker it is 0.86 s with 8 in parallel and 1.18 s with 16.
- The 17-worker run above actually averaged ≈ 2.5 s per step per worker (1327 s for 500 steps
  plus growth), so the parallel
  benchmark still undercounts at full load.

| t | 2 | 5 | 8 | 11 | 14 | 17 | 20 |
|---|---|---|---|---|---|---|---|
| D = 32, γT = 0.18 | 0.53 | 1.30 | 2.11 | 2.68 | 3.04 | 3.20 | 3.28 |
| D = 64, γT = 0.18 | 0.53 | 1.30 | 2.14 | 2.84 | 3.39 | 3.70 | 3.87 |
| D = 32 noiseless | 1.28 | 3.06 | 3.25 | 3.31 | 3.33 | 3.34 | 3.34 |
| D = 64 noiseless | 1.28 | 3.20 | 3.84 | 3.96 | 4.00 | 4.01 | 4.01 |

(ensemble means, standard error ≤ 0.05; spread of S(20) across trajectories: 3.25–3.31 at
D = 32 and 3.80–3.93 at D = 64.)

- **The noisy curve is D-independent until it meets the D = 32 ceiling.** Then each D bends
  towards its own noiseless ceiling.
- **D = 64 was still climbing at t = 20**, at 0.07 per unit time, 3.87 against a ceiling of
  4.01. That is not saturated.
- **So there is no bath-set plateau below 3.87 by t = 20**, even at the paper's strongest
  friction point. The D = 32 "saturation" of the earlier scans was the variational ceiling.
- **The noise does halve the growth rate** relative to noiseless (S = 1.30 against 3.2 at
  t = 5).
- **What would still reconcile with the paper:** an intrinsic plateau between about 3.9 and
  their D = 128 ceiling (roughly 4.6 by their Fig. 2). Seeing that needs D = 128, or a longer
  D = 64 run that levels off clearly short of 4.0.

`plot_friction_scan` (or `run_demos.py friction_scan_plot`) refits and redraws from every
`runs/friction_scan_N{N}_D{D}*.npz`, merged by γT. New γT values can therefore be added as a
separate run with its own `--tag` without recomputing the old ones.

**No saturation transition at this D.** Every curve, noiseless included, saturates at the same
S ≈ 3.34, just under ln 32 = 3.47. This is the bond-dimension ceiling. The 0.02 spread in fitted
S0 (panel c, zoomed axis) is the tanh shape not quite fitting, not physics: the late-time ⟨S⟩
is flat to 0.015. What the bath *does* change is the rate. t0 grows roughly linearly with γT
(4.8 → 13.5; the noiseless t0 is 4.1), so all the curves collapse onto one shape in `t/t0`
(panel b). That is the analogue of the paper's Fig. 2d, not of its Fig. 1c.

**The bath does not cool to T = 0.2.** The random initial state sits at E = +2.05. Every
friction/noise run settles at ⟨E⟩ ≈ −0.07, i.e. infinite temperature (Tr H = 0), whereas
T = 0.2 Gibbs is essentially the ground state (≈ −26 for N = 16, extrapolated
from the exact N = 8 value of −13.05). The earlier `schemes` demo
showed the same at T = 1, D = 8. A plausible reason, not yet tested: the TDVP-Langevin
equation is a *classical* Langevin equation on the manifold, whose natural stationary state is
the classical Gibbs measure `exp(-<H>/T)` with the Liouville volume. The manifold has ~10⁴ real
dimensions here, so equipartition puts `<H> − E0 ~ (n_dof/2) T ≈ 1500 ≫` the spectral width
(~50). Classically, T = 0.2 is then effectively infinite, and entanglement runs up to the
D-ceiling whatever γT is. If that holds, the paper's transition needs either much smaller D
(fewer degrees of freedom) or an effective temperature that scales like `T / n_dof`.

A cheap test: ⟨E⟩(T) at small D (D = 1, 2, 4; N = 8), against `n_dof T / 2`.

## Reproducing the reference code: `paper_exact` at the Fig. 1 point (2026-09-29)

The fixed-T argument: if the equation obeys fluctuation–dissipation, its stationary measure is
`exp(-<H>/T)`, which does not depend on γ, so a transition in γT at fixed T must be kinetic or
come from something that breaks it. Mechanism, seen in the D = 64 run above: averaged over
the noise, the isotropic white noise is a depolarizing channel (it relaxes each field term at
rate 8γT and each ZZ term at 16γT, at any D), while friction removes energy at `γ A^T S A` with
`A = d<F>/dt`. From a product state A is large and friction wins (E: +2.05 → −8.6 by t = 1);
as entanglement shrinks the local Bloch vectors, A → 0, friction dies, and E returns to ≈ 0.
The typo was the one remaining reference-code difference that grows with γ, vanishes at γ = 0
and is non-unitary, so it was tested directly.

N = 16, D = 32, T = 0.2, γ = 0.9, γT = 0.18, dt = 0.04, 16 trajectories, same seeds as the
`paper` run. Ensemble means (`runs/fig1_exact*_N16_D32.npz`,
`figures/fig1_exact_comparison_N16_D32.png`):

| t | 1 | 2 | 5 | 8 | 11 | 14 | 17 | 20 |
|---|---|---|---|---|---|---|---|---|
| S, `paper`, random state | 0.31 | 0.53 | 1.30 | 2.11 | 2.68 | 3.04 | 3.20 | 3.28 |
| S, `paper_exact`, random state | 0.31 | 0.53 | 1.30 | 2.12 | 2.66 | 2.99 | 3.14 | 3.20 |
| S, `paper_exact`, `\|+>`, `switch=any` | 0.43 | 0.69 | 1.49 | 2.21 | 2.70 | 3.00 | 3.14 | 3.21 |
| E, `paper`, random state | −8.63 | −8.55 | −5.06 | −2.14 | −1.03 | −0.38 | −0.22 | −0.13 |
| E, `paper_exact`, random state | −8.29 | −8.40 | −4.81 | −2.06 | −1.21 | −0.82 | −0.62 | −0.53 |
| E, `paper_exact`, `\|+>`, `switch=any` | −11.17 | −8.44 | −4.21 | −1.89 | −1.08 | −0.83 | −0.62 | −0.44 |

(S(20) across trajectories: 3.25–3.31 `paper`, 3.12–3.24 `paper_exact`, 3.17–3.26 `|+>`.)

- **The typo does not produce a plateau.** It lowers S(20) by 0.08 and holds E slightly lower
  (−0.53 against −0.13), consistent with a weak non-unitary pull on one edge spin, but
  S is still rising towards the D = 32 ceiling at t = 20.
- **The initial state is forgotten by t ≈ 5.** From `|+>` (E = −16.8) with the reference
  hand-over rule, the curve joins the random-state one; E passes −11 at t = 1 and then follows
  the same reheating.
- **The noiseless `|+>` run stays near S ≈ 2.3** at t = 20 (fitted S0 = 2.19, still rising 3%),
  below the D ceiling, because its energy is conserved at −16.8. So from the reference
  initial state the bath *raises* the late-time entanglement (to 3.2), the opposite of Fig. 1's
  trend. Not separated here: how much of the 2.3 is the energy and how much the bonds frozen by
  `switch=any`.
- Cost: the reference Krylov routine is cheaper than qtensor's two-pass one: 234 s and 296 s
  for these runs, against 12 min for the `paper` run.

## Running

From the repo root, with the Anaconda python:

    python langevin/validate_brackets.py          # ~2 s, prints PASS
    python langevin/run_demos.py saturation       # or noise | friction | schemes | all

    from langevin.models import plus_state
    from langevin.integrator import LangevinModel, run_trajectory
    res = run_trajectory(plus_state(16, 1), LangevinModel(16), dt=0.02, steps=500,
                         gamma=0.1, gammaT=0.05, scheme='heun', grow_to=32, seed=0)
    res['t'], res['S_mid'], res['energy'], res['sigma']    # sigma: (times, N, 3) Bloch vectors

Costs (1 core, idle machine): N = 16, D = 16 is about 0.2 s/step for `euler` and 0.4 s/step
for `heun`; N = 20, D = 32 is about 0.6 / 1.2 s/step. Up to D ≈ 32 the cost barely depends on D:
Python overhead in the per-site loops dominates, not the D³ contractions.

**Size sweeps against the loaded machine.** With 21 workers on this 16-core / 22-thread CPU, a
`heun` step at N = 16, D = 32 costs 1.48 s, against 0.5 s with 8 workers. The 2026-09-28
`friction_scan` was sized from the 8-worker number, so it took 2 h 24 min instead of the
planned 45 min. Budget `wall ≈ ceil(jobs / workers) × steps × 1.5 s` at this size, or use
≤ 16 workers.

## Open / next

- Does `heun` give the right thermal state? At small N one could compare the long-time ensemble
  `<E>` against `Tr H e^{-H/T}/Z`. Note, though, that the Fokker–Planck section of the notes
  suggests that even the Stratonovich SDE needs its noise-induced drift checked against the
  manifold's divergence terms before a Gibbs-like stationary state can be expected.
- Two-site Langevin during the growth phase uses the same coefficients as the single-site phase
  (A and K are exact either way). The *projector* differs, so the first few steps are two-site
  dynamics, as in the reference code.
