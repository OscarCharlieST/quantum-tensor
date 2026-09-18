# Relaxation rates from the tangent-space spectrum

Status: **implementation started.** `tangent_hamiltonian.py` builds and
verifies the projected Hamiltonian; the observable weights and rate
extraction are not written yet.

See [`../README.md`](../README.md) for shared background (purification,
symmetric vs antisymmetric thermofield Hamiltonian, the fixed-point
property) and for why this strand exists rather than a direct Lyapunov
calculation.

## Goal

The Lyapunov exponents at the fixed point are exactly zero: `P H_asym P` is
Hermitian, so `‖δψ(t)‖` is exactly conserved. What is *not* conserved is
the local content of δψ — a local observable's deviation from its
fixed-point value decays by dephasing among the real mode frequencies.
Compute that decay rate.

This is a relaxation rate / inverse relaxation time, not a Lyapunov
exponent, and for an energy-density perturbation it should reduce to a
diffusion constant.

## Approach: MPS tangent-space projection

Project H_asym onto the **MPS tangent space** at `psi_uniform` — the
polynomial-size (in N, D) subspace used throughout the
tangent-space/quasiparticle-ansatz literature (Haegeman, Verstraete et
al.), and a direct generalization of the single-site effective Hamiltonian
this codebase already builds for TDVP.

1. Bring `psi_uniform` to canonical form. Both gauges are needed — a
   tangent vector at site n uses left-orthogonal tensors to its left and
   right-orthogonal ones to its right — and they must agree bond-for-bond.
   One sweep in each direction does *not* guarantee that (each clips bond
   dimensions to its own staircase), so three sweeps are taken.
2. At each site n, compute the **null-space tensor** `V_L^n`: reshape the
   left-orthogonal tensor `A^n` from `(d, Dl, Dr)` to a `(d·Dl, Dr)`
   isometry and take an orthonormal basis for the orthogonal complement of
   its column space. Any tangent direction at site n is `B^n = V_L^n X^n`
   for a free matrix `X^n`. This is the gauge-fixing that removes the
   redundancy between "moving `A^n`" and "absorbing the opposite move into
   a neighbouring tensor", and it is what makes different sites' tangent
   vectors mutually orthogonal and orthogonal to `psi_uniform` itself.

   **Edge case: `d·Dl = Dr` exactly.** A square isometry is a full unitary,
   so its column space is everything and `V_L^n` has zero columns — that
   site contributes no tangent directions. Not a bug: every variation of a
   square-unitary `A^n` is pure gauge (`A^n Y`, cancellable against the
   neighbouring tensor), exactly what the gauge condition excludes. The
   physical freedom near that bond is carried by neighbouring sites'
   tangent vectors instead, and the dimension count
   `Σ_n (d·D_{n-1} − D_n)·D_n` stays correct with some terms zero. Observed
   in practice at the first two sites of an L=6, D=6 chain.
3. Each variation `|Φ_n(X^n)> = A_L^1...A_L^{n-1} (V_L^n X^n)
   A_R^{n+1}...A_R^N` is one basis direction; the tangent space is the span
   over all n and all `X^n`, of dimension `Σ_n (d·D_{n-1} − D_n)·D_n`.
4. Matrix elements `<Φ_n(e_i)| H_asym |Φ_m(e_j)>` come from contracting the
   H_asym MPO between two MPS copies differing only in their defect tensor.

   **Reuse note.** The sum-of-environment-projectors form of `P_tangent`
   (`Σ_n P_{≤n-1}⊗I_n⊗P_{>n} − P_{≤n}⊗P_{>n}`, the object underlying TDVP's
   site/bond effective-Hamiltonian split) is the *same* projector as the
   `V_L^n` construction: substituting `I_n = A^n(A^n)† + V_L^n(V_L^n)†` into
   the first term of each summand and cancelling against the second leaves
   exactly `P_{≤n-1}⊗V_L^n(V_L^n)†⊗P_{>n}`. It is an operator-level
   restatement, not an alternative route — TDVP only ever *applies*
   `P_tangent` to one vector, so it never needs an explicit basis and can
   skip `V_L` entirely. Diagonalizing needs the basis, so `V_L^n` stays
   necessary. The payoff is that every block, diagonal and off-diagonal
   alike, is `updatemethod.apply_Heff_parts` evaluated against the
   appropriate environments and then sandwiched by `V_L`.
5. Diagonalize with `scipy.linalg.eigh`.
6. Extract the rate — see the next section, which is the part that needs
   real care.

**Open caveat.** Because `psi_uniform` is a purification, unitary rotations
acting purely on the auxiliary copy leave the physical reduced state
unchanged. Some tangent directions may be exactly these physically-trivial
auxiliary rotations, distinct from the ordinary MPS gauge-fixing in step 2.
They need identifying and excluding, or showing not to matter for the
observable of interest. Unresolved.

## From oscillatory phases to an exponential rate

The response of a local observable is

    δ⟨O⟩(t) = Σ_k w_k e^{-iω_k t},     w_k = 2 Re[⟨ψ*|O|k⟩⟨k|δψ(0)⟩]

with every `ω_k` real. Getting `e^{-Γt}` out of this needs an argument, and
the obvious one does not work.

**Why the naive expansion fails.** Expanding in t and using that the
response is real, the linear term cancels and

    δ⟨O⟩(t) = δ⟨O⟩(0) · [1 − ½⟨ω²⟩t² + O(t⁴)]

The short-time behaviour is *quadratic*, whereas `e^{-Γt} = 1 − Γt + ...`
is linear. They disagree at first order, so no finite-order Taylor
expansion can ever produce exponential decay. This is the quantum Zeno
regime, and it is a structural obstruction, not a missing term: expansion
in t is simply the wrong tool.

**The right framing is Fourier, not Taylor.** Define the spectral function

    A_O(ω) = Σ_k w_k δ(ω − ω_k)      so that     δ⟨O⟩(t) = ∫dω A_O(ω) e^{-iωt}

The decay law is fixed entirely by the analytic structure of `A_O(ω)`.
Three consequences:

- **Discrete spectrum ⇒ no decay, ever.** A finite sum of phases is
  quasi-periodic, with recurrences at the Heisenberg time `t_H ≈ 2π/Δω`
  set by the level spacing. A 59-mode matrix cannot show relaxation.
- **A continuum is required.** As the level spacing → 0 *and* the weights
  `w_k` become a smooth function of `ω_k` (an ETH-like smoothness
  assumption on matrix elements of local operators), the sum becomes a
  smooth integral, which does decay.
- **Exponential ⟺ Lorentzian.** The Fourier transform of a Lorentzian of
  half-width Γ is exactly `e^{-iω₀t}e^{-Γ|t|}`. A Gaussian `A_O` gives
  Gaussian decay, a box gives a sinc. Exponential decay is therefore not
  generic — it is the signature of one specific lineshape, and the real
  question is why that lineshape arises.

**Where the Lorentzian comes from (Wigner–Weisskopf).** Split the tangent
space into the slow mode `|s⟩` of interest (the long-wavelength
energy-density deformation) and everything else `{|q⟩}`:

    H_tan = ε_s|s⟩⟨s| + Σ_q ε_q|q⟩⟨q| + Σ_q (V_sq|s⟩⟨q| + h.c.)

The survival amplitude `a_s(t) = ⟨s|e^{-iH_tan t}|s⟩` has an *exact*
resolvent representation

    a_s(t) = (1/2πi) ∮ dz e^{-izt} G_ss(z),    G_ss(z) = 1/(z − ε_s − Σ(z))
    Σ(z) = Σ_q |V_sq|² / (z − ε_q)

With a discrete q-spectrum, Σ has real poles, `G_ss` has only real poles,
and `a_s` is quasi-periodic — consistent with the above. When the
q-spectrum becomes dense, the sum turns into `∫dω ρ(ω)|V(ω)|²/(z−ω)`, which
has a **branch cut** along the real axis, and approaching from above

    Σ(ω + i0⁺) = Δ(ω) − i Γ(ω)/2,     Γ(ω) = 2π ρ(ω) |V(ω)|²

— Fermi's golden rule. Analytically continuing `G_ss` *through* the cut onto
the second Riemann sheet reveals a pole at `z* ≈ ε_s + Δ − iΓ/2`, and
closing the contour around it gives

    a_s(t) ≈ e^{-i(ε_s+Δ)t} e^{-Γt/2}

So the decay rate is the imaginary part of a **resonance pole on the second
sheet**. The Hermitian spectrum is the branch cut on the first sheet; the
complex pole hides behind it. Nothing contradicts Hermiticity — the
*generator* is Hermitian, the *single-mode effective propagator* obtained by
integrating out the rest of the tangent space is not. This is the same
structure as a Nakajima–Zwanzig reduced generator, and it is the honest
version of "non-Hermitian effective dynamics from Hermitian microscopics".

**Validity window.** The pole approximation needs Γ(ω) and Δ(ω) roughly
constant over a width Γ around the pole (flat-continuum / Markov
condition), which gives

    Δω  ≪  Γ  ≪  W

with Δω the level spacing and W the bandwidth. Γ ≫ Δω so that a continuum
exists on the timescale 1/Γ (else recurrences arrive first); Γ ≪ W so the
self-energy is slowly varying and the Zeno time `1/√⟨ω²⟩` is well short of
`1/Γ`. Outside the window, the known deviations: quadratic at `t ≲ t_Zeno`,
recurrences beyond `t_H`, and a power-law tail at very long times from the
branch-cut endpoint (the spectrum is bounded below — Khalfin's theorem).

**Heuristic feasibility estimate.** The tangent dimension is
`~ N(d−1)D²`, and H_asym is extensive so `W ~ N`. Hence the typical level
spacing `Δω ~ W/dim ~ 1/((d−1)D²)` — roughly **independent of N and
controlled by bond dimension**. For the longest-wavelength diffusive mode,
`Γ ~ D_diff(π/N)²`, so `Γ ≫ Δω` becomes

    D  ≳  N / √((d−1)·D_diff)

Linear in N, i.e. polynomially feasible, and it identifies **D rather than
N** as the useful lever for densifying the spectrum. Back-of-envelope only
— the density of states is not uniform — so check numerically.

**Extraction recipe.** Given `eigh(H_tangent)` and the weights `w_k` (one
more contraction: an observable MPO with a single defect site, same
environment machinery, no H):

1. Form `δ⟨O⟩(t) = Σ_k w_k e^{-iω_k t}` directly. This costs nothing — the
   whole time trace at any t from one diagonalization, no time stepping —
   and makes the Zeno onset, the exponential window and the recurrence all
   visible at once. Primary diagnostic.
2. Fit an exponential across the window; cross-check by histogramming
   `A_O(ω)` and fitting a Lorentzian.
3. For O = energy density at wavevector q, fit `Γ_q` vs `q²` to get the
   diffusion constant, and cross-check against
   `visualise.estimate_diffusion_constant_from_widths` on a direct
   `finiteTDVP.tdvp` run.

## Default parameters

Physical Hamiltonian: `ops.tilted_ising`, at its own defaults, which are
the ones used throughout this repo (taken from arXiv:1702.08894):

    H = J z_i z_{i+1} + h z_i + g x_i,   J = 1,  h = 0.25,  g = -0.525

Non-integrable, so local observables are expected to relax. The thermofield
Hamiltonians are `thermofield.thermofield_hamiltonian(H, asym=...)`:
symmetric (`a=+1`) for the imaginary-time build, antisymmetric (`a=-1`) as
the generator whose tangent projection we diagonalize. Doubled physical
dimension d = 4.

Scan defaults in `run_relaxation_scan.py`:

| parameter | value | note |
|---|---|---|
| `BETA` | 0.1 | see below |
| `D` | 8 | same for every L, so the tangent dimension grows only through L |
| `L_VALUES` | 8, 12, 16 | |
| `IMAG_STEPS` | 60 | too low for D >= 16 — see "Cost and convergence" |
| `SEED_NOISE` | 0 | none needed, see below |

**On β = 0.1** (changed from β = 1 on 2026-09-17). Hydrodynamics is a
high-temperature expectation, so the standing convention across `lyapunov/`
is now β = 0.1; this subproject follows it. The cost is real but bounded,
and is worth stating because the original choice of β = 1 was made to avoid
exactly this:

| L | β | `s_min` | bandwidth | fixed-point residual | residual/bandwidth |
|---|---|---|---|---|---|
| 4 | 1 | 3.9e-07 | 14.8 | 1.0e-02 | 7.0e-04 |
| 4 | 0.1 | 1.6e-11 | 15.0 | 2.9e-02 | 1.9e-03 |
| 4 | 0.01 | 9.2e-16 | 13.3 | 2.6e-02 | 2.0e-03 |
| 8 | 1 | 3.4e-07 | 18.6 | 2.6e-02 | 1.4e-03 |
| 8 | 0.1 | 8.2e-12 | 16.3 | 4.5e-02 | 2.8e-03 |
| 12 | 1 | 3.0e-07 | 18.8 | 3.4e-02 | 1.8e-03 |
| 12 | 0.1 | 4.5e-11 | 16.1 | 4.7e-02 | 2.9e-03 |

Going from β = 1 to β = 0.1 costs four to five orders of magnitude in the
smallest Schmidt value and roughly doubles the fixed-point residual
relative to the tangent bandwidth. It is still small — 0.3% — so β = 0.1 is
usable, and the degradation does not grow with L. **β = 0.01 is not
usable here**: `s_min` reaches 9e-16, so the tangent space is built on
numerically null directions and any rate read off it is meaningless.

Note this is the *opposite* of what the `tdvp_lyapunov` temperature scan
found (conditioning *improves* at high temperature there). There is no
contradiction: that subproject runs 160 real-time TDVP steps under `H_sym`
before switching on the tangent vectors, and that transient fills the bond
dimension. This one works at the imaginary-time fixed point with no such
transient, so it sees the bare rank collapse of the nearly-unentangled
thermofield double. **The `tdvp_lyapunov` result does not license high
temperature here** — the table above is the relevant evidence.

If the relaxation rates turn out to depend strongly on β, that is a
physical result worth having, not a nuisance. Results recorded in this
README predating the change were taken at β = 1.

**On the seeding noise — removed 2026-09-18, and it was the dominant
error.** `inf_T_thermofield` returns a rank-1 state zero-padded to bond
dimension D. Single-site TDVP is a fixed-rank method, which was taken to
mean the evolution would stay rank 1 without a noise seed. That reasoning
is wrong. `states.left_orthogonal_tensor` calls
`la.svd(..., full_matrices=False)` and keeps every singular value including
the exact zeros, so after one canonicalization the `A` tensors are dense
isometries whose columns past the rank are an arbitrary orthonormal
completion. The environments then have support on every bond index, `H_eff`
couples the centre tensor into the zero-weight directions, and the
evolution fills the padded quadrants by itself.

Measured at L = 16, D = 12: the noiseless build reaches full rank (12 of 12
Schmidt values above 1e-10) with `fixed_point_residual` = 1.1e-7, against
7.1e-2 with `noise = 1e-2`.

| D | noise | rank | `‖P H ψ*‖` | n_eff | η window | D_peak |
|---|---|---|---|---|---|---|
| 8 | 0 | 8/8 | 5.9e-07 | 114 | 0.27 dec | 0.4446 |
| 8 | 1e-2 | 8/8 | 6.7e-02 | 150 | 0.37 dec | 0.4353 |
| 12 | 0 | 12/12 | 1.1e-07 | 322 | 0.70 dec | 0.4647 |
| 12 | 1e-2 | 12/12 | 7.1e-02 | 450 | 0.85 dec | 0.5907 |

Three consequences. The residual resumes falling with D (5.9e-7 → 1.1e-7)
instead of sitting on a noise floor that was flat across D = 6…12. The
apparent bond-dimension drift in `D_peak` largely evaporates: 0.445 → 0.465
noiseless (4.5%) against 0.435 → 0.591 noisy (36%), so most of what looked
like variational non-convergence was the seed. And `n_eff` *falls* without
the noise, which narrows the admissible broadening window — the noise had
been inflating the effective mode count by smearing weight onto spurious
modes, so the narrower noiseless window is the honest one.

**Any result in this README dated before 2026-09-18 was computed at a fixed
point ≈ 6% off the thermofield double**, and the error was self-inflicted
rather than a finite-D limitation.

**Gauge pitfall, learned the hard way.** The centre tensors C^n must be
derived from the *same* A_L and A_R used everywhere else, via the bond
matrices (`build_centre_tensors`), never by running an independent
canonicalization sweep. Wherever the Schmidt spectrum is near-degenerate or
near-zero the singular vectors are numerically arbitrary, so an independent
sweep silently lands in a different gauge. The symptom is a
`fixed_point_residual` far *larger* than `||H_asym psi||` — impossible for
a projection, and the check worth keeping in mind for any new overlap built
on this machinery.

## Cost and convergence: how to choose L and D (2026-09-18)

This supersedes every earlier claim in this file about which of `L` and `D`
is the better lever.

**Cost.** The tangent dimension is `sum_n (4 D_{n-1} - D_n) D_n`, i.e.
`~ 3 L D^2` in the bulk; the formula reproduces all ten measured runs
exactly. `eigh` dominates everything else:

    t_eigh = 2.91e-9 * dim^2.942       (6 runs with dim > 2000, max resid 2.6%)
    memory = 1.48 * 2 * 16 * dim^2     bytes (matrix + eigenvectors + workspace)

so `t ~ L^2.94 D^5.88` and `mem ~ L^2 D^4`. **Doubling D costs 59x in time
and 16x in memory; doubling L costs 7.7x and 4x.**

**L and D are equally priced per unit of resolution, though.** The tangent
bandwidth is 16-19 at *every* L and D tested — it is set by local energy
scales, not extensive ones — so the mean level spacing is
`bandwidth/dim ~ 1/(L D^2)`. Halving it means doubling `dim` either way,
which costs the same 7.7x whichever lever is pulled. The choice between
them is therefore physical, not computational:

- **D controls variational error**, and saturates — see below.
- **L controls finite-size error**, and has not saturated.

So take D to its knee and spend everything else on L.

**Where D's knee is.** `convergence_probe.py` measures it without paying
for a run: `||P H_asym psi*||` needs only the tangent *vector*, so it skips
both the `dim^2` allocation and the `dim^3` eigh and costs seconds rather
than hours. Residual against D and against the number of imaginary-time
steps:

| L = 16 | steps 60 | 120 | 240 |
|---|---|---|---|
| D = 12 | 1.1e-07 | 3.0e-08 | 2.8e-08 |
| D = 16 | 3.3e-09 | 4.4e-10 | **1.3e-10** |
| D = 20 | 5.4e-09 | 8.3e-10 | 3.0e-10 |
| D = 24 | 2.8e-09 | 4.3e-10 | 9.8e-11 |

| | L = 24, 60 | L = 32, 60 | L = 32, 240 |
|---|---|---|---|
| D = 12 | 3.7e-07 | 6.3e-07 | 7.8e-08 |
| D = 16 | 1.1e-08 | 6.2e-09 | **1.2e-09** |
| D = 20 | 5.3e-09 | 2.4e-08 | 1.5e-09 |

**D = 16 is enough, because that is where it hits the build floor.** At
D = 12 the residual is manifold-limited — refining `dt` stops helping at
~3e-08. From D = 16 up it is limited by the imaginary-time build instead,
and D = 20 and D = 24 are no better than D = 16 at any step count. Past
240 steps there is scatter, not improvement (D = 16, L = 16: 1.3e-10 at
240, 4.7e-10 at 480, 1.9e-11 at 960 — roundoff, near what double precision
gives after `L * steps` TDVP updates).

**So `IMAG_STEPS` is the cheap lever, not `D`.** Raising it 60 -> 240 buys
25x in residual at D = 16 and costs seconds; raising D 16 -> 24 buys
nothing and costs 6x the eigh. The default 60 predates this measurement
and is too low for D >= 16.

**What that makes affordable** (this machine has 15.5 GB):

| run | dim | eigh | memory |
|---|---|---|---|
| L = 24, D = 16 | 15615 | 1.8 h | 10.8 GB |
| L = 32, D = 16 | 21759 | 4.7 h | 20.9 GB |
| L = 48, D = 16 | 43151 | 17.4 h | 51 GB |
| L = 32, D = 32 | 82943 | 10 days | 303 GB |

The last row is why large D is not the way: twenty times this machine's
memory, to buy directions the residual says are already resolved at D = 16.

**Caveat.** A small residual is necessary for a converged spectrum, not
sufficient — it says `psi*` is close to the fixed point, not that every
tangent mode is resolved. It is a cheap *screen* for D, not a proof.

## The current relaxation time (2026-09-17)

The question this was built for: **how long must you wait before energy
transport is diffusive?** Energy cannot flow diffusively until the current
has reached its constitutive value `j = -D ∇e`. Starting from local
equilibrium the current is zero — there are no currents in a local
equilibrium state, which is why the energy profile is stationary to
`O(t²)` — so it has to build up first, on the Maxwell–Cattaneo timescale
`τ ∂_t j + j = -D ∇e`. That `τ` is what `current_mid` measures.

Scan: `L = 8, 12, 16`, `D = 8`, `β = 0.1`, seed 0. Figures
`figures/D8_timescale_scan.png` and
`figures/L16_D8_{current,energy}_mid_{spectrum,response}.png`.

| L | τ(current) | τ(energy) | ratio | current `n_eff` | energy `n_eff` |
|---|---|---|---|---|---|
| 8 | 0.973 | 1.494 | 1.54 | 177 | 40 |
| 12 | 0.974 | 1.531 | 1.57 | 324 | 76 |
| 16 | 0.958 | 1.484 | 1.55 | 383 | 116 |

**1. τ(current) ≈ 0.96 and does not depend on L.** Flat to 2% over
L = 8–16, well converged, and sitting inside its `[t_zeno, t_heis]`
window. This is the number the exercise was for: it is *microscopic*, as a
current relaxation time must be. Against a profile-change time `L²/D` of
order 10²–10³ at these sizes, that is two to three orders of separation —
**the wait time is not the obstacle to extracting a diffusion constant.**

**2. The spectral densities have exactly the structure hydrodynamics
requires.** This is the stronger result, visible in the two `_spectrum.png`
figures. Weight within `|ω| < 0.25`, as a fraction and relative to what a
structureless distribution would put there:

| observable | fraction | vs. chance |
|---|---|---|
| energy density | 0.37 | 4.7x |
| current | 0.019 | 0.24x |

The conserved density piles weight up at zero frequency — a single mode at
ω = 0 carries ~9% of the energy weight on its own — while the current is
*depleted* there by a factor of four, and is bimodal with peaks at ω ≈ ±1.
A conserved quantity has a low-frequency pole and its current does not,
which is the precondition for a finite Green–Kubo `D ∝ A_j(ω→0)`. Nothing
here had to come out that way, so it is a real check on the whole
construction.

**3. Read τ from the 1/e crossing, not from `tau_fit`.** Eleven of sixteen
exponential fits in this scan fail at R² < 0.9, several returning NaN or a
τ an order of magnitude off (energy at L = 12: `tau_fit` = 30.3 with
R² = 0.003, against a crossing of 1.53). These responses fall fast and then
crawl through a slow oscillating tail, which is not one exponential.
`fit_relaxation_time`'s window logic does not cope, and the scan figure
therefore plots the crossing as the primary series and overlays fits only
where R² ≥ 0.9. Fixing the fitter is open work; the crossing needs no model
and is stable across L where the fit is not.

**4. Caveats, in order of how much they matter.**

- **These are local dephasing times, not hydrodynamic times.** Both τ's are
  O(1) and flat in L. A hydrodynamic relaxation would grow like L². What is
  being measured is the autocorrelation of a *local* operator at the fixed
  point, which dephases locally; the diffusive tail is a small long-time
  part of it and is invisible to a 1/e crossing. So the ratio τ(energy)/
  τ(current) ≈ 1.55 is **not** a hydrodynamic separation of scales, and
  should not be read as one — the comparison that matters for diffusion is
  τ(current) against `L²/D`, per finding 1.
- **The current's decay is not exponential.** Being bimodal in ω, its C(t)
  oscillates: down to 0.03 by t ≈ 1.5, back up to 0.35 at t ≈ 2.6, and on
  with slowly decaying revivals. τ = 0.96 is a first-crossing time, not a
  rate.
- **τ(current) is only 1.5x above its own `t_zeno` = 0.64**, so the window
  in which an exponential regime could exist is marginal for this
  observable — narrower than for the energy density.
**Followed up** in the next section: the zero-frequency weight in finding 2
*is* the Green–Kubo integrand.

## Green-Kubo: the diffusion constant (2026-09-18)

### The quantity

    kappa = (beta^2 / L) int_0^inf dt <J(t) J(0)>_c,
    c     = (beta^2 / L) Var(H),          D = kappa / c

so `beta^2` and `L` both cancel and

    D = int_0^inf dt <J(t) J(0)>_c / Var(H).

Temperature enters only through `psi*`. Both numerator and denominator are
extensive, and it is their ratio converging in `L` that has to be checked.

### It must be the *total* current

`single_copy_current` is the local `j_mid`, and its autocorrelator is only
the `r = 0` term of `sum_r <j_r(t) j_0(0)>`. The `r != 0` terms carry the
diffusive contribution, and the local term is a vanishing fraction of the
whole: `D_peak` from `j_mid` alone falls 0.028, 0.018, 0.013 across
L = 8, 12, 16, i.e. like `1/L`, while the total current gives 0.45.

`single_copy_total_current` builds `J_tot` as one operator. Summing the
three-site `j_l` over *every* site, dropping what falls off the open ends,
telescopes onto nearest-neighbour bonds:

    J_tot = J g sum_l ( y_l z_{l+1} - z_l y_{l+1} )

so it is a bond-dimension-4 finite-state machine at any `L`, rather than
the ~2L of an uncompressed `mpo.__add__` direct sum. Checked three ways:
against the dense operator (exact), against `sum_l j_l` built term by term
(exact), and against the windowed polarization identity
`sum_{l=a+1}^{b} j_l = i[H, sum_{l=a}^{b} l h_l] - a j_a + b j_{b+1}`
(3.6e-15). The *global* version of that identity is useless on an open
chain — Abel summation leaves `-(N-2) j_{N-1}`, which grows with `N`.

The denominator is `static_susceptibility` = `Var(H)`, from exact MPO
algebra and deliberately *not* tangent-projected: it is a thermodynamic
quantity, so keeping it exact confines all projection error to the
numerator. `static_variance` also gives the capture ratio
`sum_k w_k / Var(O)`, which is **1.0000 to machine precision** for every
observable here. The Green-Kubo numerator therefore carries no *static*
truncation error. (Not a vacuous test: a weight-L product of random
single-site rotations scores 0.66. And capture = 1 says only that `C(0)` is
exact — the *evolution* is still tangent-projected.)

### The integral is closed form; that is not the problem

    I(t) = int_0^t C = sum_k w_k sin(omega_k t) / omega_k

exactly, for any `t`, from the one diagonalization already done. No
quadrature, no time stepping, nothing to converge. (`green_kubo_integral`
writes it with `sinc` so the `omega = 0` modes contribute `w_k t` without a
near-zero branch.)

The problem is that on a finite chain the spectrum is a discrete set of
deltas, so `C(t)` never decays — it dephases and then recurs, and `I(t)`
oscillates about its mean forever instead of converging. Averaging over the
nominal plateau window gives `D = 0.048 +/- 0.076` at L = 16: consistent
with zero and useless. Panel c of the figures shows it.

### The broadened estimator

Regulate the finite-size problem at its source by giving each delta a
Lorentzian width `eta`:

    I(eta) = int_0^inf dt e^{-eta t} C(t)
           = sum_k w_k eta / (omega_k^2 + eta^2)

also exact, and with no time grid at all. `eta` must be large enough to
wash out the level discreteness and small enough not to eat into the
correlator's own decay, which is `broadening_window`:

    safety * spacing  <  eta  <  1 / (safety * tau)

existing only when `tau * spacing < 1/safety^2`. **A plateau in `D(eta)`
across that window is what has to replace convergence of the time
integral**, and `diffusion_constant` reports `log_slope = d log D/d log eta`
as the test: `|slope| << 1` means the value means something.

### Results

`figures/D8_green_kubo.png` (L scan), `figures/L16_green_kubo.png` (bond
scan). `D_peak` runs 0.357, 0.412, 0.445 over L = 8, 12, 16 and 0.448 →
0.465 over D = 6 → 12; the admissible `eta` window opens from empty to 0.70
decades, entirely from its lower edge as the level spacing shrinks.

**There is no plateau anywhere.** The slope sits at 0.54-0.68 and does not
improve as the window widens, so `D ~ 0.45` is a crossover bound, not a
measurement. Two specific reasons it is biased *high*:

- **`D_peak` is read outside its own validity window.** `eta_peak` is
  0.35, 0.35, 0.34 against window upper bounds of 0.140, 0.136, 0.135 — the
  maximum always sits *above* the admissible range, where the Lorentzian is
  already cutting into the correlator's decay.
- **`D_win` tracks the window, not the transport.** It falls 0.313, 0.252,
  0.217 as the window widens, because widening extends it down the rising
  flank toward the finite-size floor. Kept in the figures and labelled as
  such: watching it slide while `D_peak` holds still is the evidence that
  the window moves and the physics does not.

Not yet compared against the `D` from the nonlinear Gaussian-width fits in
`qtensor.visualise`, which remains the external validation.

### Zero-frequency weight: the current has none, the density must

The `omega = 0` weight is the sharpest thing the spectrum says, and it says
opposite things about the two observables.

**The current carries no Drude weight.** Below `|omega| < 1e-10` it is ~0
for every L >= 8, below 1e-3 at most 5e-7, below 1e-2 at most 2.3e-4. So
there is no ballistic delta and `D` is at least finite in principle.

**The energy density carries an exact one, and must.** `conserved_fraction`
measures it: 0.156, 0.100, 0.074 at L = 8, 12, 16, and it is
perfectly independent of bond dimension (0.07396 at every D from 6 to 12).
This is the Mazur bound. An observable overlapping a conserved quantity
cannot relax to zero — split it,

    h_mid = [Cov(h_mid, H)/Var(H)] H + h_perp,

and the first term is a constant of the motion, contributing the same
amount to `<h(t)h(0)>` at every `t` including infinity. Only `h_perp`
dephases. Physically: a bump of energy on one bond is partly "the chain now
holds more energy", which has nowhere to go; it spreads until uniform, and
uniform across `L` bonds still leaves ~1/L of it on the middle bond
forever.

Verified, not assumed. `H_asym (H(x)I)|psi*> = [H(x)I - I(x)H,
H(x)I]|psi*> = 0` because the two copies commute, so `(H(x)I)|psi*>` is an
*exact* zero mode. `H_tangent` has exactly one zero eigenvalue at
L = 8, 12, 16; the energy tangent vector lies in it to ten digits; and
`h_mid`'s weight there reproduces `Cov(h,H)^2/(Var h Var H)` to six digits
(0.1555070 vs 0.1555065 at L = 8). It scales as `c/L` with
`c -> Cov(h,H)/Var(h) = 1.13`, against 1.14 from the `beta -> 0`
arithmetic. Exactly one zero mode and not two because on a thermofield
double `(H(x)I)|psi> = (I(x)H^T)|psi>`.

**So the floor is a passed conservation test, not a leak.** Its absence, or
a drift with bond dimension, would have meant the tangent flow was losing
energy. `fit_relaxation_time` and `crossing_time` both take `c_inf` and
work on `dephasing_response(C, c_inf) = (C - C_inf)/(1 - C_inf)`; the
decomposition `C = C_inf + (1 - C_inf) C~` is exact, and only `C~` has a
relaxation time.

### Why the fit window is what it is (2026-09-18)

`fit_relaxation_time` takes `[t_zeno, t_end]` with

    t_end = min( t_max, 3 * t_1/e, last time above `floor`, 1/spacing )

and the last two both changed once the energy density was looked at
properly.

**The floor cut is on the last time above the floor, not the first time
below.** These coincide for a monotone decay and not otherwise. `C~` for
`energy_mid` plunges to ~1e-3 at its first zero and then *recovers to 0.3*
before decaying slowly, so "first crossing below 0.05" was stopping at the
first zero of an oscillation and discarding everything after it. At L = 16
that cut the window at t = 2.1 out of a 700-long trace.

**Nothing slower than `1/spacing` is resolvable, and the code now refuses
it.** An exponential of rate `Gamma = 1/tau` is a Lorentzian of width
`Gamma`; resolving it needs modes inside that width, and the count is
`Gamma/spacing = 1/(tau*spacing)`. This is *2 pi tighter than `t_heis`*,
which `timescales` defines as `2 pi/spacing` — so passing `t_max = t_heis`
alone permits fits four times slower than the spectrum can resolve.

That matters because the full trace does show a slow tail, and it is
tempting to fit it. Fitting the envelope of `C~` over `[t_recovery,
t_heis]` gives:

| D (at L = 16) | t_heis | 1/spacing | tau_envelope | modes per linewidth |
|---|---|---|---|---|
| 6 | 155.6 | 24.8 | 108.8 | 0.23 |
| 8 | 176.0 | 28.0 | 218.9 | 0.13 |
| 10 | 214.5 | 34.1 | 283.8 | 0.12 |
| 12 | 233.3 | 37.1 | 127.6 | 0.29 |

Every one of those has **less than one mode per linewidth**, so none is a
lineshape — it is the beating of a handful of discrete levels. The proof
is in the scatter: L and the physics are fixed across that table, only the
box changes, and the answer moves by a factor of 2.6 without a trend.
Restricting to the genuinely resolvable window `[t_recovery, 1/spacing]`
does not rescue it either — only 5–8 envelope peaks fit inside, and tau
still scatters 28 / 32 / 85 / 67.

So the window is small because the resolvable window *is* small, and the
O(100) tail is not being ignored by oversight: it is below the resolution
of the spectrum that produced it.

**Consequences for the reported numbers.** With the floor cut fixed, the
`energy_mid` window at L = 16 runs to t = 4.2 instead of 2.1, takes in the
recovery as well as the plunge, and honestly reports that one exponential
does not describe it: R² falls from 0.90 to 0.07. That is the right
outcome. The currents are unaffected — their windows extend too, but the
extra samples have `C < 0` and are already excluded, so tau and R² are
unchanged to three digits (`current_mid` 0.320 at R² = 0.933,
`current_total` 0.681 at R² = 0.857 at L = 16). Their taus, ~0.3–0.7
against a limit `1/spacing` of ~46, are resolvable by two orders of
magnitude, which is why they were never the problem.

### The spectral-density limit: why it is the right object

Everything above is one statement about the spectral density
`A_O(omega) = sum_k w_k delta(omega - omega_k)`, and saying it that way is
more useful than any of the time-domain fits.

The Green-Kubo integral *is* the zero-frequency spectral density:
`int_0^inf C(t) dt = pi A(0)`. More precisely, for `A(omega) = c|omega|^s`
near zero, substituting `omega = eta u` in

    I(eta) = int domega A(omega) eta/(omega^2 + eta^2)

gives `I(eta) = c K_s eta^s` with `K_s = int |u|^s/(u^2+1) du`. Therefore

    d log D / d log eta  =  d log A / d log omega.

**The flatness test is a measurement of the spectral density's exponent.**
`|slope| < 0.1` means `A(omega)` is flat near zero, which is the definition
of diffusive; and the two limits the figures show are both forced.
At small `eta` the Lorentzian is narrower than the level spacing and sees a
gap, `A -> 0`, so slope `-> +1`. At large `eta` it integrates the whole
band, `I -> sum_k w_k / eta`, so slope `-> -1`.

Read that way, the measured 0.54-0.68 is not merely "no plateau": it says
`A_J(omega) ~ omega^{0.5..0.7}` over the accessible window — the current's
spectral density is *vanishing* as `omega -> 0`, not approaching a
constant. Taken literally that is `D = 0`, a subdiffusive or insulating
chain. It cannot be taken literally, because the window bottoms out at the
level spacing, and a discrete spectrum has no weight at small `omega` for
the trivial reason that it has no *modes* there.

Disentangling those two is the measurement worth making, and it needs no
new machinery — only a different view of weights already computed:

1. **Bin `A_J(omega)` directly** against `omega`, with bins wider than the
   level spacing, and look at the shape rather than at one number. Whether
   `A_J` bends over to a constant, keeps falling as a power, or has a dip
   is visible there and invisible in `D(eta)`.
2. **Check the exponent against `L` and `D` separately.** A finite-size gap
   should fill in as the spacing shrinks (so the exponent should fall
   toward 0 with either knob); genuine subdiffusion should not.
3. **Cross-check on the density.** A diffusive system has a `t^{-1/2}`
   tail in the energy-density correlator above its Mazur floor, i.e.
   `A_h(omega) ~ |omega|^{-1/2}`. That is a *divergence*, so it is much
   easier to see than a flat `A_J`, and it is an independent route to the
   same answer. Fitting the time-domain tail for it gave slopes -0.05 to
   -0.63 against the predicted -0.5 (R^2 0.21-0.44) — inconclusive,
   because the tail oscillates about its envelope and the fit fights the
   oscillation. Frequency space has no such problem.

This is why the spectral route is preferred over repairing the time-domain
fit. The oscillations that wreck an exponential fit are just the beating of
discrete `omega_k`; smoothing resolves them instead of fighting them, there
is no model to choose and no window to tune, and the finite-size limit is
explicit — nothing below the level spacing is knowable, and that shows up
as the edge of the plot rather than as a plausible number.

### Measured (2026-09-18, morning): D_peak is an overestimate

> **Read the next section before using any number here.** The L = 24
> and L = 32 runs refute the bound `D <= 0.045` and the claim that the
> current's exponent was measured rather than extrapolated. What
> survives is the kernel argument, the zero-mode argument, and the
> direction of the `D_peak` result.

`spectral_density` and `spectral_exponent` do it. Figures
`figures/L16_spectral_density.png` (bond scan) and
`figures/D8_spectral_density.png` (L scan).

**Two things had to be got right first, and neither is cosmetic.**

*The kernel must be Gaussian, not Lorentzian.* A Lorentzian has `w^-2`
tails, so the current's band weight around `w ~ 0.5` leaks down into
`w ~ 0.05` as `~ W eta / w^2`. The tell is `A(w)` coming out proportional
to the width at fixed `w`, and that is exactly what happens: `A/eta` = 55,
53, 68 at D = 8, 10, 12, i.e. the entire apparent low-frequency signal was
the kernel's own tails. A Gaussian leaks `exp(-w^2/2 s^2)`. (The Lorentzian
is still correct for Green-Kubo, where it is not a kernel choice but the
physical broadening of `int e^{-eta t} C dt`; `green_kubo_broadened`
satisfies `A_lorentzian(0; eta) = (2/pi) I(eta)` to machine precision.)

*Exact zero modes must be dropped.* They are a delta, not continuum. For
the energy density the Mazur weight is large enough to leak into the
low-frequency region and bias the exponent **negative** — the direction
that makes a non-diffusive system look diffusive. Dropping it moves the
exponent by +0.08.

**The result that survives: `D_peak` is an overestimate.** Reading
`(pi/2) A_J(w_min)/Var(H)` at L = 16 gives 0.85, 0.36, 0.064, 0.054 across
D = 6, 8, 10, 12 — it collapses as the resolution improves, while `D_peak`
sits at 0.45 throughout. That is what "read outside its own validity
window" was always going to mean.

The quantitative bound that went with it (`D <= 0.045`, from a weight
budget below `w_min`) is **withdrawn**: the inequality is sound, but the
spectrum it was evaluated on was not converged. The same budget at L = 32
gives `D <= 0.107` and is still loosening. See the next section.

**The two observables disagree**: the current says `A_J -> 0` (`D = 0`),
the density says `A_h ~ w^-1/2` (`D > 0`). At the time the current looked
like the trustworthy one, because only 0.24% of its weight sat below its
resolution limit against 26% for the density.

> **That reasoning was wrong, instructively.** A small *unresolved* weight
> is not a *converged* one. Raising L put ten times more weight below the
> limit. Little weight down there says the kernel is not being asked to
> extrapolate; it says nothing about whether the modes that belong there
> exist yet.

### Measured (2026-09-18, evening): L = 24 and L = 32 at D = 12

Runs: L = 24 (dim 8879, 22 min) and L = 32 (dim 12335, 57 min), both at
D = 12, beta = 0.1, fixed-point residual 3.7e-07 and 6.3e-07. Figures
`figures/size_comparison_spectral.png` (each run at its own resolution)
and `figures/size_comparison_collapse.png` (all runs at one common
kernel).

**The collapse test.** Evaluate `A(w)` at the same `w` with the same
kernel width across runs and see whether the curves agree. This is the
question that has to be settled before any per-run exponent means
anything, and it is *not* what the earlier figures asked — they read each
run at its own best resolution, which smooths the runs by different
amounts and separates them at low frequency for that reason alone.

The current has to be compared as `D(w) = (pi/2) A_J(w)/Var(H)` rather
than as `A_J`: `J_tot` is a sum over bonds, so `A_J ~ L` and raw curves at
different L are offset by that factor before any physics enters. Var(H) is
extensive too and cancels it. `Var(J)/L` = 0.510, 0.521, 0.527 at
L = 16, 24, 32 confirms the extensivity directly.

| ratio max/min over the common band | L = 16, 24, 32 | L = 24 vs 32 only |
|---|---|---|
| energy density `A_h` | 1.30 median, 2.06 worst | **1.01 median, 1.20 worst** |
| current `D(w)` | 2.67 median, 10.0 worst | 1.08 median, 1.63 worst |

**The energy density has converged in L.** L = 24 and L = 32 agree to
about 1% across the whole resolved band — a genuine collapse, and the
first evidence in this project that `A_h` is a property of the chain
rather than of the box. L = 16 is visibly off, so the convergence happened
between 16 and 24.

**The current's low-frequency collapse at L = 16 was finite-size.** At
`w = 0.072`, `D(w)` reads 0.085 at L = 16 but 0.849 and 0.823 at L = 24
and L = 32 — a factor of ten, in the direction that had been read as
evidence for `D -> 0`. The fall-off is a depletion that recedes as the box
grows, not a property of the chain.

**The weight budget, evaluated at a *fixed* window rather than at each
run's own limit**, shows the same thing without any kernel. Fraction of
`Var(J)` carried by modes below `w = 0.05`, and the bound it implies:

| | L=16 D=8 | L=16 D=12 | L=16 D=16 | L=24 D=12 | L=32 D=12 |
|---|---|---|---|---|---|
| weight below 0.05 | 0.20% | 0.40% | 0.83% | 0.77% | 0.81% |
| implied `D <=` | 0.027 | 0.052 | 0.108 | 0.101 | **0.107** |

The bound **loosens monotonically in both L and D** and shows no sign of
saturating. It is a valid bound at every row; it is simply not yet a
useful one. What still stands is its direction — `D_peak` is 0.465, 0.513,
0.537 at L = 16, 24, 32, so the crossover estimate remains several times
larger than anything the low-frequency weight supports, and the gap has
narrowed from ~10x to ~5x rather than closing.

**Neither exponent is converged.** The current reads +0.67, +1.39, +1.41,
+0.70, +0.71 across D = 8-16 at L = 16, and +1.41, +0.20, +0.55 across
L = 16, 24, 32 at D = 12 — scatter of about +/-0.6 with no trend. The
density reads -0.56, -0.90, -0.80 across L = 16, 24, 32, consistently
steeper than the diffusive -1/2 and consistently with ~24% of its weight
below the limit. The `w^1.41` quoted in the previous section was two
points coinciding.

> **The D = 14, 16 runs also miss the collapse, and this section
> originally read that as those runs being unreliable. That was wrong** —
> see "Cost and convergence" above. Every hard diagnostic says they are
> the *better* runs: exactly one zero mode at every D, the Mazur floor
> D-independent to six digits, static capture 1.0000000, and a residual
> that keeps falling (1.1e-07, 2.9e-08, 3.3e-09 at D = 12, 14, 16).
> Disagreement with D <= 12 is equally consistent with D = 12 being
> unconverged, which the residual says it is. **The corollary is
> uncomfortable: L = 24 and L = 32 at D = 12 carry residuals of 3.7e-07
> and 6.3e-07, comparable to L = 16 at D = 8, so the collapse above may be
> two calculations agreeing at the same under-converged bond dimension.**
> Settling it needs one run at L = 24, D = 16.

**"Resolve the slowest hydrodynamic mode" was the wrong target.** The
resolution limit does improve with L — `w_min` for the density is 0.1077,
0.0561, 0.0392 at L = 16, 24, 32, falling as `L^-1.46`. But the slowest
diffusive mode sits at `D (2 pi/L)^2` and falls as `L^-2`, which is
faster, so the ratio of what must be resolved to what can be is 14, 16, 20
at L = 16, 24, 32. **The target recedes faster than the resolution
improves**, and no accessible L reaches it. Panel c of the collapse figure
is this statement.

The answerable question is the one the collapse test asks: does `A_h(w)`
agree between system sizes over the band that *is* visible? It does, from
L = 24 up. The honest gap is then interpretation rather than resolution —
whether a band that far above the hydrodynamic window should look like
`w^-1/2` at all is a physics question. Measuring a finite-`q` density
correlator, and fitting `w(q) = D q^2` at several small-but-nonzero `q`,
would sidestep the `w -> 0` limit entirely.

## Code

`tangent_hamiltonian.py`:

- `build_tangent_hamiltonian(psi_uniform, H_asym, max_bond_dim)` — wrapper,
  returns `(H_tangent, basis_index_map)`.
- `canonicalize_and_build_environments` — three-sweep canonicalization plus
  the `L_con`/`R_con`-style environment families.
- `build_null_space_tensor` — `V_L^n`, with an optional gauge-condition
  check.
- `project_H_onto_tangent_basis` — one column of the matrix: all bra sites
  for one ket direction, via mixed-gauge environments and
  `apply_Heff_parts`.
- `assemble_tangent_hamiltonian` — stacks the blocks.

`convergence_probe.py` — picks `D` and `IMAG_STEPS` before paying for a
run. `probe(L, D, steps)` returns `||P H_asym psi*||` using only the
tangent *vector*, skipping the `dim^2` assembly and the `dim^3` eigh, and
`estimate(dim)` gives the eigh time and peak memory a full run would cost.
Seconds against hours; this is what "Cost and convergence" was measured
with.

    python lyapunov/relaxation/convergence_probe.py --L 32 --D 12,16,20 --steps 60,240

Verified at L=6, D=6, d=2 (random state, tilted Ising MPO): dimension 59,
matching `Σ_n (d·D_{n-1} − D_n)·D_n` computed independently from the bond
dimensions, and Hermitian to 6e-16 — a strong check, since the `n < m` and
`n > m` blocks are built by different code paths (mixed left vs mixed right
environments) and come out exact conjugate transposes.

`response.py`:

- `single_copy_onesite` / `single_copy_energy_density` /
  `single_copy_current` — observable MPOs acting on the physical copy only
  (`kron(A, I)`, matching `thermofield.single_copy_expectation`'s
  convention).
- `single_copy_current(site)` is the energy current *through* `site`,
  `J g (y_i z_{i+1} - z_{i-1} y_i)`, a three-site operator. It is the
  current that satisfies continuity with `single_copy_energy_density`,
  `d<h_l>/dt = <j_l> - <j_{l+1}>` — note `h` is indexed by bond and `j` by
  site. The symmetrization of the density fixes the form: for the
  unsymmetrized convention `h_l = J z_l z_{l+1} + 2 a_l`, which is what
  `operators.ising_commutator` assumes, the current is instead
  `-2 J g z_{l-1} y_l`, and pairing either density with the other's current
  breaks continuity at O(1). Only `g` drives transport; the longitudinal
  field commutes with the coupling and drops out. Verified against dense
  `i[H, h_l]` on a 6-site chain (residual 8e-16) and the MPO against its
  dense form exactly.
- `observable_tangent_vector` — `v_i = <b_i|O|psi*>`, the one vector the
  whole response is built from. With the kick and the measured observable
  both equal to O, the weights are `|u_k|^2` with `u = U† v`: manifestly
  non-negative, so A_O(ω) is a genuine spectral density.
- `single_copy_total_current(sites)` — `J_tot = sum_l j_l` as a single
  bond-dimension-4 finite-state machine, needed because Green–Kubo is a
  statement about the total current (see that section).
- `single_copy_hamiltonian(sites)` — `H ⊗ I_aux`, which
  `thermofield.thermofield_hamiltonian` does not provide on its own.
- `static_variance(psi, O)` / `static_susceptibility(psi, sites)` — exact
  `Var(O)` from MPO algebra: the Green–Kubo denominator, and the yardstick
  for the tangent capture ratio `sum_k w_k / Var(O)`.
- `green_kubo_integral(omega, weights, times)` — the running `I(t)` in
  closed form; `green_kubo_broadened(omega, weights, eta)` — the
  Lorentzian-regulated `I(eta)`, which is the estimator to trust.
- `broadening_window`, `diffusion_constant` — the admissible `eta` range
  and the flatness test over it.
- `spectral_density(omega, weights, omega_eval, width)` — the one-sided
  `A(w)`, Gaussian-smoothed (see above for why not Lorentzian), with exact
  zero modes dropped. `spectral_exponent` fits `d log A/d log w` over the
  resolvable range and reports how much weight lies below it, which is the
  diagnostic that says whether the exponent is measured or extrapolated.
- `timescales` — the `t_zeno << t << t_heis` window bounds, from the
  weighted spectrum rather than the raw dimension (modes carrying no
  overlap cannot dephase anything).
- `conserved_fraction(omega, weights)` — `C(inf)`, the weight on exact zero
  modes, which is the Mazur floor for an observable overlapping a conserved
  quantity; `dephasing_response(C, c_inf)` rescales it away.
- `fit_relaxation_time`, `crossing_time` — rate extraction, both taking
  `c_inf` so they measure against the right asymptote.
  `fit_relaxation_time` also takes `spacing` and refuses any τ beyond
  `1/spacing` as unresolvable, reporting the rejected window in
  `t_fit_end` so it stays visible.

`run_relaxation_scan.py` — the L scan driver. Also computes
`||P H_asym psi*||` as a fixed-point diagnostic, which is free: it is
`observable_tangent_vector` with O = H_asym.

`plots.py` — eyeball diagnostics, pure consumers of a `run_one` result:

- `plot_spectral_weights(omega, weights, ...)` — A_O(ω) against the DOS
  (both unit area, one axis), per-mode weights on a log scale, cumulative
  weight. Optionally overlays the Lorentzian of HWHM 1/τ implied by a fitted
  τ, which is the direct test of whether that τ is a lineshape width.
- `plot_response(times, response, ...)` — C(t) over the full trace with the
  fit window and `t_zeno`/`t_heis` marked; then, zoomed to the decay,
  log|C| and the running τ(t) = −1/(d ln C/dt). A real exponential regime
  is a plateau in τ(t). The zoom is needed because `run_one` integrates to
  3·t_heis, hundreds of decay times.
- `spectral_weights_from_result` / `response_from_result` / `plot_result`
  unpack a result dict directly.

It also runs from the terminal (repo root, Anaconda base env):

    python lyapunov/relaxation/plots.py --L 8 --D 8 --obs energy_mid --seed 0

`--obs` takes `z_mid`, `x_mid`, `energy_mid` (the Hamiltonian term on the
centre bond), `current_mid`, `current_total` or `all`. `--pickle
scan_results.pkl` plots a saved scan instead of re-running, `--green-kubo`
writes the Green–Kubo figure from a scan varying either L or D, and
`--show` opens the figures. PNGs go to
`figures/L{L}_D{D}_{obs}_{spectrum,response}.png`. `--seed` is a leftover
from the noise-seeded build and no longer matters: with `SEED_NOISE = 0`
the state build is deterministic. It used to matter a great deal (τ_fit for
`z_mid` at L=8 ranged 8.7–12.7 across three unseeded runs), which in
hindsight was the noise announcing itself.

Not yet written: the wavevector-resolved energy density needed to turn
`Γ_q` vs `q²` into a diffusion constant.

## Open questions

1. **Role of the existing MPO-trace machinery.** `commutator_trace.ipynb`
   prototypes traces like `Tr([H_i,H_j]²)` and `Tr([H_i,H][H_j,H])` via
   `mpo.trace()` / `mpo.__matmul__` / `extensive_twosite_local_term` /
   `extensive_as_terms`. Does this feed into the weights `w_k`, or is it
   unrelated groundwork?
2. **Validation.** Cross-check the predicted rate against a direct
   `finiteTDVP.tdvp` run with a small non-uniform perturbation.
3. **Auxiliary-gauge modes** (caveat above) — needs resolution before the
   spectrum can be read physically.
4. **Nothing relaxes yet at L=8, D=8** (from the `plots.py` figures, seed 0;
   `z_mid` has since been dropped from the default scan, so this is a
   record), and the fitted τ values shouldn't be read as rates. `z_mid`:
   C(t) levels off at ~0.2–0.25 and stays there past t_heis, so τ_fit ≈ 9
   describes the
   approach to a plateau. `energy_mid`: C(t) crosses zero at t ≈ 2.3 and
   then oscillates between 0 and ~0.3 indefinitely. τ_fit = 0.36
   (R² = 0.96) disagrees with τ_1/e = 1.18 because the fit window closes
   before the first zero and fits the Zeno shoulder. Its weight sits in
   ω ∈ [−2, 2], comb-like, n_eff = 53 of 959 modes. That is much narrower
   than both the DOS and the implied Lorentzian (HWHM 2.75), which points
   to the discrete-spectrum obstruction above rather than a slow rate. The
   running τ(t) never plateaus for either observable.

   *Partly settled since.* τ_1/e for `energy_mid` is flat in bond
   dimension (1.401 / 1.408 / 1.395 / 1.400 at D = 6/8/10/12). Which of L
   and D buys resolution is answered in "Cost and convergence": neither —
   they cost the same per unit of level spacing, and D saturates at 16.

   The exponential fit is settled the other way: **there is no
   exponential regime for `energy_mid` and the code now says so.** See
   "Why the fit window is what it is" below. The intermediate claim that
   subtracting `C_inf` rescued it (τ_fit 11.7 → 0.42, R² 0.06 → 0.97) was
   an artefact of a window that stopped at the first zero; with that fixed
   the same fit reads τ = 2.8 at R² = 0.05. The floor subtraction was still
   right — it is what makes τ_1/e converge — but it did not make the energy
   density relax.

5. **Is `A_J(ω) → 0` physical, or a finite-size gap?** **Finite-size, so
   far as L = 32 can tell.** `D(ω = 0.072)` reads 0.085 at L = 16 but 0.85
   at L = 24 and L = 32, so the low-frequency depletion recedes with the
   box. The weight-budget bound loosens monotonically — `D ≤ 0.027, 0.052,
   0.107` as the calculation improves — and has not saturated, so no upper
   bound on `D` can be quoted yet. The exponents are not converged either
   (±0.6 scatter for the current). `A_h(ω)` agrees to ~1% between L = 24
   and L = 32 — but both are at D = 12, whose residual is 100x above the
   D = 16 floor, so that agreement is not yet evidence of convergence.
   **The next run is L = 24, D = 16** (1.8 h, 10.8 GB), which is the first
   point with both L and D past their knees. See "Cost and convergence"
   and "Measured (2026-09-18, evening)".
