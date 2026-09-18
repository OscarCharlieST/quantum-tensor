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
| `L_VALUES` | 4, 8, 12, 16 | |
| `IMAG_STEPS` | 60 | TDVP steps for the imaginary-time build |
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

## The current relaxation time (2026-09-17)

The question this was built for: **how long must you wait before energy
transport is diffusive?** Energy cannot flow diffusively until the current
has reached its constitutive value `j = -D ∇e`. Starting from local
equilibrium the current is zero — there are no currents in a local
equilibrium state, which is why the energy profile is stationary to
`O(t²)` — so it has to build up first, on the Maxwell–Cattaneo timescale
`τ ∂_t j + j = -D ∇e`. That `τ` is what `current_mid` measures.

Scan: `L = 4, 8, 12, 16`, `D = 8`, `β = 0.1`, seed 0. Figures
`figures/D8_timescale_scan.png` and
`figures/L16_D8_{current,energy}_mid_{spectrum,response}.png`.

| L | τ(current) | τ(energy) | ratio | current `n_eff` | energy `n_eff` |
|---|---|---|---|---|---|
| 4 | 0.903 | 3.024 | 3.35 | 30 | 7 |
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
- L = 4 is not converged for either observable and should be ignored.

**Open lead.** The zero-frequency weight in finding 2 *is* the Green–Kubo
integrand. Turning `A_j(ω→0)` into a diffusion constant, and comparing it
with the `D` from the nonlinear Gaussian-width fits in `qtensor.visualise`,
is the natural next step and needs no new machinery — but it is a separate
deliverable and has not been done.

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
- `timescales` — the `t_zeno << t << t_heis` window bounds, from the
  weighted spectrum rather than the raw dimension (modes carrying no
  overlap cannot dephase anything).
- `fit_relaxation_time`, `crossing_time` — rate extraction.

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
4. **Nothing relaxes yet at L=8, D=8** (from the `plots.py` figures, seed 0),
   and the fitted τ values shouldn't be read as rates. `z_mid`: C(t) levels
   off at ~0.2–0.25 and stays there past t_heis, so τ_fit ≈ 9 describes the
   approach to a plateau. `energy_mid`: C(t) crosses zero at t ≈ 2.3 and
   then oscillates between 0 and ~0.3 indefinitely. τ_fit = 0.36
   (R² = 0.96) disagrees with τ_1/e = 1.18 because the fit window closes
   before the first zero and fits the Zeno shoulder. Its weight sits in
   ω ∈ [−2, 2], comb-like, n_eff = 53 of 959 modes. That is much narrower
   than both the DOS and the implied Lorentzian (HWHM 2.75), which points
   to the discrete-spectrum obstruction above rather than a slow rate. The
   running τ(t) never plateaus for either observable. Whether larger D
   densifies the weighted spectrum enough (the feasibility estimate says D,
   not N, is the lever) is the next thing to check.
