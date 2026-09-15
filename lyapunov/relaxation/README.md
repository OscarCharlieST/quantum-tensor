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
| `BETA` | 1.0 | see below |
| `D` | 8 | same for every L, so the tangent dimension grows only through L |
| `L_VALUES` | 4, 8, 12, 16 | |
| `IMAG_STEPS` | 60 | TDVP steps for the imaginary-time build |
| `SEED_NOISE` | 1e-2 | rank seeding, see below |

**On β = 1.** Hotter than the β ≈ 1e-2 to 1e-3 used in `active.ipynb`, and
deliberately so. At β ≈ 1e-2 the thermofield double is barely entangled —
its Schmidt spectrum at L=4 is `[1, 5e-3, 1e-8, 7e-11]`, i.e. effectively
rank 2 — so most of a D=8 tangent space would be built on numerically null
directions. β = 1 gives a genuinely rank-filling state while keeping the
fixed-point residual small (~1e-2 against a tangent bandwidth of ~5). If
the relaxation rates turn out to depend strongly on β, that is a physical
result worth having, not a nuisance.

**On the seeding noise.** `inf_T_thermofield` returns a rank-1 state
zero-padded to bond dimension D, and single-site TDVP cannot grow the
Schmidt rank, so without noise the imaginary-time evolution stays rank 1.
The noise is what lets it fill the bond dimension at all. It does mean
psi_uniform is only approximately the thermofield double, which is exactly
what `fixed_point_residual` measures.

**Gauge pitfall, learned the hard way.** The centre tensors C^n must be
derived from the *same* A_L and A_R used everywhere else, via the bond
matrices (`build_centre_tensors`), never by running an independent
canonicalization sweep. Wherever the Schmidt spectrum is near-degenerate or
near-zero the singular vectors are numerically arbitrary, so an independent
sweep silently lands in a different gauge. The symptom is a
`fixed_point_residual` far *larger* than `||H_asym psi||` — impossible for
a projection, and the check worth keeping in mind for any new overlap built
on this machinery.

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

- `single_copy_onesite` / `single_copy_energy_density` — observable MPOs
  acting on the physical copy only (`kron(A, I)`, matching
  `thermofield.single_copy_expectation`'s convention).
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
