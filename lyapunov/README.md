# Perturbations of the thermofield fixed point

Umbrella folder for two related subprojects, both concerned with how a
purified, spatially non-uniform-temperature state behaves near the
uniform-temperature state under the *antisymmetric* thermofield
Hamiltonian. They split because the original single question turned out to
have two genuinely different answers (see "Why this splits in two" below).

- [`relaxation/`](relaxation/) — **active.** Relaxation rates of local
  observables, from the spectrum of H_asym projected onto the MPS tangent
  space at the fixed point. Hermitian, exactly diagonalizable, code
  started.
- [`tdvp_lyapunov/`](tdvp_lyapunov/) — **not started.** Genuine Lyapunov
  exponents of the nonlinear TDVP flow on the MPS manifold, which requires
  linearizing along a trajectory rather than at the fixed point.

(The folder is still named `lyapunov` for continuity with where the idea
started; only the second subproject really computes Lyapunov exponents.
Worth renaming if the relaxation strand becomes the main line of work.)

## Background both strands build on

- **Purification / thermofield doubling.** A (locally) thermal state is
  represented as a pure MPS on a doubled physical dimension (`d -> d^2`), via
  `qtensor.thermofield` (`inf_T_thermofield`, `near_thermal`,
  `near_thermal_delta_function`, `near_thermal_first_order_deformed`) and
  `qtensor.simulation.finiteTDVP.tdvp` for the imaginary-time evolution that
  builds these states. A *uniform*-temperature profile gives the standard
  thermofield double at inverse temperature β; a *non-uniform* profile
  (`near_thermal`'s `profile` argument) stitches together a state that is
  only locally/approximately thermal, with β varying along the chain.
- **Symmetric vs antisymmetric thermofield Hamiltonian.** Given a physical
  Hamiltonian H, `operators.thermofield_hamiltonian(H, asym=...)` (and the
  near-duplicate in `thermofield.py`) builds H⊗I + a·(I⊗H) on the doubled
  space, with `a=+1` (symmetric) or `a=-1` (antisymmetric). Prior work in
  this repo (see the "asymmetric evolution" commits) already noted the
  antisymmetric case has a much longer coherence time in direct simulation —
  consistent with the fixed-point property below.
- **Fixed point property.** For the *uniform*-temperature thermofield double
  (the genuine purification of e^{-βH}/Z), evolving under the antisymmetric
  Hamiltonian H⊗I − I⊗H leaves the state exactly invariant — this is just
  the statement that a thermal state is stationary (KMS condition). So the
  uniform-temperature purification is an exact fixed point of the
  antisymmetric flow. A *non-uniform*-temperature purification is not an
  eigenstate of that flow, and will evolve.

## Why this splits in two

The original plan was: linearize around `psi_uniform`, diagonalize the
generator, read off Lyapunov exponents as the real parts of its
eigenvalues. Quantum evolution is exactly linear in the state, so no
two-trajectory finite-difference machinery is needed — a genuine
simplification over the classical case.

**But H_asym is exactly Hermitian** (H⊗I − I⊗H, a difference of two
commuting Hermitian operators), so it, and `P H_asym P` for any orthogonal
projector P, has a strictly real spectrum. The linearized flow is therefore
unitary and `‖δψ(t)‖ = ‖δψ(0)‖` exactly, for every perturbation and all
time. **Every Lyapunov exponent at this fixed point is exactly zero.** This
is not a small effect to be worked around; it's the general statement that
linear unitary evolution has no sensitive dependence on initial conditions,
which is why quantum chaos is normally diagnosed with level statistics,
OTOCs or spectral form factors rather than Lyapunov exponents.

Crucially, **the fixed-point property is what causes this.** TDVP on the
MPS manifold *is* a nonlinear classical dynamical system — its vector field
is `F(ψ) = -i P_ψ H ψ` with the tangent projector `P_ψ` depending on ψ —
and linearizing gives

    δF = -i [ P_{ψ*} H δψ  +  (∂P·δψ) H ψ* ]

The second term is the only possible source of non-Hermiticity, and it is
multiplied by `H ψ*`. Because the thermofield double is an *exact zero
eigenvector* of H_asym (not merely a stationary point of the projected
flow), that term vanishes identically and the linearization collapses to
the Hermitian `P H P`. The shortcut that made the calculation cheap — "the
uniform state is a fixed point, so we needn't track pairs of trajectories"
— is precisely the condition that forces the exponents to zero. You cannot
keep both.

Hence two separate subprojects:

1. **Accept the zero exponents and compute what is actually nonzero**: the
   dephasing-driven relaxation rate of *local* observables. The norm of δψ
   is conserved, but its local content is not. See [`relaxation/`](relaxation/).
2. **Recover genuine exponents by giving up the fixed point**: linearize
   along a trajectory where `Hψ ≠ 0`, so the `∂P` term survives and the
   flow is genuinely nonlinear. See [`tdvp_lyapunov/`](tdvp_lyapunov/).

**Finite-bond-dimension trap, relevant to both.** At finite D,
`psi_uniform` is only an approximate eigenvector, so `H ψ* ≠ 0` and the
`∂P` term reappears with a small non-Hermitian part. Exponents found that
way measure truncation error, not physics. Any nonzero rate must be shown
to survive increasing D.

## Shared dependencies

- `qtensor.thermofield` — building uniform and non-uniform-temperature
  purified states.
- `qtensor.operators` — `thermofield_hamiltonian`/`symmetric_thermofield`,
  MPO algebra (`@`, `+`/`-`, `.trace()`), and the
  `contract_left`/`contract_right` environment primitives.
- `qtensor.states` — canonicalization, entropy/purity helpers.
- `qtensor.simulation.finiteTDVP` / `updatemethod` — the effective
  Hamiltonian and environment machinery, reused directly; and direct time
  evolution for validation.
