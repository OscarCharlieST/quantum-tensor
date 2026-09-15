# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A personal research toolkit for simulating 1D quantum spin chains with matrix product states (MPS),
built for finite-system TDVP (Time-Dependent Variational Principle) time evolution. The current physics
focus (visible in the notebooks and `qtensor/thermofield.py`) is: building finite-temperature states via
thermofield/purification doubling, evolving them (symmetric vs. antisymmetric evolution of the auxiliary
copy), and studying energy transport (e.g. fitting energy-density profiles to a diffusion equation) in a
tilted transverse-field Ising chain.

This is not a packaged library — there is no `setup.py`/`pyproject.toml`, no `__init__.py` in `qtensor/`,
no dependency manifest, and no test runner or linter configured. Code is developed and run interactively
from Jupyter notebooks at the repo root, which import `qtensor.*` relying on the notebook's cwd (repo
root) being on `sys.path`. Treat this as a research script collection, not a distributable package —
don't introduce packaging scaffolding unless asked.

## Design choice: no external tensor-network library

All MPS/MPO logic is hand-written on top of **numpy** and **scipy** (`scipy.linalg` for SVD/`expm`/`eig`,
`scipy.optimize.curve_fit` for post-processing fits). The only tensor-contraction helper used is
[`ncon`](https://github.com/mhauru/ncon) (numbered-leg `einsum`-style contraction) — it is not a
tensor-network framework, just a contraction convenience function; there is no automatic gauge tracking,
no built-in truncation, no graph/network abstraction. `quimb` and `tenpy` appear only in
`quimbtest.ipynb` / `tenpy_test.ipynb` as external references/comparisons — they are not used by
`qtensor` itself and shouldn't be treated as dependencies of the package.

Other runtime dependencies actually imported by `qtensor`: `h5py` (state checkpointing), `progressbar`
(sweep progress bar in `tdvp`), `matplotlib` (`qtensor/visualise.py`, `qtensor/thermofield.py`), and
`numba` (`jit`/`njit`/`numba.typed.List` are imported in `finiteTDVP.py` and `updatemethod.py` but not
currently applied to any function — dead/planned-only import, not active acceleration).

## Running things

There's no build/lint/test command. In practice:
- Work happens in the top-level `.ipynb` notebooks (`active.ipynb`, `normal_tdvp.ipynb`,
  `commutator_trace.ipynb`, `matexptest.ipynb`), run from the repo root so `import qtensor...` resolves.
  These are working/scratch notebooks for dynamically exercising features, not documentation — don't
  invest effort maintaining or explaining their contents beyond what's needed for a specific task.
- `qtensor/tests.py` is **not** a pytest suite — it's a handful of manual sanity-check helpers
  (`verify_norm`, `verify_lcan`, `verify_rcan`, `verify_expectation`) meant to be called ad hoc from a
  notebook on the object you just built, not run automatically. Note `verify_expectation` calls a
  `mpo_expect` that doesn't exist anywhere in `qtensor/operators.py` (likely stale after a rename to
  `expect`) — a pre-existing bug, not something to silently "fix" as a side effect of an unrelated task.
- Saved simulation states live under `states/*.h5`, written/read via `mps.save()` /
  `states.load_mps()`, named by convention like `L_{sites}_D_{bond_dim}_{tag}.h5`.

## Core data model

Two thin dict-backed container classes carry all the state; the actual math lives in module-level
functions in `states.py` / `operators.py` that operate directly on raw numpy arrays (or on `mps`/`mpo`
objects via `[]` indexing) rather than being methods on the class:

- **`states.mps`** (`qtensor/states.py`): wraps `{site_index: tensor}` where each tensor is rank-3
  `(d, D_left, D_right)`. Carries gauge bookkeeping as plain mutable attributes rather than as part of the
  type — `form` (`'none'|'left'|'right'|'center'`), `c_site` (current orthogonality centre),
  `centred`/`bond_centred`/`normalized` flags. Nothing enforces these are kept consistent; simulation code
  (`finiteTDVP.py`) trusts them via `assert state.form == ...` rather than deriving gauge from the data.
- **`operators.mpo`** (`qtensor/operators.py`): wraps `{site_index: tensor}` where each tensor is rank-4
  `(d, d', D_left, D_right)`, plus boundary vectors `l`/`r` to contract at the chain ends. Supports algebra
  as operators: `+`/`-` (direct sum of bond dimension — not compressed), `@` and `.combine()` (sequential
  composition/product of MPOs), `.trace()`.
- Both classes are keyed by arbitrary sorted integer site labels (not required to be `0..N-1`), so
  `sorted(self.sites)` is the recurring idiom for iterating the chain in order.

**Shared index convention** (repeated as an ASCII diagram at the top of `states.py`, `operators.py`, and
`finiteTDVP.py` — keep it in sync if leg ordering ever changes): MPS tensor legs are `(physical, left,
right)`; MPO tensor legs are `(physical_ket, physical_bra, left, right)`; `ncon` calls throughout the
codebase are written against this exact ordering, so changing it in one place requires updating the
matching numbered-leg lists everywhere else — there's no type system enforcing it.

## How the pieces call into each other

`qtensor/simulation/finiteTDVP.py` is the orchestration layer:

1. `tdvp(state, operator, t_f, steps, method=...)` is the entry point. It right-canonicalizes the state,
   builds initial left/right environment dictionaries (`L_con`/`R_con`, keyed by site — cached and updated
   incrementally rather than recontracted from scratch each step, so each local update stays roughly
   `O(D^3)` instead of `O(N D^3)`), then alternates `tdvp_sweep_r` / `tdvp_sweep_l` per time step.
2. Each sweep (`tdvp_sweep_r`/`l`) walks the chain calling `tdvp_step_r`/`l` per site, which: evolves the
   current centre tensor forward by `dt/2` via the pluggable `method`, re-orthogonalizes it with
   `states.left_orthogonal_tensor`/`right_orthogonal_tensor` (this is also where SVD truncation to
   `max_bond_dim` happens), evolves the resulting bond tensor *backward* by `dt/2` (the standard two-site
   TDVP splitting), and folds it into the neighboring tensor before advancing `state.c_site`.
3. The `method` argument is a `updatemethod.TDVPMethod(c_method, b_method)` — a pluggable pair of
   functions for evolving a centre tensor and a bond tensor under `exp(-i H_eff dt)`. This is the
   performance-critical, swappable layer:
   - `exact_method()`: builds the full effective Hamiltonian explicitly and calls dense
     `scipy.linalg.expm` on a `(d·D_l·D_r)²`-sized matrix — correct at any bond dimension but scales
     as roughly the cube of that size; fine for small/debug runs only.
   - `lanczos_method(epsilon, max_iters)` (the default): builds a small Krylov subspace by repeatedly
     applying the effective Hamiltonian (`apply_Heff_parts`/`apply_Heff_bond`) instead of forming it
     densely, then `expm`s only the resulting small (`max_iters`-dimensional) matrix. This is the intended
     way to reach large bond dimension — the docstring in `updatemethod.py` quotes ~4000x speedup at
     d=4, D=32, max_iters=16 vs. exact diagonalization, at the cost of `~(dt·|H|)^(max_iters+1)` error.
   - `ops.contract_left`/`contract_right` are the environment-update primitives shared by both the main
     sweep and the Lanczos environment contractions.
4. `qtensor/operators.py` supplies the Hamiltonians/observables as MPOs (`tilted_ising`, `total_z`,
   `single_site_pauli`, etc.) and the two expectation-value routines: `expect` (full-chain, assumes
   operator spans every site) and `local_expect` (centralizes the state at the operator's support first,
   for operators covering only part of the chain).
5. `qtensor/thermofield.py` builds on top of both `states.py`/`operators.py` and reuses
   `simulation.finiteTDVP.tdvp` unchanged: it doubles the physical dimension (`d -> d²`) to represent a
   purification of a mixed/thermal state, builds thermofield-doubled Hamiltonians
   (`thermofield_hamiltonian`, `symmetric_thermofield` — note there are two near-duplicate
   `thermofield_hamiltonian` implementations, one in `operators.py` and one in `thermofield.py`), and
   drives *imaginary*-time evolution (`dt -> -i·dt`) toward finite temperature
   (`finite_T_thermofield`, `near_thermal`, `near_thermal_delta_function`). `single_copy_expectation`
   computes expectation values against only the "real" copy of a purification (as opposed to the
   symmetric `local_expect`), needed when the auxiliary/disentangling copy isn't evolved symmetrically.
6. `qtensor/visualise.py` is a pure consumer of the `(state_history, expectations)` dicts returned by
   `tdvp(..., history=True, operators=[...])` — `state_history` is `{time: mps}`, `expectations` is
   `{time: [values]}`. This is the standard interchange format between simulation and plotting code.
   Also contains the Gaussian-fit / diffusion-constant-estimation helpers
   (`fit_gaussian(s)`, `estimate_D`, `estimate_diffusion_constant_from_widths`) used to analyze energy
   transport from `plot_energy_density_evolution` output.
7. `qtensor/simulation/tangentTDVP.py` currently exists but is **empty** — a stub for a planned
   tangent-space TDVP variant that hasn't been written yet.

## Research subprojects (`lyapunov/`)

`lyapunov/` is a *consumer* of `qtensor`, not part of it — self-contained research code that imports
`qtensor.*` the same way the notebooks do (run from repo root). Don't move its helpers into `qtensor`
without being asked. `lyapunov/WORKFLOW.md` is the practical record (pipeline, results, open items);
the per-folder READMEs carry the physics arguments.

Two subprojects, split because the original question had two different answers:
- `lyapunov/relaxation/` (active) — projects the antisymmetric thermofield Hamiltonian onto the MPS
  tangent space at the uniform-temperature thermofield double and diagonalizes it exactly, to get
  dephasing relaxation rates of local observables. `tangent_hamiltonian.py` builds the tangent basis
  (null-space tensors `V_L^n`) and the projected Hamiltonian, reusing `updatemethod.apply_Heff_parts`
  and `operators.contract_left/right` for every matrix element; `response.py` turns the spectrum into
  a response function and a relaxation time; `run_relaxation_scan.py` drives an L scan (~70 s).
- `lyapunov/tdvp_lyapunov/` (not started) — genuine Lyapunov exponents of the nonlinear TDVP flow.

Key result worth not re-deriving: `H_asym` is exactly Hermitian and annihilates the thermofield double
exactly, so the tangent-space projection has a real spectrum and **all Lyapunov exponents at that fixed
point are exactly zero**. Nonzero exponents require linearizing along a trajectory where `Hψ ≠ 0`, where
the tangent-projector-derivative term survives — which is why the project split.

## TODOs

- MPO tensors are frequently allocated as `np.complex64` (e.g. in `tilted_ising`, `symmetric_thermofield`)
  while MPS tensors and most `ncon` contractions default to numpy's `complex128` — this mixed precision
  is scattered through `operators.py`/`thermofield.py` and doesn't look deliberate; be aware it can
  silently upcast/downcast if you're chasing a precision or performance issue.
- Truncation only happens where `max_bond_dim` is explicitly threaded through (`left_orthogonal_tensor` /
  `right_orthogonal_tensor` and the `mps` methods that wrap them); by default it's `np.inf` (no
  truncation), so bond dimension can grow unchecked during evolution unless a caller opts in.
- `states.centralize_state`/`mps.centralize()` still silently discard the state's norm when normalizing
  the new centre tensor (the same issue `left_orthogonal_state`/`right_orthogonal_state`/`mps.apply()` had
  until it was fixed to return that norm) — not yet fixed, since `centralize` isn't on the `apply()` path.
- `lyapunov/relaxation/tangent_hamiltonian.py`'s `build_null_space_tensor` (the isometry-completion/null-space
  helper for a single MPS tensor) is a generically useful primitive that arguably belongs in `states.py`
  alongside `left_orthogonal_tensor`/`right_orthogonal_tensor`, not scoped to the Lyapunov project — left
  where it is for now to keep that subproject self-contained; revisit if another use for it turns up.
- **Canonicalization gauge is not reproducible across independent sweeps.** Wherever the Schmidt spectrum
  has near-degenerate or near-zero values (thermofield states routinely have them at 1e-11), the SVD's
  singular vectors are numerically arbitrary, so re-canonicalizing a state a second time lands in a
  *different* gauge. Anything that combines tensors from two separate canonicalization passes — centre
  tensors, environments, overlaps — must derive them from one shared pass (e.g. via bond matrices), not
  recompute them. This produces plausible-looking but meaningless numbers when violated; see
  `lyapunov/WORKFLOW.md` for the instance that caught it.
- **Single-site TDVP cannot grow bond dimension** (it's a fixed-rank manifold method), and
  `thermofield.inf_T_thermofield` returns a rank-1 state zero-padded to bond dimension D. So
  imaginary-time evolution from it stays rank 1 unless seeded — that's what the `noise` argument is for,
  and it means the resulting finite-temperature state is only approximately the thermofield double.
  Note also that `tdvp` never truncates (no `max_bond_dim` is threaded through it), so bond dimension is
  fixed by whatever the initial state carries.
- `thermofield.th_onesite` passes `[site, W]` to `ops.mpo`, where every other caller passes a list of
  `(site, tensor)` pairs — it would raise on the `W[0]`/`W[1]` unpacking in `mpo.__init__`. Looks like a
  latent bug in a function nothing currently calls.
- Several `active.ipynb` cells call `thermofield.near_thermal(H, profile, D, steps=..., initial_state=...)`,
  which doesn't match the current signature `near_thermal(H, profile, initial_state, steps=100)` and would
  raise a duplicate-argument `TypeError`. The notebook cells are stale against a since-changed API — don't
  treat them as a guide to current usage (they are still a good guide to the *parameter regimes* used).
