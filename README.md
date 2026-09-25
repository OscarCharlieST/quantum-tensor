# quantum-tensor
quantum tensor network toolkit
consolidating various tensor network things I have made

## Open work items

- **Restructure how `tdvp` is called.** `qtensor/simulation/finiteTDVP.py` now exposes
  `tdvp_step(state, operator, dt, method, env=None) -> (state, env)`, and `tdvp` is a loop
  over it, but the observation machinery still inside `tdvp` should go: `history`,
  `history_interval`, `operators`, `extensive_operators`, `termination_func` -- three of them
  undeclared `**kwargs`. A caller that wants to watch, steer or interrupt the evolution should
  own the loop, as `thermofield.finite_T_thermofield` now does for `sym_every`. Left to do:
  migrate the notebooks and the `near_thermal_*` variants, then thin `tdvp` to a convenience
  wrapper over the step. Threading `env` is not a performance matter (measured ~1% at L = 8,
  D = 6, since the Lanczos applications dominate the sweeps) but it keeps one implementation.
- **`tdvp` takes one step past `t_f`.** `times = linspace(0, t_f, steps+1)` and the step is
  taken *after* each recording, so the returned state sits at `t_f + dt`, and
  `finite_T_thermofield` lands at `beta * (1 + 1/steps)`. Kept deliberately so that existing
  builds reproduce; fix it when you are willing to re-run them.
