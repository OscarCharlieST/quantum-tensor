# Lyapunov exponents of the TDVP flow

Status: **not started** — context only, no code.

See [`../README.md`](../README.md) for shared background and for the
argument that forces the exponents at the fixed point to be exactly zero.
This strand is the response to that: recover genuine exponents by giving up
the fixed point.

## The idea

TDVP on the MPS manifold is a nonlinear classical dynamical system. Its
vector field

    F(ψ) = -i P_ψ H ψ

has a tangent projector `P_ψ` that depends on the current point, so
linearizing about a trajectory gives

    δF = -i [ P_ψ H δψ  +  (∂P·δψ) H ψ ]

At the thermofield fixed point the second term dies because `H ψ* = 0`
exactly, leaving the Hermitian `P H P`. **Away from a fixed point it
survives**, is not Hermitian, and is the source of genuine exponential
separation of nearby trajectories on the manifold. This is a real and
studied object — Hallam, Morley & Green, *The Lyapunov spectrum of quantum
thermalisation* (Nat. Commun. 10, 2708, 2019) compute exactly this
spectrum and relate it to thermalization.

Concretely, the plan would be to evolve a state under
`finiteTDVP.tdvp`, and alongside it integrate the linearized flow above to
get the growth rates, either by:

- propagating a set of tangent vectors with periodic
  Gram–Schmidt/QR reorthogonalization (the standard Benettin algorithm)
  for the full Lyapunov spectrum, or
- tracking a pair of nearby trajectories and fitting their separation, for
  the leading exponent only.

The tangent-vector machinery in
[`../relaxation/tangent_hamiltonian.py`](../relaxation/tangent_hamiltonian.py)
is directly reusable here — `build_null_space_tensor` and
`build_tangent_vector` parametrize exactly the δψ this flow acts on. What
is missing is the `∂P` term, which has no counterpart in the existing TDVP
code (TDVP only ever applies `P_ψ` at the current point; it never needs its
derivative).

## Open questions

1. **Do we need `∂P` explicitly?** The Benettin algorithm only needs the
   *action* of the linearized flow on a tangent vector, which might be
   obtainable by finite-differencing `F` along the manifold rather than by
   deriving `∂P` in closed form. Worth checking before committing to the
   algebra.
2. **Physical meaning at finite bond dimension.** These exponents are
   properties of the *variational manifold*, not of the underlying quantum
   dynamics (which has zero exponents regardless). The D-dependence is the
   whole story and needs to be studied deliberately, not treated as
   convergence error.
3. **Which trajectory?** The non-uniform-temperature states this project
   started from are near the fixed point, where the `∂P` term is small by
   construction. A meaningful Lyapunov spectrum may need to start somewhere
   genuinely far from equilibrium.
4. **Relation to the relaxation strand.** Whether the TDVP Lyapunov
   exponents and the dephasing relaxation rates of
   [`../relaxation/`](../relaxation/) are connected, or are describing
   different physics that happen to share a starting point.
