# Working-field ODA update for Vidyut

This note explains how to port the external solver's linear working-field ODA
strategy to Vidyut. The implementation described here is the corrected
Chiral_DW version.

The change is algorithmic only. It must not alter interaction channels,
normal-ordering reference, finite-\(Q\) convention, active-space gauge,
integer-filling policy, or the ODA line-minimization policy.

## Algebra and the Hartree qualification

Write the density relative to the normal-ordering reference as

\[
Q=P-P_{\rm ref}.
\]

Let \(H_H\) and \(H_F\) be the linear Hartree and Fock field maps:

\[
H[P]=h_0+H_H[Q]+H_F[Q].
\]

In Chiral_DW, the interaction energy is

\[
E_{\rm int}[Q]=E_H[Q]
 +\frac{1}{2}\operatorname{Tr}\!\left(H_F[Q]Q\right),
\]

where \(E_H[Q]\) is the explicit sum of direct-channel density amplitudes.
Do not replace \(E_H[Q]\) by
\(\tfrac12\operatorname{Tr}(H_H[Q]Q)\) unless that identity is covered by
tests for every Vidyut backend and wrapper. Evaluating the cheap explicit
direct-channel quadratic form avoids making that extra assumption.

At the start of an SCF iteration, retain \(P\), \(H_H[Q]\), and \(H_F[Q]\).
Diagonalize the current working Hamiltonian and apply the selected global or
fixed-per-\(k\) Aufbau rule to obtain \(P_A\). Evaluate

\[
H_{H,A}=H_H[P_A-P_{\rm ref}],
\qquad
H_{F,A}=H_F[P_A-P_{\rm ref}].
\]

This is one inexpensive Hartree construction and exactly one new expensive
Fock application. Define

\[
\Delta P=P_A-P,\quad
\Delta H_H=H_{H,A}-H_H[Q],\quad
\Delta H_F=H_{F,A}-H_F[Q].
\]

Linearity gives

\[
\Delta H_H=H_H[\Delta P],\qquad
\Delta H_F=H_F[\Delta P],
\]

and, for \(X\in\{H,F\}\),

\[
H_X[P+\lambda\Delta P-P_{\rm ref}]
 =H_X[Q]+\lambda\Delta H_X.
\]

The ODA quadratic

\[
E(\lambda)=E(0)+s\lambda+\frac{1}{2}c\lambda^2
\]

therefore uses

\[
s=\operatorname{Tr}(H_{\rm projected}\Delta P),
\]

\[
c=2E_H[\Delta P]
 +\operatorname{Tr}(\Delta H_F\,\Delta P)
 =2E_{\rm int}[\Delta P].
\]

After selecting the same damping fraction as before, update the density and
both fields together:

\[
P'=P+\lambda\Delta P,
\]

\[
H_H'=H_H[Q]+\lambda\Delta H_H,\qquad
H_F'=H_F[Q]+\lambda\Delta H_F.
\]

The accepted Hamiltonian and energy then require no additional Fock
application:

\[
H'=h_0+H_H'+H_F',
\]

\[
E[P']=\operatorname{Tr}(h_0P')
      +E_H[P'-P_{\rm ref}]
      +\frac{1}{2}\operatorname{Tr}
       \left(H_F'(P'-P_{\rm ref})\right).
\]

The explicit Hartree scalar contraction is much cheaper than rebuilding the
exchange field.

## Reference pseudocode

```python
P = project_density(P_initial)
Q = P - P_ref
H_H = hartree_field(Q)
H_F = fock_field(Q)
H = hermitize(h0 + H_H + H_F)
energy = energy_from_fields(P, H_H, H_F)

for iteration in range(max_iter):
    H_projected = project_operator(H)
    P_aufbau = aufbau(H_projected, particle_number)

    delta_P = hermitize(P_aufbau - P)
    Q_aufbau = P_aufbau - P_ref
    H_H_aufbau = hartree_field(Q_aufbau)
    H_F_aufbau = fock_field(Q_aufbau)  # one expensive application
    delta_H_H = hermitize(H_H_aufbau - H_H)
    delta_H_F = hermitize(H_F_aufbau - H_F)

    slope = trace_product(H_projected, delta_P)
    curvature = (
        2 * hartree_energy(delta_P)
        + trace_product(delta_H_F, delta_P)
    )
    lambda_ = choose_oda_lambda(slope, curvature)

    P = hermitize(P + lambda_ * delta_P)
    H_H = hermitize(H_H + lambda_ * delta_H_H)
    H_F = hermitize(H_F + lambda_ * delta_H_F)
    H = hermitize(h0 + H_H + H_F)
    energy = energy_from_fields(P, H_H, H_F)

    diagnostics = diagnostics_from_cached_state(P, H, energy)
    if converged(diagnostics):
        break

P_final = aufbau(project_operator(H), particle_number)
Q_final = P_final - P_ref
H_H_final = hartree_field(Q_final)
H_F_final = fock_field(Q_final)
H_final = hermitize(h0 + H_H_final + H_F_final)
energy_final = energy_from_fields(P_final, H_H_final, H_F_final)
```

For \(N\) productive iterations, this performs \(N+2\) expensive Fock
applications: one for the initial density, one for every Aufbau trial, and one
for the final idempotent projector. The old Chiral_DW loop could rebuild the
Fock field several times per iteration through separate curvature, energy,
and diagnostic calls.

## Suggested Vidyut interfaces

Expose the linear field operations and the cheap direct quadratic form:

```python
hartree_field(delta_density) -> hartree_field
fock_field(delta_density) -> fock_field
hartree_energy(delta_density) -> float
```

The solver should own the matched working tuple
`(density, hartree_field, fock_field)`. This remains valid whether Vidyut uses
dense operators, compact symmetry sectors, matrix-free contractions, CPU
kernels, or GPU kernels.

Composite backends require special care. The bug found in Chiral_DW's
`C3SymmetrizedACBackend` came from forwarding an inherited `self_energy`
method to the unsymmetrized base backend: the forwarded method invoked the
base Hartree and Fock maps instead of the wrapper's C3-averaged maps. Vidyut
should either make composed field methods dispatch through the wrapper or have
the solver call the wrapper's Hartree and Fock primitives explicitly. Add a
test that the composite self-energy equals the sum of its own two primitive
fields and differs from the unwrapped field on a non-symmetric density.

Add an energy helper that consumes already-matched fields:

```python
energy_from_fields(density, hartree_field, fock_field, h0, p_ref)
```

Scalar diagnostic helpers should accept the cached Hamiltonian and energy.
They may repeat the inexpensive diagonalization needed for a fixed-point
residual, but must not apply the Fock map again.

Use production precision throughout. Hermitize both field increments and both
accepted fields so anti-Hermitian roundoff does not accumulate.

## Constraints and filling

The update is identical for fixed-per-\(k\) and global Aufbau filling; only the
routine constructing \(P_A\) changes.

For a symmetry-constrained solve, the density projection must be linear. If
both \(P\) and \(P_A\) obey the same linear constraint, their convex mixture
does too, and the mixed-field identity remains exact. This covers Chiral_DW's
valley-U(1) and non-Kramers \(T'\) projectors.

If Vidyut applies a nonlinear operation such as eigenvalue clipping,
renormalization, or penalty minimization after mixing, the cached fields no
longer necessarily match the transformed density. Either keep that operation
outside ODA or explicitly rebuild both fields after it.

For finite meshes, convert the nominal filling to an integer particle count
before solving and record both nominal and realized fillings.

## DIIS integration

DIIS may extrapolate the working HF field, but it must not introduce repeated
Fock evaluations. A DIIS iteration should:

1. extrapolate a working field from accepted history;
2. construct the Aufbau trial density;
3. evaluate its Hartree field and Fock field once;
4. apply the acceptance or damping policy;
5. store the accepted density and matched fields together;
6. record the commutator and accepted field in DIIS history.

A rejected extrapolation may reset the DIIS subspace, but must not replace the
last accepted matched tuple.

## Required regression tests

Before making this path the Vidyut default, require:

1. linearity of both field maps for arbitrary scalar combinations;
2. \(2E_H[\Delta P]+\operatorname{Tr}(\Delta H_F\Delta P)
   =2E_{\rm int}[\Delta P]\);
3. identical ODA lambda from direct and working-field calculations;
4. both mixed fields equal fresh evaluations at the mixed density;
5. cached energy equals the full energy functional;
6. identical single-Aufbau and single-ODA updates;
7. identical short iteration trajectories and converged results from the same
   seeds;
8. the same checks at global integer filling;
9. the same checks under every supported linear symmetry constraint;
10. exactly one trial Fock application per productive iteration.

For every composite or symmetry-averaged backend, also verify that cached
energy equals that wrapper's full energy, not merely the base backend's
energy.

Run the tests first on a synthetic dense backend, then at the matched
\(12\times12\) tMoTe2 point in the \(Q_0\), \(Q_+\), and \(Q_-\) active
frames. Gauge-adapter benchmarks must remain unchanged because the
optimization does not alter vertices or basis transport.

## Chiral_DW implementation map

The implementation is in `src/chiral_dw/continuum/hf.py`:

- `ContinuumHFBackend.hartree_energy`;
- `ContinuumHFBackend.total_energy_from_fields`;
- `_working_hf_state`;
- `solve_hf`;
- `solve_global_hf`;
- cached-state arguments to both diagnostic helpers.

Regression coverage is in `tests/test_continuum_hf.py` and
`tests/test_set_signal.py`.
