# Chiral Domain-Wall Variational Codebase Plan

## Summary

Build a self-contained Python package for chiral domain-wall variational
projector calculations. Chiral_DW owns the runnable AC, continuum/Hartree-Fock,
projector-response, and artifact workflows. TMD_HF and the old variational repo
are read-only reference material only; do not import their modules at runtime.

The continuum workflow uses symmetry-constrained HF references:

- VP+ from a K-valley seed with continuous valley-U(1) projection.
- VP- from a Kprime-valley seed with continuous valley-U(1) projection.
- IVC from a Q=0 coherent seed with non-Kramers T-prime projection.
- For Taige sweeps, a finite-Q IVC active-frame branch can also be solved using
  the symmetric convention `K: k-Q/2`, `Kprime: k+Q/2`. When branch selection is
  enabled, the lower-energy IVC branch supplies the whole VP+/VP-/IVC
  interpolation frame; exact ties prefer Q=0.

The variational path is a convex interpolation of full raw reference HF
Hamiltonians:

```text
H_var(theta, phi) =
    max(cos(theta), 0)^2 H_VP+
  + max(-cos(theta), 0)^2 H_VP-
  + sin(theta)^2 U_phi H_IVC U_phi^\dagger
```

Report `K(theta)`, `cG`, and charge density in dimensionless moire units first;
optional physical conversion divides charge density by `a_M^2`.

## Key Changes

- Use Pydantic v2 frozen models for public parameters, conventions, and run
  summaries: AC params, native continuum/HF params, response params,
  domain-wall params, artifact manifests, and output summaries.
- Keep the nonideal finite-LL AC workflow for AC `cG` and charge-profile
  studies.
- Add `src/chiral_dw/continuum/` as the native continuum/HF package:
  grid/active-space construction, density vertices, screened-Coulomb HF
  backend, symmetry constraints, seeds, reference solves, convex Hamiltonians,
  embedded Bloch-basis charge response, and end-to-end workflow.
- Store raw Hermitian HF Hamiltonians by default. Do not silently subtract
  scalar, identity, or traceless channels; instead report channel diagnostics.
- Support Taige lower-energy Q=0/finite-Q IVC branch selection without mixing
  active frames: selected finite-Q responses use finite-Q VP+/VP-/IVC
  references and the finite-Q active basis.
- Generated outputs stay under `results/` unless tests intentionally write to a
  temporary directory.

## Implementation Sequence

1. **Docs and Governance**
   - Keep `AGENTS.md`, `README.md`, and this plan aligned with the
     self-contained runtime policy.
   - Record TMD_HF and the old variational repo as read-only references only.

2. **Package Scaffold and Config**
   - Maintain `src/chiral_dw`, `tests`, and `pyproject.toml`.
   - Define frozen models for AC and native continuum/HF workflows.

3. **Nonideal AC Backend and Response**
   - Maintain the finite-LL nonideal AC backend, projected physical energies,
     source-field projectors, `K(theta)`, dimensionless `cG`, and radial charge
     profiles.

4. **Native Continuum HF**
   - Implement Chiral_DW-native active-space construction, density vertices,
     screened-Coulomb HF backend, fixed-per-k zero-temperature solver, and
     seeds.
   - Maintain finite-Q active-frame source remapping for Taige IVC energy
     comparisons.
   - Implement `ValleyU1Constraint` and `TPrimeConstraint` with
     `project_density`, `project_operator`, `symmetry_error`, and final
     idempotent Aufbau support.

5. **Symmetric Variational Hamiltonians**
   - Solve VP+, VP-, and IVC references.
   - Build the convex full-HF variational Hamiltonian and fixed-per-k projector
     path.
   - Record final idempotency, self-consistency residuals, gaps, trace errors,
     constraint errors, and channel diagnostics.

6. **Examples and Artifacts**
   - Provide CLI/Python examples for AC and native continuum workflows.
   - Write JSON summaries, `.npz` projector arrays, CSV response/charge tables,
     and manifests under `results/`.

## Test And Review Protocol

- Before each substantive implementation step: re-read this plan, `AGENTS.md`,
  and relevant reference files or local modules.
- Run targeted pytest for the current slice and affected neighbors.
- Inspect `git diff` and verify the changes still match the plan before
  committing.
- If a symmetry or convention ambiguity appears, resolve it before expanding
  the implementation.

## Assumptions And Defaults

- Hole filling one uses one occupied active state per momentum block.
- Primary AC backend: nonideal finite-LL AC.
- Continuum path: self-contained native HF references plus convex full-HF
  Hamiltonian interpolation.
- Charge response: dimensionless first; physical units are optional
  postprocessing.
- Final HF reference projectors should be idempotent. If the final idempotent
  projector has a large Aufbau residual, report a self-consistency warning
  rather than silently accepting the reference.
