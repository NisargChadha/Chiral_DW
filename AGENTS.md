# Agent Workflow Notes

This repository implements chiral domain-wall variational calculations for
twisted MoTe2 and related Aharonov-Casher models.

## Source Of Truth

- Use `/Users/nisargchadha/Documents/TMD_HF` as read-only reference material for
  continuum-model, Hartree-Fock, hole-basis, density-vertex, and T' symmetry
  conventions.
- Use `/Users/nisargchadha/Desktop/Variational_Calculation_tMoTe2` as
  read-only reference material for variational projector workflows, nonideal AC
  prototypes, projector response logic, and notes.
- Do not import old `Variational_Calculation_tMoTe2` modules at runtime.
  Borrow logic deliberately and rewrite it into this package with explicit
  tests and conventions.
- Do not import `ttmd` or `hartree_fock` from TMD_HF at runtime. Rewrite needed
  continuum/HF logic into this package with explicit tests and conventions.
- In v1, use symmetry-constrained reference HF states and convex full-HF
  Hamiltonian interpolation. Do not implement penalty-constrained Hartree-Fock.

## Required Step Loop

Before every substantive implementation step:

1. Re-read `docs/IMPLEMENTATION_PLAN.md` and this file.
2. Re-read the relevant TMD_HF or old-reference files for the current slice.
3. Check whether the preceding committed step satisfied its stated goal.
4. Implement only the current planned slice.
5. Run targeted tests for that slice and affected neighboring behavior.
6. Inspect `git diff` and confirm the changes still match the plan.
7. Commit only after the slice is working and aligned with the plan.

If a step drifts from the plan or reveals a physics/convention ambiguity, stop
and resolve that mismatch before committing and moving on.

## Implementation Conventions

- All user-facing parameters, conventions, and artifact summaries should be
  Pydantic v2 `BaseModel` objects, preferably frozen.
- JSON files are generated run artifacts only, not source configuration.
- Report charge response in dimensionless moire units first. The coefficient
  `cG` is dimensionless. Optional physical charge-density conversion divides by
  `a_M^2`.
- Treat trial source fields as projector generators only. Do not include source
  or penalty energies in reported physical energies.
- Store raw Hermitian source fields by default. Do not silently subtract scalar,
  identity, or traceless channels.
- Keep generated outputs under `results/` unless a test intentionally writes to
  a temporary directory.

## Commit Discipline

- Commit regular working slices after tests pass.
- Keep commits linear and focused on one implementation step.
- Do not revert user changes or unrelated work.
- Use sub-agents for parallel test design, convention audits, and diff review
  when useful, but keep final integration and commit history under the main
  agent's control.
