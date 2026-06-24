# Chiral Domain-Wall Variational Codebase Plan

## Summary

Build a new Python package from this repo, using `TMD_HF` as the canonical
continuum/Hartree-Fock source and the old `Variational_Calculation_tMoTe2` repo
only for variational/projector-response logic. The primary AC workflow uses the
nonideal finite-Landau-level AC backend so `cG` and charge profiles can be
studied versus residual periodic potential `U(r)` and periodic magnetic field
`B'(r)`.

Use simple VP/IVC source-field interpolation only. Do not implement
penalty-constrained HF in v1. Report `K(theta)`, `cG`, and charge density in
dimensionless moire units first; optional physical conversion divides charge
density by `a_M^2`.

## Key Changes

- Create repo guidance first: `AGENTS.md`, `README.md`, and this plan.
- Add a mandatory step loop to `AGENTS.md`: before every implementation step,
  re-read the full plan and relevant references; check the previous step
  against its stated goal; run targeted tests; only commit after alignment is
  confirmed.
- Use Pydantic v2 frozen models for all parameters, conventions, and artifacts:
  AC params, continuum params, projector-response params, domain-wall params,
  run manifests, and output summaries.
- Package layout: `src/chiral_dw/` with modules for `config`, `ac`,
  `response`, `domain_wall`, `ttmd_adapter`, `artifacts`, and `cli`.
- Dependency policy: new repo imports `ttmd`/`hartree_fock` from an
  editable/local install of `/Users/nisargchadha/Documents/TMD_HF`; the old repo
  is read-only reference material, not a runtime dependency.

## Implementation Sequence

1. **Docs and Governance Commit**
   - Add `README.md`, `AGENTS.md`, `docs/IMPLEMENTATION_PLAN.md`.
   - Record: TMD_HF owns physics conventions; old repo owns variational logic;
     nonideal AC is primary; simple source interpolation only; `cG`
     dimensionless.
   - Commit: `docs: establish variational workflow and conventions`.

2. **Package Scaffold and Pydantic Config Commit**
   - Add `pyproject.toml`, `src/chiral_dw`, `tests`.
   - Define frozen Pydantic models for grids, units, AC parameters, response
     parameters, source interpolation, artifacts, and output summaries.
   - Commit after config/model tests pass.

3. **Nonideal AC Backend Commit**
   - Port/refactor finite-LL nonideal AC logic from old `nonideal_ac_ll.py` into
     typed, tested package code.
   - Support first-shell toy params, Fourier params, and tMoTe2-derived
     adiabatic Fourier params using the TMD_HF canonical parameter choice.
   - Tests: Hermitian Hamiltonian, zero-harmonic flat C=1 smoke test, finite LL
     gap, Fourier coefficient sanity, TR partner projector.

4. **AC Projected Energy and Source-Field Projectors Commit**
   - Add AC form-factor and projected physical-energy evaluator.
   - Add source-field projector generation where trial fields generate
     projectors but are excluded from physical energy.
   - Tests: source term excluded from reported energy, density form-factor
     Hermiticity, neutralized Hartree convention, uniform fixed-spinor
     benchmark.

5. **Projector Response and Domain-Wall Commit**
   - Add U(1) phi reconstruction, projector-grid derivatives, `K(theta)`,
     dimensionless `cG`, and circular domain-wall charge profile.
   - Use moire-length coordinates by default, so `rho_dimless` has units of
     charge per `a_M^2`; optional physical output is `rho_dimless / a_M^2`.
   - Tests: Hermiticity/idempotency preservation, phi covariance, zero response
     for k-independent trivial projector, oddness check
     `K(theta) ~= -K(pi-theta)`, radial charge/dipole consistency.

6. **Nonideal AC `cG` Workflow Commit**
   - Wire nonideal AC projector paths into the response module.
   - Add CLI/script to sweep `b1`, `u1`, C3 harmonics, `n_ll`, and `n_k`,
     producing JSON manifests, `.npz` projectors, CSV summaries, and optional
     plots.
   - Tests: small-grid end-to-end AC run produces finite `K`, finite
     dimensionless `cG`, and valid artifacts.

7. **TMD_HF VP/IVC Source-Interpolation Commit**
   - Add adapter that uses TMD_HF precompute/HF machinery to obtain VP and IVC
     reference projectors and contracted HF source fields.
   - Build
     `H_var(k; theta, phi) = H0(k) + source_scale * [cos(theta) Delta_VP(k)
     + sin(theta) U_phi Delta_IVC(k) U_phi^dagger]`.
   - Use raw Hermitian contracted fields by default; no hidden scalar/traceless
     cleanup. Store diagnostics for scalar/traceless parts.
   - Tests: import guard for missing TMD_HF, reference-shape checks, endpoint
     diagnostics, phi-rotation covariance, small source-interpolation smoke
     test.

8. **Final Docs and Examples Commit**
   - Add concise usage examples for AC and TMD_HF workflows.
   - Document artifact schema, dimensionless `cG` interpretation, and
     recommended convergence scans.
   - Run full test suite and commit final docs.

## Test And Review Protocol

- At each step: re-read this plan, `AGENTS.md`, and the relevant source notes
  before editing.
- Use sub-agents during implementation for parallel test design, diff review,
  and convention audits. The main agent keeps the canonical repo history linear
  and commits only after reviewing their findings.
- Before every commit: run targeted pytest for the current slice, run affected
  existing tests, inspect `git diff`, verify the step matches the plan, then
  commit.
- If a step fails its own goal, fix or revise before committing; do not move to
  the next step with known drift.

## Assumptions And Defaults

- Primary AC backend: nonideal finite-LL AC.
- Exact ideal AC/theta-function construction: validation and future upgrade
  path, not the first production backend.
- Continuum path: simple VP/IVC source-field interpolation only.
- Canonical tMoTe2 conventions and parameters: from
  `/Users/nisargchadha/Documents/TMD_HF`.
- Charge response: dimensionless first; physical units are optional
  postprocessing.
- Repo guidance goes in `AGENTS.md`; no separate `skills.md` unless later
  needed for a reusable Codex skill outside this repo.
