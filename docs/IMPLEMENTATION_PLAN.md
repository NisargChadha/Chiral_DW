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
   - Regulate projected interaction channels with a C3-invariant radial cutoff
     on the combined physical momentum `q+G`; do not use the candidate
     reciprocal box itself as the physical cutoff.
   - On the finite magnetic-Bloch torus, C3-orbit-average the AC HF functional
     in the coefficient gauge. Do not C3-average the iterated projector, so
     spontaneous nematic reference states remain allowed.

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

7. **Scanning-SET Thermodynamics**
   - Add a zero-temperature global-filling HF workflow for homogeneous
     scanning-SET chemical-potential and inverse-compressibility calculations.
     Keep the existing fixed-per-k solver and variational workflow unchanged.
   - At hole filling one, retain a separate fixed-one-state-per-k VP reference
     for HF-band Chern and direct/indirect-gap diagnostics. Flag a nonpositive
     indirect gap as invalidating the insulating fixed-per-k interpretation;
     treat a vanishing direct gap as making the band Chern unresolved.
   - Compute relaxed total energies over a particle-number window around
     filling one, then form addition/removal chemical potentials, the SET
     charge gap, and finite-difference inverse compressibility. Preserve raw
     zero-temperature data and apply experimental broadening only in
     postprocessing.
   - Report both the full dual-gate result and an intrinsic result with the
     uniform q=0 capacitive contribution separated using the native backend
     normalization.
   - Near self-consistent first-order transitions, support upward and downward
     displacement-field continuation independently at every particle number.
     Form SET finite differences only after selecting the lower-energy
     converged branch separately for each particle number; retain both
     hysteresis branches and their topology diagnostics in the artifacts.
   - Allow broad cluster continuations to use an explicit projector-only
     storage mode.  Retain the complete model, grid, interaction, and HF
     parameters beside every projector archive so later analyses can rebuild
     the deterministic mean-field Hamiltonian as `backend.hf_hamiltonian(P)`;
     do not require duplicated stored HF Hamiltonian arrays in this mode.
   - Provide restartable point artifacts, merged displacement/filling tables,
     and a local/cluster CLI whose exact single-point path is smoke-tested
     before any sweep is scaled up.

8. **Orbital Magnetization And Remote-Band Convergence**
   - Evaluate the hole-filling-one VP orbital magnetization with the
     gauge-covariant occupied/empty-projector formula in the common Taige
     continuum plane-wave/layer basis.  Report the electron-state result
     relative to the filled-valence reference; do not attach the filled sea as
     an additive magnetization to the direct-hole HF result.
   - For the first physical point use `theta_deg=3.7`, `u_D=0`, `n_k=18`,
     `plane_wave_shell=5`, two HF-active bands per locked spin-valley, and one
     occupied hole per momentum.  Hold the two-band HF self-energy fixed while
     increasing the number of deeper bare valence-continuum bands from zero to
     six per spin-valley.
   - Call this a valence-continuum remote-band convergence study.  The current
     Taige model does not include a separate physical conduction-band
     manifold, so it cannot establish convergence with respect to true
     conduction remote states.
   - Keep the frozen HF correction in the active subspace, but embed it into
     the common microscopic basis before differentiating.  Reciprocal-boundary
     finite differences must sew plane-wave coefficients and must record the
     retained state weight at the finite plane-wave boundary.
   - Report orbital magnetization at the electron VBM, midgap, and CBM, plus
     the in-gap chemical-potential slope.  Validate that slope against the
     occupied-electron Chern number with the same sign and unit convention.
   - After frozen-remote convergence, solve VP HF self-consistently with two,
     three, and four active bands per spin-valley and no additional frozen
     bands.  Compare matched total cutoffs (two-band HF plus one/two frozen
     bands against three/four-band HF) to isolate remote-band relaxation from
     observable completeness.
   - Record orbital magnetization, self-rotation, direct/indirect gaps,
     microscopic occupied-projector overlap, valley polarization, Chern
     number, and active-remote HF mixing.  Describe the final result as tested
     only through the largest completed active and remote cutoffs.
   - Smoke-test the exact artifact path on a tiny mesh, then run one physical
     cutoff, the nested frozen sequence, and finally the enlarged HF sequence.
     Write generated arrays and tables under `results/`; plotting scripts and
     rendered convergence figures remain under `Plots/` and `Plots/figures/`.

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
- Scanning-SET thermodynamics is the exception: charged states and fillings
  away from one use global particle-number filling. The fixed-per-k filling-one
  state remains a topology diagnostic and is physically insulating only when
  its indirect HF gap is positive.
- Primary AC backend: nonideal finite-LL AC.
- Continuum path: self-contained native HF references plus convex full-HF
  Hamiltonian interpolation.
- Charge response: dimensionless first; physical units are optional
  postprocessing.
- Final HF reference projectors should be idempotent. If the final idempotent
  projector has a large Aufbau residual, report a self-consistency warning
  rather than silently accepting the reference.
- Orbital magnetization uses positive elementary charge in the standard
  electron formula and is reported as a moment in Bohr magnetons per moire
  cell.  The primary observable is the doped electron state minus the
  filled-valence reference at the same retained-band cutoff.
