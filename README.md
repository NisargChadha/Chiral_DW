# Chiral Domain-Wall Variational Calculations

This repository contains a clean implementation of variational projector
calculations for chiral domain-wall charge response in twisted MoTe2-inspired
models.

The first production workflow focuses on nonideal finite-Landau-level
Aharonov-Casher bands. This lets us study how the dimensionless geometric
coefficient `cG` and the induced domain-wall charge profile respond to residual
periodic potential `U(r)` and periodic magnetic field `B'(r)`.

## Design Principles

- `TMD_HF` is the canonical source for continuum-model and Hartree-Fock
  conventions.
- The old variational repository is read-only reference material for algorithmic
  workflow and tests.
- Parameters and artifacts are represented by Pydantic models.
- Trial source fields generate projectors; physical energies exclude trial
  source terms.
- Charge response is dimensionless by default. Physical charge-density units are
  optional postprocessing.

## Planned Workflows

1. Nonideal AC backend in a finite Landau-level magnetic Bloch basis.
2. AC projected energy and source-field projector scans.
3. Projector response calculation for `K(theta)`, dimensionless `cG`, and
   circular domain-wall charge profiles.
4. TMD_HF-backed VP/IVC source-field interpolation for continuum tMoTe2.

See `docs/IMPLEMENTATION_PLAN.md` for the staged implementation plan and the
test/commit protocol.
