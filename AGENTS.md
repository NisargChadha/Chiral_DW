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

## Cluster Run Discipline

- Before recommending, submitting, or scaling any large SLURM array, cluster
  phase-diagram sweep, finite-size sweep, or thousands-point recompute, first
  run a single-point or single-shard smoke test through the exact same code path
  and verify that the expected logs, scalar outputs, projector/output records,
  and merge inputs are written correctly.
- Do not launch or ask the user to launch thousand-point sweeps before the
  single-point test has completed successfully. If the user explicitly asks to
  bypass this rule, state the fairshare and queue-risk tradeoff clearly before
  proceeding.
- Scale cluster jobs gradually after the smoke test: one shard/point, then a
  small batch or one mesh, then the full sweep. Prefer low-concurrency repair
  runs over flooding the scheduler with many small jobs.
- For cluster repair workflows, audit existing outputs first and rerun only the
  missing or invalid work. Treat whole-mesh missing outputs as an orchestration
  failure to diagnose, not as a reason to blindly submit all shards again.

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

## Plotting Discipline

- For plotting tasks, render and visually review the actual output before
  reporting completion. Check for label, legend, colorbar, tick, and annotation
  overlap; clipping; unreadable text; misplaced colorbars; distorted aspect
  ratios; and phase-boundary visibility. Fix visible layout problems before
  handing the plot back.
- Display colorbar labels horizontally by default, placed above or near the
  upper-right of the colorbar unless a plot-specific layout requires otherwise.

## Paired Notebook Workflow

- Use Jupytext percent-format `.py` companions for substantial notebooks, e.g.
  `notebooks/foo.ipynb` paired with `notebooks/foo.py`.
- Treat the `.ipynb` as the live user-editable source. Before answering
  questions about, inspecting, or editing a paired `.py` file, first sync the
  notebook into the script with Jupytext so the `.py` is not stale.
- After editing a paired `.py` file, sync it back into the `.ipynb` before
  running tests or reporting completion.
- Prefer `python3 -m jupytext --set-formats ipynb,py:percent <notebook> --sync`
  when creating a pair, and use explicit Jupytext sync commands whenever one
  side may have changed.

## Commit Discipline

- Commit regular working slices after tests pass.
- Keep commits linear and focused on one implementation step.
- Do not revert user changes or unrelated work.
- Use sub-agents for parallel test design, convention audits, and diff review
  when useful, but keep final integration and commit history under the main
  agent's control.
