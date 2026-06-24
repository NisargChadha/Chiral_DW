# Chiral Domain-Wall Variational Calculations

This repository implements variational projector calculations for chiral
domain-wall charge response in twisted MoTe2-inspired models. The v1 codebase
has two paths:

- a production nonideal finite-Landau-level Aharonov-Casher workflow for
  studying dimensionless `cG` and charge profiles versus periodic `U(r)` and
  `B'(r)`;
- a TMD_HF adapter that consumes continuum Hartree-Fock VP/IVC reference
  projectors and builds the simple source-field interpolation used by the
  variational workflow.

`/Users/nisargchadha/Documents/TMD_HF` is the canonical source for continuum
physics conventions, hole-basis conventions, density vertices, and Hartree-Fock
machinery. `/Users/nisargchadha/Desktop/Variational_Calculation_tMoTe2` remains
read-only reference material for variational/projector-response logic.

## Setup

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest -q
```

For continuum TMD_HF-backed workflows, also install the local TMD_HF repo:

```bash
python3 -m pip install -e /Users/nisargchadha/Documents/TMD_HF
```

or set:

```bash
export PYTHONPATH=/Users/nisargchadha/Documents/TMD_HF/src:$PYTHONPATH
```

## Nonideal AC Workflow

Command-line example:

```bash
python3 scripts/run_ac_cg.py \
  --output-dir results/ac_cg_smoke \
  --b1 0.20 \
  --u1 0.10 \
  --b1-c3 0.02 \
  --u1-c3 0.00 \
  --n-ll 5 \
  --n-k 7 \
  --n-theta 41 \
  --source-scale 1.0 \
  --interaction-shell 2 \
  --gate-distance 2.0 \
  --plots
```

Python example:

```python
from chiral_dw.ac.workflow import run_ac_cg_workflow
from chiral_dw.config import (
    ACResponseWorkflowParams,
    FirstShellACParams,
    MomentumGridParams,
    ResponseParams,
)

params = ACResponseWorkflowParams(
    grid=MomentumGridParams(n_k=7),
    ac=FirstShellACParams(b1=0.20, u1=0.10, n_ll=5),
    response=ResponseParams(n_theta=41),
    output_dir="results/ac_cg_python",
)
result = run_ac_cg_workflow(params, write_outputs=True)

print(result.response.cG)
print(result.summary.model_dump())
```

The AC workflow uses source fields only to generate projectors. Reported
physical energies come from the projected band, Hartree, and Fock terms and do
not include the trial source field.

## TMD_HF Source Interpolation

The TMD_HF adapter is deliberately thin. TMD_HF builds the continuum/HF problem
and supplies either converged VP/IVC projectors or seed projectors; this package
extracts raw contracted fields and builds:

```text
H_var(k; theta, phi) =
    H0(k) + source_scale * [
        cos(theta) Delta_VP(k)
      + sin(theta) U_phi Delta_IVC(k) U_phi^dagger
    ]
```

No scalar, identity, or traceless channel is subtracted silently. Diagnostics
report those channels so the run can be audited.

```python
import numpy as np

from chiral_dw.config import SourceInterpolationParams, TMDHFReferenceParams
from chiral_dw.ttmd_adapter import (
    diagnose_reference_projectors,
    endpoint_diagnostics,
    references_from_tmd_hf_bundle,
    source_interpolation_path,
)

# Build this with TMD_HF, for example from ttmd.problem.build_ttmd_hf_bundle.
# bundle = ...

# Prefer converged TMD_HF reference projectors in production.
# P_vp = ...
# P_ivc = ...

refs = references_from_tmd_hf_bundle(
    bundle,
    P_vp=P_vp,
    P_ivc=P_ivc,
    params=TMDHFReferenceParams(n_occ_per_block=1),
)

print(diagnose_reference_projectors(refs).model_dump())
print(endpoint_diagnostics(refs).model_dump())

theta = np.linspace(0.0, np.pi, 41)
projectors, path_diagnostics = source_interpolation_path(
    refs,
    theta,
    params=SourceInterpolationParams(source_scale=1.0, n_occ_per_block=1),
)
```

If `P_vp` or `P_ivc` is omitted, the adapter can call TMD_HF seed helpers, but
production runs should normally pass self-consistent references from TMD_HF.

## Artifacts

The AC command writes a directory containing:

- `projectors.npz`: theta nodes, projector path, gaps, `K_theta`, scalar `cG`,
  and physical-energy components.
- `K_theta.csv`: `theta`, `theta_over_pi`, dimensionless `K_theta`, repeated
  scalar `cG`, source gap, and physical `energy_total`.
- `charge_profile.csv`: radial coordinate `r/a_M`, local texture angle,
  interpolated `K_theta`, and `rho_dimless`.
- `summary.json`: frozen Pydantic parameter dump, scalar response summary,
  projector errors, and normalization notes.
- `artifact_manifest.json`: Pydantic `RunManifest` with artifact names, paths,
  kinds, descriptions, required flags, existence flags, byte sizes, missing
  required artifacts, and pass/fail status.
- `K_theta.png`: optional plot when `--plots` is supplied.

Generated outputs should stay under `results/`.

## Dimensionless Units

`K(theta)`, `cG`, and `rho_dimless` are reported in moire units. The coefficient
`cG` is dimensionless:

```text
cG = integral dtheta K(theta) log[tan(theta/2)]
```

The radial charge profile `rho_dimless` is charge density per `a_M^2`. To
convert to a physical areal density, divide by `a_M^2` in the chosen length
units. The sign of `cG` can change with orientation conventions; the magnitude
is the primary quantity to compare across convergence scans.

## Recommended Scans

For the AC workflow, check convergence against:

- `n_ll`, because the nonideal backend is a finite-Landau-level truncation;
- `n_k`, because `K(theta)` is a Brillouin-zone response;
- `n_theta`, especially near endpoints where the `log[tan(theta/2)]` weight is
  large;
- `source_scale`, verifying the local projector gap stays positive;
- `b1`, `u1`, `b1_c3`, and `u1_c3`, which control residual `B'(r)` and `U(r)`;
- `interaction_shell`, `gate_distance`, and `v0` when comparing energies.

For the TMD_HF path, run the TMD_HF convergence checks first, then audit this
repo's `diagnose_reference_projectors` and `endpoint_diagnostics` before using
the resulting projector path in charge-response calculations.

## Development Protocol

See `AGENTS.md` and `docs/IMPLEMENTATION_PLAN.md`. In short: reread the plan
before each implementation step, use Pydantic models for user-facing parameters
and artifact summaries, run targeted tests plus affected tests before each
commit, inspect the diff, and keep commits focused.
