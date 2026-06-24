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

## Progress So Far

The current codebase has the core scaffolding and validation path needed before
moving back to the nonideal conjugate AC/domain-wall calculation:

- Pydantic v2 frozen models now define the AC parameters, source-interpolation
  controls, real-space benchmark controls, run summaries, and artifact records.
- The finite-Landau-level nonideal AC backend is ported into `chiral_dw.ac`,
  with tests for Hermiticity, flat LLL behavior, finite-LL gaps, Fourier
  coefficients, and time-reversed partner projectors.
- The AC `cG` workflow computes source-field projectors, projected physical
  energies, `K(theta)`, dimensionless `cG`, radial charge profiles, and
  old-compatible conjugate-AC C3 bias sweeps.
- The TMD_HF adapter is in place for simple VP/IVC source-field interpolation.
  It keeps TMD_HF as the physics source of truth and records raw source-field
  diagnostics without hidden scalar or traceless cleanup.
- The same-Chern QHFM benchmark validates the real-space 4D charge evaluator
  against `rho_top=-q_sk` in a controlled Chern-1 limit.
- The ideal opposite-Chern conjugate LLL benchmark validates the circular
  domain-wall charge evaluator against the discrete analytic plaquette result
  `rho=-n_z q_sk`.

The latest ideal conjugate LLL run used `R=10`, `w=3.5`, `n_k=7`, and `n_r=41`
in magnetic-length units. It wrote artifacts under
`results/ideal_conjugate_lll_R10_w3p5` and passed with:

- `up_chern = 1`, `down_chern = -1`
- `charge_error_max = 9.87e-05`
- `charge_error_rms = 2.47e-05`
- `integrated_charge = 0.00272`
- `integrated_analytic_charge = 0.00268`
- `valid_analytic_charge = true`

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

## QHFM Real-Space Charge Benchmark

The same-Chern QHFM benchmark validates the real-space charge evaluator and
plot normalization in a controlled limit. It uses two identical copies of the
same Chern-1 AC band and a periodic skyrmion-lattice texture. In this
factorized validation limit, the 4D link-variable charge map should satisfy:

```text
rho_top = -q_sk
```

with the current orientation convention. This is not the target opposite-Chern
domain-wall physics; it is a normalization certificate for the real-space
charge machinery.

```bash
python3 scripts/run_qhfm_charge_benchmark.py \
  --output-dir results/qhfm_charge_benchmark \
  --b1 0.20 \
  --u1 0.10 \
  --n-ll 5 \
  --n-k 7 \
  --n-r 9 \
  --plots
```

The command writes:

- `qhfm_charge_density.csv`: `rho_top`, `q_sk`, `-q_sk`, and their difference
  on real-space plaquettes.
- `qhfm_charge_summary.json`: orbital Chern, mixed-curvature residual, maximum
  charge error, integrated charge, integrated skyrmion charge, fit slope,
  intercept, and correlation.
- `qhfm_curvature_components.npz`: raw 4D curvature components and charge
  arrays.
- `qhfm_charge_maps.png`: computed charge, target charge, and error maps.
- `artifact_manifest.json`: generated artifact status.

Expected passing diagnostics are `orbital_chern ~= 1`,
`mixed_curvature_max` near numerical zero, `charge_error_max` small, and
`integrated_charge ~= -integrated_skyrmion_charge`.

## Ideal Conjugate LLL Charge Benchmark

The ideal conjugate LLL benchmark is the first opposite-Chern real-space
domain-wall check. It sets the AC backend to the exactly flat `n_ll=1`,
`U=B'=0` limit, builds `C=+1` and time-reversed `C=-1` active bands, and uses
the local source Hamiltonian:

```text
h_tr(k,r) = h0(k) - m0 M(r).sigma
```

The benchmark compares the centered 4D link-variable charge map against the
discrete ideal plaquette answer:

```text
rho_analytic = -n_z(center) q_sk
```

where `q_sk` is the spinor Berry phase on the same real-space plaquettes. The
default wall parameters are `R=10`, `w=3.5`, and lengths are in magnetic-length
units.

```bash
python3 scripts/run_ideal_conjugate_lll_charge_benchmark.py \
  --output-dir results/ideal_conjugate_lll_charge \
  --n-k 7 \
  --n-r 41 \
  --radius-lb 10 \
  --width-lb 3.5 \
  --patch-length-lb 56 \
  --m0 1.0 \
  --plots
```

The command writes:

- `ideal_conjugate_lll_charge.npz`: centered 4D charge, analytic target,
  skyrmion plaquette charge, `n_z(center)`, charge error, and curvature
  components.
- `ideal_conjugate_lll_summary.json`: Chern numbers, flat-band diagnostics,
  projector checks, charge errors, integrated charges, and `m0`.
- `ideal_conjugate_lll_profiles.csv`: radial averages for the direct charge,
  analytic charge, skyrmion charge, and continuum radial-shape diagnostic.
- `ideal_conjugate_lll_charge.png`: optional charge, analytic-target, and error
  maps when `--plots` is supplied.
- `artifact_manifest.json`: generated artifact status.

Changing positive `m0` should change only the source gap, not the projectors or
charge map, in this exactly flat validation limit.

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
