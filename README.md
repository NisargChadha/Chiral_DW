# Chiral Domain-Wall Variational Calculations

This repository implements variational projector calculations for chiral
domain-wall charge response in twisted MoTe2-inspired models. The v1 codebase
has two paths:

- a production nonideal finite-Landau-level Aharonov-Casher workflow for
  studying dimensionless `cG` and charge profiles versus periodic `U(r)` and
  `B'(r)`;
- a native continuum/Hartree-Fock workflow that solves symmetry-constrained
  VP+/VP-/IVC reference states and builds the symmetric convex variational
  Hamiltonian path.

`/Users/nisargchadha/Documents/TMD_HF` remains reference material for continuum
physics conventions, hole-basis conventions, density vertices, and Hartree-Fock
machinery. It is not a runtime dependency. Everything runnable in this repo is
self-contained inside Chiral_DW.

## Progress So Far

The current codebase has the core scaffolding and validation path needed before
moving back to the nonideal conjugate AC/domain-wall calculation:

- Pydantic v2 frozen models now define the AC parameters, native continuum/HF
  controls, real-space benchmark controls, run summaries, and artifact records.
- The finite-Landau-level nonideal AC backend is ported into `chiral_dw.ac`,
  with tests for Hermiticity, flat LLL behavior, finite-LL gaps, Fourier
  coefficients, and time-reversed partner projectors.
- The AC `cG` workflow computes source-field projectors, projected physical
  energies, `K(theta)`, dimensionless `cG`, radial charge profiles, and
  old-compatible conjugate-AC C3 bias sweeps.
- The native continuum/HF workflow builds self-contained VP+, VP-, and IVC
  reference Hamiltonians, reports final projector idempotency, and constructs a
  convex full-HF variational path.
- The Taige notebook and cluster sweep can solve both Q=0 and finite-Q IVC
  branches, then use the lower-energy IVC branch as the whole interpolation
  frame for `K(theta)` and `c_G`.
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

## Native Continuum Symmetric HF

The continuum path is self-contained in `chiral_dw.continuum`. It solves three
reference HF states:

- VP+ from a `K`-polarized seed with a continuous valley-U(1) constraint;
- VP- from a `Kprime`-polarized seed with the same constraint;
- IVC from an intervalley-coherent seed with the non-Kramers `T'` constraint.

For Taige-parameter notebooks and sweeps, the finite-Q IVC branch is available
in the symmetric active frame `K: k-Q/2`, `Kprime: k+Q/2`. By default the code
compares Q=0 and finite-Q IVC energies per moire cell and uses the lower-energy
IVC branch for the whole convex `K(theta)` and `cG` response. If finite-Q wins,
the response uses finite-Q VP+/VP-/IVC references and the finite-Q active basis;
ties prefer Q=0.

The variational Hamiltonian is a convex combination of the full raw HF
Hamiltonians:

```text
H_var(theta, phi) =
    max(cos(theta), 0)^2 H_VP+
  + max(-cos(theta), 0)^2 H_VP-
  + sin(theta)^2 U_phi H_IVC U_phi^dagger
```

Because the weights sum to one, scalar and kinetic pieces common to the HF
Hamiltonians are not artificially sign-flipped. The HF solver records
idempotency during iteration and finishes with an idempotent fixed-per-k Aufbau
projector; the summary also reports the residual that measures whether that
final projector is a true self-consistent fixed point.

Command-line example:

```bash
chiral-dw-continuum-symmetric-hf \
  --output-dir results/continuum_symmetric_hf_smoke \
  --n-k 5 \
  --n-theta 21 \
  --v0 0.5 \
  --q-shell 1
```

Python example:

```python
import numpy as np

from chiral_dw.config import ContinuumGridParams, ContinuumHFParams, ContinuumWorkflowParams
from chiral_dw.continuum import (
    build_continuum_bundle,
    build_symmetric_hf_references,
    reference_diagnostics,
    symmetric_convex_path,
)

params = ContinuumWorkflowParams(
    grid=ContinuumGridParams(n_k=5),
    hf=ContinuumHFParams(n_occ_per_k=1),
)
bundle = build_continuum_bundle(params.model, params.grid, params.interaction)
refs = build_symmetric_hf_references(bundle, params.hf)

theta = np.linspace(0.0, np.pi, 41)
projectors, path_diagnostics = symmetric_convex_path(refs, theta)

print(refs.vp_plus.diagnostics.idempotency_error_fro)
print(refs.ivc.diagnostics.aufbau_residual_norm)
print(reference_diagnostics(refs)["ivc"].model_dump())
```

### Taige Continuum Cluster Sweep

The cluster sweep wrapper runs one `(u_D, theta_deg)` point per SLURM array
task and writes every point under `results/`. On a fresh cluster checkout,
create the environment once:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

For a cheap launch sanity check, use `--dry-run`; it writes only
`sweep_plan.csv/json` and does not run HF:

```bash
python scripts/scan_taige_continuum_cg.py \
  --output-root results/taige_cg_grid \
  --n-u-d 13 \
  --n-twist 9 \
  --n-k 18 \
  --dry-run
```

Submit the array after checking the plan:

```bash
sbatch jobs/scan_taige_continuum_cg_array.sh
```

Useful overrides follow the same style as the old TMD_HF job scripts:

```bash
sbatch --export=ALL,N_U_D=13,N_TWIST=9,N_K=18,OUTPUT_ROOT=results/taige_cg_grid \
  jobs/scan_taige_continuum_cg_array.sh
```

Monitor with `squeue -u "$USER"` and inspect `logs/taige_continuum_cG_*` for
per-task output. By default each point records scalar-rich diagnostics:
`c_G`, `K(theta)`, trial energy and gap versus theta, HF reference gaps and
energies, selected/Q=0/finite-Q VP and IVC order-parameter magnitudes,
noninteracting Chern numbers, HF Chern numbers, and the Taige IVC- finite-Q
energy comparison. The default `IVC_BRANCH_POLICY=lower-energy` selects the
lower-energy Q=0/finite-Q IVC branch for interpolation; use
`IVC_BRANCH_POLICY=q0` to force the Q=0 path. The merged `sweep.csv` includes
plotting columns such as `vp_reference_order_abs_nz`,
`ivc_q0_ivc_amplitude_block`, `ivc_finite_q_ivc_amplitude_block`,
`vp_reference_direct_gap`, and `selected_ivc_direct_gap`. Disable expensive
branches with `COMPUTE_CHERN=0` or `COMPUTE_FINITE_Q_IVC=0`; disabling finite-Q
forces Q=0 selection. Set
`WRITE_HF_PATH_SPECTRA=1` only when path spectra are needed for every point.

After the array finishes, manually merge per-point summaries:

```bash
python3 scripts/scan_taige_continuum_cg.py \
  --output-root results/taige_cg_grid \
  --merge-only
```

The merge writes `sweep.csv/json` plus stacked long-form tables:
`sweep_trial_theta.csv`, `sweep_reference_energies.csv`,
`sweep_noninteracting_chern_numbers.csv`, `sweep_hf_chern_numbers.csv`, and
`sweep_hf_path_spectra.csv` when that optional branch was enabled.

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

For the native continuum path, check convergence against `n_k`, `n_theta`,
`v0`, `q_shell`, `mixing`, and the reported final idempotency and
Aufbau-residual diagnostics for VP+, VP-, and IVC references.

## Development Protocol

See `AGENTS.md` and `docs/IMPLEMENTATION_PLAN.md`. In short: reread the plan
before each implementation step, use Pydantic models for user-facing parameters
and artifact summaries, run targeted tests plus affected tests before each
commit, inspect the diff, and keep commits focused.
