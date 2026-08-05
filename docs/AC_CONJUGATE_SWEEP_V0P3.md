# Conjugate-AC `v0=0.3` finite-size sweep

## Production parameters

The production wrapper `jobs/scan_ac_projected_hf_b1_u1_array.sh` runs the
finite-LL conjugate Aharonov-Casher projected-HF calculation with:

- `b1,u1 = -0.1,-0.09,...,0.09,0.1` (a `21 x 21` grid);
- `n_k = 18,20,21,22,24`;
- `N_LL = 8`, with the interaction projected to the lowest AC band;
- the C3-invariant full-q channel construction with local-field cutoff `L=1`;
- dual-gate screening with `d=30 nm` and smearing length `0.347 nm`;
- `v0 = E_C/(hbar*omega_c) = 0.3`, with no dielectric-amplitude parameter;
- four vertex and exchange workers per SLURM task, while BLAS/OpenMP thread
  pools remain fixed to one thread to prevent nested oversubscription.

There are `5 * 21 * 21 = 2205` shards.  Every shard is restartable through
`--skip-existing`.  A point writes the scalar summary, reference diagnostics,
HF Chern table, response table, projector arrays, and interpolated-path gap
records required by the per-mesh merge.

## Staged cluster launch

From the repository root on the cluster, first run exactly one shard through
the production wrapper:

```bash
mkdir -p logs
sbatch --array=0-0 jobs/scan_ac_projected_hf_b1_u1_array.sh
```

After it completes, verify that
`results/ac_b1_u1_cg_dual_gate_omega_v0p3_nll8_grid21_nk18_24/nk18/points/b_000_u_000/`
contains `point_summary.json`, `point_params.json`, `reference_states.npz`,
`response.npz`, `reference_diagnostics.csv`, `hf_chern_numbers.csv`,
`response_K_theta.csv`, and `path_theta_edges.csv`.  Then test one complete
mesh at conservative concurrency:

```bash
sbatch --array=0-440%12 jobs/scan_ac_projected_hf_b1_u1_array.sh
```

Merge and inspect the `n_k=18` mesh before scaling further:

```bash
python scripts/scan_ac_projected_hf_b1_u1.py \
  --output-root results/ac_b1_u1_cg_dual_gate_omega_v0p3_nll8_grid21_nk18_24/nk18 \
  --merge-only
```

The merged `sweep.csv` must have 441 unique parameter rows.  HF convergence
controls whether `cG` and the path gaps are computed; Chern admissibility is a
separate diagnostic and never gates the response.  The file
`sweep_path_theta_edges.csv` stores the interpolated-path gaps.  Once the
one-mesh audit passes, submit the full restartable array:

```bash
sbatch --array=0-2204%12 jobs/scan_ac_projected_hf_b1_u1_array.sh
```

The already-completed `n_k=18` points are skipped automatically.  Merge each
`nk*` directory independently after the array finishes.

## Magnetic-Bloch sewing correction and stored-point repair

The AC active band is a magnetic Bloch bundle.  Reciprocal-boundary links must
therefore include the exact Landau-level transport phase and cocycle parity.
The current overlap provider folds both endpoints into the archived active
frame, constructs the `Kprime` overlap by applying `Tprime` to the already-sewn
`K` frame, and uses the resulting sewn overlaps for both lattice-Chern and
mixed-curvature `cG` calculations.

The topology artifact records the minimum occupied-link magnitude, number of
links below tolerance, integer residual, plaquette branch margin, and
translated-edge closure.  A positive Hamiltonian gap does not guarantee that
this particular link discretization is admissible: a projector can remain
gapped while a neighboring occupied-state overlap is zero or nearly zero.
Such a point is labeled `numerically_unresolved`, but its sewn `cG` and path
gaps are still computed when HF converged.

Completed points can be repaired directly from `reference_states.npz` without
rerunning HF.  The command below writes a new `sewn_response_v1/` sidecar and
refuses to overwrite either the archived point or an existing sidecar:

```bash
python scripts/recompute_ac_sewn_response.py \
  results/ac_b1_u1_cg_dual_gate_omega_v0p3_nll8_grid21_nk18_24/nk18/points/b_009_u_015
```

The sidecar contains `summary.json`, `chern_diagnostics.csv`,
`path_theta_edges.csv`, `response_K_theta.csv`, and `response.npz`.  Before any
cluster repair array is launched, run this exact command on one archived point
and verify all five outputs.  Then select only the points requiring repair and
scale gradually; do not rerun the HF sweep.

## Finite-size analysis note

The meshes split into two commensurability classes: `n_k=18,21,24` are
divisible by three, whereas `n_k=20,22` are not.  Preserve all five raw values,
but inspect the two sequences separately before fitting a single extrapolation.
Report the spread between commensurate-only and combined fits as a finite-size
systematic if the two classes do not collapse smoothly.

At `v0=0.3`, the characteristic interaction exceeds the earlier `0.25`
projection-warning threshold by construction.  The wrapper therefore uses a
tight guard value of `0.31`: the requested run passes, while an accidental
increase of `v0` still fails before the expensive vertex construction.
