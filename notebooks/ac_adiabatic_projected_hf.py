# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Finite-LL Aharonov-Casher Projected HF
#
# The workflow is deliberately split into four visible stages:
#
# 1. diagonalize the K-valley finite-LL adiabatic Hamiltonian;
# 2. construct the K' valley by T' and form the two-valley active space;
# 3. project the density-density interaction into that active space by building
#    the `Lambda(q,G,k)` density vertices;
# 4. run the same symmetry-constrained HF references, convex interpolation, and
#    an AC-specific link-variable `cG` response with magnetic-Bloch overlaps.
#
# The key point is that the interaction problem is not inferred from the band
# dispersion alone. The HF backend only sees the one-body blocks `h0(k)` plus
# the projected density vertices built from finite-LL wave-function form
# factors. The response step also needs the AC orbital overlap matrix; ordinary
# LL-coefficient dot products miss the LLL Chern number.

# %%
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chiral_dw.ac.kahler import IdealACKahlerModel
from chiral_dw.ac.nonideal import NonIdealACLLModel
from chiral_dw.ac.projected import (
    build_ac_active_space,
    build_ac_density_vertices,
)
from chiral_dw.ac.response import (
    ACBandOverlapProvider,
    ac_projector_chern,
    k_theta_from_ac_projectors,
)
from chiral_dw.config import (
    ACProjectedHFParams,
    ContinuumGridParams,
    ContinuumHFParams,
    ContinuumInteractionParams,
    FirstShellACParams,
    ResponseParams,
)
from chiral_dw.continuum import (
    ContinuumBundle,
    ContinuumHFBackend,
    MomentumGrid,
    TPrimeConstraint,
    ValleyU1Constraint,
    build_seed,
    build_symmetric_hf_references,
    mix_projector_seeds,
    order_diagnostics,
    projector_maps,
    random_projector_like_seed,
    reference_diagnostics,
    symmetric_convex_path,
)

# %% [markdown]
# ## Parameters and Conventions
#
# The first cell fixes the phenomenological AC model and the numerical controls.
# The noninteracting Hamiltonian is parameterized by low Fourier harmonics of
# the magnetic-field variation and residual scalar potential:
#
# - `b1` controls the first-shell variation of the effective magnetic field in
#   the convention `-B'(r) A_M/(2*pi)`.
# - `u1` controls the first-shell residual scalar potential `U(r)/omega_c`,
#   where the adiabatic paper defines `U = Delta_+ - omega_c xi`.
# - `b2` and `u2` are the corresponding second harmonics on the six vectors
#   `2*G_j`.
#
# This notebook evaluates one fixed point `(b1,u1,b2,u2)`. The companion cluster
# script `scripts/scan_ac_projected_hf_b2_u2.py` sweeps the `b2,u2` plane while
# holding `b1,u1` fixed.
#
# Energies in this low-harmonic AC model are measured in units of the average
# cyclotron energy `omega_c`. The default interaction is the dimensionless
# dual-gate kernel `2*pi*v0*tanh(q*d)/q`, with `d` measured in moire lattice
# spacings. The `"dual_gate"` option instead uses physical units and requires
# consistent `moire_length_nm` and `energy_unit_mev` values.
#
# HF uses ODA mixing by default here because the IVC reference is a constrained
# self-consistency problem and simple linear mixing can stall even when the final
# projector is symmetry-clean. The notebook refuses to run the convex
# interpolation and `cG` response from unconverged HF references unless
# `allow_nonconverged_references` is set explicitly for diagnostics. The
# default parameters below are the ideal LLL benchmark because that is the
# limit where the AC response should reproduce the analytic opposite-Chern
# result `cG ~= -1/(4*pi)` in the current sign convention.

# %%
b1 = 0.3
u1 = 0.0
b2 = 0.0
u2 = 0.0
n_ll = 8
active_band = 0

n_k = 12
q_shell = 1
local_field_cutoff = 1
interaction_strength_scale = 0.2
dimensionless_gate_distance = 2.0

coulomb_kind = "dimensionless_dual_gate"
epsilon = 16.7
gate_distance_nm = 30.0
smear_length_nm = 0.347
moire_length_nm = 1.0
energy_unit_mev = 1.0

hf_max_iter = 800
hf_mixing_method = "oda"
hf_mixing = 0.45
allow_nonconverged_references = False
n_theta = 81
n_phi = 5
phi_step = 0.2
output_dir = ROOT / "results" / "ac_projected_hf_dual_gate"
output_dir.mkdir(parents=True, exist_ok=True)

params = ACProjectedHFParams(
    grid=ContinuumGridParams(n_k=n_k),
    ac=FirstShellACParams(b1=b1, u1=u1, b2=b2, u2=u2, n_ll=n_ll),
    interaction=ContinuumInteractionParams(
        coulomb_kind=coulomb_kind,
        v0=interaction_strength_scale,
        gate_distance=dimensionless_gate_distance,
        q_shell=q_shell,
        local_field_cutoff=local_field_cutoff,
        epsilon=epsilon,
        gate_distance_nm=gate_distance_nm,
        smear_length_nm=smear_length_nm,
    ),
    hf=ContinuumHFParams(
        max_iter=hf_max_iter,
        mixing_method=hf_mixing_method,
        mixing=hf_mixing,
    ),
    response=ResponseParams(n_theta=n_theta, n_phi=n_phi, phi_step=phi_step),
    active_band=active_band,
    moire_length_nm=moire_length_nm,
    energy_unit_mev=energy_unit_mev,
    output_dir=str(output_dir),
)
params

# %% [markdown]
# ## Optional b2-u2 Cluster Sweep
#
# The single-point notebook above is useful for inspecting one Hamiltonian and
# its projected HF references. For a phase diagram, submit the Slurm array below
# from the cluster checkout. It keeps the first harmonics fixed at the notebook
# values `b1,u1` and sweeps the second harmonics over the `B2_*` and `U2_*`
# ranges.

# %%
b2_u2_sweep_command = (
    "sbatch --export=ALL,"
    f"B1_FIXED={b1},U1_FIXED={u1},"
    "B2_MIN=-0.3,B2_MAX=0.3,N_B2=11,"
    "U2_MIN=-0.3,U2_MAX=0.3,N_U2=11 "
    "jobs/scan_ac_projected_hf_b2_u2_array.sh"
)
print(b2_u2_sweep_command)
print(
    "merge with: python scripts/scan_ac_projected_hf_b2_u2.py "
    "--output-root results/ac_b2_u2_cg_dual_gate_n11_nk12_nll8 --merge-only"
)

# %% [markdown]
# ## Single-Particle Finite-LL Band
#
# This section only solves the one-body adiabatic Hamiltonian. The K-valley
# Hamiltonian is diagonalized in a truncated average-field LL basis. The
# selected active hole band is `active_band=0`, and the K' valley is generated
# from the K valley by the non-Kramers T' relation
# `u_Kprime(k) = conj(u_K(-k))`.
#
# The output `active` contains the two-valley one-body blocks `h0(k)`, the
# active-band wave-function coefficients, and the T'-related K' data. No
# interaction has been projected yet in this cell.

# %%
model = NonIdealACLLModel(params.ac)
momentum_grid = MomentumGrid(params.grid.n_k)
active, band_data = build_ac_active_space(
    model,
    momentum_grid,
    active_band=params.active_band,
    diagnostics_n_k=params.band_diagnostics_n_k,
)

print("active h0 shape:", active.h0.shape)
print("active basis coefficients shape:", active.band_vectors.shape)
print("band diagnostics:")
for key, value in band_data.diagnostics.items():
    print(f"  {key}: {value:.8g}")

with (output_dir / "single_particle_band_diagnostics.json").open("w") as f:
    json.dump(band_data.diagnostics, f, indent=2)

# %% [markdown]
# The next plot is a conventional high-symmetry-path view of the finite-LL
# Hamiltonian itself. It is not the interacting HF spectrum. Its role is to show
# whether the selected active band is well separated from the other LL-derived
# minibands before we trust a one-band projection.

# %%
def high_symmetry_path(b1_vec: np.ndarray, b2_vec: np.ndarray, n_per_segment: int = 80):
    gamma = np.zeros(2)
    k_corner = (2.0 * b1_vec + b2_vec) / 3.0
    m_edge = 0.5 * (b1_vec + b2_vec)
    nodes = [gamma, k_corner, m_edge, gamma]
    labels = ["Gamma", "K", "M", "Gamma"]
    points = []
    ticks = [0]
    for start, stop in zip(nodes[:-1], nodes[1:]):
        for t in np.linspace(0.0, 1.0, n_per_segment, endpoint=False):
            points.append((1.0 - t) * start + t * stop)
        ticks.append(len(points))
    points.append(nodes[-1])
    distances = [0.0]
    for a, b in zip(points[:-1], points[1:]):
        distances.append(distances[-1] + float(np.linalg.norm(b - a)))
    return np.asarray(points), np.asarray(distances), ticks, labels


b1_vec, b2_vec = model.fields.G_shell[0], model.fields.G_shell[1]
path_k, path_x, path_ticks, path_labels = high_symmetry_path(b1_vec, b2_vec)
n_plot_bands = min(n_ll, 5)
path_energies = np.asarray([model.solve(k).eigenvalues[:n_plot_bands] for k in path_k])

fig, ax = plt.subplots(figsize=(6.5, 4.0))
for band in range(n_plot_bands):
    ax.plot(path_x, path_energies[:, band], lw=1.5)
for tick in path_ticks:
    ax.axvline(path_x[tick], color="0.75", lw=0.8)
ax.set_xticks([path_x[t] for t in path_ticks])
ax.set_xticklabels(path_labels)
ax.set_ylabel("energy / omega_c")
ax.set_title(f"finite-LL AC bands: b1={b1:g}, u1={u1:g}, b2={b2:g}, u2={u2:g}")
fig.tight_layout()
fig.savefig(output_dir / "single_particle_band_structure.png", dpi=180)
plt.show()

# %% [markdown]
# The Chern number and Berry-curvature spread are computed from the finite-LL
# eigenvectors using the backend's Fukui-link calculation. The projection used
# later is meaningful only when the active band has the expected Chern number
# and a direct gap large compared with the interaction scale.

# %%
centers, berry_dimless, chern = model.berry_curvature_fukui(
    n_k=max(5, params.band_diagnostics_n_k),
    active_band=active_band,
)
fig, ax = plt.subplots(figsize=(5.2, 4.2))
sc = ax.scatter(centers[..., 0].ravel(), centers[..., 1].ravel(), c=berry_dimless.ravel(), s=28)
fig.colorbar(sc, ax=ax, label="Omega / <Omega>")
ax.set_aspect("equal")
ax.set_title(f"finite-LL Berry curvature, C = {chern:.6g}")
fig.tight_layout()
fig.savefig(output_dir / "finite_ll_berry_curvature.png", dpi=180)
plt.show()

# %% [markdown]
# ## Ideal-AC Kahler Diagnostic

# %%
kahler = IdealACKahlerModel(model.fields)
G_b, b_coeff, _A_coeff = model.vector_potential_coefficients()
chi_solution = kahler.solve_chi_from_fourier(G_b, b_coeff, n_grid=64)
G_chi, weighted_phi, raw_phi = kahler.exp2chi_fourier_coeffs(chi_solution)
omega_ac = kahler.dimensionless_berry_curvature(centers, G_chi, weighted_phi)

print("chi mean:", float(np.mean(chi_solution.chi)))
print("ideal AC Berry mean:", float(np.mean(omega_ac)))
print("ideal AC Berry std:", float(np.std(omega_ac)))

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)
im0 = axes[0].imshow(chi_solution.chi.T, origin="lower", cmap="coolwarm")
fig.colorbar(im0, ax=axes[0], label="chi")
axes[0].set_title("Kahler potential chi")
sc = axes[1].scatter(centers[..., 0].ravel(), centers[..., 1].ravel(), c=omega_ac.ravel(), s=28)
fig.colorbar(sc, ax=axes[1], label="Omega_AC / <Omega>")
axes[1].set_aspect("equal")
axes[1].set_title("ideal AC curvature")
fig.savefig(output_dir / "kahler_diagnostic.png", dpi=180)
plt.show()

# %% [markdown]
# ## Project the Interaction
#
# This is the missing bridge between the single-particle band and the HF/cG
# calculation. The projected density operator is represented by vertices
#
# `Lambda[q_index, G_index, k_index, alpha, beta]`.
#
# For this first AC workflow the microscopic density is valley diagonal, so the
# K and K' blocks are filled while intervalley density vertices are zero. For K,
# each vertex is computed from the finite-LL coefficients as
#
# `c_target^dagger M(k_target, k_source, -q + G) c_source`,
#
# where `M` is the LL-basis matrix element of `exp(iG.r)` and `k_source =
# k_target - q` on the discrete mesh. The K' block is built from the T'-related
# active states. The HF backend combines these vertices with `v_over_a(q,G)` to
# build Hartree and Fock terms. Only after this cell do we have an interacting
# projected Hamiltonian problem.

# %%
vertices = build_ac_density_vertices(
    model,
    active,
    params.interaction,
    moire_length_nm=params.moire_length_nm,
    energy_unit_mev=params.energy_unit_mev,
)
backend = ContinuumHFBackend(active.h0, vertices, params.interaction)
bundle = ContinuumBundle(
    grid=active.grid,
    active=active,
    vertices=vertices,
    backend=backend,
    params=active.model,
    interaction=params.interaction,
    bands=band_data,
    geometry=None,
    form_factors=model,
)

q0 = vertices.q_shifts.index((0, 0))
g0 = vertices.g_channels.index((0, 0))
lambda_q0_g0 = vertices.lambda_blocks[q0, g0]
interaction_scale = float(np.max(np.abs(vertices.v_over_a)))
min_direct_gap = float(band_data.diagnostics["min_direct_gap"])
interaction_gap_ratio = interaction_scale / max(min_direct_gap, 1e-15)
backend_diagnostics = {
    "n_blocks": int(backend.n_blocks),
    "dim": int(backend.dim),
    "n_q": int(backend.n_q),
    "n_g": int(backend.n_g),
    "exchange_tVE_shape": list(backend.tVE.shape),
    "h0_norm": float(np.linalg.norm(backend.h0)),
    "exchange_tVE_norm": float(np.linalg.norm(backend.tVE)),
    "hartree_channel_count": int(len(backend.hartree_channels)),
}

print("q transfers:", vertices.q_shifts)
print("local-field G channels:", vertices.g_channels)
print("Lambda shape:", vertices.lambda_blocks.shape)
print("v_over_a range:", float(np.min(vertices.v_over_a)), float(np.max(vertices.v_over_a)))
print("q=0,G=0 Lambda diagonal mean:", np.mean(np.diagonal(lambda_q0_g0, axis1=-2, axis2=-1), axis=0))
print("max interaction / min direct gap:", interaction_gap_ratio)
print("HF backend diagnostics:")
for key, value in backend_diagnostics.items():
    print(f"  {key}: {value}")

with (output_dir / "interaction_projection_diagnostics.json").open("w") as f:
    json.dump(
        {
            "q_shifts": vertices.q_shifts,
            "g_channels": vertices.g_channels,
            "lambda_shape": list(vertices.lambda_blocks.shape),
            "v_over_a_min": float(np.min(vertices.v_over_a)),
            "v_over_a_max": float(np.max(vertices.v_over_a)),
            "q0_g0_identity_error": float(np.max(np.abs(lambda_q0_g0 - np.eye(active.dim)[None, :, :]))),
            "interaction_gap_ratio": float(interaction_gap_ratio),
            "backend": backend_diagnostics,
        },
        f,
        indent=2,
    )

# %% [markdown]
# ## Symmetry Checks
#
# These checks are before HF. They verify that the one-body projected problem
# respects the two symmetries we rely on later:
#
# - valley U(1): the one-body Hamiltonian and density vertices have no
#   intervalley density blocks;
# - T': the K' one-body data is the T'-conjugate of the K data.
#
# A small number here is a convention check. It is not a statement that the HF
# solution has converged.

# %%
tprime = TPrimeConstraint(active)
valley_u1 = ValleyU1Constraint(active)
symmetry_checks = {
    "h0_tprime_error": tprime.symmetry_error(active.h0),
    "h0_valley_u1_error": valley_u1.symmetry_error(active.h0),
    "sample_vertex_valley_u1_error": valley_u1.symmetry_error(vertices.lambda_blocks[q0, g0]),
    "q0_g0_identity_error": float(np.max(np.abs(lambda_q0_g0 - np.eye(active.dim)[None, :, :]))),
    "interaction_gap_ratio": interaction_gap_ratio,
}
for key, value in symmetry_checks.items():
    print(f"{key}: {value:.3e}")

with (output_dir / "symmetry_checks.json").open("w") as f:
    json.dump(symmetry_checks, f, indent=2)

# %% [markdown]
# ## Symmetry-Constrained Hartree-Fock References
#
# The three reference states are solved with the same projected interaction
# vertices constructed above:
#
# - `vp_plus`: valley-polarized seed biased toward K, with valley-U(1)
#   projection applied during HF;
# - `vp_minus`: valley-polarized seed biased toward K', with the same U(1)
#   projection;
# - `ivc`: Q=0 intervalley-coherent seed, with T' projection applied during HF.
#
# The reported energy is the full mesh HF energy in the dimensionless units of
# this projected problem. The per-momentum value is printed alongside it because
# that is easier to compare across mesh sizes. `Nz` is the average valley
# polarization and `IVC` is the intervalley-coherence amplitude from the final
# projector. `converged=False` is not harmless: it means the final idempotent
# Aufbau projector did not satisfy the requested commutator/residual tolerance,
# even though the symmetry constraint itself may be exactly satisfied. In that
# case increase `hf_max_iter`, reduce the interaction scale, or adjust the ODA
# controls before trusting the reference energetics.
#
# For small meshes this section can run quickly because the exchange operator is
# already stored as a dense tensor `tVE`. The timing is therefore not the
# convergence criterion; the residual history and final `converged` flag are.

# %%
hf_constraints = {
    "vp_plus": ValleyU1Constraint(active),
    "vp_minus": ValleyU1Constraint(active),
    "ivc": TPrimeConstraint(active),
}
hf_seed_names = {
    "vp_plus": "vp_plus",
    "vp_minus": "vp_minus",
    "ivc": "ivc",
}


def make_reference_seed(seed_name: str, constraint) -> np.ndarray:
    P0 = build_seed(
        seed_name,
        active,
        n_occ_per_k=params.hf.n_occ_per_k,
        ivc_angle=params.hf.ivc_angle,
        ivc_phase=params.hf.ivc_phase,
        random_seed_value=params.hf.random_seed,
    )
    if params.hf.seed_random_weight > 0.0:
        P_noise = random_projector_like_seed(P0, seed=params.hf.random_seed)
        P_noise = constraint.project_density(P_noise)
        P0 = mix_projector_seeds(
            P0,
            P_noise,
            ordered_weight=params.hf.seed_ordered_weight,
            random_weight=params.hf.seed_random_weight,
        )
    return constraint.project_density(P0)


initial_projectors = {
    name: make_reference_seed(seed_name, hf_constraints[name])
    for name, seed_name in hf_seed_names.items()
}
initial_hf_rows = []
for name, P0 in initial_projectors.items():
    constraint = hf_constraints[name]
    H0 = backend.hf_hamiltonian(P0)
    H0_projected = constraint.project_operator(H0)
    P_aufbau, _evals, direct_gap, _indirect_gap = backend.update_density_per_k(
        H0_projected,
        params.hf.n_occ_per_k,
        constraint,
    )
    components = backend.energy(P0)
    row = {
        "reference": name,
        "seed_energy": float(components.total),
        "seed_one_body": float(components.one_body),
        "seed_hartree": float(components.hartree),
        "seed_fock": float(components.fock),
        "seed_hf_minus_h0_norm": float(np.linalg.norm(H0 - backend.h0)),
        "seed_aufbau_residual_norm": float(np.linalg.norm(P_aufbau - P0)),
        "seed_direct_gap_min": float(direct_gap),
    }
    initial_hf_rows.append(row)
    print(
        f"{name} seed: E={row['seed_energy']:.8g}, "
        f"H_hf-h0={row['seed_hf_minus_h0_norm']:.3e}, "
        f"Aufbau residual={row['seed_aufbau_residual_norm']:.3e}, "
        f"Fock={row['seed_fock']:.8g}"
    )

with (output_dir / "hf_initial_summary.json").open("w") as f:
    json.dump(initial_hf_rows, f, indent=2)

overlap_provider = ACBandOverlapProvider(
    model,
    active_band=params.active_band,
    active=active,
)
hf_start = time.perf_counter()
refs = build_symmetric_hf_references(bundle, params.hf)
hf_elapsed_s = time.perf_counter() - hf_start
ref_rows = {
    "vp_plus": refs.vp_plus,
    "vp_minus": refs.vp_minus,
    "ivc": refs.ivc,
}
hf_summary_rows = []
hf_chern_rows = []
print(f"HF reference solve wall time: {hf_elapsed_s:.3f} s")
for name, result in ref_rows.items():
    diag = result.diagnostics
    order = order_diagnostics(result.P, active, n_occ_per_k=params.hf.n_occ_per_k)
    components = backend.energy(result.P)
    energy_per_k = result.energy / active.n_k
    seed_motion = float(np.linalg.norm(result.P - initial_projectors[name]))
    history_initial_residual = (
        float(result.history[0].aufbau_residual_norm) if result.history else float("nan")
    )
    history_final_residual = (
        float(result.history[-1].aufbau_residual_norm) if result.history else float("nan")
    )
    ac_chern = ac_projector_chern(overlap_provider, active.grid, result.P)
    hf_chern_rows.append({"reference": name, "ac_overlap_chern": float(ac_chern)})
    hf_summary_rows.append(
        {
            "reference": name,
            "energy_total": float(result.energy),
            "energy_per_k": float(energy_per_k),
            "one_body": float(components.one_body),
            "hartree": float(components.hartree),
            "fock": float(components.fock),
            "converged": bool(result.converged),
            "iterations": int(result.n_iter),
            "direct_gap_min": float(diag.direct_gap_min),
            "constraint_error": float(diag.constraint_error),
            "aufbau_residual_norm": float(diag.aufbau_residual_norm),
            "commutator_norm": float(diag.commutator_norm),
            "projector_motion_from_seed": seed_motion,
            "history_initial_aufbau_residual_norm": history_initial_residual,
            "history_final_aufbau_residual_norm": history_final_residual,
            "history_iterations": len(result.history),
            "ac_overlap_chern": float(ac_chern),
            "Nz": float(order.Nz_block),
            "IVC_amplitude": float(order.IVC_amplitude_block),
        }
    )
    print(
        f"{name}: E={result.energy:.8g}, E/Nk={energy_per_k:.8g}, "
        f"converged={result.converged}, it={result.n_iter}, "
        f"gap={diag.direct_gap_min:.6g}, residual={diag.aufbau_residual_norm:.3e}, "
        f"comm={diag.commutator_norm:.3e}, constraint={diag.constraint_error:.3e}, "
        f"||P-P_seed||={seed_motion:.3e}, "
        f"C_AC={ac_chern:.6g}, "
        f"Nz={order.Nz_block:.5g}, IVC={order.IVC_amplitude_block:.5g}"
    )
    if not result.converged:
        print(f"  WARNING: {name} did not meet the HF convergence tolerances.")

channel_diagnostics = reference_diagnostics(refs)
with (output_dir / "reference_channel_diagnostics.json").open("w") as f:
    json.dump({k: v.model_dump() for k, v in channel_diagnostics.items()}, f, indent=2)
with (output_dir / "hf_reference_summary.json").open("w") as f:
    json.dump(
        {
            "wall_time_s": float(hf_elapsed_s),
            "rows": hf_summary_rows,
        },
        f,
        indent=2,
    )
with (output_dir / "hf_chern_diagnostics.json").open("w") as f:
    json.dump(hf_chern_rows, f, indent=2)

fig, ax = plt.subplots(figsize=(5.8, 3.8))
for name, result in ref_rows.items():
    if not result.history:
        continue
    residuals = [row.aufbau_residual_norm for row in result.history]
    ax.semilogy(range(1, len(residuals) + 1), residuals, label=name)
ax.axhline(params.hf.tolerance, color="0.25", ls="--", lw=0.9, label="solver tolerance")
ax.axhline(
    params.hf.final_residual_tolerance,
    color="0.55",
    ls=":",
    lw=0.9,
    label="final residual tolerance",
)
ax.set_xlabel("HF iteration")
ax.set_ylabel("Aufbau residual norm")
ax.set_title("HF self-consistency residuals")
ax.legend()
fig.tight_layout()
fig.savefig(output_dir / "hf_residual_history.png", dpi=180)
plt.show()

# %% [markdown]
# The following maps visualize the final projectors. VP references should put
# nearly all weight in one valley block. The IVC reference should show balanced
# valley occupation and nonzero intervalley coherence. These plots are a
# qualitative check on the labels above, not a substitute for the convergence
# diagnostics.

# %%
def plot_projector_maps(P: np.ndarray, title: str):
    maps = projector_maps(P, active)
    panels = [
        ("K", maps["K"], "viridis", 0.0, 1.0),
        ("Kprime", maps["Kprime"], "viridis", 0.0, 1.0),
        ("VP", maps["VP"], "coolwarm", -1.0, 1.0),
        ("|IVC|", maps["IVC_abs"], "viridis", 0.0, 0.5),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0), constrained_layout=True)
    for ax, (label, data, cmap, vmin, vmax) in zip(axes.flat, panels):
        im = ax.imshow(data.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    return fig


for name, result in ref_rows.items():
    fig = plot_projector_maps(result.P, name)
    fig.savefig(output_dir / f"{name}_projector_maps.png", dpi=180)
    plt.show()

nonconverged_references = [name for name, result in ref_rows.items() if not result.converged]
if nonconverged_references:
    message = (
        "HF references did not converge: "
        + ", ".join(nonconverged_references)
        + ". The convex interpolation and cG response are disabled by default "
        "because they should be built from self-consistent reference Hamiltonians."
    )
    print("WARNING:", message)
    if not allow_nonconverged_references:
        raise RuntimeError(
            message
            + " Increase hf_max_iter, reduce the interaction strength, or set "
            "allow_nonconverged_references=True only for diagnostic runs."
        )

# %% [markdown]
# ## Convex HF Interpolation And cG
#
# The response calculation starts only after the three HF references exist. We
# form the same convex trial Hamiltonian used in the continuum workflow:
#
# `H_var(theta,phi) = w_+ H_VP+ + w_- H_VP- + w_IVC U_phi H_IVC U_phi^dagger`.
#
# For each `theta` edge, the occupied projector is the fixed-per-k Aufbau
# projector of this trial Hamiltonian. The AC `cG` calculation must then use
# magnetic-Bloch orbital overlaps, not ordinary LL-coefficient dot products.
# This is essential in the LLL: the coefficient vector is k-independent, but
# the magnetic-translation overlap still carries Chern number. The link-variable
# evaluator below interleaves active projectors with
# `diag(S_K(k,p), S_Kprime(k,p))`, where `S_Kprime(k,p)=conj(S_K(-k,-p))`.

# %%
response_method = "ac_link_variable_orbital_overlap"
ideal_lll_cg_expected = -1.0 / (4.0 * np.pi)
theta_edges = np.linspace(params.response.theta_min, params.response.theta_max, params.response.n_theta + 1)
phi_nodes = np.arange(params.response.n_phi, dtype=float) * params.response.phi_step
projectors, path_diagnostics = symmetric_convex_path(refs, theta_edges)
projector_grid = projectors.reshape(len(theta_edges), n_k, n_k, active.dim, active.dim)
response = k_theta_from_ac_projectors(overlap_provider, projector_grid, theta_edges, phi_nodes)

print("cG:", response.cG)
print("ideal LLL expected cG:", ideal_lll_cg_expected)
print("K(theta) min/max:", float(np.min(response.K)), float(np.max(response.K)))
print("projector grid shape:", projector_grid.shape)
print("phi nodes:", phi_nodes)
print("response method:", response_method)

fig, ax = plt.subplots(figsize=(5.8, 3.8))
ax.plot(response.theta / np.pi, response.K, marker="o", ms=3)
ax.set_xlabel("theta / pi")
ax.set_ylabel("K(theta)")
ax.set_title(f"dimensionless cG = {response.cG:.6g}")
fig.tight_layout()
fig.savefig(output_dir / "K_theta_cG.png", dpi=180)
plt.show()

# %% [markdown]
# The final cell writes the numerical response and enough metadata to audit the
# run later. JSON summaries are generated artifacts under `results/`, not source
# configuration. The dimensionless `cG` is reported first, following the repo
# convention.

# %%
with (output_dir / "response_K_theta.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["theta", "theta_over_pi", "K_theta", "cG"])
    writer.writeheader()
    for theta_value, K_value in zip(response.theta, response.K):
        writer.writerow(
            {
                "theta": float(theta_value),
                "theta_over_pi": float(theta_value / np.pi),
                "K_theta": float(K_value),
                "cG": float(response.cG),
            }
        )

np.savez_compressed(
    output_dir / "ac_projected_hf_response.npz",
    theta=response.theta,
    theta_edges=theta_edges,
    phi_nodes=phi_nodes,
    K=response.K,
    cG=np.array(response.cG),
    projectors=projector_grid,
)

summary = {
    "params": params.model_dump(mode="json"),
    "band_diagnostics": band_data.diagnostics,
    "symmetry_checks": symmetry_checks,
    "interaction_projection_diagnostics": backend_diagnostics,
    "hf_initial_summary": initial_hf_rows,
    "hf_reference_summary": hf_summary_rows,
    "hf_chern_summary": hf_chern_rows,
    "hf_wall_time_s": float(hf_elapsed_s),
    "allow_nonconverged_references": bool(allow_nonconverged_references),
    "response_method": response_method,
    "response_phi_nodes": [float(x) for x in phi_nodes],
    "ideal_lll_cg_expected_current_sign": float(ideal_lll_cg_expected),
    "cG": float(response.cG),
    "cG_dimension": "dimensionless",
}
with (output_dir / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)

summary
