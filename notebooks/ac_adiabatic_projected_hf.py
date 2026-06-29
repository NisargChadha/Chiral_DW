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
# This notebook builds the adiabatic finite-LL Aharonov-Casher band structure,
# checks the band isolation diagnostics, projects the interaction into the
# lowest hole band per valley, and reuses the symmetric HF and response
# machinery from the continuum workflow.

# %%
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = ROOT.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chiral_dw.ac.kahler import IdealACKahlerModel
from chiral_dw.ac.projected import build_ac_projected_bundle
from chiral_dw.config import (
    ACProjectedHFParams,
    ContinuumGridParams,
    ContinuumHFParams,
    ContinuumInteractionParams,
    FirstShellACParams,
    ResponseParams,
)
from chiral_dw.continuum import (
    TPrimeConstraint,
    ValleyU1Constraint,
    active_basis_frames,
    build_symmetric_hf_references,
    order_diagnostics,
    projector_maps,
    reference_diagnostics,
    symmetric_convex_path,
)
from chiral_dw.response import k_theta_from_projectors_with_basis

# %% [markdown]
# ## Parameters

# %%
b1 = 0.2
u1 = 0.05
n_ll = 5
active_band = 0

n_k = 5
q_shell = 1
local_field_cutoff = 1
interaction_strength_scale = 0.2
dimensionless_gate_distance = 2.0

coulomb_kind = "dimensionless_screened"
epsilon = 16.7
gate_distance_nm = 30.0
smear_length_nm = 0.347
moire_length_nm = 1.0
energy_unit_mev = 1.0

hf_max_iter = 60
hf_mixing = 0.45
n_theta = 21
output_dir = ROOT / "results" / "ac_projected_hf"
output_dir.mkdir(parents=True, exist_ok=True)

params = ACProjectedHFParams(
    grid=ContinuumGridParams(n_k=n_k),
    ac=FirstShellACParams(b1=b1, u1=u1, n_ll=n_ll),
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
    hf=ContinuumHFParams(max_iter=hf_max_iter, mixing=hf_mixing),
    response=ResponseParams(n_theta=n_theta),
    active_band=active_band,
    moire_length_nm=moire_length_nm,
    energy_unit_mev=energy_unit_mev,
    output_dir=str(output_dir),
)
params

# %% [markdown]
# ## Single-Particle Finite-LL Band

# %%
bundle = build_ac_projected_bundle(params)
active = bundle.active
model = bundle.form_factors
band_data = bundle.bands

print("active h0 shape:", active.h0.shape)
print("density vertex shape:", bundle.vertices.lambda_blocks.shape)
print("band diagnostics:")
for key, value in band_data.diagnostics.items():
    print(f"  {key}: {value:.8g}")

with (output_dir / "single_particle_band_diagnostics.json").open("w") as f:
    json.dump(band_data.diagnostics, f, indent=2)

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
ax.set_title("finite-LL AC band structure")
fig.tight_layout()
fig.savefig(output_dir / "single_particle_band_structure.png", dpi=180)
plt.show()

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
# ## Symmetry Checks

# %%
tprime = TPrimeConstraint(active)
valley_u1 = ValleyU1Constraint(active)
symmetry_checks = {
    "h0_tprime_error": tprime.symmetry_error(active.h0),
    "h0_valley_u1_error": valley_u1.symmetry_error(active.h0),
    "sample_vertex_valley_u1_error": valley_u1.symmetry_error(bundle.vertices.lambda_blocks[0, 0]),
}
for key, value in symmetry_checks.items():
    print(f"{key}: {value:.3e}")

with (output_dir / "symmetry_checks.json").open("w") as f:
    json.dump(symmetry_checks, f, indent=2)

# %% [markdown]
# ## Symmetry-Constrained Hartree-Fock References

# %%
refs = build_symmetric_hf_references(bundle, params.hf)
ref_rows = {
    "vp_plus": refs.vp_plus,
    "vp_minus": refs.vp_minus,
    "ivc": refs.ivc,
}
for name, result in ref_rows.items():
    diag = result.diagnostics
    order = order_diagnostics(result.P, active, n_occ_per_k=params.hf.n_occ_per_k)
    print(
        f"{name}: E={result.energy:.8g}, converged={result.converged}, "
        f"gap={diag.direct_gap_min:.6g}, constraint={diag.constraint_error:.3e}, "
        f"Nz={order.Nz_block:.5g}, IVC={order.IVC_amplitude_block:.5g}"
    )

channel_diagnostics = reference_diagnostics(refs)
with (output_dir / "reference_channel_diagnostics.json").open("w") as f:
    json.dump({k: v.model_dump() for k, v in channel_diagnostics.items()}, f, indent=2)

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

# %% [markdown]
# ## Convex HF Interpolation And cG

# %%
theta_nodes = np.linspace(
    params.response.endpoint_eps,
    np.pi - params.response.endpoint_eps,
    params.response.n_theta,
)
projectors, path_diagnostics = symmetric_convex_path(refs, theta_nodes)
frames = active_basis_frames(active).reshape(n_k, n_k, -1, active.dim)
projector_grid = projectors.reshape(len(theta_nodes), n_k, n_k, active.dim, active.dim)
response = k_theta_from_projectors_with_basis(projector_grid, theta_nodes, frames)

print("cG:", response.cG)
print("K(theta) min/max:", float(np.min(response.K)), float(np.max(response.K)))

fig, ax = plt.subplots(figsize=(5.8, 3.8))
ax.plot(response.theta / np.pi, response.K, marker="o", ms=3)
ax.set_xlabel("theta / pi")
ax.set_ylabel("K(theta)")
ax.set_title(f"dimensionless cG = {response.cG:.6g}")
fig.tight_layout()
fig.savefig(output_dir / "K_theta_cG.png", dpi=180)
plt.show()

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
    K=response.K,
    cG=np.array(response.cG),
    projectors=projector_grid,
)

summary = {
    "params": params.model_dump(mode="json"),
    "band_diagnostics": band_data.diagnostics,
    "symmetry_checks": symmetry_checks,
    "cG": float(response.cG),
    "cG_dimension": "dimensionless",
}
with (output_dir / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)

summary
