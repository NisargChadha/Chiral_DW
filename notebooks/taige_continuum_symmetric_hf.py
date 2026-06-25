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
# # Taige Continuum Symmetric HF
#
# Native Chiral_DW workflow for Taige-parameter tMoTe2: continuum bands, Chern numbers, density vertices/form factors, dual-gated smeared Coulomb HF, three constrained references, the symmetric convex trial Hamiltonian, and the post-HF chiral-domain-wall charge response.
#
# The notebook is organized so that parameters appear only when the next stage needs them. You should be able to change HF iteration controls without rerunning the continuum band plot, and change the domain-wall texture after HF without rerunning HF.
#

# %% [markdown]
# ## Setup Imports
#
# This cell imports the numerical stack and Chiral_DW APIs used throughout the notebook. It also makes the local `src/` tree importable when the notebook is launched from either the repository root or the `notebooks/` directory.

# %%

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd().resolve()
if not (ROOT / "src").exists() and (ROOT.parent / "src").exists():
    ROOT = ROOT.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from chiral_dw.config import ContinuumGridParams, ContinuumHFParams, DomainWallParams
from chiral_dw.continuum import (
    SymmetricHFReferences,
    TPrimeConstraint,
    ValleyU1Constraint,
    build_continuum_bundle,
    build_seed,
    chern_number_table,
    compute_taige_path_spectrum,
    mix_projector_seeds,
    projector_maps,
    random_projector_like_seed,
    reference_diagnostics,
    solve_hf,
    symmetric_convex_path,
    taige_interaction_params,
    taige_model_params,
)
from chiral_dw.domain_wall import charge_density_radial
from chiral_dw.response import k_theta_from_projectors

plt.rcParams.update({"figure.dpi": 120})


# %% [markdown]
# ## Continuum Model Parameters
#
# These are the only parameters needed to inspect the non-interacting continuum bands. The defaults are Taige's MoTe2 values with the requested starting point `theta_deg = 3.5` and `u_D = 0`.
#
# `plane_wave_shell` controls the plane-wave cutoff in the continuum diagonalization. The band plot is sensitive to this cutoff: small values can visibly break the expected K/Kprime time-reversal overlap along the path.
#

# %%

# Continuum model parameters.
theta_deg = 3.5
u_D = 0.0
plane_wave_shell = 5
n_bands = 2

# Band-plot resolution along each high-symmetry path segment.
path_n_per_segment = 48

run_label = f"theta{theta_deg:g}_uD{u_D:g}_shell{plane_wave_shell}"
result_dir = ROOT / "results" / "taige_continuum_symmetric_hf" / run_label
result_dir.mkdir(parents=True, exist_ok=True)

model = taige_model_params(
    theta_deg=theta_deg,
    u_D=u_D,
    plane_wave_shell=plane_wave_shell,
    n_bands=n_bands,
)

print(model)
print("result_dir:", result_dir)


# %% [markdown]
# ## Continuum Band Structure
#
# This cell diagonalizes the non-interacting continuum Hamiltonian along the standard Taige path. The Kprime branch is generated in the same T-prime convention used for the active-space mesh: it is evaluated from the K valley at folded `-k`, rather than by comparing to a separately gauged direct Kprime Hamiltonian.
#
# The diagnostic printout checks the maximum K/Kprime mismatch in the plotted hole bands and whether the lowest plotted hole band has its maximum at Gamma.
#

# %%

path_data = compute_taige_path_spectrum(model, n_per_segment=path_n_per_segment)
distances = path_data["distances"]
ticks = path_data["ticks"]
labels = path_data["labels"]
hole_path = path_data["hole_energies"]
gamma_index = ticks[0]
valley_mismatch = float(np.max(np.abs(hole_path[:, 0, :] - hole_path[:, 1, :])))
lowest_hole_peak_index = int(np.argmax(hole_path[:, 0, 0]))

fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
for iv, valley in enumerate(("K", "Kprime")):
    style = "-" if valley == "K" else "--"
    for band in range(hole_path.shape[-1]):
        ax.plot(distances, hole_path[:, iv, band], style, lw=1.3, label=f"{valley} hole band {band}")
for tick in ticks:
    ax.axvline(distances[tick], color="0.85", lw=0.8)
ax.set_xticks([distances[t] for t in ticks], labels)
ax.set_ylabel("hole energy (meV)")
ax.set_title(f"Taige continuum path, theta={theta_deg:g} deg, u_D={u_D:g} meV")
ax.legend(fontsize="small", ncols=2)
fig.savefig(result_dir / "continuum_band_path.png", dpi=180)
plt.show()

print("max |K - Kprime| in plotted hole bands:", valley_mismatch)
print("lowest plotted hole band peak index:", lowest_hole_peak_index)
print("Gamma index:", gamma_index)
print("lowest plotted hole band peak at Gamma:", lowest_hole_peak_index == gamma_index)


# %% [markdown]
# ## Active Space And Interaction Parameters
#
# The continuum band plot above does not require an HF mesh or Coulomb kernel. Those controls enter here, where we build the active hole basis, density vertices/form factors, and dual-gated smeared Coulomb weights used by HF.
#
# Increase `n_k`, `q_mesh`, `q_shell`, and `local_field_cutoff` for more faithful production runs. The default is intentionally modest so the notebook remains interactive.
#

# %%

n_active_bands_per_valley = 1
n_k = 6

q_mesh = "full"  # "shell" for quick scans, "full" for all mesh transfers
q_shell = 1
local_field_cutoff = 0
include_q0 = True

model = model.model_copy(
    update={"n_active_bands_per_valley": n_active_bands_per_valley}
)
interaction = taige_interaction_params(
    include_q0=include_q0,
    q_mesh=q_mesh,
    q_shell=q_shell,
    local_field_cutoff=local_field_cutoff,
)
grid_params = ContinuumGridParams(n_k=n_k)

bundle = build_continuum_bundle(model=model, grid=grid_params, interaction=interaction)
active = bundle.active
vertices = bundle.vertices
backend = bundle.backend

print(model)
print(interaction)
print("grid blocks:", active.n_k)
print("active block dimension:", active.dim)
print("plane waves per layer:", active.n_plane_waves)
print("h0:", active.h0.shape)
print("hole vectors:", active.band_vectors.shape)
print("lambda blocks:", vertices.lambda_blocks.shape)
print("v_over_a:", vertices.v_over_a.shape)
print("nonzero interaction channels:", int(np.count_nonzero(vertices.v_over_a)))
print("T-prime sewing quality min:", float(np.min(bundle.bands.tprime_sewing_quality)))


# %% [markdown]
# ## Chern Numbers
#
# This cell computes Fukui Chern numbers for the selected active hole band and its electron counterpart on the same coarse mesh used for HF. The signs provide a quick convention check for the opposite-Chern K and Kprime active bands.
#

# %%

chern_rows = chern_number_table(bundle.bands, band_indices=tuple(range(n_active_bands_per_valley)))
for row in chern_rows:
    print(f"{row.basis:8s} {row.valley:6s} band {row.band}: C = {row.chern:+.6f}")


# %% [markdown]
# ## Density Vertex And Coulomb Diagnostics
#
# This cell inspects the projected density vertex table and the interaction weights. The `q=0` density vertex should be the identity in the active subspace; a large error here would indicate a form-factor convention problem before HF begins.
#

# %%

q_norm = vertices.q_norm_nm_inv.reshape(-1) if vertices.q_norm_nm_inv is not None else np.arange(vertices.v_over_a.size)
weights = vertices.v_over_a.reshape(-1)

fig, ax = plt.subplots(figsize=(5.5, 3.8), constrained_layout=True)
ax.scatter(q_norm, weights, s=28)
ax.set_xlabel(r"|q+G| (nm$^{-1}$)")
ax.set_ylabel(r"V(q+G)/A")
ax.set_title("dual-gated smeared Coulomb weights")
fig.savefig(result_dir / "interaction_weights.png", dpi=180)
plt.show()

iq0 = vertices.q_shifts.index((0, 0))
q0_identity_error = np.max(np.abs(vertices.lambda_blocks[iq0, 0] - np.eye(active.dim)))
print("q=0 identity error:", float(q0_identity_error))


# %% [markdown]
# ## Projector Visualization And HF Helpers
#
# These helper functions are used by the HF cells below. They do not run HF or change the model; they only define common plots, seed construction, and diagnostic summaries so the three reference solves have the same presentation.
#

# %%

def plot_projector(P, title):
    maps = projector_maps(P, active)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2), constrained_layout=True)
    specs = [
        ("K occupation", maps["K"], "viridis"),
        ("Kprime occupation", maps["Kprime"], "viridis"),
        ("valley polarization", maps["VP"], "coolwarm"),
        ("|IVC|", maps["IVC_abs"], "magma"),
    ]
    for ax, (label, values, cmap) in zip(axes, specs):
        image = ax.imshow(values, origin="lower", cmap=cmap)
        ax.set_title(label)
        ax.set_xlabel("k2")
        ax.set_ylabel("k1")
        fig.colorbar(image, ax=ax, shrink=0.78)
    fig.suptitle(title)
    return fig


def noisy_initial_projector(seed_name, rng_seed):
    ordered = build_seed(
        seed_name,
        active,
        n_occ_per_k=hf_params.n_occ_per_k,
        ivc_angle=hf_params.ivc_angle,
        ivc_phase=hf_params.ivc_phase,
        random_seed_value=rng_seed,
    )
    noise = random_projector_like_seed(ordered, seed=rng_seed)
    mixed = mix_projector_seeds(
        ordered,
        noise,
        ordered_weight=hf_params.seed_ordered_weight,
        random_weight=hf_params.seed_random_weight,
    )
    return mixed, ordered, noise


def diagnostics_rows(result):
    rows = [diag.model_dump(mode="json") for diag in result.history]
    rows.append(result.diagnostics.model_dump(mode="json"))
    return rows


def plot_hf_history(result, title):
    rows = diagnostics_rows(result)
    iterations = [row["iteration"] for row in rows]
    energies = [row["energy"] for row in rows]
    residual = [row["aufbau_residual_norm"] for row in rows]
    commutator = [row["commutator_norm"] for row in rows]
    idem = [row["idempotency_error_fro"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    axes[0].plot(iterations, energies, marker="o", ms=3)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("HF energy")
    axes[0].set_title(title)
    axes[1].semilogy(iterations, np.maximum(residual, 1e-16), label="Aufbau")
    axes[1].semilogy(iterations, np.maximum(commutator, 1e-16), label="commutator")
    axes[1].semilogy(iterations, np.maximum(idem, 1e-16), label="idempotency")
    axes[1].set_xlabel("iteration")
    axes[1].legend(fontsize="small")
    return fig


def summarize_result(name, result):
    d = result.diagnostics
    print(f"{name}: energy={result.energy:.12g}, converged={result.converged}, iterations={result.n_iter}")
    print("  idempotency fro/max:", d.idempotency_error_fro, d.idempotency_error_max)
    print("  residual/commutator:", d.aufbau_residual_norm, d.commutator_norm)
    print("  trace/constraint:", d.trace_error, d.constraint_error)
    print("  direct/indirect gap:", d.direct_gap_min, d.indirect_gap)
    print("  snapshots:", [snapshot.iteration for snapshot in result.snapshots])



# %% [markdown]
# ## Hartree-Fock Run Parameters
#
# These controls are only needed once the active space and interaction backend exist. You can change iteration counts, tolerances, seed noise, and snapshot cadence here without rebuilding the continuum band plot above.
#
# The seed weights implement the requested 0.8 ordered / 0.2 random projector-like mixture for VP+, VP-, and IVC initial conditions.
#

# %%

hf_params = ContinuumHFParams(
    n_occ_per_k=1,
    max_iter=80,
    min_iter=3,
    mixing_method="oda",
    mixing=0.45,
    tolerance=1e-8,
    energy_tolerance=1e-10,
    seed_ordered_weight=0.8,
    seed_random_weight=0.2,
    random_seed=7,
    store_projector_snapshots=True,
    snapshot_interval=5,
    first_iteration_snapshot=True,
)

print(hf_params)


# %% [markdown]
# ## Initial Noisy Projectors
#
# This cell constructs and displays the actual initial projectors for the three HF references. Each ordered VP/IVC seed is mixed with a projector-like random Slater seed using the weights in `hf_params`.
#
# The figures are meant to catch obvious seed mistakes before the self-consistency loop starts: VP seeds should be valley diagonal, while the IVC seed should show intervalley coherence.
#

# %%

P0_vp_plus, P_ordered_vp_plus, P_noise_vp_plus = noisy_initial_projector("vp_plus", hf_params.random_seed + 1)
P0_vp_minus, P_ordered_vp_minus, P_noise_vp_minus = noisy_initial_projector("vp_minus", hf_params.random_seed + 2)
P0_ivc, P_ordered_ivc, P_noise_ivc = noisy_initial_projector("ivc", hf_params.random_seed + 3)

for label, P0 in [("VP+ initial", P0_vp_plus), ("VP- initial", P0_vp_minus), ("IVC initial", P0_ivc)]:
    fig = plot_projector(P0, label)
    fig.savefig(result_dir / f"{label.lower().replace(' ', '_').replace('+', 'plus').replace('-', 'minus')}.png", dpi=180)
    plt.show()


# %% [markdown]
# ## VP+ Valley-U(1) HF
#
# This cell runs the K-valley polarized reference. The `ValleyU1Constraint` removes intervalley density/operator blocks, and `pinned_valley="K"` keeps the Aufbau update in the VP+ sector rather than allowing momentum-by-momentum valley flips.
#
# The final diagnostics report both projector idempotency and self-consistency residuals. A converged reference should have an idempotent final projector and a small Aufbau residual.
#

# %%

vp_plus_constraint = ValleyU1Constraint(active, pinned_valley="K")
vp_plus = solve_hf(
    backend,
    P0_vp_plus,
    hf_params,
    constraint=vp_plus_constraint,
    seed="vp_plus_0p8_ordered_0p2_random",
)
summarize_result("VP+", vp_plus)
fig = plot_hf_history(vp_plus, "VP+ constrained HF")
fig.savefig(result_dir / "vp_plus_hf_history.png", dpi=180)
plt.show()
fig = plot_projector(vp_plus.P, "VP+ final projector")
fig.savefig(result_dir / "vp_plus_final_projector.png", dpi=180)
plt.show()


# %% [markdown]
# ## VP- Valley-U(1) HF
#
# This cell repeats the same U(1)-preserving solve in the Kprime-polarized sector. At `u_D = 0`, VP+ and VP- should become degenerate once all continuum, form-factor, and interaction conventions are sufficiently symmetric.
#
# If the printed energy splitting remains visible, treat it as a convention or finite-cutoff diagnostic rather than a physical displacement-field effect.
#

# %%

vp_minus_constraint = ValleyU1Constraint(active, pinned_valley="Kprime")
vp_minus = solve_hf(
    backend,
    P0_vp_minus,
    hf_params,
    constraint=vp_minus_constraint,
    seed="vp_minus_0p8_ordered_0p2_random",
)
summarize_result("VP-", vp_minus)
fig = plot_hf_history(vp_minus, "VP- constrained HF")
fig.savefig(result_dir / "vp_minus_hf_history.png", dpi=180)
plt.show()
fig = plot_projector(vp_minus.P, "VP- final projector")
fig.savefig(result_dir / "vp_minus_final_projector.png", dpi=180)
plt.show()


# %% [markdown]
# ## Q=0 IVC T-Prime HF
#
# This cell runs the Q=0 intervalley-coherent reference with the non-Kramers T-prime constraint. The seed carries explicit K/Kprime coherence, and the constraint relates the projector at `k` to the valley-swapped complex conjugate at `-k`.
#
# The final idempotent projector is always reported, even when the residual says the idempotent projector is not yet a fully self-consistent HF fixed point.
#

# %%

ivc_constraint = TPrimeConstraint(active)
ivc = solve_hf(
    backend,
    P0_ivc,
    hf_params,
    constraint=ivc_constraint,
    seed="ivc_0p8_ordered_0p2_random",
)
summarize_result("IVC", ivc)
fig = plot_hf_history(ivc, "IVC T-prime constrained HF")
fig.savefig(result_dir / "ivc_hf_history.png", dpi=180)
plt.show()
fig = plot_projector(ivc.P, "IVC final projector")
fig.savefig(result_dir / "ivc_final_projector.png", dpi=180)
plt.show()


# %% [markdown]
# ## Reference Energies And Channel Diagnostics
#
# The VP splitting printed below is a convention/truncation check. At `u_D = 0`, a nonzero value indicates the finite active-space/form-factor setup or the selected mesh controls are still breaking the expected VP+/VP- equivalence.
#

# %%

refs = SymmetricHFReferences(vp_plus=vp_plus, vp_minus=vp_minus, ivc=ivc, n_occ_per_k=hf_params.n_occ_per_k)
energies = {
    "VP+": vp_plus.energy,
    "VP-": vp_minus.energy,
    "IVC": ivc.energy,
}
for key, value in energies.items():
    print(f"{key:4s}: {value:.12g}")
print("VP splitting E(VP+) - E(VP-):", energies["VP+"] - energies["VP-"])

for name, diag in reference_diagnostics(refs).items():
    print(name, diag.model_dump(mode="json"))


# %% [markdown]
# ## Symmetric Convex Trial Hamiltonian Path
#
# The HF references are now fixed, so this is the first place where the `theta` path and global IVC phase are needed. Changing these values recomputes the trial Hamiltonian path and `K(theta)` without rerunning the three HF solves.
#

# %%

n_theta = 41
endpoint_eps = 1e-4
phi = 0.0

print("n_theta:", n_theta)
print("endpoint_eps:", endpoint_eps)
print("phi:", phi)


# %% [markdown]
# This cell builds the convex Hamiltonian path from the three raw HF Hamiltonians, diagonalizes each point into a fixed-occupation projector, and computes the dimensionless response kernel `K(theta)` and coefficient `c_G`.
#

# %%

theta_nodes = np.linspace(endpoint_eps, np.pi - endpoint_eps, n_theta)
projectors_flat, path_diagnostics = symmetric_convex_path(refs, theta_nodes, phi=phi)
projectors = projectors_flat.reshape(n_theta, n_k, n_k, active.dim, active.dim)
if active.dim != 2:
    raise ValueError("charge-response cell currently expects one active band per valley")
response = k_theta_from_projectors(projectors, theta_nodes)

gaps = np.array([row.direct_gap_min for row in path_diagnostics])
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
axes[0].plot(response.theta / np.pi, response.K, marker="o", ms=3)
axes[0].set_xlabel(r"$\theta/\pi$")
axes[0].set_ylabel(r"$K(\theta)$")
axes[0].set_title(f"c_G = {response.cG:.6g}")
axes[1].plot(theta_nodes / np.pi, gaps, marker="o", ms=3)
axes[1].set_xlabel(r"$\theta/\pi$")
axes[1].set_ylabel("direct gap")
axes[1].set_title("trial Hamiltonian gap")
fig.savefig(result_dir / "convex_path_response.png", dpi=180)
plt.show()
print("cG:", response.cG)


# %% [markdown]
# ## Post-HF Texture Controls
#
# The domain-wall texture parameters enter only after `K(theta)` is known. Edit `R` and `w` here to recompute the radial texture and charge density from the already computed HF references and convex path.
#
# This cell does not rerun continuum bands, density vertices, or HF.
#

# %%

R = 20.0
w = 3.0
winding = 1

domain_wall = DomainWallParams(radius=R, width=w, winding=winding)
r_max = max(2.0 * R, R + 8.0 * w)
r = np.linspace(max(1e-6, r_max / 800.0), r_max, 800)
profile = charge_density_radial(r, response.theta, response.K, domain_wall)

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
axes[0].plot(profile.r, profile.theta / np.pi)
axes[0].set_xlabel("r / a_M")
axes[0].set_ylabel(r"$\theta(r)/\pi$")
axes[0].set_title(f"R={R:g}, w={w:g}")
axes[1].plot(profile.r, profile.rho_dimless)
axes[1].set_xlabel("r / a_M")
axes[1].set_ylabel(r"$\rho a_M^2$")
axes[1].set_title("dimensionless charge density")
fig.savefig(result_dir / "charge_profile.png", dpi=180)
plt.show()

integrated_charge = float(np.trapezoid(2.0 * np.pi * profile.r * profile.rho_dimless, profile.r))
print("cG:", response.cG)
print("integrated radial charge:", integrated_charge)


# %% [markdown]
# ## Save Arrays
#
# This cell writes the converged reference projectors, convex-path projectors, `K(theta)`, `c_G`, and the current post-HF charge profile into a compressed NumPy artifact under `results/`.
#

# %%

np.savez_compressed(
    result_dir / "taige_symmetric_hf_references_and_response.npz",
    theta=response.theta,
    K=response.K,
    cG=np.array(response.cG),
    vp_plus_P=vp_plus.P,
    vp_minus_P=vp_minus.P,
    ivc_P=ivc.P,
    convex_projectors=projectors,
    charge_r=profile.r,
    charge_theta=profile.theta,
    charge_rho_dimless=profile.rho_dimless,
)
print("saved:", result_dir / "taige_symmetric_hf_references_and_response.npz")
