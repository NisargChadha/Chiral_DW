"""End-to-end nonideal AC cG workflow."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chiral_dw.ac.energy import ProjectedPhysicalEnergy
from chiral_dw.ac.nonideal import NonIdealACLLModel
from chiral_dw.ac.source import FlavorSourceProjector, target_vector
from chiral_dw.artifacts import RunArtifact, RunManifest
from chiral_dw.config import ACResponseWorkflowParams, ChargeResponseSummary
from chiral_dw.domain_wall import DomainWallChargeProfile, charge_density_radial
from chiral_dw.response import KThetaResult, k_theta_from_projectors, projector_errors


@dataclass(frozen=True)
class ACCGWorkflowResult:
    """In-memory result for one nonideal AC cG run."""

    params: ACResponseWorkflowParams
    theta: np.ndarray
    projectors: np.ndarray
    gaps: np.ndarray
    energy_total: np.ndarray
    energy_band: np.ndarray
    energy_hartree: np.ndarray
    energy_fock: np.ndarray
    response: KThetaResult
    charge_profile: DomainWallChargeProfile
    summary: ChargeResponseSummary
    manifest: RunManifest | None = None


def theta_nodes(params: ACResponseWorkflowParams) -> np.ndarray:
    response = params.response
    return np.linspace(response.theta_min, response.theta_max, response.n_theta)


def build_source_projector_path(
    params: ACResponseWorkflowParams,
) -> tuple[FlavorSourceProjector, np.ndarray, np.ndarray, np.ndarray]:
    model = NonIdealACLLModel(params.ac)
    source = FlavorSourceProjector(
        model,
        n_k=params.grid.n_k,
        active_band=0,
        occupy=params.source.occupy,
    )
    theta = theta_nodes(params)
    projectors = np.zeros((len(theta), params.grid.n_k, params.grid.n_k, 2, 2), dtype=complex)
    gaps = np.zeros(len(theta), dtype=float)
    for it, th in enumerate(theta):
        P_flat, gap = source.projector_for_direction(
            target_vector(float(th)),
            amplitude=params.source.source_scale,
        )
        projectors[it] = P_flat.reshape(params.grid.n_k, params.grid.n_k, 2, 2)
        gaps[it] = gap
    return source, theta, projectors, gaps


def evaluate_projector_path_energy(
    source: FlavorSourceProjector,
    projectors: np.ndarray,
    params: ACResponseWorkflowParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    energy = ProjectedPhysicalEnergy(source, params.interaction)
    totals = []
    bands = []
    hartrees = []
    focks = []
    for P_grid in projectors:
        breakdown = energy.energy(P_grid.reshape(source.n_total, 2, 2))
        totals.append(breakdown.total)
        bands.append(breakdown.band)
        hartrees.append(breakdown.hartree)
        focks.append(breakdown.fock)
    return (
        np.asarray(totals, dtype=float),
        np.asarray(bands, dtype=float),
        np.asarray(hartrees, dtype=float),
        np.asarray(focks, dtype=float),
    )


def run_ac_cg_workflow(
    params: ACResponseWorkflowParams | None = None,
    *,
    write_outputs: bool = False,
    write_plots: bool = False,
) -> ACCGWorkflowResult:
    workflow_params = params or ACResponseWorkflowParams()
    source, theta, projectors, gaps = build_source_projector_path(workflow_params)
    energy_total, energy_band, energy_hartree, energy_fock = evaluate_projector_path_energy(
        source, projectors, workflow_params
    )
    response = k_theta_from_projectors(projectors, theta)
    r_max = max(
        2.0 * workflow_params.domain_wall.radius,
        workflow_params.domain_wall.radius + 8.0 * workflow_params.domain_wall.width,
    )
    r = np.linspace(max(1e-6, r_max / 500.0), r_max, 500)
    profile = charge_density_radial(
        r,
        response.theta,
        response.K,
        workflow_params.domain_wall,
    )
    summary = ChargeResponseSummary(
        cG=response.cG,
        kappa_min=float(np.min(response.K)),
        kappa_max=float(np.max(response.K)),
        gap_min=float(np.min(gaps)),
        valid_local_gap=bool(np.min(gaps) > 0.0),
    )
    result = ACCGWorkflowResult(
        params=workflow_params,
        theta=theta,
        projectors=projectors,
        gaps=gaps,
        energy_total=energy_total,
        energy_band=energy_band,
        energy_hartree=energy_hartree,
        energy_fock=energy_fock,
        response=response,
        charge_profile=profile,
        summary=summary,
    )
    if write_outputs:
        manifest = write_ac_cg_outputs(result, write_plots=write_plots)
        result = ACCGWorkflowResult(**{**result.__dict__, "manifest": manifest})
    return result


def _artifact(path: Path, name: str, kind: str, description: str, required: bool = True) -> RunArtifact:
    return RunArtifact(
        name=name,
        path=str(path),
        kind=kind,  # type: ignore[arg-type]
        description=description,
        required=required,
        exists=path.exists(),
        size_bytes=path.stat().st_size if path.exists() else None,
    )


def write_ac_cg_outputs(result: ACCGWorkflowResult, *, write_plots: bool = False) -> RunManifest:
    out_dir = Path(result.params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    projectors_path = out_dir / "projectors.npz"
    ktheta_path = out_dir / "K_theta.csv"
    charge_path = out_dir / "charge_profile.csv"
    summary_path = out_dir / "summary.json"
    manifest_path = out_dir / "artifact_manifest.json"

    np.savez_compressed(
        projectors_path,
        theta=result.theta,
        projectors=result.projectors,
        gap=result.gaps,
        K_theta=result.response.K,
        cG=np.array(result.response.cG),
        energy_total=result.energy_total,
        energy_band=result.energy_band,
        energy_hartree=result.energy_hartree,
        energy_fock=result.energy_fock,
    )
    _write_k_theta_csv(ktheta_path, result)
    _write_charge_csv(charge_path, result)
    payload = {
        "params": result.params.model_dump(mode="json"),
        "summary": result.summary.model_dump(mode="json"),
        "projector_errors": projector_errors(result.projectors),
        "normalization": "K(theta), cG, and rho_dimless are dimensionless moire-unit quantities.",
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))

    plot_path = out_dir / "K_theta.png"
    if write_plots:
        _write_plot(plot_path, result)

    artifacts = [
        _artifact(projectors_path, "projectors", "array", "Theta projector path and response arrays"),
        _artifact(ktheta_path, "K_theta", "table", "Dimensionless K(theta) table"),
        _artifact(charge_path, "charge_profile", "table", "Radial dimensionless charge profile"),
        _artifact(summary_path, "summary", "json", "Run parameters and scalar response summary"),
        _artifact(plot_path, "K_theta_plot", "plot", "Optional K(theta) plot", required=False),
    ]
    manifest = RunManifest.from_artifacts(
        run_id="ac_cg",
        result_dir=str(out_dir),
        artifacts=artifacts,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return manifest


def _write_k_theta_csv(path: Path, result: ACCGWorkflowResult) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["theta", "theta_over_pi", "K_theta", "cG", "gap", "energy_total"],
        )
        writer.writeheader()
        for row in zip(
            result.theta,
            result.response.K,
            result.gaps,
            result.energy_total,
        ):
            writer.writerow(
                {
                    "theta": float(row[0]),
                    "theta_over_pi": float(row[0] / np.pi),
                    "K_theta": float(row[1]),
                    "cG": float(result.response.cG),
                    "gap": float(row[2]),
                    "energy_total": float(row[3]),
                }
            )


def _write_charge_csv(path: Path, result: ACCGWorkflowResult) -> None:
    profile = result.charge_profile
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["r", "theta", "K_theta", "rho_dimless"],
        )
        writer.writeheader()
        for row in zip(profile.r, profile.theta, profile.K_theta, profile.rho_dimless):
            writer.writerow(
                {
                    "r": float(row[0]),
                    "theta": float(row[1]),
                    "K_theta": float(row[2]),
                    "rho_dimless": float(row[3]),
                }
            )


def _write_plot(path: Path, result: ACCGWorkflowResult) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0), constrained_layout=True)
    axes[0].plot(result.theta / np.pi, result.response.K, marker="o")
    axes[0].set_xlabel("theta/pi")
    axes[0].set_ylabel("K(theta)")
    axes[0].set_title(f"dimensionless cG={result.response.cG:.6g}")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(result.charge_profile.r, result.charge_profile.rho_dimless)
    axes[1].set_xlabel("r / a_M")
    axes[1].set_ylabel("rho_dimless")
    axes[1].grid(True, alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)
