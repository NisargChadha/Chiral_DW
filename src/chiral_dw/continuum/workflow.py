"""End-to-end native continuum symmetric-HF response workflow."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chiral_dw.artifacts import RunArtifact, RunManifest
from chiral_dw.config import ChargeResponseSummary, ContinuumWorkflowParams
from chiral_dw.continuum.builder import build_continuum_bundle
from chiral_dw.continuum.observables import active_basis_frames
from chiral_dw.continuum.references import (
    build_symmetric_hf_references,
    reference_diagnostics,
    symmetric_convex_path,
)
from chiral_dw.domain_wall import DomainWallChargeProfile, charge_density_radial
from chiral_dw.response import KThetaResult, k_theta_from_projectors_with_basis, projector_errors


@dataclass(frozen=True)
class ContinuumSymmetricHFWorkflowResult:
    """In-memory output for one native continuum symmetric-HF run."""

    params: ContinuumWorkflowParams
    theta: np.ndarray
    projectors: np.ndarray
    response: KThetaResult
    charge_profile: DomainWallChargeProfile
    summary: ChargeResponseSummary
    reference_summary: dict
    manifest: RunManifest | None = None


def continuum_theta_nodes(params: ContinuumWorkflowParams) -> np.ndarray:
    response = params.response
    return np.linspace(response.theta_min, response.theta_max, response.n_theta)


def run_continuum_symmetric_hf_workflow(
    params: ContinuumWorkflowParams | None = None,
    *,
    write_outputs: bool = False,
) -> ContinuumSymmetricHFWorkflowResult:
    """Run the native continuum HF references, convex path, and charge response."""

    controls = params or ContinuumWorkflowParams()
    bundle = build_continuum_bundle(
        model=controls.model,
        grid=controls.grid,
        interaction=controls.interaction,
    )
    refs = build_symmetric_hf_references(bundle, controls.hf)
    theta = continuum_theta_nodes(controls)
    projectors_flat, path_diagnostics = symmetric_convex_path(refs, theta)
    if bundle.grid.n_k * bundle.grid.n_k != projectors_flat.shape[1]:
        raise ValueError("projector path does not match the continuum momentum grid")
    if projectors_flat.shape[-1] != bundle.active.dim:
        raise ValueError("projector path active dimension does not match the continuum active space")
    projectors = projectors_flat.reshape(
        theta.size,
        bundle.grid.n_k,
        bundle.grid.n_k,
        bundle.active.dim,
        bundle.active.dim,
    )
    basis = active_basis_frames(bundle.active).reshape(
        bundle.grid.n_k,
        bundle.grid.n_k,
        -1,
        bundle.active.dim,
    )
    response = k_theta_from_projectors_with_basis(projectors, theta, basis)
    r_max = max(
        2.0 * controls.domain_wall.radius,
        controls.domain_wall.radius + 8.0 * controls.domain_wall.width,
    )
    r = np.linspace(max(1e-6, r_max / 500.0), r_max, 500)
    profile = charge_density_radial(r, response.theta, response.K, controls.domain_wall)
    gaps = np.asarray([row.direct_gap_min for row in path_diagnostics], dtype=float)
    summary = ChargeResponseSummary(
        cG=response.cG,
        kappa_min=float(np.min(response.K)),
        kappa_max=float(np.max(response.K)),
        gap_min=float(np.min(gaps)),
        valid_local_gap=bool(np.min(gaps) > 0.0),
    )
    reference_summary = {
        "vp_plus": refs.vp_plus.diagnostics.model_dump(mode="json"),
        "vp_minus": refs.vp_minus.diagnostics.model_dump(mode="json"),
        "ivc": refs.ivc.diagnostics.model_dump(mode="json"),
        "hamiltonian_channels": {
            name: diag.model_dump(mode="json")
            for name, diag in reference_diagnostics(refs).items()
        },
    }
    result = ContinuumSymmetricHFWorkflowResult(
        params=controls,
        theta=theta,
        projectors=projectors,
        response=response,
        charge_profile=profile,
        summary=summary,
        reference_summary=reference_summary,
    )
    if write_outputs:
        manifest = write_continuum_symmetric_hf_outputs(result)
        result = ContinuumSymmetricHFWorkflowResult(**{**result.__dict__, "manifest": manifest})
    return result


def _artifact(path: Path, name: str, kind: str, description: str) -> RunArtifact:
    return RunArtifact(
        name=name,
        path=str(path),
        kind=kind,  # type: ignore[arg-type]
        description=description,
        exists=path.exists(),
        size_bytes=path.stat().st_size if path.exists() else None,
    )


def write_continuum_symmetric_hf_outputs(
    result: ContinuumSymmetricHFWorkflowResult,
) -> RunManifest:
    out_dir = Path(result.params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = out_dir / "projectors.npz"
    ktheta_path = out_dir / "K_theta.csv"
    charge_path = out_dir / "charge_profile.csv"
    summary_path = out_dir / "summary.json"
    manifest_path = out_dir / "artifact_manifest.json"

    np.savez_compressed(
        arrays_path,
        theta=result.theta,
        projectors=result.projectors,
        K_theta=result.response.K,
        cG=np.array(result.response.cG),
    )
    with ktheta_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["theta", "theta_over_pi", "K_theta", "cG"])
        writer.writeheader()
        for theta, K in zip(result.theta, result.response.K):
            writer.writerow(
                {
                    "theta": float(theta),
                    "theta_over_pi": float(theta / np.pi),
                    "K_theta": float(K),
                    "cG": float(result.response.cG),
                }
            )
    with charge_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["r", "theta", "K_theta", "rho_dimless"])
        writer.writeheader()
        profile = result.charge_profile
        for row in zip(profile.r, profile.theta, profile.K_theta, profile.rho_dimless):
            writer.writerow(
                {
                    "r": float(row[0]),
                    "theta": float(row[1]),
                    "K_theta": float(row[2]),
                    "rho_dimless": float(row[3]),
                }
            )
    payload = {
        "params": result.params.model_dump(mode="json"),
        "summary": result.summary.model_dump(mode="json"),
        "reference_summary": result.reference_summary,
        "projector_errors": projector_errors(result.projectors),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    artifacts = [
        _artifact(arrays_path, "projectors", "array", "Theta projector path and response arrays"),
        _artifact(ktheta_path, "K_theta", "table", "Dimensionless K(theta) table"),
        _artifact(charge_path, "charge_profile", "table", "Radial dimensionless charge profile"),
        _artifact(summary_path, "summary", "json", "Run parameters, HF diagnostics, and response summary"),
    ]
    manifest = RunManifest.from_artifacts(
        run_id="continuum_symmetric_hf",
        result_dir=str(out_dir),
        artifacts=artifacts,
    )
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    return manifest


__all__ = [
    "ContinuumSymmetricHFWorkflowResult",
    "continuum_theta_nodes",
    "run_continuum_symmetric_hf_workflow",
    "write_continuum_symmetric_hf_outputs",
]
