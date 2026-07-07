"""End-to-end native continuum symmetric-HF response workflow."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from chiral_dw.artifacts import RunArtifact, RunManifest
from chiral_dw.config import ChargeResponseSummary, ContinuumWorkflowParams
from chiral_dw.config import ContinuumFiniteQParams
from chiral_dw.continuum.builder import build_continuum_bundle
from chiral_dw.continuum.models import ContinuumBundle, ConvexPathDiagnostics, SymmetricHFReferences
from chiral_dw.continuum.models import finite_q_shift_metadata
from chiral_dw.continuum.observables import active_basis_frames
from chiral_dw.continuum.references import (
    build_symmetric_hf_references,
    convex_weights,
    reference_diagnostics,
    symmetric_convex_path,
)
from chiral_dw.continuum.sweep import trial_theta_rows
from chiral_dw.continuum.taige import (
    TaigeFiniteQShiftPolicy,
    taige_ivc_minus_shift_choice,
)
from chiral_dw.domain_wall import DomainWallChargeProfile, charge_density_radial
from chiral_dw.response import KThetaResult, k_theta_from_projectors_with_basis, projector_errors

IVCBranchPolicy = Literal["q0", "lower_energy"]


@dataclass(frozen=True)
class ContinuumSymmetricHFBranch:
    """A self-consistent reference set in one active momentum frame."""

    name: Literal["q0", "finite_q"]
    bundle: ContinuumBundle
    references: SymmetricHFReferences
    metadata: dict


@dataclass(frozen=True)
class ContinuumSymmetricHFWorkflowResult:
    """In-memory output for one native continuum symmetric-HF run."""

    params: ContinuumWorkflowParams
    bundle: ContinuumBundle
    references: SymmetricHFReferences
    theta: np.ndarray
    projectors_flat: np.ndarray
    projectors: np.ndarray
    path_diagnostics: tuple[ConvexPathDiagnostics, ...]
    response: KThetaResult
    charge_profile: DomainWallChargeProfile
    summary: ChargeResponseSummary
    reference_summary: dict
    selected_ivc_branch: Literal["q0", "finite_q"] = "q0"
    ivc_branch_policy: IVCBranchPolicy = "q0"
    q0_branch: ContinuumSymmetricHFBranch | None = None
    finite_q_branch: ContinuumSymmetricHFBranch | None = None
    branch_selection: dict = field(default_factory=dict)
    manifest: RunManifest | None = None


def continuum_theta_nodes(params: ContinuumWorkflowParams) -> np.ndarray:
    response = params.response
    return np.linspace(response.theta_min, response.theta_max, response.n_theta)


def _vp_reference_energy_per_cell(
    branch: ContinuumSymmetricHFBranch,
) -> tuple[str, float]:
    refs = branch.references
    norm = float(branch.bundle.backend.n_blocks)
    if np.isclose(refs.vp_plus.energy, refs.vp_minus.energy, rtol=1e-9, atol=1e-9):
        return "VP+", float(refs.vp_plus.energy / norm)
    if refs.vp_plus.energy <= refs.vp_minus.energy:
        return "VP+", float(refs.vp_plus.energy / norm)
    return "VP-", float(refs.vp_minus.energy / norm)


def _branch_reference_summary(refs: SymmetricHFReferences) -> dict:
    return {
        "vp_plus": refs.vp_plus.diagnostics.model_dump(mode="json"),
        "vp_minus": refs.vp_minus.diagnostics.model_dump(mode="json"),
        "ivc": refs.ivc.diagnostics.model_dump(mode="json"),
        "hamiltonian_channels": {
            name: diag.model_dump(mode="json")
            for name, diag in reference_diagnostics(refs).items()
        },
    }


def _nan_path_diagnostics(theta: np.ndarray) -> tuple[ConvexPathDiagnostics, ...]:
    rows: list[ConvexPathDiagnostics] = []
    for angle in theta:
        w_plus, w_minus, w_ivc = convex_weights(float(angle))
        rows.append(
            ConvexPathDiagnostics(
                theta=float(angle),
                phi=0.0,
                w_vp_plus=float(w_plus),
                w_vp_minus=float(w_minus),
                w_ivc=float(w_ivc),
                direct_gap_min=float("nan"),
                indirect_gap=float("nan"),
                projector_idempotency_error_fro=float("nan"),
                projector_idempotency_error_max=float("nan"),
            )
        )
    return tuple(rows)


def _build_suppressed_texture_result(
    *,
    controls: ContinuumWorkflowParams,
    bundle: ContinuumBundle,
    refs: SymmetricHFReferences,
    selected_ivc_branch: Literal["q0", "finite_q"],
    ivc_branch_policy: IVCBranchPolicy,
    q0_branch: ContinuumSymmetricHFBranch | None,
    finite_q_branch: ContinuumSymmetricHFBranch | None,
    branch_selection: dict,
) -> ContinuumSymmetricHFWorkflowResult:
    theta = continuum_theta_nodes(controls)
    projectors_flat = np.full(
        (theta.size, bundle.grid.size, bundle.active.dim, bundle.active.dim),
        np.nan + 0.0j,
        dtype=complex,
    )
    projectors = projectors_flat.reshape(
        theta.size,
        bundle.grid.n_k,
        bundle.grid.n_k,
        bundle.active.dim,
        bundle.active.dim,
    )
    response = KThetaResult(
        theta=theta,
        K=np.full(theta.shape, np.nan, dtype=float),
        cG=float("nan"),
    )
    r_max = max(
        2.0 * controls.domain_wall.radius,
        controls.domain_wall.radius + 8.0 * controls.domain_wall.width,
    )
    r = np.linspace(max(1e-6, r_max / 500.0), r_max, 500)
    profile = charge_density_radial(r, response.theta, response.K, controls.domain_wall)
    summary = ChargeResponseSummary(
        cG=float("nan"),
        kappa_min=float("nan"),
        kappa_max=float("nan"),
        gap_min=float("nan"),
        valid_local_gap=False,
    )
    return ContinuumSymmetricHFWorkflowResult(
        params=controls,
        bundle=bundle,
        references=refs,
        theta=theta,
        projectors_flat=projectors_flat,
        projectors=projectors,
        path_diagnostics=_nan_path_diagnostics(theta),
        response=response,
        charge_profile=profile,
        summary=summary,
        reference_summary=_branch_reference_summary(refs),
        selected_ivc_branch=selected_ivc_branch,
        ivc_branch_policy=ivc_branch_policy,
        q0_branch=q0_branch,
        finite_q_branch=finite_q_branch,
        branch_selection=branch_selection,
    )


def _build_response_result(
    *,
    controls: ContinuumWorkflowParams,
    bundle: ContinuumBundle,
    refs: SymmetricHFReferences,
    selected_ivc_branch: Literal["q0", "finite_q"],
    ivc_branch_policy: IVCBranchPolicy,
    q0_branch: ContinuumSymmetricHFBranch | None,
    finite_q_branch: ContinuumSymmetricHFBranch | None,
    branch_selection: dict,
) -> ContinuumSymmetricHFWorkflowResult:
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
    return ContinuumSymmetricHFWorkflowResult(
        params=controls,
        bundle=bundle,
        references=refs,
        theta=theta,
        projectors_flat=projectors_flat,
        projectors=projectors,
        path_diagnostics=path_diagnostics,
        response=response,
        charge_profile=profile,
        summary=summary,
        reference_summary=_branch_reference_summary(refs),
        selected_ivc_branch=selected_ivc_branch,
        ivc_branch_policy=ivc_branch_policy,
        q0_branch=q0_branch,
        finite_q_branch=finite_q_branch,
        branch_selection=branch_selection,
    )


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
    q0_branch = ContinuumSymmetricHFBranch(
        name="q0",
        bundle=bundle,
        references=refs,
        metadata=finite_q_shift_metadata(bundle.finite_q, bundle.grid),
    )
    branch_selection = {
        "ivc_branch_policy": "q0",
        "selected_ivc_branch": "q0",
        "q0_ivc_energy_per_cell": float(refs.ivc.energy / bundle.backend.n_blocks),
        "finite_q_ivc_energy_per_cell": None,
        "finite_q_minus_q0_ivc_energy_per_cell": None,
        "tie_atol": 0.0,
    }
    result = _build_response_result(
        controls=controls,
        bundle=bundle,
        refs=refs,
        selected_ivc_branch="q0",
        ivc_branch_policy="q0",
        q0_branch=q0_branch,
        finite_q_branch=None,
        branch_selection=branch_selection,
    )
    if write_outputs:
        manifest = write_continuum_symmetric_hf_outputs(result)
        result = ContinuumSymmetricHFWorkflowResult(**{**result.__dict__, "manifest": manifest})
    return result


def taige_ivc_minus_finite_q_params(
    n_k: int,
    *,
    finite_q_shift_policy: TaigeFiniteQShiftPolicy = "exact",
) -> ContinuumFiniteQParams:
    """Return the default Taige IVC- finite-Q active-frame controls."""

    choice = taige_ivc_minus_shift_choice(
        int(n_k),
        policy=finite_q_shift_policy,
    )
    return ContinuumFiniteQParams(
        enabled=True,
        q_coord=choice.q_coord,
        half_shift_coord=choice.half_shift_coord,
    )


def _taige_finite_q_metadata(
    finite_q: ContinuumFiniteQParams,
    grid,
    *,
    finite_q_shift_policy: TaigeFiniteQShiftPolicy,
) -> dict:
    choice = taige_ivc_minus_shift_choice(
        grid.n_k,
        policy=finite_q_shift_policy,
    )
    return {
        **finite_q_shift_metadata(finite_q, grid),
        "taige_ivc_minus_shift_choice": choice.model_dump(mode="json"),
        "finite_q_shift_policy": choice.policy,
        "finite_q_exact": choice.exact,
        "q_error_grid_units": choice.q_error_grid_units,
        "half_shift_error_grid_units": choice.half_shift_error_grid_units,
        "q_error_fractional_norm": choice.q_error_fractional_norm,
        "half_shift_error_fractional_norm": choice.half_shift_error_fractional_norm,
    }


def select_ivc_branch_by_energy(
    *,
    q0_branch: ContinuumSymmetricHFBranch,
    finite_q_branch: ContinuumSymmetricHFBranch | None,
    ivc_branch_policy: IVCBranchPolicy = "lower_energy",
    tie_atol: float = 1e-9,
) -> tuple[Literal["q0", "finite_q"], dict]:
    """Choose the branch whose IVC reference supplies the interpolation path."""

    policy = str(ivc_branch_policy).replace("-", "_")
    if policy not in {"q0", "lower_energy"}:
        raise ValueError("ivc_branch_policy must be 'q0' or 'lower_energy'")
    q0_ivc_per_cell = float(
        q0_branch.references.ivc.energy / q0_branch.bundle.backend.n_blocks
    )
    finite_ivc_per_cell = (
        None
        if finite_q_branch is None
        else float(finite_q_branch.references.ivc.energy / finite_q_branch.bundle.backend.n_blocks)
    )
    delta = None if finite_ivc_per_cell is None else float(finite_ivc_per_cell - q0_ivc_per_cell)
    selected_name: Literal["q0", "finite_q"] = "q0"
    if policy == "lower_energy" and delta is not None and delta < -float(tie_atol):
        selected_name = "finite_q"
    branch_selection = {
        "ivc_branch_policy": policy,
        "selected_ivc_branch": selected_name,
        "q0_ivc_energy_per_cell": q0_ivc_per_cell,
        "finite_q_ivc_energy_per_cell": finite_ivc_per_cell,
        "finite_q_minus_q0_ivc_energy_per_cell": delta,
        "selected_ivc_energy_per_cell": (
            q0_ivc_per_cell if selected_name == "q0" else finite_ivc_per_cell
        ),
        "tie_atol": float(tie_atol),
    }
    return selected_name, branch_selection


def run_taige_branch_selected_symmetric_hf_workflow(
    params: ContinuumWorkflowParams | None = None,
    *,
    finite_q_enabled: bool = True,
    finite_q_shift_policy: TaigeFiniteQShiftPolicy = "exact",
    ivc_branch_policy: IVCBranchPolicy = "lower_energy",
    tie_atol: float = 1e-9,
    suppress_texture_when_ivc_below_vp: bool = False,
    texture_energy_tie_atol: float = 1e-9,
    write_outputs: bool = False,
) -> ContinuumSymmetricHFWorkflowResult:
    """Run Taige Q=0 and optional finite-Q HF branches, then select the IVC path.

    If the finite-Q IVC energy per moire cell is strictly lower than the Q=0
    IVC energy by more than ``tie_atol``, the whole VP+/VP-/IVC interpolation
    is built in the finite-Q active frame. Ties prefer Q=0.
    """

    controls = params or ContinuumWorkflowParams()
    if controls.model.active_model != "taige":
        raise ValueError("branch-selected finite-Q IVC workflow requires a Taige continuum model")
    policy = str(ivc_branch_policy).replace("-", "_")
    if policy not in {"q0", "lower_energy"}:
        raise ValueError("ivc_branch_policy must be 'q0' or 'lower_energy'")

    q0_bundle = build_continuum_bundle(
        model=controls.model,
        grid=controls.grid,
        interaction=controls.interaction,
    )
    q0_refs = build_symmetric_hf_references(q0_bundle, controls.hf)
    q0_branch = ContinuumSymmetricHFBranch(
        name="q0",
        bundle=q0_bundle,
        references=q0_refs,
        metadata=finite_q_shift_metadata(q0_bundle.finite_q, q0_bundle.grid),
    )

    finite_q_branch: ContinuumSymmetricHFBranch | None = None
    if finite_q_enabled:
        finite_q = taige_ivc_minus_finite_q_params(
            controls.grid.n_k,
            finite_q_shift_policy=finite_q_shift_policy,
        )
        finite_bundle = build_continuum_bundle(
            model=controls.model,
            grid=controls.grid,
            interaction=controls.interaction,
            finite_q=finite_q,
        )
        finite_refs = build_symmetric_hf_references(finite_bundle, controls.hf)
        finite_q_branch = ContinuumSymmetricHFBranch(
            name="finite_q",
            bundle=finite_bundle,
            references=finite_refs,
            metadata=_taige_finite_q_metadata(
                finite_q,
                finite_bundle.grid,
                finite_q_shift_policy=finite_q_shift_policy,
            ),
        )

    selected_name, branch_selection = select_ivc_branch_by_energy(
        q0_branch=q0_branch,
        finite_q_branch=finite_q_branch,
        ivc_branch_policy=policy,  # type: ignore[arg-type]
        tie_atol=tie_atol,
    )
    selected_branch = q0_branch if selected_name == "q0" else finite_q_branch
    if selected_branch is None:
        raise RuntimeError("finite-Q branch was selected but no finite-Q branch was solved")
    vp_reference_name, vp_reference_per_cell = _vp_reference_energy_per_cell(selected_branch)
    selected_ivc_per_cell = float(branch_selection["selected_ivc_energy_per_cell"])
    selected_ivc_minus_vp_per_cell = float(selected_ivc_per_cell - vp_reference_per_cell)
    texture_valid = selected_ivc_minus_vp_per_cell >= -float(texture_energy_tie_atol)
    branch_selection = {
        **branch_selection,
        "vp_reference_name": vp_reference_name,
        "vp_reference_energy_per_cell": vp_reference_per_cell,
        "selected_ivc_minus_vp_energy_per_cell": selected_ivc_minus_vp_per_cell,
        "hf_ground_state": (
            "VP"
            if texture_valid
            else ("IVC_0" if selected_name == "q0" else "IVC_-")
        ),
        "texture_valid": bool(texture_valid),
        "texture_invalid_reason": (
            None
            if texture_valid
            else "ivc_energy_below_vp_reference"
        ),
        "texture_nan_policy": bool(suppress_texture_when_ivc_below_vp),
        "texture_energy_tie_atol": float(texture_energy_tie_atol),
    }
    if suppress_texture_when_ivc_below_vp and not texture_valid:
        result = _build_suppressed_texture_result(
            controls=controls,
            bundle=selected_branch.bundle,
            refs=selected_branch.references,
            selected_ivc_branch=selected_name,
            ivc_branch_policy=policy,  # type: ignore[arg-type]
            q0_branch=q0_branch,
            finite_q_branch=finite_q_branch,
            branch_selection=branch_selection,
        )
        if write_outputs:
            manifest = write_continuum_symmetric_hf_outputs(result)
            result = ContinuumSymmetricHFWorkflowResult(
                **{**result.__dict__, "manifest": manifest}
            )
        return result
    result = _build_response_result(
        controls=controls,
        bundle=selected_branch.bundle,
        refs=selected_branch.references,
        selected_ivc_branch=selected_name,
        ivc_branch_policy=policy,  # type: ignore[arg-type]
        q0_branch=q0_branch,
        finite_q_branch=finite_q_branch,
        branch_selection=branch_selection,
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
    trial_theta_path = out_dir / "trial_theta.csv"
    charge_path = out_dir / "charge_profile.csv"
    summary_path = out_dir / "summary.json"
    manifest_path = out_dir / "artifact_manifest.json"
    trial_rows = trial_theta_rows(result)

    np.savez_compressed(
        arrays_path,
        theta=result.theta,
        projectors=result.projectors,
        K_theta=result.response.K,
        cG=np.array(result.response.cG),
        selected_ivc_branch=np.array(result.selected_ivc_branch),
        q0_ivc_energy_per_cell=np.array(
            result.branch_selection.get("q0_ivc_energy_per_cell", np.nan),
            dtype=float,
        ),
        finite_q_ivc_energy_per_cell=np.array(
            (
                np.nan
                if result.branch_selection.get("finite_q_ivc_energy_per_cell") is None
                else result.branch_selection.get("finite_q_ivc_energy_per_cell")
            ),
            dtype=float,
        ),
        trial_direct_gap=np.asarray([row["direct_gap"] for row in trial_rows], dtype=float),
        trial_indirect_gap=np.asarray([row["indirect_gap"] for row in trial_rows], dtype=float),
        trial_energy_total_per_cell=np.asarray(
            [row["energy_total_per_cell"] for row in trial_rows],
            dtype=float,
        ),
        trial_energy_relative_per_cell=np.asarray(
            [row["energy_relative_per_cell"] for row in trial_rows],
            dtype=float,
        ),
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
    with trial_theta_path.open("w", newline="") as f:
        fieldnames = [
            "theta",
            "theta_over_pi",
            "trial_interpolation",
            "K_theta",
            "cG",
            "w_vp_plus",
            "w_vp_minus",
            "w_ivc",
            "direct_gap",
            "indirect_gap",
            "energy_total_per_cell",
            "energy_relative_per_cell",
            "energy_one_body_per_cell",
            "energy_hartree_per_cell",
            "energy_fock_per_cell",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trial_rows)
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
        "branch_selection": result.branch_selection,
        "reference_summary": result.reference_summary,
        "path_diagnostics": [asdict(row) for row in result.path_diagnostics],
        "projector_errors": projector_errors(result.projectors),
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    artifacts = [
        _artifact(arrays_path, "projectors", "array", "Theta projector path and response arrays"),
        _artifact(ktheta_path, "K_theta", "table", "Dimensionless K(theta) table"),
        _artifact(trial_theta_path, "trial_theta", "table", "Trial path gaps and physical energies"),
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
    "ContinuumSymmetricHFBranch",
    "ContinuumSymmetricHFWorkflowResult",
    "IVCBranchPolicy",
    "continuum_theta_nodes",
    "run_continuum_symmetric_hf_workflow",
    "run_taige_branch_selected_symmetric_hf_workflow",
    "select_ivc_branch_by_energy",
    "taige_ivc_minus_finite_q_params",
    "write_continuum_symmetric_hf_outputs",
]
