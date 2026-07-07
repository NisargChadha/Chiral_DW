"""Taige Q=0 IVC displacement-hysteresis branch helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field

from chiral_dw.config import ChargeResponseSummary, ContinuumWorkflowParams
from chiral_dw.continuum.ivc_diagnostics import (
    ProjectorOverlapDiagnostics,
    projector_overlap_diagnostics_with_frames,
)
from chiral_dw.continuum.models import (
    ContinuumBundle,
    ContinuumHFResult,
    ConvexPathDiagnostics,
    SymmetricHFReferences,
)
from chiral_dw.continuum.observables import active_basis_frames
from chiral_dw.continuum.references import (
    TrialInterpolationMode,
    interpolation_weights,
    projector_path_for_interpolation,
    reference_diagnostics,
)
from chiral_dw.continuum.workflow import (
    ContinuumSymmetricHFWorkflowResult,
    continuum_theta_nodes,
)
from chiral_dw.domain_wall import charge_density_radial
from chiral_dw.response import KThetaResult, k_theta_from_projectors_with_basis

HysteresisDirection = Literal["up", "down"]
HysteresisSweepAxis = Literal["u_D", "theta"]


class TaigeHysteresisPoint(BaseModel):
    """One displacement/twist point in a Taige hysteresis grid."""

    model_config = ConfigDict(frozen=True)

    u_index: int = Field(ge=0)
    theta_index: int = Field(ge=0)
    u_D: float
    theta_deg: float

    @computed_field
    @property
    def label(self) -> str:
        return f"u_{self.u_index:03d}_theta_{self.theta_index:03d}"


class TaigeHysteresisBranchRecord(BaseModel):
    """Scalar record for one branch-continuation point."""

    model_config = ConfigDict(frozen=True, extra="allow")

    sweep_axis: HysteresisSweepAxis = "u_D"
    branch_id: str | None = None
    fixed_axis: str | None = None
    fixed_index: int | None = None
    fixed_value: float | None = None
    continuation_axis: str | None = None
    continuation_index: int | None = None
    continuation_value: float | None = None
    u_index: int
    theta_index: int
    u_D_meV: float
    theta_deg: float
    direction: HysteresisDirection
    point_label: str
    run_id: str
    seed_label: str
    warm_start_source: str
    warm_start_from_run_id: str | None = None
    energy_total_per_cell: float
    direct_gap_min: float
    indirect_gap: float
    ivc_amplitude_block: float
    c_ivc_block: float
    aufbau_residual_norm: float
    warning_flag: bool
    converged: bool
    iteration_count: int
    max_iter: int | None = None
    hit_max_iter: bool = False
    self_consistency_warning: bool = False
    delta_P: float | None = None
    delta_energy: float | None = None
    commutator_norm: float | None = None
    idempotency_error_fro: float | None = None
    idempotency_error_max: float | None = None
    constraint_error: float | None = None
    trace_error: float | None = None
    clean_branch: bool = True
    branch_reliability: Literal["clean", "unclean", "unreliable_no_clean_candidate"] = "clean"
    transport_mean_retained_weight: float | None = None
    transport_min_retained_weight: float | None = None
    cG: float
    cG_diagnostic: float | None = None
    cG_warning_flag: bool = False
    cG_warning_reason: str | None = None
    texture_valid: bool
    texture_invalid_reason: str | None = None
    vp_reference_name: str
    vp_reference_energy_per_cell: float
    ivc_minus_vp_energy_per_cell: float
    projector_path: str
    cache_path: str | None = None

    def as_csv_row(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class TaigeHysteresisComparisonRecord(BaseModel):
    """Merged up/down comparison for one phase point."""

    model_config = ConfigDict(frozen=True, extra="allow")

    u_index: int
    theta_index: int
    u_D_meV: float
    theta_deg: float
    energy_up_minus_down: float
    direct_gap_up_minus_down: float
    ivc_order_up_minus_down: float
    selected_lower_energy_branch: HysteresisDirection
    high_gap_branch: HysteresisDirection
    low_gap_branch: HysteresisDirection
    lowest_energy_ivc_cG: float
    cG_up: float
    cG_down: float
    cG_high_gap: float
    cG_low_gap: float
    warning_count: int
    mean_projector_overlap: float
    one_minus_mean_projector_overlap: float

    def as_csv_row(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _row_get(row: Mapping[str, Any] | BaseModel, key: str, default: Any = None) -> Any:
    if isinstance(row, BaseModel):
        return getattr(row, key, default)
    return row.get(key, default)


def _row_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        key = value.strip().lower()
        if key in {"", "0", "false", "f", "no", "n", "none", "null"}:
            return False
        if key in {"1", "true", "t", "yes", "y"}:
            return True
    return bool(value)


def _row_energy(row: Mapping[str, Any] | BaseModel) -> float:
    for key in ("energy_total_per_cell", "energy", "final_energy"):
        value = _row_get(row, key)
        if value is not None:
            return float(value)
    raise KeyError("record has no energy_total_per_cell, energy, or final_energy field")


def _row_warning(row: Mapping[str, Any] | BaseModel) -> bool:
    for key in ("warning_flag", "final_self_consistency_warning", "self_consistency_warning"):
        value = _row_get(row, key)
        if value is not None:
            return _row_bool(value)
    return False


def is_clean_hysteresis_record(row: Mapping[str, Any] | BaseModel) -> bool:
    """Return whether a candidate is clean enough for physical branch selection."""

    clean_value = _row_get(row, "clean_branch")
    if clean_value is not None:
        return _row_bool(clean_value)
    if not _row_bool(_row_get(row, "converged", False)):
        return False
    if _row_warning(row):
        return False
    if _row_bool(_row_get(row, "hit_max_iter", False)):
        return False
    return True


def select_lowest_energy_clean_record(
    records: Sequence[Mapping[str, Any] | BaseModel],
) -> Mapping[str, Any] | BaseModel | None:
    """Return the lowest-energy clean candidate, or None when no clean row exists."""

    rows = list(records)
    if not rows:
        raise ValueError("cannot select from an empty record list")
    clean = [row for row in rows if is_clean_hysteresis_record(row)]
    if not clean:
        return None
    return min(clean, key=_row_energy)


def select_lowest_energy_raw_record(
    records: Sequence[Mapping[str, Any] | BaseModel],
) -> Mapping[str, Any] | BaseModel:
    """Return the lowest-energy candidate regardless of convergence quality."""

    rows = list(records)
    if not rows:
        raise ValueError("cannot select from an empty record list")
    return min(rows, key=_row_energy)


def select_lowest_energy_record(
    records: Sequence[Mapping[str, Any] | BaseModel],
) -> tuple[Mapping[str, Any] | BaseModel, str]:
    """Select the lowest-energy clean row, falling back to raw with a reliability label."""

    rows = list(records)
    if not rows:
        raise ValueError("cannot select from an empty record list")
    clean = select_lowest_energy_clean_record(rows)
    if clean is not None:
        return clean, "clean"
    return select_lowest_energy_raw_record(rows), "all_unclean_raw_fallback"


def _vp_reference(refs: SymmetricHFReferences) -> tuple[str, ContinuumHFResult]:
    if np.isclose(refs.vp_plus.energy, refs.vp_minus.energy, rtol=1e-9, atol=1e-9):
        return "VP+", refs.vp_plus
    if refs.vp_plus.energy <= refs.vp_minus.energy:
        return "VP+", refs.vp_plus
    return "VP-", refs.vp_minus


def _reference_summary(refs: SymmetricHFReferences) -> dict[str, Any]:
    return {
        "vp_plus": refs.vp_plus.diagnostics.model_dump(mode="json"),
        "vp_minus": refs.vp_minus.diagnostics.model_dump(mode="json"),
        "ivc": refs.ivc.diagnostics.model_dump(mode="json"),
        "hamiltonian_channels": {
            name: diag.model_dump(mode="json")
            for name, diag in reference_diagnostics(refs).items()
        },
    }


def _nan_path_diagnostics(
    theta: np.ndarray,
    *,
    trial_interpolation: TrialInterpolationMode = "convex_full_hf",
) -> tuple[ConvexPathDiagnostics, ...]:
    rows: list[ConvexPathDiagnostics] = []
    for angle in theta:
        w_plus, w_minus, w_ivc = interpolation_weights(float(angle), trial_interpolation)
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


def build_branch_response_result(
    *,
    params: ContinuumWorkflowParams,
    bundle: ContinuumBundle,
    vp_plus: ContinuumHFResult,
    vp_minus: ContinuumHFResult,
    ivc: ContinuumHFResult,
    branch_label: str,
    suppress_texture_when_ivc_below_vp: bool = True,
    texture_energy_tie_atol: float = 1e-9,
    trial_interpolation: TrialInterpolationMode = "convex_full_hf",
) -> ContinuumSymmetricHFWorkflowResult:
    """Build a VP+/VP-/branch-IVC response without re-solving cold IVC refs."""

    refs = SymmetricHFReferences(
        vp_plus=vp_plus,
        vp_minus=vp_minus,
        ivc=ivc,
        n_occ_per_k=params.hf.n_occ_per_k,
    )
    norm = float(bundle.backend.n_blocks)
    vp_name, vp_ref = _vp_reference(refs)
    vp_per_cell = float(vp_ref.energy / norm)
    ivc_per_cell = float(ivc.energy / norm)
    ivc_minus_vp = float(ivc_per_cell - vp_per_cell)
    texture_valid = ivc_minus_vp >= -float(texture_energy_tie_atol)
    branch_selection = {
        "ivc_branch_policy": "q0_hysteresis",
        "selected_ivc_branch": "q0",
        "hysteresis_branch": str(branch_label),
        "q0_ivc_energy_per_cell": ivc_per_cell,
        "selected_ivc_energy_per_cell": ivc_per_cell,
        "finite_q_ivc_energy_per_cell": None,
        "finite_q_minus_q0_ivc_energy_per_cell": None,
        "vp_reference_name": vp_name,
        "vp_reference_energy_per_cell": vp_per_cell,
        "selected_ivc_minus_vp_energy_per_cell": ivc_minus_vp,
        "hf_ground_state": "VP" if texture_valid else "IVC_Q0_hysteresis",
        "texture_valid": bool(texture_valid),
        "texture_invalid_reason": None if texture_valid else "ivc_energy_below_vp_reference",
        "texture_nan_policy": bool(suppress_texture_when_ivc_below_vp),
        "texture_energy_tie_atol": float(texture_energy_tie_atol),
        "trial_interpolation": str(trial_interpolation),
    }
    theta = continuum_theta_nodes(params)
    if suppress_texture_when_ivc_below_vp and not texture_valid:
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
        path_diagnostics = _nan_path_diagnostics(
            theta,
            trial_interpolation=trial_interpolation,
        )
        summary = ChargeResponseSummary(
            cG=float("nan"),
            kappa_min=float("nan"),
            kappa_max=float("nan"),
            gap_min=float("nan"),
            valid_local_gap=False,
        )
    else:
        projectors_flat, path_diagnostics = projector_path_for_interpolation(
            refs,
            theta,
            trial_interpolation=trial_interpolation,
            h0=bundle.backend.h0,
        )
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
        gaps = np.asarray([row.direct_gap_min for row in path_diagnostics], dtype=float)
        summary = ChargeResponseSummary(
            cG=response.cG,
            kappa_min=float(np.min(response.K)),
            kappa_max=float(np.max(response.K)),
            gap_min=float(np.min(gaps)),
            valid_local_gap=bool(np.min(gaps) > 0.0),
        )
    r_max = max(
        2.0 * params.domain_wall.radius,
        params.domain_wall.radius + 8.0 * params.domain_wall.width,
    )
    r = np.linspace(max(1e-6, r_max / 500.0), r_max, 500)
    charge_profile = charge_density_radial(r, response.theta, response.K, params.domain_wall)
    return ContinuumSymmetricHFWorkflowResult(
        params=params,
        bundle=bundle,
        references=refs,
        theta=theta,
        projectors_flat=projectors_flat,
        projectors=projectors,
        path_diagnostics=path_diagnostics,
        response=response,
        charge_profile=charge_profile,
        summary=summary,
        reference_summary=_reference_summary(refs),
        selected_ivc_branch="q0",
        ivc_branch_policy="q0",  # type: ignore[arg-type]
        q0_branch=None,
        finite_q_branch=None,
        branch_selection=branch_selection,
    )


def compare_hysteresis_records(
    *,
    up: Mapping[str, Any],
    down: Mapping[str, Any],
    up_projector: np.ndarray,
    down_projector: np.ndarray,
    up_frames: np.ndarray,
    down_frames: np.ndarray,
    n_occ_per_k: int = 1,
) -> tuple[TaigeHysteresisComparisonRecord, ProjectorOverlapDiagnostics]:
    """Compare up/down branch rows and embedded projectors at one phase point."""

    overlap = projector_overlap_diagnostics_with_frames(
        up_projector,
        down_projector,
        up_frames,
        down_frames,
        n_occ_per_k=n_occ_per_k,
    )
    e_up = float(up["energy_total_per_cell"])
    e_down = float(down["energy_total_per_cell"])
    gap_up = float(up["direct_gap_min"])
    gap_down = float(down["direct_gap_min"])
    order_up = float(up["ivc_amplitude_block"])
    order_down = float(down["ivc_amplitude_block"])
    lower: HysteresisDirection = "up" if e_up <= e_down else "down"
    high_gap: HysteresisDirection = "up" if gap_up >= gap_down else "down"
    low_gap: HysteresisDirection = "down" if high_gap == "up" else "up"
    cG_up = float(up.get("cG", float("nan")))
    cG_down = float(down.get("cG", float("nan")))

    def cG_for(direction: HysteresisDirection) -> float:
        return cG_up if direction == "up" else cG_down

    record = TaigeHysteresisComparisonRecord(
        u_index=int(up["u_index"]),
        theta_index=int(up["theta_index"]),
        u_D_meV=float(up["u_D_meV"]),
        theta_deg=float(up["theta_deg"]),
        energy_up_minus_down=float(e_up - e_down),
        direct_gap_up_minus_down=float(gap_up - gap_down),
        ivc_order_up_minus_down=float(order_up - order_down),
        selected_lower_energy_branch=lower,
        high_gap_branch=high_gap,
        low_gap_branch=low_gap,
        lowest_energy_ivc_cG=cG_for(lower),
        cG_up=cG_up,
        cG_down=cG_down,
        cG_high_gap=cG_for(high_gap),
        cG_low_gap=cG_for(low_gap),
        warning_count=int(bool(up.get("warning_flag", False)))
        + int(bool(down.get("warning_flag", False))),
        mean_projector_overlap=float(overlap.mean_overlap),
        one_minus_mean_projector_overlap=float(overlap.one_minus_mean_overlap),
        min_block_projector_overlap=float(overlap.min_block_overlap),
        max_block_projector_overlap=float(overlap.max_block_overlap),
        run_id_up=up.get("run_id"),
        run_id_down=down.get("run_id"),
        projector_path_up=up.get("projector_path"),
        projector_path_down=down.get("projector_path"),
        cG_lowest_energy_ivc=cG_for(lower),
    )
    return record, overlap


def records_to_csv_rows(records: Sequence[BaseModel | Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return JSON-compatible dict rows for CSV writing."""

    rows: list[dict[str, Any]] = []
    for row in records:
        if isinstance(row, BaseModel):
            rows.append(row.model_dump(mode="json"))
        else:
            rows.append(dict(row))
    return rows


def phase_table_rows(comparisons: Sequence[TaigeHysteresisComparisonRecord]) -> dict[str, list[dict[str, Any]]]:
    """Return phase-diagram-ready tables derived from comparison records."""

    rows = [row.model_dump(mode="json") for row in comparisons]
    return {
        "energy_crossing": [
            {
                "u_index": row["u_index"],
                "theta_index": row["theta_index"],
                "u_D_meV": row["u_D_meV"],
                "theta_deg": row["theta_deg"],
                "energy_up_minus_down": row["energy_up_minus_down"],
                "selected_lower_energy_branch": row["selected_lower_energy_branch"],
            }
            for row in rows
        ],
        "gap_jump": [
            {
                "u_index": row["u_index"],
                "theta_index": row["theta_index"],
                "u_D_meV": row["u_D_meV"],
                "theta_deg": row["theta_deg"],
                "direct_gap_up_minus_down": row["direct_gap_up_minus_down"],
                "high_gap_branch": row["high_gap_branch"],
                "low_gap_branch": row["low_gap_branch"],
            }
            for row in rows
        ],
        "overlap_discontinuity": [
            {
                "u_index": row["u_index"],
                "theta_index": row["theta_index"],
                "u_D_meV": row["u_D_meV"],
                "theta_deg": row["theta_deg"],
                "mean_projector_overlap": row["mean_projector_overlap"],
                "one_minus_mean_projector_overlap": row["one_minus_mean_projector_overlap"],
            }
            for row in rows
        ],
        "selected_branch_cg": [
            {
                "u_index": row["u_index"],
                "theta_index": row["theta_index"],
                "u_D_meV": row["u_D_meV"],
                "theta_deg": row["theta_deg"],
                "cG_lowest_energy_ivc": row["lowest_energy_ivc_cG"],
                "cG_high_gap": row["cG_high_gap"],
                "cG_low_gap": row["cG_low_gap"],
            }
            for row in rows
        ],
    }


__all__ = [
    "HysteresisDirection",
    "HysteresisSweepAxis",
    "TaigeHysteresisBranchRecord",
    "TaigeHysteresisComparisonRecord",
    "TaigeHysteresisPoint",
    "build_branch_response_result",
    "compare_hysteresis_records",
    "is_clean_hysteresis_record",
    "phase_table_rows",
    "records_to_csv_rows",
    "select_lowest_energy_clean_record",
    "select_lowest_energy_raw_record",
    "select_lowest_energy_record",
]
