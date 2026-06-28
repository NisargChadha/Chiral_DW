"""Taige cluster-sweep diagnostics built from native continuum workflows."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field

from chiral_dw.config import ContinuumFiniteQParams
from chiral_dw.continuum.builder import build_continuum_bundle
from chiral_dw.continuum.hf_bands import evaluate_hf_high_symmetry_path, hf_band_chern_table
from chiral_dw.continuum.models import (
    ContinuumBundle,
    ContinuumHFResult,
    SymmetricHFReferences,
    finite_q_shift_metadata,
)
from chiral_dw.continuum.references import solve_reference_hf
from chiral_dw.continuum.symmetry import TPrimeConstraint
from chiral_dw.continuum.taige import (
    chern_number_table,
    taige_ivc_minus_half_shift_coord,
    taige_ivc_minus_q_coord,
)


class TaigeSweepPoint(BaseModel):
    """One displacement/twist point in the Taige cluster sweep."""

    model_config = ConfigDict(frozen=True)

    u_index: int = Field(ge=0)
    theta_index: int = Field(ge=0)
    u_D: float
    theta_deg: float

    @computed_field
    @property
    def label(self) -> str:
        return f"u_{self.u_index:03d}_theta_{self.theta_index:03d}"


class TaigeSweepDiagnosticsParams(BaseModel):
    """Controls for expensive per-point Taige sweep diagnostics."""

    model_config = ConfigDict(frozen=True)

    compute_chern_numbers: bool = True
    compute_finite_q_ivc: bool = True
    write_hf_path_spectra: bool = False
    hf_path_n_per_segment: int = Field(default=36, ge=1)


class TaigeSweepPointSummary(BaseModel):
    """Scalar summary for one Taige sweep point."""

    model_config = ConfigDict(frozen=True)

    u_index: int
    theta_index: int
    u_D_meV: float
    theta_deg: float
    cG: float
    K_min: float
    K_max: float
    gap_min: float
    valid_local_gap: bool
    vp_plus_energy: float
    vp_minus_energy: float
    ivc_energy: float
    vp_plus_energy_per_cell: float
    vp_minus_energy_per_cell: float
    ivc_q0_energy_per_cell: float
    vp_reference_name: str
    vp_reference_energy_per_cell: float
    ivc_q0_minus_vp_energy_per_cell: float
    ivc_finite_q_energy: float | None = None
    ivc_finite_q_energy_per_cell: float | None = None
    ivc_finite_q_minus_vp_energy_per_cell: float | None = None
    ivc_finite_q_minus_q0_energy_per_cell: float | None = None
    vp_plus_direct_gap: float
    vp_minus_direct_gap: float
    ivc_direct_gap: float
    vp_plus_indirect_gap: float
    vp_minus_indirect_gap: float
    ivc_indirect_gap: float
    ivc_finite_q_direct_gap: float | None = None
    ivc_finite_q_indirect_gap: float | None = None
    vp_plus_idempotency_error_fro: float
    vp_minus_idempotency_error_fro: float
    ivc_idempotency_error_fro: float
    ivc_finite_q_idempotency_error_fro: float | None = None
    vp_plus_self_consistency_warning: bool
    vp_minus_self_consistency_warning: bool
    ivc_self_consistency_warning: bool
    ivc_finite_q_self_consistency_warning: bool | None = None
    finite_q_ivc_enabled: bool
    chern_enabled: bool
    elapsed_seconds: float
    point_dir: str
    trial_theta_csv: str
    reference_energies_csv: str
    noninteracting_chern_numbers_csv: str | None = None
    hf_chern_numbers_csv: str | None = None
    hf_path_spectra_csv: str | None = None
    chern_columns: dict[str, float] = Field(default_factory=dict)

    def as_csv_row(self) -> dict[str, Any]:
        """Return a flattened row with dynamic Chern columns included."""

        row = self.model_dump(mode="json", exclude={"chern_columns"})
        row.update(self.chern_columns)
        return row


@dataclass(frozen=True)
class FiniteQIVCDiagnostic:
    """Finite-Q IVC branch used only for energy/topology diagnostics."""

    bundle: ContinuumBundle
    result: ContinuumHFResult
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TaigeSweepDiagnostics:
    """All table outputs for one Taige sweep point."""

    summary: TaigeSweepPointSummary
    trial_theta_rows: list[dict[str, Any]]
    reference_energy_rows: list[dict[str, Any]]
    noninteracting_chern_rows: list[dict[str, Any]]
    hf_chern_rows: list[dict[str, Any]]
    hf_path_spectrum_rows: list[dict[str, Any]]
    finite_q_ivc: FiniteQIVCDiagnostic | None = None


def _safe_column_fragment(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("+", "plus").replace("-", "minus").replace("=", "_")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unnamed"


def _point_fields(point: TaigeSweepPoint) -> dict[str, Any]:
    return {
        "u_index": int(point.u_index),
        "theta_index": int(point.theta_index),
        "u_D_meV": float(point.u_D),
        "theta_deg": float(point.theta_deg),
        "point_label": point.label,
    }


def _with_point_fields(point: TaigeSweepPoint, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prefix = _point_fields(point)
    return [{**prefix, **row} for row in rows]


def _energy_zero(energies: np.ndarray, n_occ: int) -> float:
    n = int(n_occ)
    if n <= 0 or n >= energies.shape[1]:
        return float(np.mean(energies))
    occupied_top = float(np.max(energies[:, :n]))
    empty_bottom = float(np.min(energies[:, n:]))
    return 0.5 * (occupied_top + empty_bottom)


def _vp_reference(refs: SymmetricHFReferences) -> tuple[str, ContinuumHFResult]:
    if np.isclose(refs.vp_plus.energy, refs.vp_minus.energy, rtol=1e-9, atol=1e-9):
        return "VP+", refs.vp_plus
    if refs.vp_plus.energy <= refs.vp_minus.energy:
        return "VP+", refs.vp_plus
    return "VP-", refs.vp_minus


def trial_theta_rows(workflow_result: Any) -> list[dict[str, Any]]:
    """Return theta-dependent response, gap, and physical-energy diagnostics."""

    theta = np.asarray(workflow_result.theta, dtype=float)
    path_diagnostics = tuple(workflow_result.path_diagnostics)
    projectors_flat = np.asarray(workflow_result.projectors_flat, dtype=complex)
    backend = workflow_result.bundle.backend
    energy_norm = float(backend.n_blocks)
    energy_components = [backend.energy(P_theta) for P_theta in projectors_flat]
    total = np.asarray([item.total for item in energy_components], dtype=float) / energy_norm
    one_body = np.asarray([item.one_body for item in energy_components], dtype=float) / energy_norm
    hartree = np.asarray([item.hartree for item in energy_components], dtype=float) / energy_norm
    fock = np.asarray([item.fock for item in energy_components], dtype=float) / energy_norm
    relative = total - float(np.min(total))
    rows: list[dict[str, Any]] = []
    for idx, angle in enumerate(theta):
        diag = path_diagnostics[idx]
        rows.append(
            {
                "theta": float(angle),
                "theta_over_pi": float(angle / np.pi),
                "K_theta": float(workflow_result.response.K[idx]),
                "cG": float(workflow_result.response.cG),
                "w_vp_plus": float(diag.w_vp_plus),
                "w_vp_minus": float(diag.w_vp_minus),
                "w_ivc": float(diag.w_ivc),
                "direct_gap": float(diag.direct_gap_min),
                "indirect_gap": float(diag.indirect_gap),
                "energy_total_per_cell": float(total[idx]),
                "energy_relative_per_cell": float(relative[idx]),
                "energy_one_body_per_cell": float(one_body[idx]),
                "energy_hartree_per_cell": float(hartree[idx]),
                "energy_fock_per_cell": float(fock[idx]),
            }
        )
    return rows


def run_finite_q_ivc_diagnostic(workflow_result: Any) -> FiniteQIVCDiagnostic:
    """Solve the Taige IVC- finite-Q branch for comparison only."""

    n_k = int(workflow_result.params.grid.n_k)
    try:
        q_coord = taige_ivc_minus_q_coord(n_k)
        half_shift_coord = taige_ivc_minus_half_shift_coord(n_k)
    except ValueError as exc:
        raise ValueError(
            "finite-Q IVC diagnostics require n_k divisible by 6; "
            "use --no-finite-q-ivc to skip this diagnostic"
        ) from exc
    finite_q = ContinuumFiniteQParams(
        enabled=True,
        q_coord=q_coord,
        half_shift_coord=half_shift_coord,
    )
    bundle = build_continuum_bundle(
        model=workflow_result.params.model,
        grid=workflow_result.params.grid,
        interaction=workflow_result.params.interaction,
        finite_q=finite_q,
    )
    constraint = TPrimeConstraint(bundle.active)
    result = solve_reference_hf(
        bundle,
        "finite_q_ivc",
        workflow_result.params.hf,
        constraint=constraint,
    )
    return FiniteQIVCDiagnostic(
        bundle=bundle,
        result=result,
        metadata=finite_q_shift_metadata(finite_q, bundle.grid),
    )


def reference_energy_rows(
    workflow_result: Any,
    finite_q_ivc: FiniteQIVCDiagnostic | None,
) -> list[dict[str, Any]]:
    """Return reference energies and IVC energy costs per moire cell."""

    refs = workflow_result.references
    energy_norm = float(workflow_result.bundle.backend.n_blocks)
    finite_q_norm = (
        None if finite_q_ivc is None else float(finite_q_ivc.bundle.backend.n_blocks)
    )
    vp_reference_name, vp_reference = _vp_reference(refs)
    vp_reference_per_cell = float(vp_reference.energy / energy_norm)
    ivc_q0_per_cell = float(refs.ivc.energy / energy_norm)
    finite_q_per_cell = (
        None
        if finite_q_ivc is None or finite_q_norm is None
        else float(finite_q_ivc.result.energy / finite_q_norm)
    )
    rows = [
        {
            "quantity": "E_VP_plus_per_cell",
            "value": float(refs.vp_plus.energy / energy_norm),
            "reference": "VP+",
        },
        {
            "quantity": "E_VP_minus_per_cell",
            "value": float(refs.vp_minus.energy / energy_norm),
            "reference": "VP-",
        },
        {
            "quantity": "E_VP_reference_per_cell",
            "value": vp_reference_per_cell,
            "reference": vp_reference_name,
        },
        {
            "quantity": "E_IVC_Q0_per_cell",
            "value": ivc_q0_per_cell,
            "reference": "IVC Q=0",
        },
        {
            "quantity": "E_IVC_finite_Q_per_cell",
            "value": finite_q_per_cell,
            "reference": "Taige IVC-" if finite_q_ivc is not None else "disabled",
        },
        {
            "quantity": "Delta_IVC_Q0_vs_VP_per_cell",
            "value": float(ivc_q0_per_cell - vp_reference_per_cell),
            "reference": vp_reference_name,
        },
        {
            "quantity": "Delta_IVC_finite_Q_vs_VP_per_cell",
            "value": (
                None
                if finite_q_per_cell is None
                else float(finite_q_per_cell - vp_reference_per_cell)
            ),
            "reference": vp_reference_name,
        },
        {
            "quantity": "Delta_finite_Q_minus_Q0_per_cell",
            "value": (
                None
                if finite_q_per_cell is None
                else float(finite_q_per_cell - ivc_q0_per_cell)
            ),
            "reference": "",
        },
    ]
    return rows


def noninteracting_chern_rows(workflow_result: Any) -> list[dict[str, Any]]:
    """Return noninteracting electron/hole Chern numbers for active Taige bands."""

    bands = workflow_result.bundle.bands
    if bands is None:
        return []
    n_active = int(workflow_result.bundle.active.n_active)
    rows = chern_number_table(bands, band_indices=tuple(range(n_active)))
    return [row.model_dump(mode="json") for row in rows]


def hf_chern_rows(
    workflow_result: Any,
    finite_q_ivc: FiniteQIVCDiagnostic | None,
) -> list[dict[str, Any]]:
    """Return embedded Chern numbers of the physical HF bands."""

    refs = workflow_result.references
    rows: list[dict[str, Any]] = []
    for reference, bundle, result in (
        ("VP+", workflow_result.bundle, refs.vp_plus),
        ("VP-", workflow_result.bundle, refs.vp_minus),
        ("IVC Q=0", workflow_result.bundle, refs.ivc),
    ):
        rows.extend(
            row.model_dump(mode="json")
            for row in hf_band_chern_table(
                bundle.active,
                result.H_hf,
                reference=reference,
            )
        )
    if finite_q_ivc is not None:
        rows.extend(
            row.model_dump(mode="json")
            for row in hf_band_chern_table(
                finite_q_ivc.bundle.active,
                finite_q_ivc.result.H_hf,
                reference="IVC finite Q",
            )
        )
    return rows


def hf_path_spectrum_rows(
    workflow_result: Any,
    finite_q_ivc: FiniteQIVCDiagnostic | None,
    *,
    n_per_segment: int,
) -> list[dict[str, Any]]:
    """Return optional high-symmetry fixed-density HF path spectra."""

    refs = workflow_result.references
    vp_reference_name, vp_reference = _vp_reference(refs)
    references = [
        ("VP", workflow_result.bundle, vp_reference, vp_reference_name),
        ("IVC_Q0", workflow_result.bundle, refs.ivc, "IVC Q=0"),
    ]
    if finite_q_ivc is not None:
        references.append(
            ("IVC_finite_Q", finite_q_ivc.bundle, finite_q_ivc.result, "IVC finite Q")
        )
    rows: list[dict[str, Any]] = []
    for file_key, bundle, result, display_label in references:
        spectrum = evaluate_hf_high_symmetry_path(
            bundle,
            result.P,
            n_per_segment=int(n_per_segment),
            reference=display_label,
        )
        energy_zero = _energy_zero(spectrum.energies, workflow_result.params.hf.n_occ_per_k)
        for row in spectrum.rows:
            payload = row.model_dump(mode="json")
            payload["file_key"] = file_key
            payload["energy_zero"] = energy_zero
            payload["energy_shifted"] = float(payload["energy"] - energy_zero)
            rows.append(payload)
    return rows


def _chern_columns(
    noninteracting_rows: list[dict[str, Any]],
    hf_rows: list[dict[str, Any]],
) -> dict[str, float]:
    columns: dict[str, float] = {}
    for row in noninteracting_rows:
        key = (
            "chern_nonint_"
            f"{_safe_column_fragment(row['basis'])}_"
            f"{_safe_column_fragment(row['valley'])}_"
            f"band_{int(row['band'])}"
        )
        columns[key] = float(row["chern"])
    for row in hf_rows:
        key = (
            "chern_hf_"
            f"{_safe_column_fragment(row['reference'])}_"
            f"band_{int(row['band'])}"
        )
        columns[key] = float(row["chern"])
    return columns


def build_taige_sweep_diagnostics(
    *,
    point: TaigeSweepPoint,
    workflow_result: Any,
    controls: TaigeSweepDiagnosticsParams,
    elapsed_seconds: float,
    point_dir: str | Path,
) -> TaigeSweepDiagnostics:
    """Build all scalar and table diagnostics for one Taige sweep point."""

    out_dir = Path(point_dir)
    finite_q = run_finite_q_ivc_diagnostic(workflow_result) if controls.compute_finite_q_ivc else None
    trial_rows = _with_point_fields(point, trial_theta_rows(workflow_result))
    energy_rows = _with_point_fields(point, reference_energy_rows(workflow_result, finite_q))
    nonint_rows = (
        _with_point_fields(point, noninteracting_chern_rows(workflow_result))
        if controls.compute_chern_numbers
        else []
    )
    hf_rows = (
        _with_point_fields(point, hf_chern_rows(workflow_result, finite_q))
        if controls.compute_chern_numbers
        else []
    )
    path_rows = (
        _with_point_fields(
            point,
            hf_path_spectrum_rows(
                workflow_result,
                finite_q,
                n_per_segment=controls.hf_path_n_per_segment,
            ),
        )
        if controls.write_hf_path_spectra
        else []
    )

    refs = workflow_result.reference_summary
    reference_rows_by_quantity = {row["quantity"]: row for row in energy_rows}
    vp_reference_name, _vp_reference_result = _vp_reference(workflow_result.references)
    finite_q_diag = None if finite_q is None else finite_q.result.diagnostics
    finite_q_energy_norm = None if finite_q is None else float(finite_q.bundle.backend.n_blocks)
    summary = TaigeSweepPointSummary(
        u_index=point.u_index,
        theta_index=point.theta_index,
        u_D_meV=point.u_D,
        theta_deg=point.theta_deg,
        cG=float(workflow_result.summary.cG),
        K_min=float(workflow_result.summary.kappa_min),
        K_max=float(workflow_result.summary.kappa_max),
        gap_min=float(workflow_result.summary.gap_min),
        valid_local_gap=bool(workflow_result.summary.valid_local_gap),
        vp_plus_energy=float(refs["vp_plus"]["energy"]),
        vp_minus_energy=float(refs["vp_minus"]["energy"]),
        ivc_energy=float(refs["ivc"]["energy"]),
        vp_plus_energy_per_cell=float(reference_rows_by_quantity["E_VP_plus_per_cell"]["value"]),
        vp_minus_energy_per_cell=float(reference_rows_by_quantity["E_VP_minus_per_cell"]["value"]),
        ivc_q0_energy_per_cell=float(reference_rows_by_quantity["E_IVC_Q0_per_cell"]["value"]),
        vp_reference_name=vp_reference_name,
        vp_reference_energy_per_cell=float(
            reference_rows_by_quantity["E_VP_reference_per_cell"]["value"]
        ),
        ivc_q0_minus_vp_energy_per_cell=float(
            reference_rows_by_quantity["Delta_IVC_Q0_vs_VP_per_cell"]["value"]
        ),
        ivc_finite_q_energy=None if finite_q is None else float(finite_q.result.energy),
        ivc_finite_q_energy_per_cell=(
            None
            if finite_q is None or finite_q_energy_norm is None
            else float(finite_q.result.energy / finite_q_energy_norm)
        ),
        ivc_finite_q_minus_vp_energy_per_cell=reference_rows_by_quantity[
            "Delta_IVC_finite_Q_vs_VP_per_cell"
        ]["value"],
        ivc_finite_q_minus_q0_energy_per_cell=reference_rows_by_quantity[
            "Delta_finite_Q_minus_Q0_per_cell"
        ]["value"],
        vp_plus_direct_gap=float(refs["vp_plus"]["direct_gap_min"]),
        vp_minus_direct_gap=float(refs["vp_minus"]["direct_gap_min"]),
        ivc_direct_gap=float(refs["ivc"]["direct_gap_min"]),
        vp_plus_indirect_gap=float(refs["vp_plus"]["indirect_gap"]),
        vp_minus_indirect_gap=float(refs["vp_minus"]["indirect_gap"]),
        ivc_indirect_gap=float(refs["ivc"]["indirect_gap"]),
        ivc_finite_q_direct_gap=None if finite_q_diag is None else finite_q_diag.direct_gap_min,
        ivc_finite_q_indirect_gap=None if finite_q_diag is None else finite_q_diag.indirect_gap,
        vp_plus_idempotency_error_fro=float(refs["vp_plus"]["idempotency_error_fro"]),
        vp_minus_idempotency_error_fro=float(refs["vp_minus"]["idempotency_error_fro"]),
        ivc_idempotency_error_fro=float(refs["ivc"]["idempotency_error_fro"]),
        ivc_finite_q_idempotency_error_fro=(
            None if finite_q_diag is None else finite_q_diag.idempotency_error_fro
        ),
        vp_plus_self_consistency_warning=bool(refs["vp_plus"]["self_consistency_warning"]),
        vp_minus_self_consistency_warning=bool(refs["vp_minus"]["self_consistency_warning"]),
        ivc_self_consistency_warning=bool(refs["ivc"]["self_consistency_warning"]),
        ivc_finite_q_self_consistency_warning=(
            None if finite_q_diag is None else finite_q_diag.self_consistency_warning
        ),
        finite_q_ivc_enabled=bool(controls.compute_finite_q_ivc),
        chern_enabled=bool(controls.compute_chern_numbers),
        elapsed_seconds=float(elapsed_seconds),
        point_dir=str(out_dir),
        trial_theta_csv=str(out_dir / "trial_theta.csv"),
        reference_energies_csv=str(out_dir / "reference_energies.csv"),
        noninteracting_chern_numbers_csv=(
            str(out_dir / "noninteracting_chern_numbers.csv")
            if controls.compute_chern_numbers
            else None
        ),
        hf_chern_numbers_csv=(
            str(out_dir / "hf_chern_numbers.csv") if controls.compute_chern_numbers else None
        ),
        hf_path_spectra_csv=(
            str(out_dir / "hf_path_spectra.csv")
            if controls.write_hf_path_spectra
            else None
        ),
        chern_columns=_chern_columns(nonint_rows, hf_rows),
    )
    return TaigeSweepDiagnostics(
        summary=summary,
        trial_theta_rows=trial_rows,
        reference_energy_rows=energy_rows,
        noninteracting_chern_rows=nonint_rows,
        hf_chern_rows=hf_rows,
        hf_path_spectrum_rows=path_rows,
        finite_q_ivc=finite_q,
    )


__all__ = [
    "FiniteQIVCDiagnostic",
    "TaigeSweepDiagnostics",
    "TaigeSweepDiagnosticsParams",
    "TaigeSweepPoint",
    "TaigeSweepPointSummary",
    "build_taige_sweep_diagnostics",
    "hf_chern_rows",
    "hf_path_spectrum_rows",
    "noninteracting_chern_rows",
    "reference_energy_rows",
    "run_finite_q_ivc_diagnostic",
    "trial_theta_rows",
]
