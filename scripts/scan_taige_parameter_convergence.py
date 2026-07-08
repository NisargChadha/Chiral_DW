#!/usr/bin/env python3
"""Single-point Taige continuum convergence scans over cutoff-like parameters."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from math import pi
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chiral_dw.config import (  # noqa: E402
    ContinuumGridParams,
    ContinuumHFParams,
    ContinuumWorkflowParams,
    DomainWallParams,
    ResponseParams,
)
from chiral_dw.continuum.sweep import (  # noqa: E402
    TaigeSweepDiagnosticsParams,
    TaigeSweepPoint,
    build_taige_sweep_diagnostics,
)
from chiral_dw.continuum.taige import (  # noqa: E402
    reciprocal_shell,
    taige_interaction_params,
    taige_material_label,
    taige_model_params,
)
from chiral_dw.continuum.workflow import run_taige_branch_selected_symmetric_hf_workflow  # noqa: E402

ScanAxis = Literal["plane_wave_shell", "active_bands"]

AXIS_ALIASES = {
    "plane-wave-shell": "plane_wave_shell",
    "plane_wave_shell": "plane_wave_shell",
    "active-bands": "active_bands",
    "active_bands": "active_bands",
}

DEFAULT_VALUE_LISTS = {
    "plane_wave_shell": "3,4,5,6,7,8",
    "active_bands": "1,2,3,4,5",
}

DEFAULT_OUTPUT_ROOTS = {
    "plane_wave_shell": "results/taige_convergence_plane_wave_shell_theta35_u0",
    "active_bands": "results/taige_convergence_active_bands_theta35_u0",
}

DELTA_METRICS = (
    "cG",
    "K_min",
    "K_max",
    "gap_min",
    "vp_plus_energy_per_cell",
    "vp_minus_energy_per_cell",
    "vp_reference_energy_per_cell",
    "ivc_q0_energy_per_cell",
    "selected_ivc_energy_per_cell",
    "ivc_q0_minus_vp_energy_per_cell",
    "selected_ivc_minus_vp_energy_per_cell",
    "vp_reference_direct_gap",
    "vp_reference_indirect_gap",
    "selected_ivc_direct_gap",
    "selected_ivc_indirect_gap",
    "vp_reference_order_abs_nz",
    "selected_ivc_ivc_amplitude_block",
    "ivc_q0_ivc_amplitude_block",
)

SUMMARY_COLUMNS = (
    "scan_axis",
    "scan_value",
    "value_index",
    "theta_deg",
    "u_D_meV",
    "n_k",
    "plane_wave_shell",
    "n_plane_waves",
    "n_bands",
    "n_active_bands_per_valley",
    "cG",
    "K_min",
    "K_max",
    "gap_min",
    "valid_local_gap",
    "texture_valid",
    "texture_invalid_reason",
    "hf_ground_state",
    "ivc_branch_policy",
    "selected_ivc_branch",
    "finite_q_ivc_enabled",
    "finite_q_shift_policy",
    "finite_q_exact",
    "q0_ivc_energy_per_cell",
    "finite_q_ivc_energy_per_cell",
    "finite_q_minus_q0_ivc_energy_per_cell",
    "selected_ivc_energy_per_cell",
    "vp_plus_energy_per_cell",
    "vp_minus_energy_per_cell",
    "vp_reference_name",
    "vp_reference_energy_per_cell",
    "ivc_q0_energy_per_cell",
    "ivc_q0_minus_vp_energy_per_cell",
    "selected_ivc_minus_vp_energy_per_cell",
    "vp_reference_direct_gap",
    "vp_reference_indirect_gap",
    "selected_ivc_direct_gap",
    "selected_ivc_indirect_gap",
    "vp_plus_self_consistency_warning",
    "vp_minus_self_consistency_warning",
    "ivc_self_consistency_warning",
    "ivc_finite_q_self_consistency_warning",
    "vp_reference_order_abs_nz",
    "selected_ivc_order_abs_nz",
    "selected_ivc_ivc_amplitude_block",
    "ivc_q0_ivc_amplitude_block",
    "elapsed_seconds",
    "point_dir",
)


class TaigeParameterConvergencePoint(BaseModel):
    """One parameter value in a single-point Taige convergence scan."""

    model_config = ConfigDict(frozen=True)

    value_index: int = Field(ge=0)
    scan_axis: ScanAxis
    scan_value: int = Field(ge=0)
    theta_deg: float
    u_D: float

    @computed_field
    @property
    def label(self) -> str:
        prefix = "plane_wave_shell" if self.scan_axis == "plane_wave_shell" else "active_bands"
        return f"{prefix}_{self.scan_value:03d}"

    def as_taige_point(self) -> TaigeSweepPoint:
        return TaigeSweepPoint(
            u_index=0,
            theta_index=int(self.value_index),
            u_D=float(self.u_D),
            theta_deg=float(self.theta_deg),
        )


def _normalize_axis(value: str) -> ScanAxis:
    try:
        return AXIS_ALIASES[str(value)]
    except KeyError as exc:
        raise ValueError("scan axis must be plane-wave-shell or active-bands") from exc


def _parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("value list must contain at least one integer")
    if any(value < 0 for value in values):
        raise ValueError("scan values must be nonnegative")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-axis", choices=["plane-wave-shell", "active-bands"], default="plane-wave-shell")
    parser.add_argument("--value-list", default=None)
    parser.add_argument("--material", choices=["mote2", "wse2"], default="mote2")
    parser.add_argument("--output-root", default=None)

    parser.add_argument("--u-d", type=float, default=0.0)
    parser.add_argument("--theta-deg", type=float, default=3.5)
    parser.add_argument("--task-id", type=int, default=None, help="Run one scan value by flat index.")

    parser.add_argument("--n-k", type=int, default=24)
    parser.add_argument("--plane-wave-shell", type=int, default=5, help="Fixed shell for active-band scans.")
    parser.add_argument("--n-bands", type=int, default=2, help="Fixed computed bands for plane-wave scans.")
    parser.add_argument("--n-active-bands-per-valley", type=int, default=2)

    parser.add_argument("--q-mesh", choices=["shell", "full"], default="full")
    parser.add_argument("--q-shell", type=int, default=0)
    parser.add_argument("--local-field-cutoff", type=int, default=4)
    parser.add_argument("--omit-q0", action="store_true")
    parser.add_argument("--epsilon", type=float, default=16.7)
    parser.add_argument("--gate-distance-nm", type=float, default=30.0)
    parser.add_argument("--smear-length-nm", type=float, default=0.347)
    parser.add_argument("--v0", type=float, default=1.0)
    parser.add_argument("--exchange-scale", type=float, default=1.0)
    parser.add_argument("--hartree-scale", type=float, default=1.0)
    parser.add_argument("--vertex-workers", type=int, default=1)
    parser.add_argument("--exchange-workers", type=int, default=1)
    parser.add_argument(
        "--density-vertex-retention",
        choices=["full", "hartree_only"],
        default="hartree_only",
    )
    parser.add_argument(
        "--density-vertex-layout",
        choices=["auto", "dense", "valley_compact"],
        default="auto",
    )
    parser.add_argument(
        "--exchange-representation",
        choices=["auto", "dense", "valley_sector"],
        default="auto",
    )
    parser.add_argument(
        "--form-factor-backend",
        choices=["auto", "scalar", "cached_gather", "vectorized"],
        default="auto",
    )

    parser.add_argument("--n-occ-per-k", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--min-iter", type=int, default=3)
    parser.add_argument("--mixing-method", choices=["linear", "oda"], default="oda")
    parser.add_argument("--mixing", type=float, default=0.45)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--energy-tolerance", type=float, default=1e-10)
    parser.add_argument("--seed-ordered-weight", type=float, default=0.8)
    parser.add_argument("--seed-random-weight", type=float, default=0.2)
    parser.add_argument("--random-seed", type=int, default=7)

    parser.add_argument("--n-theta", type=int, default=41)
    parser.add_argument("--endpoint-eps", type=float, default=1e-5)
    parser.add_argument("--domain-radius", type=float, default=20.0)
    parser.add_argument("--domain-width", type=float, default=3.0)
    parser.add_argument("--domain-winding", type=int, default=1)

    parser.add_argument("--no-chern", action="store_true")
    parser.add_argument("--compute-finite-q-ivc", action="store_true")
    parser.add_argument("--no-finite-q-ivc", action="store_true")
    parser.add_argument(
        "--finite-q-shift-policy",
        choices=["exact", "nearest-half", "nearest_half"],
        default="exact",
    )
    parser.add_argument(
        "--ivc-branch-policy",
        choices=["lower-energy", "q0"],
        default="q0",
    )
    parser.add_argument("--ivc-branch-tie-atol", type=float, default=1e-9)
    parser.add_argument("--allow-texture-in-ivc-ground-state", action="store_true")
    parser.add_argument("--texture-energy-tie-atol", type=float, default=1e-9)
    parser.add_argument("--write-hf-path-spectra", action="store_true")
    parser.add_argument("--hf-path-n-per-segment", type=int, default=36)

    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    return parser


def _axis_and_values(args: argparse.Namespace) -> tuple[ScanAxis, list[int]]:
    axis = _normalize_axis(args.scan_axis)
    text = args.value_list if args.value_list is not None else DEFAULT_VALUE_LISTS[axis]
    values = _parse_int_list(text)
    if axis == "active_bands" and any(value < 1 for value in values):
        raise ValueError("active-band scan values must be at least one")
    if axis == "plane_wave_shell" and int(args.n_active_bands_per_valley) > int(args.n_bands):
        raise ValueError("n_active_bands_per_valley cannot exceed n_bands")
    return axis, values


def _all_points(args: argparse.Namespace) -> list[TaigeParameterConvergencePoint]:
    axis, values = _axis_and_values(args)
    return [
        TaigeParameterConvergencePoint(
            value_index=idx,
            scan_axis=axis,
            scan_value=int(value),
            theta_deg=float(args.theta_deg),
            u_D=float(args.u_d),
        )
        for idx, value in enumerate(values)
    ]


def _selected_points(args: argparse.Namespace) -> list[TaigeParameterConvergencePoint]:
    points = _all_points(args)
    if args.task_id is None:
        return points
    task_id = int(args.task_id)
    if task_id < 0:
        raise ValueError("--task-id must be nonnegative")
    if task_id >= len(points):
        print(f"Task {task_id} is outside scan size {len(points)}; exiting.")
        return []
    return [points[task_id]]


def _output_root_from_args(args: argparse.Namespace) -> Path:
    axis = _normalize_axis(args.scan_axis)
    output = args.output_root or DEFAULT_OUTPUT_ROOTS[axis]
    root = Path(output)
    if not root.is_absolute():
        root = ROOT / root
    return root


def _point_dir(output_root: Path, point: TaigeParameterConvergencePoint) -> Path:
    return output_root / "points" / point.label


def _point_model_sizes(args: argparse.Namespace, point: TaigeParameterConvergencePoint) -> tuple[int, int, int]:
    if point.scan_axis == "plane_wave_shell":
        return int(point.scan_value), int(args.n_bands), int(args.n_active_bands_per_valley)
    active = int(point.scan_value)
    return int(args.plane_wave_shell), active, active


def _point_fields(args: argparse.Namespace, point: TaigeParameterConvergencePoint) -> dict[str, Any]:
    plane_wave_shell, n_bands, n_active = _point_model_sizes(args, point)
    return {
        "scan_axis": point.scan_axis,
        "scan_value": int(point.scan_value),
        "value_index": int(point.value_index),
        "convergence_point_label": point.label,
        "n_k": int(args.n_k),
        "plane_wave_shell": int(plane_wave_shell),
        "n_plane_waves": int(len(reciprocal_shell(plane_wave_shell))),
        "n_bands": int(n_bands),
        "n_active_bands_per_valley": int(n_active),
    }


def _with_point_fields(
    args: argparse.Namespace,
    point: TaigeParameterConvergencePoint,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prefix = _point_fields(args, point)
    return [{**prefix, **row} for row in rows]


def _params_for_point(
    args: argparse.Namespace,
    point: TaigeParameterConvergencePoint,
    point_dir: Path,
) -> ContinuumWorkflowParams:
    plane_wave_shell, n_bands, n_active = _point_model_sizes(args, point)
    model = taige_model_params(
        material=args.material,
        theta_deg=point.theta_deg,
        u_D=point.u_D,
        plane_wave_shell=plane_wave_shell,
        n_bands=n_bands,
        n_active_bands_per_valley=n_active,
    )
    interaction = taige_interaction_params(
        material=args.material,
        include_q0=not args.omit_q0,
        q_mesh=args.q_mesh,
        q_shell=args.q_shell,
        local_field_cutoff=args.local_field_cutoff,
        epsilon=float(args.epsilon),
        gate_distance_nm=float(args.gate_distance_nm),
        smear_length_nm=args.smear_length_nm,
        interaction_strength_scale=float(args.v0),
        exchange_scale=float(args.exchange_scale),
        hartree_scale=float(args.hartree_scale),
        vertex_workers=int(args.vertex_workers),
        exchange_workers=int(args.exchange_workers),
        density_vertex_retention=args.density_vertex_retention,
        density_vertex_layout=args.density_vertex_layout,
        exchange_representation=args.exchange_representation,
        form_factor_backend=args.form_factor_backend,
    )
    hf = ContinuumHFParams(
        n_occ_per_k=args.n_occ_per_k,
        max_iter=args.max_iter,
        min_iter=args.min_iter,
        mixing_method=args.mixing_method,
        mixing=args.mixing,
        tolerance=args.tolerance,
        energy_tolerance=args.energy_tolerance,
        seed_ordered_weight=args.seed_ordered_weight,
        seed_random_weight=args.seed_random_weight,
        random_seed=args.random_seed,
        store_projector_snapshots=False,
    )
    response = ResponseParams(
        n_theta=args.n_theta,
        theta_min=float(args.endpoint_eps),
        theta_max=float(pi - args.endpoint_eps),
        endpoint_eps=float(args.endpoint_eps),
    )
    return ContinuumWorkflowParams(
        model=model,
        grid=ContinuumGridParams(n_k=args.n_k),
        interaction=interaction,
        hf=hf,
        response=response,
        domain_wall=DomainWallParams(
            radius=args.domain_radius,
            width=args.domain_width,
            winding=args.domain_winding,
        ),
        output_dir=str(point_dir),
    )


def _normalize_policy(text: str) -> str:
    return str(text).replace("-", "_")


def _diagnostic_params(args: argparse.Namespace) -> TaigeSweepDiagnosticsParams:
    compute_finite_q = bool(args.compute_finite_q_ivc and not args.no_finite_q_ivc)
    branch_policy = _normalize_policy(args.ivc_branch_policy)
    if not compute_finite_q:
        branch_policy = "q0"
    return TaigeSweepDiagnosticsParams(
        compute_chern_numbers=not args.no_chern,
        compute_finite_q_ivc=compute_finite_q,
        finite_q_shift_policy=_normalize_policy(args.finite_q_shift_policy),  # type: ignore[arg-type]
        ivc_branch_policy=branch_policy,  # type: ignore[arg-type]
        ivc_branch_tie_atol=float(args.ivc_branch_tie_atol),
        nan_texture_when_ivc_lower=not args.allow_texture_in_ivc_ground_state,
        texture_energy_tie_atol=float(args.texture_energy_tie_atol),
        write_hf_path_spectra=bool(args.write_hf_path_spectra),
        hf_path_n_per_segment=args.hf_path_n_per_segment,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _load_point_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if "row" not in data:
        raise ValueError(f"{path} does not contain a point-summary row")
    return dict(data["row"])


def _stack_point_table(output_root: Path, filename: str, output_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_root / "points").rglob(filename)):
        rows.extend(_read_csv(path))
    _write_csv(output_root / output_name, rows)
    return rows


def _float_or_nan(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _summary_base_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in SUMMARY_COLUMNS:
        if key in row:
            out[key] = row[key]
    return out


def _add_metric_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    sorted_rows = sorted(rows, key=lambda row: (int(row["scan_value"]), int(row["value_index"])))
    largest = sorted_rows[-1]
    out_rows: list[dict[str, Any]] = []
    prev: dict[str, Any] | None = None
    for row in sorted_rows:
        out = dict(row)
        for metric in DELTA_METRICS:
            value = _float_or_nan(row.get(metric))
            prev_value = float("nan") if prev is None else _float_or_nan(prev.get(metric))
            largest_value = _float_or_nan(largest.get(metric))
            delta_prev = value - prev_value if np.isfinite(value) and np.isfinite(prev_value) else float("nan")
            delta_largest = (
                value - largest_value
                if np.isfinite(value) and np.isfinite(largest_value)
                else float("nan")
            )
            out[f"delta_prev_{metric}"] = delta_prev
            out[f"delta_largest_{metric}"] = delta_largest
            out[f"abs_delta_largest_{metric}"] = abs(delta_largest) if np.isfinite(delta_largest) else float("nan")
        out_rows.append(out)
        prev = row
    return out_rows


def _convergence_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    base = [_summary_base_row(row) for row in rows]
    return _add_metric_deltas(base)


def merge_point_summaries(output_root: Path) -> list[dict[str, Any]]:
    rows = [
        _load_point_summary(path)
        for path in sorted((output_root / "points").rglob("point_summary.json"))
    ]
    rows.sort(key=lambda row: (int(row["value_index"]), int(row["scan_value"])))
    _write_csv(output_root / "sweep.csv", rows)
    summary_rows = _convergence_summary_rows(rows)
    _write_csv(output_root / "convergence_summary.csv", summary_rows)
    stacked_counts = {
        "trial_theta": len(_stack_point_table(output_root, "trial_theta.csv", "sweep_trial_theta.csv")),
        "reference_energies": len(
            _stack_point_table(output_root, "reference_energies.csv", "sweep_reference_energies.csv")
        ),
        "noninteracting_chern_numbers": len(
            _stack_point_table(
                output_root,
                "noninteracting_chern_numbers.csv",
                "sweep_noninteracting_chern_numbers.csv",
            )
        ),
        "hf_chern_numbers": len(
            _stack_point_table(output_root, "hf_chern_numbers.csv", "sweep_hf_chern_numbers.csv")
        ),
        "hf_path_spectra": len(
            _stack_point_table(output_root, "hf_path_spectra.csv", "sweep_hf_path_spectra.csv")
        ),
    }
    _write_json(
        output_root / "convergence_summary.json",
        {
            "rows": summary_rows,
            "n_points": len(summary_rows),
            "delta_metrics": list(DELTA_METRICS),
        },
    )
    _write_json(
        output_root / "sweep.json",
        {
            "rows": rows,
            "n_points": len(rows),
            "convergence_summary_csv": str(output_root / "convergence_summary.csv"),
            "stacked_counts": stacked_counts,
            "tables": {
                "sweep_csv": str(output_root / "sweep.csv"),
                "convergence_summary_csv": str(output_root / "convergence_summary.csv"),
                "trial_theta_csv": str(output_root / "sweep_trial_theta.csv"),
                "reference_energies_csv": str(output_root / "sweep_reference_energies.csv"),
                "noninteracting_chern_numbers_csv": str(output_root / "sweep_noninteracting_chern_numbers.csv"),
                "hf_chern_numbers_csv": str(output_root / "sweep_hf_chern_numbers.csv"),
                "hf_path_spectra_csv": str(output_root / "sweep_hf_path_spectra.csv"),
            },
        },
    )
    return rows


def _write_plan(
    output_root: Path,
    points: list[TaigeParameterConvergencePoint],
    args: argparse.Namespace,
) -> None:
    rows = [
        point.model_dump(mode="json")
        | _point_fields(args, point)
        | {"point_dir": str(_point_dir(output_root, point))}
        for point in points
    ]
    args_json = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    _write_csv(output_root / "sweep_plan.csv", rows)
    _write_json(
        output_root / "sweep_plan.json",
        {
            "points": rows,
            "n_points": len(rows),
            "scan_axis": _normalize_axis(args.scan_axis),
            "args": args_json,
        },
    )


def _write_point_diagnostic_tables(
    args: argparse.Namespace,
    point: TaigeParameterConvergencePoint,
    point_dir: Path,
    diagnostics: Any,
) -> None:
    _write_csv(
        point_dir / "trial_theta.csv",
        _with_point_fields(args, point, diagnostics.trial_theta_rows),
    )
    _write_csv(
        point_dir / "reference_energies.csv",
        _with_point_fields(args, point, diagnostics.reference_energy_rows),
    )
    if diagnostics.summary.chern_enabled:
        _write_csv(
            point_dir / "noninteracting_chern_numbers.csv",
            _with_point_fields(args, point, diagnostics.noninteracting_chern_rows),
        )
        _write_csv(
            point_dir / "hf_chern_numbers.csv",
            _with_point_fields(args, point, diagnostics.hf_chern_rows),
        )
    if diagnostics.summary.hf_path_spectra_csv is not None:
        _write_csv(
            point_dir / "hf_path_spectra.csv",
            _with_point_fields(args, point, diagnostics.hf_path_spectrum_rows),
        )


def run_point(
    args: argparse.Namespace,
    output_root: Path,
    point: TaigeParameterConvergencePoint,
) -> dict[str, Any]:
    point_dir = _point_dir(output_root, point)
    point_summary = point_dir / "point_summary.json"
    if args.skip_existing and point_summary.exists():
        print(f"Skipping existing {point.label}: {point_summary}")
        return _load_point_summary(point_summary)

    params = _params_for_point(args, point, point_dir)
    _write_json(point_dir / "point_params.json", params.model_dump(mode="json"))
    diagnostic_controls = _diagnostic_params(args)
    fields = _point_fields(args, point)
    print(
        f"Running {taige_material_label(args.material)} parameter convergence "
        f"{fields['scan_axis']}={fields['scan_value']} "
        f"theta={point.theta_deg:.8g} deg u_D={point.u_D:.8g} meV "
        f"n_k={args.n_k} plane_wave_shell={fields['plane_wave_shell']} "
        f"active_bands={fields['n_active_bands_per_valley']} "
        f"finite_q_enabled={diagnostic_controls.compute_finite_q_ivc}"
    )
    start = time.perf_counter()
    result = run_taige_branch_selected_symmetric_hf_workflow(
        params,
        finite_q_enabled=diagnostic_controls.compute_finite_q_ivc,
        finite_q_shift_policy=diagnostic_controls.finite_q_shift_policy,  # type: ignore[arg-type]
        ivc_branch_policy=diagnostic_controls.ivc_branch_policy,
        tie_atol=diagnostic_controls.ivc_branch_tie_atol,
        suppress_texture_when_ivc_below_vp=diagnostic_controls.nan_texture_when_ivc_lower,
        texture_energy_tie_atol=diagnostic_controls.texture_energy_tie_atol,
        write_outputs=True,
    )
    diagnostics = build_taige_sweep_diagnostics(
        point=point.as_taige_point(),
        workflow_result=result,
        controls=diagnostic_controls,
        elapsed_seconds=0.0,
        point_dir=point_dir,
    )
    seconds = time.perf_counter() - start
    _write_point_diagnostic_tables(args, point, point_dir, diagnostics)
    row = diagnostics.summary.model_copy(update={"elapsed_seconds": seconds}).as_csv_row()
    row = {**fields, **row}
    _write_json(
        point_summary,
        {
            "point": point.model_dump(mode="json"),
            "row": row,
            "params": params.model_dump(mode="json"),
            "diagnostics_params": diagnostic_controls.model_dump(mode="json"),
            "summary": result.summary.model_dump(mode="json"),
            "branch_selection": result.branch_selection,
            "reference_summary": result.reference_summary,
            "finite_q_ivc": (
                None
                if diagnostics.finite_q_ivc is None
                else {
                    "metadata": diagnostics.finite_q_ivc.metadata,
                    "diagnostics": diagnostics.finite_q_ivc.result.diagnostics.model_dump(mode="json"),
                }
            ),
        },
    )
    print(
        f"Finished {point.label}: cG={row['cG']:.12g} "
        f"selected_ivc_branch={row.get('selected_ivc_branch')} "
        f"texture_valid={row.get('texture_valid')} elapsed={seconds:.1f}s"
    )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _axis_and_values(args)
    except ValueError as exc:
        parser.error(str(exc))
    finite_q_policy = _normalize_policy(args.finite_q_shift_policy)
    compute_finite_q = bool(args.compute_finite_q_ivc and not args.no_finite_q_ivc)
    if compute_finite_q and finite_q_policy == "exact" and int(args.n_k) % 6:
        parser.error(
            "exact finite-Q IVC diagnostics require --n-k divisible by 6; "
            "use --finite-q-shift-policy nearest-half or --no-finite-q-ivc."
        )

    output_root = _output_root_from_args(args)
    args.output_root = str(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        rows = merge_point_summaries(output_root)
        print(f"Merged {len(rows)} point summaries into {output_root / 'sweep.csv'}")
        return 0

    points = _selected_points(args)
    _write_plan(output_root, points, args)
    if args.dry_run:
        print(f"Wrote dry-run plan with {len(points)} selected point(s) to {output_root}")
        return 0
    if not points:
        return 0

    rows = [run_point(args, output_root, point) for point in points]
    if args.task_id is None:
        rows = merge_point_summaries(output_root)
        print(f"Wrote serial convergence summary to {output_root / 'convergence_summary.csv'}")
    else:
        print("Single task complete. Run with --merge-only after all tasks finish to build summaries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
