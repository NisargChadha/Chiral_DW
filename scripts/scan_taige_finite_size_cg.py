#!/usr/bin/env python3
"""Finite-size sweep of Taige-continuum symmetric-HF cG over momentum meshes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from math import pi
from pathlib import Path
from typing import Any

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
from chiral_dw.continuum.taige import taige_interaction_params, taige_model_params  # noqa: E402
from chiral_dw.continuum.workflow import run_taige_branch_selected_symmetric_hf_workflow  # noqa: E402


class TaigeFiniteSizePhasePoint(BaseModel):
    """One displacement/twist point in the finite-size phase diagram."""

    model_config = ConfigDict(frozen=True)

    u_index: int = Field(ge=0)
    theta_index: int = Field(ge=0)
    u_D: float
    theta_deg: float

    @computed_field
    @property
    def label(self) -> str:
        return f"u_{self.u_index:03d}_theta_{self.theta_index:03d}"

    def as_taige_point(self) -> TaigeSweepPoint:
        return TaigeSweepPoint(
            u_index=self.u_index,
            theta_index=self.theta_index,
            u_D=self.u_D,
            theta_deg=self.theta_deg,
        )


class TaigeFiniteSizePoint(BaseModel):
    """One mesh evaluation inside a finite-size phase point."""

    model_config = ConfigDict(frozen=True)

    n_k_index: int = Field(ge=0)
    n_k: int = Field(ge=1)
    u_index: int = Field(ge=0)
    theta_index: int = Field(ge=0)
    u_D: float
    theta_deg: float

    @computed_field
    @property
    def phase_label(self) -> str:
        return f"u_{self.u_index:03d}_theta_{self.theta_index:03d}"

    @computed_field
    @property
    def mesh_label(self) -> str:
        return f"nk_{self.n_k:03d}"

    @computed_field
    @property
    def label(self) -> str:
        return f"{self.phase_label}_{self.mesh_label}"

    def as_taige_point(self) -> TaigeSweepPoint:
        return TaigeSweepPoint(
            u_index=self.u_index,
            theta_index=self.theta_index,
            u_D=self.u_D,
            theta_deg=self.theta_deg,
        )


def _parse_int_list(text: str) -> list[int]:
    values = [int(part) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("n-k-list must contain at least one integer")
    if any(n < 1 for n in values):
        raise ValueError("all n_k values must be positive")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="results/taige_cg_finite_size_nk18_24_u0_15_theta2_4p2",
    )

    parser.add_argument("--n-k-list", default="18,20,22,24")
    parser.add_argument("--u-d", type=float, default=None, help="Run one explicit displacement value in meV.")
    parser.add_argument("--theta-deg", type=float, default=None, help="Run one explicit twist angle in degrees.")
    parser.add_argument("--u-d-min", type=float, default=0.0)
    parser.add_argument("--u-d-max", type=float, default=15.0)
    parser.add_argument("--n-u-d", type=int, default=21)
    parser.add_argument("--theta-min-deg", type=float, default=2.0)
    parser.add_argument("--theta-max-deg", type=float, default=4.2)
    parser.add_argument("--n-twist", type=int, default=21)
    parser.add_argument("--task-id", type=int, default=None, help="SLURM-style flat phase-grid index.")

    parser.add_argument("--plane-wave-shell", type=int, default=5)
    parser.add_argument("--n-bands", type=int, default=2)
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
    parser.add_argument(
        "--vertex-workers",
        type=int,
        default=1,
        help="Number of joblib worker processes for Taige density-vertex q-slabs.",
    )
    parser.add_argument(
        "--exchange-workers",
        type=int,
        default=1,
        help="Number of joblib worker processes for dense exchange-kernel q-slabs.",
    )
    parser.add_argument(
        "--density-vertex-retention",
        choices=["full", "hartree_only"],
        default="hartree_only",
        help="Retain full density vertices or only Hartree channels after dense exchange build.",
    )
    parser.add_argument(
        "--density-vertex-layout",
        choices=["auto", "dense", "valley_compact"],
        default="auto",
        help="Density-vertex storage layout; auto uses valley-compact Taige vertices.",
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

    parser.add_argument("--no-chern", action="store_true", help="Skip noninteracting and HF Chern diagnostics.")
    parser.add_argument("--compute-finite-q-ivc", action="store_true", help="Opt into the Taige IVC- finite-Q diagnostic branch.")
    parser.add_argument("--no-finite-q-ivc", action="store_true", help="Skip the Taige IVC- finite-Q diagnostic branch.")
    parser.add_argument(
        "--finite-q-shift-policy",
        choices=["exact", "nearest-half", "nearest_half"],
        default="nearest-half",
        help="Finite-Q IVC- mesh policy; nearest-half records approximate shifts for incommensurate nk.",
    )
    parser.add_argument(
        "--ivc-branch-policy",
        choices=["lower-energy", "q0"],
        default="q0",
        help="Choose the IVC branch used for interpolation; q0 is the finite-size default.",
    )
    parser.add_argument("--ivc-branch-tie-atol", type=float, default=1e-9)
    parser.add_argument(
        "--allow-texture-in-ivc-ground-state",
        action="store_true",
        help="Keep cG/K(theta)/trial texture diagnostics even when IVC is below the VP reference.",
    )
    parser.add_argument("--texture-energy-tie-atol", type=float, default=1e-9)
    parser.add_argument("--write-hf-path-spectra", action="store_true")
    parser.add_argument("--hf-path-n-per-segment", type=int, default=36)

    parser.add_argument("--fit-degree", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write the selected scan plan without running HF.")
    parser.add_argument("--merge-only", action="store_true", help="Merge existing point summaries into sweep.csv/json.")
    return parser


def _axis_values(single: float | None, lower: float, upper: float, count: int, name: str) -> np.ndarray:
    if single is not None:
        return np.asarray([float(single)], dtype=float)
    n = int(count)
    if n < 1:
        raise ValueError(f"{name} count must be positive")
    return np.linspace(float(lower), float(upper), n)


def _all_phase_points(args: argparse.Namespace) -> list[TaigeFiniteSizePhasePoint]:
    if (args.u_d is None) ^ (args.theta_deg is None):
        raise ValueError("--u-d and --theta-deg must be supplied together for an explicit single point")
    u_values = _axis_values(args.u_d, args.u_d_min, args.u_d_max, args.n_u_d, "u_D")
    theta_values = _axis_values(
        args.theta_deg,
        args.theta_min_deg,
        args.theta_max_deg,
        args.n_twist,
        "theta",
    )
    return [
        TaigeFiniteSizePhasePoint(
            u_index=iu,
            theta_index=it,
            u_D=float(u),
            theta_deg=float(theta),
        )
        for iu, u in enumerate(u_values)
        for it, theta in enumerate(theta_values)
    ]


def _mesh_points_for_phases(
    args: argparse.Namespace,
    phases: list[TaigeFiniteSizePhasePoint],
) -> list[TaigeFiniteSizePoint]:
    nks = _parse_int_list(args.n_k_list)
    return [
        TaigeFiniteSizePoint(
            n_k_index=ink,
            n_k=int(n_k),
            u_index=phase.u_index,
            theta_index=phase.theta_index,
            u_D=phase.u_D,
            theta_deg=phase.theta_deg,
        )
        for phase in phases
        for ink, n_k in enumerate(nks)
    ]


def _all_points(args: argparse.Namespace) -> list[TaigeFiniteSizePoint]:
    return _mesh_points_for_phases(args, _all_phase_points(args))


def _selected_phase_points(args: argparse.Namespace) -> list[TaigeFiniteSizePhasePoint]:
    phases = _all_phase_points(args)
    if args.task_id is None:
        return phases
    task_id = int(args.task_id)
    if task_id < 0:
        raise ValueError("--task-id must be nonnegative")
    if task_id >= len(phases):
        print(f"Task {task_id} is outside phase-grid size {len(phases)}; exiting.")
        return []
    return [phases[task_id]]


def _selected_points(args: argparse.Namespace) -> list[TaigeFiniteSizePoint]:
    return _mesh_points_for_phases(args, _selected_phase_points(args))


def _normalize_policy(text: str) -> str:
    return str(text).replace("-", "_")


def _params_for_point(
    args: argparse.Namespace,
    point: TaigeFiniteSizePoint,
    point_dir: Path,
) -> ContinuumWorkflowParams:
    model = taige_model_params(
        theta_deg=point.theta_deg,
        u_D=point.u_D,
        plane_wave_shell=args.plane_wave_shell,
        n_bands=args.n_bands,
        n_active_bands_per_valley=args.n_active_bands_per_valley,
    )
    interaction = taige_interaction_params(
        include_q0=not args.omit_q0,
        q_mesh=args.q_mesh,
        q_shell=args.q_shell,
        local_field_cutoff=args.local_field_cutoff,
    ).model_copy(
        update={
            "epsilon": float(args.epsilon),
            "gate_distance_nm": float(args.gate_distance_nm),
            "smear_length_nm": float(args.smear_length_nm),
            "v0": float(args.v0),
            "exchange_scale": float(args.exchange_scale),
            "hartree_scale": float(args.hartree_scale),
            "vertex_workers": int(args.vertex_workers),
            "exchange_workers": int(args.exchange_workers),
            "density_vertex_retention": args.density_vertex_retention,
            "density_vertex_layout": args.density_vertex_layout,
        }
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
        grid=ContinuumGridParams(n_k=point.n_k),
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


def _diagnostic_params(args: argparse.Namespace) -> TaigeSweepDiagnosticsParams:
    branch_policy = str(args.ivc_branch_policy).replace("-", "_")
    compute_finite_q = bool(args.compute_finite_q_ivc and not args.no_finite_q_ivc)
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


def _point_dir(output_root: Path, point: TaigeFiniteSizePoint) -> Path:
    return output_root / "points" / point.phase_label / point.mesh_label


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


def _finite_size_fields(point: TaigeFiniteSizePoint) -> dict[str, Any]:
    return {
        "n_k_index": int(point.n_k_index),
        "n_k": int(point.n_k),
        "inv_n_k": float(1.0 / point.n_k),
        "inv_n_k_squared": float(1.0 / (point.n_k * point.n_k)),
        "phase_point_label": point.phase_label,
        "finite_size_point_label": point.label,
    }


def _add_finite_size_fields(
    point: TaigeFiniteSizePoint,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    prefix = _finite_size_fields(point)
    return [{**prefix, **row} for row in rows]


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


def _bool_or_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _largest_mesh_row(group: list[dict[str, Any]]) -> dict[str, Any]:
    return max(group, key=lambda row: int(row["n_k"]))


def _finite_min(values: list[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    return float(min(finite)) if finite else float("nan")


def _fit_cg_rows(rows: list[dict[str, Any]], fit_degree: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["u_index"]), int(row["theta_index"])), []).append(row)

    fit_rows: list[dict[str, Any]] = []
    for (u_index, theta_index), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: int(row["n_k"]))
        largest = _largest_mesh_row(group)
        usable = []
        for row in group:
            cG = _float_or_nan(row.get("cG"))
            inv_n = _float_or_nan(row.get("inv_n_k"))
            if np.isfinite(cG) and np.isfinite(inv_n):
                usable.append((inv_n, cG, row))
        texture_valid_count = sum(_bool_or_false(row.get("texture_valid")) for row in group)
        texture_invalid_count = len(group) - texture_valid_count
        degree = 0
        coeffs = np.asarray([], dtype=float)
        cG_extrapolated = float("nan")
        cG_slope = float("nan")
        rmse = float("nan")
        if usable:
            usable.sort(key=lambda item: item[0])
            x = np.asarray([item[0] for item in usable], dtype=float)
            y = np.asarray([item[1] for item in usable], dtype=float)
            degree = min(max(int(fit_degree), 0), max(x.size - 1, 0))
            coeffs = np.polyfit(x, y, degree) if degree > 0 else np.asarray([float(np.mean(y))])
            fit = np.poly1d(coeffs)
            residual = y - fit(x)
            cG_extrapolated = float(fit(0.0))
            cG_slope = float(coeffs[-2]) if degree >= 1 else 0.0
            rmse = float(np.sqrt(np.mean(residual * residual)))

        vp_plus_largest = _float_or_nan(largest.get("vp_plus_energy_per_cell"))
        vp_minus_largest = _float_or_nan(largest.get("vp_minus_energy_per_cell"))
        row_out: dict[str, Any] = {
            "u_index": u_index,
            "theta_index": theta_index,
            "u_D_meV": _float_or_nan(largest.get("u_D_meV")),
            "theta_deg": _float_or_nan(largest.get("theta_deg")),
            "fit_degree": int(degree),
            "fit_degree_requested": int(fit_degree),
            "n_mesh": int(len(group)),
            "n_fit": int(len(usable)),
            "n_texture_valid": int(texture_valid_count),
            "n_texture_invalid": int(texture_invalid_count),
            "n_k_min": int(min(int(row["n_k"]) for row in group)),
            "n_k_max": int(max(int(row["n_k"]) for row in group)),
            "largest_n_k": int(largest["n_k"]),
            "cG_extrapolated_inv_n0": cG_extrapolated,
            "cG_fit_slope_inv_n": cG_slope,
            "cG_at_largest_nk": _float_or_nan(largest.get("cG")),
            "rmse": rmse,
            "coefficients_descending_json": json.dumps([float(c) for c in coeffs]),
            "texture_valid_largest_nk": _bool_or_false(largest.get("texture_valid")),
            "texture_invalid_reason_largest_nk": largest.get("texture_invalid_reason"),
            "hf_ground_state_largest_nk": largest.get("hf_ground_state"),
            "gap_min_largest_nk": _float_or_nan(largest.get("gap_min")),
            "gap_min_over_meshes": _finite_min([_float_or_nan(row.get("gap_min")) for row in group]),
            "vp_reference_name_largest_nk": largest.get("vp_reference_name"),
            "vp_reference_energy_per_cell_largest_nk": _float_or_nan(
                largest.get("vp_reference_energy_per_cell")
            ),
            "vp_plus_energy_per_cell_largest_nk": vp_plus_largest,
            "vp_minus_energy_per_cell_largest_nk": vp_minus_largest,
            "vp_plus_minus_vp_minus_energy_per_cell_largest_nk": (
                float(vp_plus_largest - vp_minus_largest)
                if np.isfinite(vp_plus_largest) and np.isfinite(vp_minus_largest)
                else float("nan")
            ),
            "ivc_q0_energy_per_cell_largest_nk": _float_or_nan(
                largest.get("ivc_q0_energy_per_cell")
            ),
            "ivc_q0_minus_vp_energy_per_cell_largest_nk": _float_or_nan(
                largest.get("ivc_q0_minus_vp_energy_per_cell")
            ),
            "selected_ivc_minus_vp_energy_per_cell_largest_nk": _float_or_nan(
                largest.get("selected_ivc_minus_vp_energy_per_cell")
            ),
        }
        for key, value in largest.items():
            if str(key).startswith("chern_"):
                row_out[f"largest_nk_{key}"] = value
        fit_rows.append(row_out)
    return fit_rows


def merge_point_summaries(output_root: Path, *, fit_degree: int) -> list[dict[str, Any]]:
    rows = [
        _load_point_summary(path)
        for path in sorted((output_root / "points").rglob("point_summary.json"))
    ]
    rows.sort(key=lambda row: (int(row["u_index"]), int(row["theta_index"]), int(row["n_k_index"])))
    _write_csv(output_root / "sweep.csv", rows)
    fit_rows = _fit_cg_rows(rows, fit_degree)
    _write_csv(output_root / "finite_size_fits.csv", fit_rows)
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
        output_root / "finite_size_fits.json",
        {
            "rows": fit_rows,
            "n_fits": len(fit_rows),
            "fit_variable": "inv_n_k",
            "fit_target": "cG",
            "fit_degree_requested": int(fit_degree),
        },
    )
    _write_json(
        output_root / "sweep.json",
        {
            "rows": rows,
            "n_points": len(rows),
            "finite_size_fits": str(output_root / "finite_size_fits.csv"),
            "stacked_counts": stacked_counts,
            "tables": {
                "sweep_csv": str(output_root / "sweep.csv"),
                "finite_size_fits_csv": str(output_root / "finite_size_fits.csv"),
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
    phases: list[TaigeFiniteSizePhasePoint],
    points: list[TaigeFiniteSizePoint],
    args: argparse.Namespace,
) -> None:
    phase_rows = [
        phase.model_dump(mode="json") | {"phase_dir": str(output_root / "points" / phase.label)}
        for phase in phases
    ]
    mesh_rows = [
        point.model_dump(mode="json") | {"point_dir": str(_point_dir(output_root, point))}
        for point in points
    ]
    _write_csv(output_root / "sweep_phase_plan.csv", phase_rows)
    _write_csv(output_root / "sweep_plan.csv", mesh_rows)
    args_json = {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, type(None)))
    }
    _write_json(
        output_root / "sweep_phase_plan.json",
        {
            "phase_points": phase_rows,
            "n_phase_points": len(phase_rows),
            "args": args_json,
        },
    )
    _write_json(
        output_root / "sweep_plan.json",
        {
            "points": mesh_rows,
            "mesh_points": mesh_rows,
            "phase_points": phase_rows,
            "n_points": len(mesh_rows),
            "n_mesh_points": len(mesh_rows),
            "n_phase_points": len(phase_rows),
            "args": args_json,
        },
    )


def _write_point_diagnostic_tables(
    point: TaigeFiniteSizePoint,
    point_dir: Path,
    diagnostics,
) -> None:
    _write_csv(
        point_dir / "trial_theta.csv",
        _add_finite_size_fields(point, diagnostics.trial_theta_rows),
    )
    _write_csv(
        point_dir / "reference_energies.csv",
        _add_finite_size_fields(point, diagnostics.reference_energy_rows),
    )
    if diagnostics.summary.chern_enabled:
        _write_csv(
            point_dir / "noninteracting_chern_numbers.csv",
            _add_finite_size_fields(point, diagnostics.noninteracting_chern_rows),
        )
        _write_csv(
            point_dir / "hf_chern_numbers.csv",
            _add_finite_size_fields(point, diagnostics.hf_chern_rows),
        )
    if diagnostics.summary.hf_path_spectra_csv is not None:
        _write_csv(
            point_dir / "hf_path_spectra.csv",
            _add_finite_size_fields(point, diagnostics.hf_path_spectrum_rows),
        )


def run_point(
    args: argparse.Namespace,
    output_root: Path,
    point: TaigeFiniteSizePoint,
) -> dict[str, Any]:
    point_dir = _point_dir(output_root, point)
    point_summary = point_dir / "point_summary.json"
    if args.skip_existing and point_summary.exists():
        print(f"Skipping existing {point.label}: {point_summary}")
        return _load_point_summary(point_summary)

    params = _params_for_point(args, point, point_dir)
    _write_json(point_dir / "point_params.json", params.model_dump(mode="json"))
    diagnostic_controls = _diagnostic_params(args)
    print(
        "Running Taige finite-size cG "
        f"n_k={point.n_k} u_D={point.u_D:.8g} meV theta={point.theta_deg:.8g} deg "
        f"finite_q_enabled={diagnostic_controls.compute_finite_q_ivc} "
        f"vertex_workers={args.vertex_workers} "
        f"exchange_workers={args.exchange_workers}"
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
    _write_point_diagnostic_tables(point, point_dir, diagnostics)
    row = diagnostics.summary.model_copy(update={"elapsed_seconds": seconds}).as_csv_row()
    row = {**_finite_size_fields(point), **row}
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
        f"texture_valid={row.get('texture_valid')} elapsed={seconds:.1f}s"
    )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    finite_q_policy = _normalize_policy(args.finite_q_shift_policy)
    nks = _parse_int_list(args.n_k_list)
    compute_finite_q = bool(args.compute_finite_q_ivc and not args.no_finite_q_ivc)
    if compute_finite_q and finite_q_policy == "exact":
        bad = [n for n in nks if n % 6]
        if bad:
            parser.error(
                "exact finite-Q IVC diagnostics require every n_k divisible by 6; "
                f"bad n_k values: {bad}. Use --finite-q-shift-policy nearest-half "
                "or --no-finite-q-ivc."
            )
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        rows = merge_point_summaries(output_root, fit_degree=args.fit_degree)
        print(f"Merged {len(rows)} point summaries into {output_root / 'sweep.csv'}")
        return 0

    phases = _selected_phase_points(args)
    points = _mesh_points_for_phases(args, phases)
    _write_plan(output_root, phases, points, args)
    if args.dry_run:
        print(
            "Wrote dry-run plan with "
            f"{len(phases)} selected phase point(s) and {len(points)} mesh evaluation(s) "
            f"to {output_root}"
        )
        return 0
    if not points:
        return 0

    _rows = [run_point(args, output_root, point) for point in points]
    if args.task_id is None:
        rows = merge_point_summaries(output_root, fit_degree=args.fit_degree)
        print(f"Wrote serial finite-size summary to {output_root / 'sweep.csv'}")
    else:
        print("Single task complete. Run with --merge-only after the array finishes to build sweep.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
