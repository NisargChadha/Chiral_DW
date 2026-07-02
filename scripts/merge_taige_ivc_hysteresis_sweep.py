#!/usr/bin/env python3
"""Merge Taige IVC hysteresis branch outputs into comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chiral_dw.continuum import (  # noqa: E402
    compare_hysteresis_records,
    is_clean_hysteresis_record,
    projector_overlap_diagnostics_with_frames,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/taige_ivc_hysteresis")
    parser.add_argument("--n-occ-per-k", type=int, default=1)
    parser.add_argument("--allow-missing-directions", action="store_true")
    return parser


def _output_root(args: argparse.Namespace) -> Path:
    root = Path(args.output_root)
    if not root.is_absolute():
        root = ROOT / root
    return root


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
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _record_paths(output_root: Path) -> list[Path]:
    return sorted((output_root / "branches").rglob("point_record.json"))


def _load_record(path: Path) -> dict[str, Any]:
    row = json.loads(path.read_text())
    row.setdefault("point_record_path", str(path))
    return row


def _load_projector_payload(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    projector_path = Path(row["projector_path"])
    with np.load(projector_path, allow_pickle=False) as data:
        return (
            np.asarray(data["final_projector"], dtype=complex),
            np.asarray(data["active_frames"], dtype=complex),
        )


def _branch_id(row: dict[str, Any]) -> str:
    branch = row.get("branch_id")
    if branch:
        return str(branch)
    return f"{row.get('sweep_axis', 'u_D')}_{row.get('direction', 'unknown')}"


def _safe_float(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    return float(value)


def _write_pair_comparison(
    *,
    output_root: Path,
    rows_by_point: dict[tuple[int, int], list[dict[str, Any]]],
    sweep_axis: str,
    n_occ_per_k: int,
) -> list[dict[str, Any]]:
    comparison_rows: list[dict[str, Any]] = []
    for key, rows in sorted(rows_by_point.items()):
        by_direction = {
            str(row["direction"]): row
            for row in rows
            if str(row.get("sweep_axis", "u_D")) == sweep_axis
        }
        if "up" not in by_direction or "down" not in by_direction:
            continue
        up_projector, up_frames = _load_projector_payload(by_direction["up"])
        down_projector, down_frames = _load_projector_payload(by_direction["down"])
        comparison, _overlap = compare_hysteresis_records(
            up=by_direction["up"],
            down=by_direction["down"],
            up_projector=up_projector,
            down_projector=down_projector,
            up_frames=up_frames,
            down_frames=down_frames,
            n_occ_per_k=n_occ_per_k,
        )
        row = comparison.model_dump(mode="json")
        row["sweep_axis"] = sweep_axis
        row["branch_up"] = _branch_id(by_direction["up"])
        row["branch_down"] = _branch_id(by_direction["down"])
        row["lowest_energy_raw_branch"] = (
            _branch_id(by_direction["up"])
            if float(row["energy_up_minus_down"]) <= 0.0
            else _branch_id(by_direction["down"])
        )
        clean_candidates = [
            item for item in by_direction.values() if is_clean_hysteresis_record(item)
        ]
        clean = min(clean_candidates, key=lambda item: float(item["energy_total_per_cell"])) if clean_candidates else None
        row["lowest_energy_clean_branch"] = None if clean is None else _branch_id(clean)
        row["row_reliability"] = "clean" if clean is not None else "unreliable_no_clean_candidate"
        row["warning_count"] = sum(
            int(not is_clean_hysteresis_record(item)) for item in by_direction.values()
        )
        comparison_rows.append(row)
    name = "displacement" if sweep_axis == "u_D" else "twist"
    _write_csv(output_root / f"hysteresis_{name}_comparison.csv", comparison_rows)
    return comparison_rows


def _all_candidate_summary(
    *,
    key: tuple[int, int],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(rows, key=_branch_id)
    raw = min(ordered, key=lambda item: float(item["energy_total_per_cell"]))
    clean_candidates = [row for row in ordered if is_clean_hysteresis_record(row)]
    clean = min(clean_candidates, key=lambda item: float(item["energy_total_per_cell"])) if clean_candidates else None
    high_gap = max(ordered, key=lambda item: _safe_float(item, "direct_gap_min"))
    low_gap = min(ordered, key=lambda item: _safe_float(item, "direct_gap_min"))
    summary: dict[str, Any] = {
        "theta_index": key[0],
        "u_index": key[1],
        "theta_deg": float(ordered[0]["theta_deg"]),
        "u_D_meV": float(ordered[0]["u_D_meV"]),
        "n_branch_candidates": len(ordered),
        "available_branches": ";".join(_branch_id(row) for row in ordered),
        "lowest_energy_raw_branch": _branch_id(raw),
        "lowest_energy_raw_cG": _safe_float(raw, "cG"),
        "lowest_energy_raw_clean": is_clean_hysteresis_record(raw),
        "lowest_energy_clean_branch": None if clean is None else _branch_id(clean),
        "lowest_energy_clean_cG": None if clean is None else _safe_float(clean, "cG"),
        "high_gap_branch": _branch_id(high_gap),
        "high_gap_cG": _safe_float(high_gap, "cG"),
        "low_gap_branch": _branch_id(low_gap),
        "low_gap_cG": _safe_float(low_gap, "cG"),
        "row_reliability": "clean" if clean is not None else "unreliable_no_clean_candidate",
        "warning_count": sum(int(not is_clean_hysteresis_record(row)) for row in ordered),
        "max_aufbau_residual_norm": max(_safe_float(row, "aufbau_residual_norm") for row in ordered),
        "max_commutator_norm": max(_safe_float(row, "commutator_norm") for row in ordered),
        "max_delta_P": max(_safe_float(row, "delta_P") for row in ordered),
        "max_delta_energy_abs": max(abs(_safe_float(row, "delta_energy", 0.0)) for row in ordered),
    }
    for row in ordered:
        label = _branch_id(row)
        summary[f"energy_{label}"] = _safe_float(row, "energy_total_per_cell")
        summary[f"direct_gap_{label}"] = _safe_float(row, "direct_gap_min")
        summary[f"ivc_amplitude_{label}"] = _safe_float(row, "ivc_amplitude_block")
        summary[f"cG_{label}"] = _safe_float(row, "cG")
        summary[f"clean_{label}"] = is_clean_hysteresis_record(row)
        summary[f"warning_{label}"] = bool(row.get("warning_flag", False))
        summary[f"aufbau_residual_{label}"] = _safe_float(row, "aufbau_residual_norm")
        summary[f"commutator_norm_{label}"] = _safe_float(row, "commutator_norm")
    return summary


def merge_outputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_root = _output_root(args)
    records = [_load_record(path) for path in _record_paths(output_root)]
    records.sort(
        key=lambda row: (
            int(row["theta_index"]),
            int(row["u_index"]),
            str(row.get("sweep_axis", "u_D")),
            str(row["direction"]),
        )
    )
    _write_csv(output_root / "hysteresis_sweep.csv", records)
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in records:
        key = (int(row["theta_index"]), int(row["u_index"]))
        grouped.setdefault(key, []).append(row)
    comparison_rows = [
        _all_candidate_summary(key=key, rows=rows)
        for key, rows in sorted(grouped.items())
    ]
    _write_csv(output_root / "hysteresis_comparison.csv", comparison_rows)
    candidate_rows = []
    for key, rows in sorted(grouped.items()):
        for row in rows:
            candidate_rows.append(
                {
                    "theta_index": key[0],
                    "u_index": key[1],
                    "branch_id": _branch_id(row),
                    "sweep_axis": row.get("sweep_axis", "u_D"),
                    "direction": row.get("direction"),
                    "energy_total_per_cell": row.get("energy_total_per_cell"),
                    "direct_gap_min": row.get("direct_gap_min"),
                    "ivc_amplitude_block": row.get("ivc_amplitude_block"),
                    "clean_branch": is_clean_hysteresis_record(row),
                    "branch_reliability": row.get("branch_reliability"),
                    "cG": row.get("cG"),
                    "cG_warning_flag": row.get("cG_warning_flag"),
                    "aufbau_residual_norm": row.get("aufbau_residual_norm"),
                    "commutator_norm": row.get("commutator_norm"),
                    "warning_flag": row.get("warning_flag"),
                    "run_id": row.get("run_id"),
                    "projector_path": row.get("projector_path"),
                }
            )
    _write_csv(output_root / "hysteresis_all_branch_candidates.csv", candidate_rows)
    displacement_rows = _write_pair_comparison(
        output_root=output_root,
        rows_by_point=grouped,
        sweep_axis="u_D",
        n_occ_per_k=args.n_occ_per_k,
    )
    twist_rows = _write_pair_comparison(
        output_root=output_root,
        rows_by_point=grouped,
        sweep_axis="theta",
        n_occ_per_k=args.n_occ_per_k,
    )
    overlap_rows = []
    for key, rows in sorted(grouped.items()):
        payloads = {id(row): _load_projector_payload(row) for row in rows}
        for i, left in enumerate(rows):
            for right in rows[i + 1 :]:
                left_projector, left_frames = payloads[id(left)]
                right_projector, right_frames = payloads[id(right)]
                overlap = projector_overlap_diagnostics_with_frames(
                    left_projector,
                    right_projector,
                    left_frames,
                    right_frames,
                    n_occ_per_k=args.n_occ_per_k,
                )
                overlap_rows.append(
                    {
                        "theta_index": key[0],
                        "u_index": key[1],
                        "branch_left": _branch_id(left),
                        "branch_right": _branch_id(right),
                        **overlap.model_dump(mode="json"),
                    }
                )
    _write_csv(output_root / "hysteresis_projector_overlaps.csv", overlap_rows)
    phase_tables = {
        "energy_crossing": [
            {
                "u_index": row["u_index"],
                "theta_index": row["theta_index"],
                "u_D_meV": row["u_D_meV"],
                "theta_deg": row["theta_deg"],
                "lowest_energy_raw_branch": row["lowest_energy_raw_branch"],
                "lowest_energy_clean_branch": row["lowest_energy_clean_branch"],
                "row_reliability": row["row_reliability"],
            }
            for row in comparison_rows
        ],
        "gap_jump": [
            {
                "u_index": row["u_index"],
                "theta_index": row["theta_index"],
                "u_D_meV": row["u_D_meV"],
                "theta_deg": row["theta_deg"],
                "high_gap_branch": row["high_gap_branch"],
                "low_gap_branch": row["low_gap_branch"],
            }
            for row in comparison_rows
        ],
        "overlap_discontinuity": overlap_rows,
        "selected_branch_cg": [
            {
                "u_index": row["u_index"],
                "theta_index": row["theta_index"],
                "u_D_meV": row["u_D_meV"],
                "theta_deg": row["theta_deg"],
                "cG_lowest_energy_raw": row["lowest_energy_raw_cG"],
                "cG_lowest_energy_clean": row["lowest_energy_clean_cG"],
                "lowest_energy_raw_branch": row["lowest_energy_raw_branch"],
                "lowest_energy_clean_branch": row["lowest_energy_clean_branch"],
                "row_reliability": row["row_reliability"],
            }
            for row in comparison_rows
        ],
    }
    table_paths = {}
    for name, rows in phase_tables.items():
        path = output_root / f"hysteresis_{name}.csv"
        _write_csv(path, rows)
        table_paths[name] = str(path)
    _write_json(
        output_root / "hysteresis_summary.json",
        {
            "n_branch_rows": len(records),
            "n_comparison_rows": len(comparison_rows),
            "tables": {
                "hysteresis_sweep_csv": str(output_root / "hysteresis_sweep.csv"),
                "hysteresis_comparison_csv": str(output_root / "hysteresis_comparison.csv"),
                "hysteresis_all_branch_candidates_csv": str(
                    output_root / "hysteresis_all_branch_candidates.csv"
                ),
                "hysteresis_displacement_comparison_csv": str(
                    output_root / "hysteresis_displacement_comparison.csv"
                ),
                "hysteresis_twist_comparison_csv": str(
                    output_root / "hysteresis_twist_comparison.csv"
                ),
                "hysteresis_projector_overlaps_csv": str(output_root / "hysteresis_projector_overlaps.csv"),
                **table_paths,
            },
            "n_displacement_comparison_rows": len(displacement_rows),
            "n_twist_comparison_rows": len(twist_rows),
        },
    )
    return records, comparison_rows


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    records, comparisons = merge_outputs(args)
    print(
        f"Merged {len(records)} branch rows and {len(comparisons)} comparisons "
        f"under {_output_root(args)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
