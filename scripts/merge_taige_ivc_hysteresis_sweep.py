#!/usr/bin/env python3
"""Merge Taige IVC hysteresis branch outputs into comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
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
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--n-occ-per-k", type=int, default=1)
    parser.add_argument("--allow-missing-directions", action="store_true")
    return parser


def _output_root(args: argparse.Namespace) -> Path:
    root = Path(args.output_root)
    if not root.is_absolute():
        root = ROOT / root
    return root


def _cache_root(args: argparse.Namespace) -> Path:
    root = Path(args.cache_root) if args.cache_root is not None else _output_root(args) / "backend_cache"
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


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


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


def _trial_interpolation(row: dict[str, Any]) -> str:
    value = row.get("trial_interpolation")
    if value in {None, ""}:
        return "convex_full_hf"
    return str(value)


def _require_single_trial_interpolation(
    rows: list[dict[str, Any]],
    *,
    context: str,
) -> str:
    modes = sorted({_trial_interpolation(row) for row in rows if row})
    if not modes:
        return "convex_full_hf"
    if len(modes) > 1:
        raise ValueError(
            f"{context} mixes trial_interpolation modes {modes}; "
            "write linear_interaction and convex_full_hf outputs to separate roots"
        )
    mode = modes[0]
    for row in rows:
        row.setdefault("trial_interpolation", mode)
    return mode


def _chern_columns(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in row.items() if str(key).startswith("chern_")}


def _load_cache_vp_chern_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(_cache_root(args).rglob("*.summary.json")):
        summary = json.loads(path.read_text())
        point_fields = {
            "u_index": summary.get("u_index"),
            "theta_index": summary.get("theta_index"),
            "u_D_meV": summary.get("u_D"),
            "theta_deg": summary.get("theta_deg"),
            "cache_hash": summary.get("cache_hash"),
            "cache_path": summary.get("cache_path"),
        }
        for chern in summary.get("vp_hf_chern_rows", []) or []:
            rows.append({**point_fields, **chern})
    return rows


def _vp_chern_rows_from_flattened(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str, int]] = set()
    pattern = re.compile(r"^chern_hf_(vpplus|vpminus)_band_(\d+)$")
    for row in rows:
        for key, value in _chern_columns(row).items():
            match = pattern.match(str(key))
            if match is None:
                continue
            reference = "VP+" if match.group(1) == "vpplus" else "VP-"
            band = int(match.group(2))
            ident = (int(row["theta_index"]), int(row["u_index"]), reference, band)
            if ident in seen:
                continue
            seen.add(ident)
            out.append(
                {
                    "theta_index": row.get("theta_index"),
                    "u_index": row.get("u_index"),
                    "theta_deg": row.get("theta_deg"),
                    "u_D_meV": row.get("u_D_meV"),
                    "reference": reference,
                    "band": band,
                    "chern": value,
                }
            )
    return out


def _gap_family_by_branch(rows: list[dict[str, Any]]) -> dict[str, str]:
    finite = [
        (_branch_id(row), _safe_float(row, "direct_gap_min"))
        for row in rows
        if np.isfinite(_safe_float(row, "direct_gap_min"))
    ]
    if not finite:
        return {_branch_id(row): "unknown_gap" for row in rows}
    finite.sort(key=lambda item: item[1])
    clusters: list[list[tuple[str, float]]] = []
    for item in finite:
        if not clusters:
            clusters.append([item])
            continue
        prev_gap = clusters[-1][-1][1]
        tol = max(1e-8, 0.05 * max(abs(prev_gap), abs(item[1]), 1e-12))
        if abs(item[1] - prev_gap) <= tol:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    labels: dict[str, str] = {}
    if len(clusters) == 1:
        for branch, _gap in clusters[0]:
            labels[branch] = "single_gap"
    elif len(clusters) == 2:
        for branch, _gap in clusters[0]:
            labels[branch] = "small_gap"
        for branch, _gap in clusters[1]:
            labels[branch] = "large_gap"
    else:
        for idx, cluster in enumerate(clusters):
            label = "small_gap" if idx == 0 else "large_gap" if idx == len(clusters) - 1 else "intermediate_gap"
            for branch, _gap in cluster:
                labels[branch] = label
    for row in rows:
        labels.setdefault(_branch_id(row), "unknown_gap")
    return labels


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
        row["trial_interpolation"] = _trial_interpolation(by_direction["up"])
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
        row.update(_chern_columns(by_direction["up"]))
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
    gap_families = _gap_family_by_branch(ordered)
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
        "trial_interpolation": _trial_interpolation(ordered[0]),
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
        "small_gap_branch": next(
            (_branch_id(row) for row in ordered if gap_families.get(_branch_id(row)) == "small_gap"),
            _branch_id(low_gap),
        ),
        "intermediate_gap_branch": next(
            (
                _branch_id(row)
                for row in ordered
                if gap_families.get(_branch_id(row)) == "intermediate_gap"
            ),
            None,
        ),
        "large_gap_branch": next(
            (_branch_id(row) for row in reversed(ordered) if gap_families.get(_branch_id(row)) == "large_gap"),
            _branch_id(high_gap),
        ),
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
        summary[f"gap_family_{label}"] = gap_families.get(label)
    summary.update(_chern_columns(ordered[0]))
    return summary


def merge_outputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_root = _output_root(args)
    records = [_load_record(path) for path in _record_paths(output_root)]
    trial_interpolation = _require_single_trial_interpolation(
        records,
        context=f"branch records under {output_root}",
    )
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
        gap_families = _gap_family_by_branch(rows)
        for row in rows:
            branch_id = _branch_id(row)
            candidate_rows.append(
                {
                    "theta_index": key[0],
                    "u_index": key[1],
                    "theta_deg": row.get("theta_deg"),
                    "u_D_meV": row.get("u_D_meV"),
                    "branch_id": branch_id,
                    "trial_interpolation": _trial_interpolation(row),
                    "sweep_axis": row.get("sweep_axis", "u_D"),
                    "direction": row.get("direction"),
                    "gap_family_label": gap_families.get(branch_id),
                    "energy_total_per_cell": row.get("energy_total_per_cell"),
                    "energy_one_body_per_cell": row.get("energy_one_body_per_cell"),
                    "energy_hartree_per_cell": row.get("energy_hartree_per_cell"),
                    "energy_fock_per_cell": row.get("energy_fock_per_cell"),
                    "direct_gap_min": row.get("direct_gap_min"),
                    "indirect_gap": row.get("indirect_gap"),
                    "ivc_amplitude_block": row.get("ivc_amplitude_block"),
                    "clean_branch": is_clean_hysteresis_record(row),
                    "branch_reliability": row.get("branch_reliability"),
                    "cG": row.get("cG"),
                    "cG_diagnostic": row.get("cG_diagnostic"),
                    "cG_warning_flag": row.get("cG_warning_flag"),
                    "aufbau_residual_norm": row.get("aufbau_residual_norm"),
                    "commutator_norm": row.get("commutator_norm"),
                    "delta_P": row.get("delta_P"),
                    "delta_energy": row.get("delta_energy"),
                    "warning_flag": row.get("warning_flag"),
                    "vp_reference_name": row.get("vp_reference_name"),
                    "vp_reference_energy_per_cell": row.get("vp_reference_energy_per_cell"),
                    "vp_plus_energy_per_cell": row.get("vp_plus_energy_per_cell"),
                    "vp_minus_energy_per_cell": row.get("vp_minus_energy_per_cell"),
                    "vp_plus_direct_gap": row.get("vp_plus_direct_gap"),
                    "vp_minus_direct_gap": row.get("vp_minus_direct_gap"),
                    "vp_plus_clean": row.get("vp_plus_clean"),
                    "vp_minus_clean": row.get("vp_minus_clean"),
                    "run_id": row.get("run_id"),
                    "projector_path": row.get("projector_path"),
                    **_chern_columns(row),
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
    trial_rows: list[dict[str, Any]] = []
    for row in records:
        trial_path = Path(row.get("trial_theta_csv") or Path(row["point_record_path"]).parent / "trial_theta.csv")
        for trial in _read_csv(trial_path):
            trial_rows.append(trial)
    _require_single_trial_interpolation(
        [*records, *trial_rows],
        context=f"trial-theta rows under {output_root}",
    )
    _write_csv(output_root / "hysteresis_trial_theta.csv", trial_rows)

    selected_by_point = {
        (int(row["theta_index"]), int(row["u_index"])): (
            row["lowest_energy_clean_branch"]
            if row.get("lowest_energy_clean_branch") not in {None, ""}
            else row["lowest_energy_raw_branch"]
        )
        for row in comparison_rows
    }
    selected_reliability = {
        (int(row["theta_index"]), int(row["u_index"])): row["row_reliability"]
        for row in comparison_rows
    }
    selected_trial_rows = []
    for row in trial_rows:
        key = (int(row["theta_index"]), int(row["u_index"]))
        if row.get("branch_id") == selected_by_point.get(key):
            selected_trial_rows.append(
                {
                    **row,
                    "selected_branch_label": selected_by_point.get(key),
                    "row_reliability": selected_reliability.get(key),
                }
            )
    _write_csv(output_root / "hysteresis_selected_trial_theta.csv", selected_trial_rows)

    vp_chern_rows = _load_cache_vp_chern_rows(args)
    if not vp_chern_rows:
        vp_chern_rows = _vp_chern_rows_from_flattened(comparison_rows)
    _write_csv(output_root / "hysteresis_vp_chern_numbers.csv", vp_chern_rows)

    phase_tables = {
        "energy_crossing": [
            {
                "u_index": row["u_index"],
                "theta_index": row["theta_index"],
                "u_D_meV": row["u_D_meV"],
                "theta_deg": row["theta_deg"],
                "lowest_energy_raw_branch": row["lowest_energy_raw_branch"],
                "lowest_energy_clean_branch": row["lowest_energy_clean_branch"],
                "trial_interpolation": row.get("trial_interpolation", trial_interpolation),
                "row_reliability": row["row_reliability"],
                **_chern_columns(row),
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
                "small_gap_branch": row.get("small_gap_branch"),
                "intermediate_gap_branch": row.get("intermediate_gap_branch"),
                "large_gap_branch": row.get("large_gap_branch"),
                "trial_interpolation": row.get("trial_interpolation", trial_interpolation),
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
                "trial_interpolation": row.get("trial_interpolation", trial_interpolation),
                "row_reliability": row["row_reliability"],
                **_chern_columns(row),
            }
            for row in comparison_rows
        ],
        "gap_families": [
            {
                "u_index": row["u_index"],
                "theta_index": row["theta_index"],
                "u_D_meV": row["u_D_meV"],
                "theta_deg": row["theta_deg"],
                "small_gap_branch": row.get("small_gap_branch"),
                "intermediate_gap_branch": row.get("intermediate_gap_branch"),
                "large_gap_branch": row.get("large_gap_branch"),
                "low_gap_branch": row["low_gap_branch"],
                "high_gap_branch": row["high_gap_branch"],
                "trial_interpolation": row.get("trial_interpolation", trial_interpolation),
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
                "hysteresis_trial_theta_csv": str(output_root / "hysteresis_trial_theta.csv"),
                "hysteresis_selected_trial_theta_csv": str(
                    output_root / "hysteresis_selected_trial_theta.csv"
                ),
                "hysteresis_vp_chern_numbers_csv": str(
                    output_root / "hysteresis_vp_chern_numbers.csv"
                ),
                **table_paths,
            },
            "n_trial_theta_rows": len(trial_rows),
            "n_selected_trial_theta_rows": len(selected_trial_rows),
            "n_vp_chern_rows": len(vp_chern_rows),
            "n_displacement_comparison_rows": len(displacement_rows),
            "n_twist_comparison_rows": len(twist_rows),
            "trial_interpolation": trial_interpolation,
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
