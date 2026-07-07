#!/usr/bin/env python3
"""Audit sharded recompute outputs and report missing phase-point tasks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-output-root",
        required=True,
        help="Per-mesh source hysteresis output root containing hysteresis_all_branch_candidates.csv.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Per-mesh recomputed output root to audit.",
    )
    parser.add_argument(
        "--n-point-tasks",
        type=int,
        default=1681,
        help="Number of phase-point shards used for the recompute array.",
    )
    parser.add_argument(
        "--json-output",
        default=None,
        help="Optional path for a JSON audit report.",
    )
    parser.add_argument(
        "--array-output",
        default=None,
        help="Optional path that receives only the compressed missing Slurm array string.",
    )
    parser.add_argument(
        "--require-source-projectors",
        action="store_true",
        help="Treat missing source projector files as failures in the summary.",
    )
    return parser


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _path_from_row(row: dict[str, Any], key: str, *, source_root: Path) -> Path:
    value = row.get(key)
    if value in {None, ""}:
        raise ValueError(f"source row is missing {key}")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = source_root / path
    return path


def _path_key(path: str | Path, *, source_root: Path) -> str:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = source_root / p
    return str(p)


def _source_row_lookup(
    source_rows: list[dict[str, str]],
    *,
    source_root: Path,
) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in source_rows:
        projector = row.get("projector_path")
        if projector:
            out[f"projector:{_path_key(projector, source_root=source_root)}"] = row
        run_id = row.get("run_id")
        if run_id:
            branch = row.get("branch_id") or f"{row.get('sweep_axis', '')}_{row.get('direction', '')}"
            out[f"run:{branch}:{run_id}"] = row
    return out


def _enriched_source_row(
    candidate: dict[str, str],
    lookup: dict[str, dict[str, str]],
    *,
    source_root: Path,
) -> dict[str, str]:
    full: dict[str, str] = {}
    projector = candidate.get("projector_path")
    if projector:
        full.update(lookup.get(f"projector:{_path_key(projector, source_root=source_root)}", {}))
    run_id = candidate.get("run_id")
    branch = candidate.get("branch_id") or f"{candidate.get('sweep_axis', '')}_{candidate.get('direction', '')}"
    if run_id:
        full.update(lookup.get(f"run:{branch}:{run_id}", {}))
    full.update(candidate)
    return full


def _int_from_row(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if value in {None, ""}:
        raise ValueError(f"source row is missing {key}")
    return int(float(str(value)))


def _safe_fragment(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(text))


def _point_label(row: dict[str, Any]) -> str:
    if row.get("point_label"):
        return str(row["point_label"])
    return f"u_{_int_from_row(row, 'u_index'):03d}_theta_{_int_from_row(row, 'theta_index'):03d}"


def _branch_id(row: dict[str, Any]) -> str:
    return str(row.get("branch_id") or f"{row.get('sweep_axis', 'unknown')}_{row.get('direction', 'unknown')}")


def _record_path(output_root: Path, row: dict[str, Any]) -> Path:
    return (
        output_root
        / "branches"
        / "recomputed"
        / _safe_fragment(_branch_id(row))
        / "points"
        / _safe_fragment(_point_label(row))
        / "point_record.json"
    )


def _phase_key(row: dict[str, Any]) -> tuple[int, int]:
    return (_int_from_row(row, "theta_index"), _int_from_row(row, "u_index"))


def _compressed_ranges(values: list[int]) -> str:
    if not values:
        return ""
    parts: list[str] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        parts.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = value
    parts.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(parts)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_output_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    candidates = _read_csv(source_root / "hysteresis_all_branch_candidates.csv")
    sweep_path = source_root / "hysteresis_sweep.csv"
    lookup = _source_row_lookup(_read_csv(sweep_path) if sweep_path.exists() else [], source_root=source_root)
    rows = [_enriched_source_row(row, lookup, source_root=source_root) for row in candidates]

    grouped: "OrderedDict[tuple[int, int], list[dict[str, str]]]" = OrderedDict()
    for row in rows:
        grouped.setdefault(_phase_key(row), []).append(row)

    n_point_tasks = int(args.n_point_tasks)
    missing_task_ids: list[int] = []
    complete_task_ids: list[int] = []
    missing_record_count = 0
    missing_source_projectors = 0
    incomplete_groups: list[dict[str, Any]] = []

    for phase_index, (key, group_rows) in enumerate(grouped.items()):
        task_id = phase_index % n_point_tasks
        expected_records = [_record_path(output_root, row) for row in group_rows]
        missing_records = [path for path in expected_records if not path.exists()]
        source_projectors = [
            _path_from_row(row, "projector_path", source_root=source_root)
            for row in group_rows
            if row.get("projector_path")
        ]
        missing_sources = [path for path in source_projectors if not path.exists()]
        if missing_records or (args.require_source_projectors and missing_sources):
            missing_task_ids.append(task_id)
            missing_record_count += len(missing_records)
            missing_source_projectors += len(missing_sources)
            incomplete_groups.append(
                {
                    "task_id": task_id,
                    "phase_index": phase_index,
                    "theta_index": key[0],
                    "u_index": key[1],
                    "n_expected_records": len(expected_records),
                    "n_missing_records": len(missing_records),
                    "n_missing_source_projectors": len(missing_sources),
                    "missing_records": [str(path) for path in missing_records],
                    "missing_source_projectors": [str(path) for path in missing_sources],
                }
            )
        else:
            complete_task_ids.append(task_id)

    missing_task_ids = sorted(set(missing_task_ids))
    complete_task_ids = sorted(set(complete_task_ids))
    summary = {
        "source_output_root": str(source_root),
        "output_root": str(output_root),
        "n_source_rows": len(rows),
        "n_phase_points": len(grouped),
        "n_point_tasks": n_point_tasks,
        "n_complete_phase_tasks": len(complete_task_ids),
        "n_missing_phase_tasks": len(missing_task_ids),
        "n_missing_records": missing_record_count,
        "n_missing_source_projectors": missing_source_projectors,
        "missing_task_ids": missing_task_ids,
        "missing_slurm_array": _compressed_ranges(missing_task_ids),
        "incomplete_groups": incomplete_groups,
    }
    if args.json_output is not None:
        path = Path(args.json_output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.array_output is not None:
        path = Path(args.array_output).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary["missing_slurm_array"] + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    summary = audit(_build_parser().parse_args(argv))
    print(
        "phase_tasks complete={complete}/{total} missing={missing} missing_records={records}".format(
            complete=summary["n_complete_phase_tasks"],
            total=summary["n_phase_points"],
            missing=summary["n_missing_phase_tasks"],
            records=summary["n_missing_records"],
        )
    )
    print(f"missing_slurm_array={summary['missing_slurm_array'] or '<none>'}")
    if summary["n_missing_source_projectors"]:
        print(f"missing_source_projectors={summary['n_missing_source_projectors']}")
    return 1 if summary["n_missing_phase_tasks"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
