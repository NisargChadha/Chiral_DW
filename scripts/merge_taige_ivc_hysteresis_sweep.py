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
    phase_table_rows,
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
    return sorted((output_root / "branches").glob("*/[du][op]*/points/*/point_record.json"))


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


def merge_outputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_root = _output_root(args)
    records = [_load_record(path) for path in _record_paths(output_root)]
    records.sort(key=lambda row: (int(row["theta_index"]), int(row["u_index"]), str(row["direction"])))
    _write_csv(output_root / "hysteresis_sweep.csv", records)
    grouped: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
    for row in records:
        key = (int(row["theta_index"]), int(row["u_index"]))
        grouped.setdefault(key, {})[str(row["direction"])] = row
    comparison_records = []
    overlap_rows = []
    for key, item in sorted(grouped.items()):
        if "up" not in item or "down" not in item:
            if args.allow_missing_directions:
                continue
            raise ValueError(f"missing up/down pair for theta_index={key[0]} u_index={key[1]}")
        up_projector, up_frames = _load_projector_payload(item["up"])
        down_projector, down_frames = _load_projector_payload(item["down"])
        comparison, overlap = compare_hysteresis_records(
            up=item["up"],
            down=item["down"],
            up_projector=up_projector,
            down_projector=down_projector,
            up_frames=up_frames,
            down_frames=down_frames,
            n_occ_per_k=args.n_occ_per_k,
        )
        comparison_records.append(comparison)
        overlap_rows.append(
            {
                "theta_index": key[0],
                "u_index": key[1],
                **overlap.model_dump(mode="json"),
            }
        )
    comparison_rows = [row.model_dump(mode="json") for row in comparison_records]
    _write_csv(output_root / "hysteresis_comparison.csv", comparison_rows)
    _write_csv(output_root / "hysteresis_projector_overlaps.csv", overlap_rows)
    phase_tables = phase_table_rows(comparison_records)
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
                "hysteresis_projector_overlaps_csv": str(output_root / "hysteresis_projector_overlaps.csv"),
                **table_paths,
            },
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
