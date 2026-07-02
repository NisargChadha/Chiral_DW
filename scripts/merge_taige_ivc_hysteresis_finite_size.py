#!/usr/bin/env python3
"""Merge per-mesh Taige IVC hysteresis outputs into finite-size diagnostics."""

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/taige_ivc_hysteresis_finite_size")
    parser.add_argument("--n-k-list", default="18,19,20")
    parser.add_argument("--mesh-dir-template", default="nk_{n_k:03d}")
    parser.add_argument("--fit-min-clean", type=int, default=3)
    parser.add_argument("--fit-degree", type=int, default=1)
    return parser


def _parse_int_list(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("--n-k-list must contain at least one integer")
    return values


def _output_root(args: argparse.Namespace) -> Path:
    root = Path(args.output_root)
    if not root.is_absolute():
        root = ROOT / root
    return root


def _mesh_root(output_root: Path, template: str, n_k: int) -> Path:
    return output_root / template.format(n_k=int(n_k))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


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
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return False


def _mesh_fields(n_k: int, mesh_root: Path) -> dict[str, Any]:
    return {
        "n_k": int(n_k),
        "inv_n_k": float(1.0 / n_k),
        "inv_n_k_squared": float(1.0 / (n_k * n_k)),
        "mesh_output_root": str(mesh_root),
    }


def _with_mesh_fields(rows: list[dict[str, Any]], *, n_k: int, mesh_root: Path) -> list[dict[str, Any]]:
    prefix = _mesh_fields(n_k, mesh_root)
    return [{**prefix, **row} for row in rows]


def _stack_table(
    *,
    output_root: Path,
    nks: list[int],
    template: str,
    filename: str,
    output_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_k in nks:
        mesh_root = _mesh_root(output_root, template, n_k)
        rows.extend(_with_mesh_fields(_read_csv(mesh_root / filename), n_k=n_k, mesh_root=mesh_root))
    _write_csv(output_root / output_name, rows)
    return rows


def _point_key(row: dict[str, Any]) -> tuple[int, int]:
    return (int(row["theta_index"]), int(row["u_index"]))


def _fit_source_from_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        common = {
            "n_k": int(row["n_k"]),
            "inv_n_k": float(row["inv_n_k"]),
            "theta_index": int(row["theta_index"]),
            "u_index": int(row["u_index"]),
            "theta_deg": _float_or_nan(row.get("theta_deg")),
            "u_D_meV": _float_or_nan(row.get("u_D_meV")),
        }
        clean_branch = row.get("lowest_energy_clean_branch")
        if clean_branch not in {None, ""}:
            out.append(
                {
                    **common,
                    "branch_label": "lowest_energy_clean",
                    "source_branch": clean_branch,
                    "cG": _float_or_nan(row.get("lowest_energy_clean_cG")),
                    "clean": row.get("row_reliability") == "clean",
                    "selection_kind": "clean",
                }
            )
        out.append(
            {
                **common,
                "branch_label": "lowest_energy_raw",
                "source_branch": row.get("lowest_energy_raw_branch"),
                "cG": _float_or_nan(row.get("lowest_energy_raw_cG")),
                "clean": _bool_or_false(row.get("lowest_energy_raw_clean")),
                "selection_kind": "raw",
            }
        )
    return out


def _best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [row for row in rows if _bool_or_false(row.get("clean_branch"))]
    pool = clean if clean else rows
    return min(pool, key=lambda row: _float_or_nan(row.get("energy_total_per_cell")))


def _fit_source_from_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    by_family: dict[tuple[int, int, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        common = {
            "n_k": int(row["n_k"]),
            "inv_n_k": float(row["inv_n_k"]),
            "theta_index": int(row["theta_index"]),
            "u_index": int(row["u_index"]),
            "theta_deg": _float_or_nan(row.get("theta_deg")),
            "u_D_meV": _float_or_nan(row.get("u_D_meV")),
            "cG": _float_or_nan(row.get("cG")),
            "clean": _bool_or_false(row.get("clean_branch")),
        }
        out.append(
            {
                **common,
                "branch_label": f"branch_{row.get('branch_id')}",
                "source_branch": row.get("branch_id"),
                "selection_kind": "branch_id",
            }
        )
        family = str(row.get("gap_family_label") or "")
        if family in {"small_gap", "intermediate_gap", "large_gap"}:
            by_family.setdefault(
                (int(row["n_k"]), int(row["theta_index"]), int(row["u_index"]), family),
                [],
            ).append(row)
    for (_n_k, _theta_index, _u_index, family), group in sorted(by_family.items()):
        row = _best_candidate(group)
        out.append(
            {
                "n_k": int(row["n_k"]),
                "inv_n_k": float(row["inv_n_k"]),
                "theta_index": int(row["theta_index"]),
                "u_index": int(row["u_index"]),
                "theta_deg": _float_or_nan(row.get("theta_deg")),
                "u_D_meV": _float_or_nan(row.get("u_D_meV")),
                "branch_label": family,
                "source_branch": row.get("branch_id"),
                "cG": _float_or_nan(row.get("cG")),
                "clean": _bool_or_false(row.get("clean_branch")),
                "selection_kind": "gap_family",
            }
        )
    return out


def _fit_one(group: list[dict[str, Any]], *, fit_min_clean: int, fit_degree: int) -> dict[str, Any]:
    group = sorted(group, key=lambda row: int(row["n_k"]))
    clean = [
        row
        for row in group
        if _bool_or_false(row.get("clean")) and np.isfinite(_float_or_nan(row.get("cG")))
    ]
    raw_finite = [row for row in group if np.isfinite(_float_or_nan(row.get("cG")))]
    largest_clean = max(clean, key=lambda row: int(row["n_k"])) if clean else None
    largest_raw = max(raw_finite, key=lambda row: int(row["n_k"])) if raw_finite else None
    status = "fit_ok" if len(clean) >= fit_min_clean else (
        "insufficient_clean_meshes" if clean else "no_clean_finite_values"
    )
    cG_extrap = float("nan")
    abs_extrap = float("nan")
    cG_slope = float("nan")
    abs_slope = float("nan")
    rmse = float("nan")
    abs_rmse = float("nan")
    coeffs: list[float] = []
    abs_coeffs: list[float] = []
    if status == "fit_ok":
        x = np.asarray([_float_or_nan(row["inv_n_k"]) for row in clean], dtype=float)
        y = np.asarray([_float_or_nan(row["cG"]) for row in clean], dtype=float)
        degree = min(int(fit_degree), max(1, x.size - 1))
        poly = np.polyfit(x, y, degree)
        fit = np.poly1d(poly)
        cG_extrap = float(fit(0.0))
        cG_slope = float(poly[-2]) if degree >= 1 else 0.0
        rmse = float(np.sqrt(np.mean((y - fit(x)) ** 2)))
        coeffs = [float(value) for value in poly]
        ay = np.abs(y)
        apoly = np.polyfit(x, ay, degree)
        afit = np.poly1d(apoly)
        abs_extrap = float(afit(0.0))
        abs_slope = float(apoly[-2]) if degree >= 1 else 0.0
        abs_rmse = float(np.sqrt(np.mean((ay - afit(x)) ** 2)))
        abs_coeffs = [float(value) for value in apoly]
    first = group[-1]
    return {
        "theta_index": int(first["theta_index"]),
        "u_index": int(first["u_index"]),
        "theta_deg": _float_or_nan(first.get("theta_deg")),
        "u_D_meV": _float_or_nan(first.get("u_D_meV")),
        "branch_label": first["branch_label"],
        "selection_kind": first.get("selection_kind"),
        "fit_status": status,
        "fit_min_clean": int(fit_min_clean),
        "fit_degree_requested": int(fit_degree),
        "n_mesh": len(group),
        "n_clean_finite": len(clean),
        "n_raw_finite": len(raw_finite),
        "n_k_values": ",".join(str(int(row["n_k"])) for row in group),
        "cG_extrapolated_inv_n0": cG_extrap,
        "abs_cG_extrapolated_inv_n0": abs_extrap,
        "cG_fit_slope_inv_n": cG_slope,
        "abs_cG_fit_slope_inv_n": abs_slope,
        "cG_fit_rmse": rmse,
        "abs_cG_fit_rmse": abs_rmse,
        "cG_fit_coefficients_descending_json": json.dumps(coeffs),
        "abs_cG_fit_coefficients_descending_json": json.dumps(abs_coeffs),
        "largest_clean_n_k": None if largest_clean is None else int(largest_clean["n_k"]),
        "cG_largest_clean_nk": None if largest_clean is None else _float_or_nan(largest_clean["cG"]),
        "abs_cG_largest_clean_nk": (
            None if largest_clean is None else abs(_float_or_nan(largest_clean["cG"]))
        ),
        "largest_raw_n_k": None if largest_raw is None else int(largest_raw["n_k"]),
        "cG_largest_raw_nk": None if largest_raw is None else _float_or_nan(largest_raw["cG"]),
        "abs_cG_largest_raw_nk": None if largest_raw is None else abs(_float_or_nan(largest_raw["cG"])),
    }


def _fit_rows(source_rows: list[dict[str, Any]], *, fit_min_clean: int, fit_degree: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in source_rows:
        grouped.setdefault(
            (int(row["theta_index"]), int(row["u_index"]), str(row["branch_label"])),
            [],
        ).append(row)
    return [
        _fit_one(group, fit_min_clean=fit_min_clean, fit_degree=fit_degree)
        for _key, group in sorted(grouped.items())
    ]


def merge_finite_size(args: argparse.Namespace) -> dict[str, Any]:
    output_root = _output_root(args)
    nks = _parse_int_list(args.n_k_list)
    comparison_rows = _stack_table(
        output_root=output_root,
        nks=nks,
        template=args.mesh_dir_template,
        filename="hysteresis_comparison.csv",
        output_name="hysteresis_finite_size_selected.csv",
    )
    candidate_rows = _stack_table(
        output_root=output_root,
        nks=nks,
        template=args.mesh_dir_template,
        filename="hysteresis_all_branch_candidates.csv",
        output_name="hysteresis_finite_size_branch_candidates.csv",
    )
    trial_rows = _stack_table(
        output_root=output_root,
        nks=nks,
        template=args.mesh_dir_template,
        filename="hysteresis_selected_trial_theta.csv",
        output_name="hysteresis_finite_size_selected_trial_theta.csv",
    )
    vp_chern_rows = _stack_table(
        output_root=output_root,
        nks=nks,
        template=args.mesh_dir_template,
        filename="hysteresis_vp_chern_numbers.csv",
        output_name="hysteresis_finite_size_vp_chern_numbers.csv",
    )
    fit_source = [
        *_fit_source_from_comparisons(comparison_rows),
        *_fit_source_from_candidates(candidate_rows),
    ]
    _write_csv(output_root / "hysteresis_finite_size_fit_source.csv", fit_source)
    fit_rows = _fit_rows(
        fit_source,
        fit_min_clean=args.fit_min_clean,
        fit_degree=args.fit_degree,
    )
    _write_csv(output_root / "hysteresis_finite_size_cg_fits.csv", fit_rows)
    boundary_rows = [
        row
        for row in fit_rows
        if row["branch_label"] in {"lowest_energy_clean", "small_gap", "intermediate_gap", "large_gap"}
    ]
    _write_csv(output_root / "hysteresis_finite_size_ivc_gap_family_boundary.csv", boundary_rows)
    vp_boundary = [
        row
        for row in vp_chern_rows
        if str(row.get("band")) == "0" and str(row.get("reference")) in {"VP+", "VP-"}
    ]
    _write_csv(output_root / "hysteresis_finite_size_vp_chern_boundary.csv", vp_boundary)
    summary = {
        "n_k_list": nks,
        "n_comparison_rows": len(comparison_rows),
        "n_candidate_rows": len(candidate_rows),
        "n_selected_trial_theta_rows": len(trial_rows),
        "n_vp_chern_rows": len(vp_chern_rows),
        "n_fit_rows": len(fit_rows),
        "tables": {
            "selected": str(output_root / "hysteresis_finite_size_selected.csv"),
            "branch_candidates": str(output_root / "hysteresis_finite_size_branch_candidates.csv"),
            "selected_trial_theta": str(output_root / "hysteresis_finite_size_selected_trial_theta.csv"),
            "vp_chern_numbers": str(output_root / "hysteresis_finite_size_vp_chern_numbers.csv"),
            "fit_source": str(output_root / "hysteresis_finite_size_fit_source.csv"),
            "cg_fits": str(output_root / "hysteresis_finite_size_cg_fits.csv"),
            "vp_chern_boundary": str(output_root / "hysteresis_finite_size_vp_chern_boundary.csv"),
            "ivc_gap_family_boundary": str(output_root / "hysteresis_finite_size_ivc_gap_family_boundary.csv"),
        },
    }
    _write_json(output_root / "hysteresis_finite_size_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = merge_finite_size(args)
    print(
        f"Merged hysteresis finite-size outputs for {summary['n_k_list']} "
        f"into {_output_root(args)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
