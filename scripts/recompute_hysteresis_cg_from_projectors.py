#!/usr/bin/env python3
"""Recompute hysteresis cG from stored branch projectors and HF Hamiltonians."""

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
    ContinuumHFDiagnostics,
    ContinuumHFResult,
    TaigeHysteresisPoint,
    ValleyU1Constraint,
    solve_reference_hf,
)
from chiral_dw.continuum.models import projector_idempotency_errors  # noqa: E402
from scan_taige_ivc_hysteresis_linecut import (  # noqa: E402
    _build_parser as _linecut_parser,
    _cache_root,
    _hf_params,
    _load_or_build_cache,
    _loaded_vp_chern_columns,
    _output_root,
    _response_fields,
    _vp_inputs_clean,
    _vp_reference_fields,
    _write_csv,
    _write_json,
)
from merge_taige_ivc_hysteresis_sweep import merge_outputs  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = _linecut_parser()
    parser.description = __doc__
    parser.add_argument(
        "--source-output-root",
        required=True,
        help="Existing per-mesh hysteresis output root containing projector_final.npz files.",
    )
    parser.add_argument(
        "--source-cache-root",
        default=None,
        help="Optional existing backend cache root; defaults to source-output-root/backend_cache.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def _resolve_root(path: str | Path) -> Path:
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    return root.resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def _bool_value(value: Any, *, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}
    return bool(value)


def _float_value(row: dict[str, Any], key: str, default: float = float("nan")) -> float:
    value = row.get(key)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key)
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _trial_interpolation(row: dict[str, Any]) -> str:
    value = row.get("trial_interpolation")
    return "convex_full_hf" if value in {None, ""} else str(value)


def _point_from_row(row: dict[str, Any]) -> TaigeHysteresisPoint:
    return TaigeHysteresisPoint(
        u_index=int(row["u_index"]),
        theta_index=int(row["theta_index"]),
        u_D=_float_value(row, "u_D_meV", _float_value(row, "u_D")),
        theta_deg=_float_value(row, "theta_deg"),
    )


def _safe_fragment(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(text))


def _point_output_dir(output_root: Path, row: dict[str, Any], point: TaigeHysteresisPoint) -> Path:
    branch_id = row.get("branch_id") or f"{row.get('sweep_axis', 'unknown')}_{row.get('direction', 'unknown')}"
    label = row.get("point_label") or point.label
    return output_root / "branches" / "recomputed" / _safe_fragment(str(branch_id)) / "points" / _safe_fragment(str(label))


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
) -> dict[str, Any]:
    full: dict[str, Any] = {}
    projector = candidate.get("projector_path")
    if projector:
        full.update(lookup.get(f"projector:{_path_key(projector, source_root=source_root)}", {}))
    run_id = candidate.get("run_id")
    branch = candidate.get("branch_id") or f"{candidate.get('sweep_axis', '')}_{candidate.get('direction', '')}"
    if run_id:
        full.update(lookup.get(f"run:{branch}:{run_id}", {}))
    full.update(candidate)
    return full


def _load_projector_payload(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing source projector payload {path}")
    with np.load(path, allow_pickle=False) as data:
        if "final_projector" not in data or "final_h_hf" not in data:
            raise KeyError(f"{path} must contain final_projector and final_h_hf")
        return (
            np.asarray(data["final_projector"], dtype=complex),
            np.asarray(data["final_h_hf"], dtype=complex),
        )


def _augment_trial_theta_csv(path: Path, fields: dict[str, Any]) -> None:
    rows = _read_csv(path)
    if not rows:
        return
    _write_csv(path, [{**row, **fields} for row in rows])


def _ivc_result_from_stored_payload(
    *,
    row: dict[str, Any],
    P: np.ndarray,
    H_hf: np.ndarray,
    energy_total: float,
    direct_gap_min: float,
    indirect_gap: float,
) -> ContinuumHFResult:
    idem_fro, idem_max = projector_idempotency_errors(P)
    diagnostics = ContinuumHFDiagnostics(
        energy=float(energy_total),
        delta_energy=_float_value(row, "delta_energy", 0.0),
        delta_P=_float_value(row, "delta_P", 0.0),
        idempotency_error_fro=_float_value(
            row,
            "idempotency_error_fro",
            _float_value(row, "final_idempotency_error_fro", float(idem_fro)),
        ),
        idempotency_error_max=_float_value(row, "idempotency_error_max", float(idem_max)),
        constraint_error=_float_value(row, "constraint_error", 0.0),
        aufbau_residual_norm=_float_value(row, "aufbau_residual_norm"),
        commutator_norm=_float_value(row, "commutator_norm"),
        trace_error=_float_value(row, "trace_error", _float_value(row, "final_trace_error", 0.0)),
        direct_gap_min=float(direct_gap_min),
        indirect_gap=float(indirect_gap),
        iteration=_int_value(row, "iteration_count", _int_value(row, "n_iter", 0)),
        constraint_name="TPrimeConstraint",
        density_kind="final_idempotent",
        self_consistency_warning=_bool_value(row.get("self_consistency_warning")),
    )
    return ContinuumHFResult(
        P=P,
        H_hf=H_hf,
        energy=float(energy_total),
        converged=_bool_value(row.get("converged"), default=_bool_value(row.get("clean_branch"))),
        n_iter=int(diagnostics.iteration),
        diagnostics=diagnostics,
        seed=str(row.get("seed_label") or row.get("run_id") or "stored_projector"),
        constraint_name="TPrimeConstraint",
    )


def _load_vp_references(args: argparse.Namespace, loaded: Any) -> tuple[Any, Any]:
    if loaded.vp_plus is not None and loaded.vp_minus is not None:
        return loaded.vp_plus, loaded.vp_minus
    controls = _hf_params(args)
    constraint = ValleyU1Constraint(loaded.bundle.active)
    return (
        solve_reference_hf(loaded.bundle, "vp_plus", controls, constraint=constraint),
        solve_reference_hf(loaded.bundle, "vp_minus", controls, constraint=constraint),
    )


def _recompute_one(
    *,
    args: argparse.Namespace,
    source_root: Path,
    output_root: Path,
    row: dict[str, Any],
) -> dict[str, Any]:
    point = _point_from_row(row)
    point_dir = _point_output_dir(output_root, row, point)
    record_path = point_dir / "point_record.json"
    if record_path.exists() and not args.rerun_existing:
        return json.loads(record_path.read_text())

    projector_path = _path_from_row(row, "projector_path", source_root=source_root)
    P_ivc, H_ivc = _load_projector_payload(projector_path)
    loaded, cache_path = _load_or_build_cache(args, point)
    bundle = loaded.bundle
    vp_plus, vp_minus = _load_vp_references(args, loaded)
    energy = bundle.backend.energy(P_ivc)
    _P_hf, _evals, direct_gap, indirect_gap = bundle.backend.update_density_per_k(
        H_ivc,
        int(args.n_occ_per_k),
    )
    ivc = _ivc_result_from_stored_payload(
        row=row,
        P=P_ivc,
        H_hf=H_ivc,
        energy_total=float(energy.total),
        direct_gap_min=_float_value(row, "direct_gap_min", float(direct_gap)),
        indirect_gap=_float_value(row, "indirect_gap", float(indirect_gap)),
    )
    clean_branch = _bool_value(row.get("clean_branch"), default=not _bool_value(row.get("warning_flag")))
    response = _response_fields(
        args=args,
        point=point,
        point_dir=point_dir,
        bundle=bundle,
        vp_plus=vp_plus,
        vp_minus=vp_minus,
        ivc=ivc,
        direction=str(row.get("direction") or "up"),
        branch_id=str(row.get("branch_id") or f"{row.get('sweep_axis', 'unknown')}_{row.get('direction', 'unknown')}"),
        clean_branch=clean_branch,
        vp_inputs_clean=_vp_inputs_clean(args, vp_plus, vp_minus),
    )
    source_trial = _trial_interpolation(row)
    provenance = {
        "source_output_root": str(source_root),
        "source_projector_path": str(projector_path),
        "source_trial_interpolation": source_trial,
        "recomputed_from_stored_projector": True,
    }
    _augment_trial_theta_csv(Path(response["trial_theta_csv"]), provenance)
    vp_fields = {
        **_vp_reference_fields(
            "vp_plus",
            vp_plus,
            n_blocks=bundle.backend.n_blocks,
            max_iter=args.max_iter,
        ),
        **_vp_reference_fields(
            "vp_minus",
            vp_minus,
            n_blocks=bundle.backend.n_blocks,
            max_iter=args.max_iter,
        ),
        **_loaded_vp_chern_columns(loaded, vp_plus, vp_minus),
    }
    record = {
        **row,
        "u_index": int(point.u_index),
        "theta_index": int(point.theta_index),
        "u_D_meV": float(point.u_D),
        "theta_deg": float(point.theta_deg),
        "point_label": row.get("point_label") or point.label,
        "branch_id": row.get("branch_id") or f"{row.get('sweep_axis', 'unknown')}_{row.get('direction', 'unknown')}",
        "projector_path": str(projector_path),
        **provenance,
        "source_cache_path": row.get("cache_path"),
        "cache_path": str(cache_path),
        "energy_total_per_cell": float(energy.total / bundle.backend.n_blocks),
        "energy_one_body_per_cell": float(energy.one_body / bundle.backend.n_blocks),
        "energy_hartree_per_cell": float(energy.hartree / bundle.backend.n_blocks),
        "energy_fock_per_cell": float(energy.fock / bundle.backend.n_blocks),
        "direct_gap_min": float(ivc.diagnostics.direct_gap_min),
        "indirect_gap": float(ivc.diagnostics.indirect_gap),
        "clean_branch": clean_branch,
        **vp_fields,
        **response,
    }
    _write_json(record_path, record)
    return record


def _prepare_args(args: argparse.Namespace) -> tuple[Path, Path]:
    source_root = _resolve_root(args.source_output_root)
    if args.output_root is None:
        args.output_root = str(source_root.with_name(f"{source_root.name}_linear_interaction_recomputed"))
    output_root = _output_root(args)
    if args.source_cache_root is not None and args.cache_root is None:
        args.cache_root = str(_resolve_root(args.source_cache_root))
    elif args.cache_root is None:
        source_cache = source_root / "backend_cache"
        args.cache_root = str(source_cache if source_cache.exists() else output_root / "backend_cache")
    return source_root, output_root


def _plan_rows(
    *,
    args: argparse.Namespace,
    source_root: Path,
    output_root: Path,
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        point = _point_from_row(row)
        projector_path = _path_from_row(row, "projector_path", source_root=source_root)
        point_dir = _point_output_dir(output_root, row, point)
        out.append(
            {
                "theta_index": int(point.theta_index),
                "u_index": int(point.u_index),
                "theta_deg": float(point.theta_deg),
                "u_D_meV": float(point.u_D),
                "branch_id": row.get("branch_id"),
                "sweep_axis": row.get("sweep_axis"),
                "direction": row.get("direction"),
                "source_projector_path": str(projector_path),
                "source_projector_exists": projector_path.exists(),
                "output_point_record": str(point_dir / "point_record.json"),
                "would_skip_existing": (point_dir / "point_record.json").exists() and not args.rerun_existing,
                "source_trial_interpolation": _trial_interpolation(row),
                "trial_interpolation": args.trial_interpolation,
            }
        )
    return out


def recompute(args: argparse.Namespace) -> dict[str, Any]:
    source_root, output_root = _prepare_args(args)
    source_candidates = _read_csv(source_root / "hysteresis_all_branch_candidates.csv")
    if not source_candidates:
        raise FileNotFoundError(
            f"{source_root / 'hysteresis_all_branch_candidates.csv'} is missing or empty"
        )
    source_sweep = _read_csv(source_root / "hysteresis_sweep.csv")
    lookup = _source_row_lookup(source_sweep, source_root=source_root)
    rows: list[dict[str, Any]] = [
        _enriched_source_row(row, lookup, source_root=source_root)
        for row in source_candidates
    ]
    if args.max_rows is not None:
        rows = rows[: int(args.max_rows)]
    plan_rows = _plan_rows(args=args, source_root=source_root, output_root=output_root, rows=rows)
    _write_csv(output_root / "hysteresis_recompute_plan.csv", plan_rows)
    _write_json(
        output_root / "hysteresis_recompute_plan.json",
        {
            "source_output_root": str(source_root),
            "output_root": str(output_root),
            "cache_root": str(_cache_root(args)),
            "n_rows": len(rows),
            "trial_interpolation": args.trial_interpolation,
            "material": args.material,
            "dry_run": bool(args.dry_run),
            "rows": plan_rows,
        },
    )
    if args.dry_run:
        return {
            "source_output_root": str(source_root),
            "output_root": str(output_root),
            "cache_root": str(_cache_root(args)),
            "n_rows": len(rows),
            "n_recomputed": 0,
            "n_skipped_existing": sum(int(row["would_skip_existing"]) for row in plan_rows),
            "dry_run": True,
        }

    records: list[dict[str, Any]] = []
    skipped = 0
    for row, plan in zip(rows, plan_rows):
        if plan["would_skip_existing"]:
            skipped += 1
            print(f"Skipping existing recomputed point {plan['output_point_record']}")
        records.append(
            _recompute_one(
                args=args,
                source_root=source_root,
                output_root=output_root,
                row=row,
            )
        )
    merge_args = argparse.Namespace(
        output_root=str(output_root),
        cache_root=str(_cache_root(args)),
        n_occ_per_k=int(args.n_occ_per_k),
        allow_missing_directions=True,
    )
    merge_outputs(merge_args)
    summary = {
        "source_output_root": str(source_root),
        "output_root": str(output_root),
        "cache_root": str(_cache_root(args)),
        "n_rows": len(rows),
        "n_recomputed_or_loaded": len(records),
        "n_skipped_existing": skipped,
        "trial_interpolation": args.trial_interpolation,
        "material": args.material,
        "recomputed_from_stored_projector": True,
    }
    _write_json(output_root / "hysteresis_recompute_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = recompute(args)
    if summary.get("dry_run"):
        print(
            f"Dry-run recompute plan for {summary['n_rows']} branch rows under "
            f"{summary['output_root']}"
        )
    else:
        print(
            f"Recomputed {summary['n_recomputed_or_loaded']} branch rows from stored projectors "
            f"under {summary['output_root']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
