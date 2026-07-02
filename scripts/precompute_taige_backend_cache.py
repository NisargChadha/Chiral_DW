#!/usr/bin/env python3
"""Precompute loadable Taige HF backend caches for hysteresis sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chiral_dw.config import ContinuumGridParams, ContinuumHFParams  # noqa: E402
from chiral_dw.continuum import (  # noqa: E402
    TaigeHysteresisPoint,
    ValleyU1Constraint,
    build_continuum_bundle,
    load_taige_backend_cache,
    save_taige_backend_cache,
    solve_reference_hf,
    taige_backend_cache_hash,
    taige_backend_cache_path,
    taige_backend_cache_signature,
    taige_interaction_params,
    taige_model_params,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/taige_ivc_hysteresis")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--u-d", type=float, default=None)
    parser.add_argument("--theta-deg", type=float, default=None)
    parser.add_argument("--u-d-min", type=float, default=0.0)
    parser.add_argument("--u-d-max", type=float, default=20.0)
    parser.add_argument("--n-u-d", type=int, default=21)
    parser.add_argument("--theta-min-deg", type=float, default=2.0)
    parser.add_argument("--theta-max-deg", type=float, default=4.0)
    parser.add_argument("--n-twist", type=int, default=21)
    parser.add_argument("--task-id", type=int, default=None)

    parser.add_argument("--n-k", type=int, default=24)
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
    parser.add_argument("--max-iter", type=int, default=800)
    parser.add_argument("--min-iter", type=int, default=3)
    parser.add_argument("--mixing-method", choices=["linear", "oda"], default="oda")
    parser.add_argument("--mixing", type=float, default=0.45)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--energy-tolerance", type=float, default=1e-10)
    parser.add_argument("--seed-ordered-weight", type=float, default=1.0)
    parser.add_argument("--seed-random-weight", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--no-vp-references", action="store_true")
    parser.add_argument("--no-vp-chern", action="store_true")

    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    return parser


def _axis_values(single: float | None, lower: float, upper: float, count: int, name: str) -> np.ndarray:
    if single is not None:
        return np.asarray([float(single)], dtype=float)
    if int(count) < 1:
        raise ValueError(f"{name} count must be positive")
    return np.linspace(float(lower), float(upper), int(count))


def _all_points(args: argparse.Namespace) -> list[TaigeHysteresisPoint]:
    if (args.u_d is None) ^ (args.theta_deg is None):
        raise ValueError("--u-d and --theta-deg must be supplied together")
    u_values = _axis_values(args.u_d, args.u_d_min, args.u_d_max, args.n_u_d, "u_D")
    theta_values = _axis_values(
        args.theta_deg,
        args.theta_min_deg,
        args.theta_max_deg,
        args.n_twist,
        "theta",
    )
    return [
        TaigeHysteresisPoint(
            u_index=iu,
            theta_index=it,
            u_D=float(u),
            theta_deg=float(theta),
        )
        for it, theta in enumerate(theta_values)
        for iu, u in enumerate(u_values)
    ]


def _selected_points(args: argparse.Namespace) -> list[TaigeHysteresisPoint]:
    points = _all_points(args)
    if args.task_id is None:
        return points
    task_id = int(args.task_id)
    if task_id < 0:
        raise ValueError("--task-id must be nonnegative")
    if task_id >= len(points):
        print(f"Task {task_id} is outside cache mesh size {len(points)}; exiting.")
        return []
    return [points[task_id]]


def _cache_root(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    root = Path(args.cache_root) if args.cache_root is not None else output_root / "backend_cache"
    if not root.is_absolute():
        root = ROOT / root
    return root


def _model_for_point(args: argparse.Namespace, point: TaigeHysteresisPoint):
    return taige_model_params(
        theta_deg=point.theta_deg,
        u_D=point.u_D,
        plane_wave_shell=args.plane_wave_shell,
        n_bands=args.n_bands,
        n_active_bands_per_valley=args.n_active_bands_per_valley,
    )


def _interaction(args: argparse.Namespace):
    return taige_interaction_params(
        include_q0=not args.omit_q0,
        q_mesh=args.q_mesh,
        q_shell=args.q_shell,
        local_field_cutoff=args.local_field_cutoff,
        epsilon=args.epsilon,
        gate_distance_nm=args.gate_distance_nm,
        smear_length_nm=args.smear_length_nm,
        interaction_strength_scale=args.v0,
        exchange_scale=args.exchange_scale,
        hartree_scale=args.hartree_scale,
        vertex_workers=args.vertex_workers,
        exchange_workers=args.exchange_workers,
        density_vertex_retention=args.density_vertex_retention,
        density_vertex_layout=args.density_vertex_layout,
        exchange_representation=args.exchange_representation,
        form_factor_backend=args.form_factor_backend,
    )


def _hf_params(args: argparse.Namespace) -> ContinuumHFParams:
    return ContinuumHFParams(
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


def _reference_quality_fields(
    prefix: str,
    result: Any | None,
    *,
    n_blocks: int,
    max_iter: int | None,
) -> dict[str, Any]:
    if result is None:
        return {
            f"{prefix}_energy_per_cell": None,
            f"{prefix}_converged": None,
            f"{prefix}_hit_max_iter": None,
            f"{prefix}_warning_flag": None,
            f"{prefix}_clean": None,
            f"{prefix}_direct_gap": None,
            f"{prefix}_indirect_gap": None,
        }
    diagnostics = result.diagnostics
    hit_max_iter = bool(
        max_iter is not None
        and not result.converged
        and int(result.n_iter) >= int(max_iter)
    )
    warning = bool(diagnostics.self_consistency_warning or hit_max_iter or not result.converged)
    return {
        f"{prefix}_energy_per_cell": float(result.energy / n_blocks),
        f"{prefix}_converged": bool(result.converged),
        f"{prefix}_hit_max_iter": hit_max_iter,
        f"{prefix}_warning_flag": warning,
        f"{prefix}_self_consistency_warning": bool(diagnostics.self_consistency_warning),
        f"{prefix}_clean": bool(result.converged and not hit_max_iter and not diagnostics.self_consistency_warning),
        f"{prefix}_iteration_count": int(result.n_iter),
        f"{prefix}_max_iter": None if max_iter is None else int(max_iter),
        f"{prefix}_direct_gap": float(diagnostics.direct_gap_min),
        f"{prefix}_indirect_gap": float(diagnostics.indirect_gap),
        f"{prefix}_aufbau_residual_norm": float(diagnostics.aufbau_residual_norm),
        f"{prefix}_commutator_norm": float(diagnostics.commutator_norm),
        f"{prefix}_delta_P": float(diagnostics.delta_P),
        f"{prefix}_delta_energy": float(diagnostics.delta_energy),
        f"{prefix}_idempotency_error_fro": float(diagnostics.idempotency_error_fro),
        f"{prefix}_constraint_error": float(diagnostics.constraint_error),
        f"{prefix}_trace_error": float(diagnostics.trace_error),
    }


def _load_cache_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _merge_cache_summaries(args: argparse.Namespace) -> list[dict[str, Any]]:
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    cache_root = _cache_root(args)
    rows = [
        _load_cache_summary(path)
        for path in sorted(cache_root.rglob("*.summary.json"))
    ]
    rows.sort(key=lambda row: (int(row.get("theta_index", 0)), int(row.get("u_index", 0))))
    _write_csv(output_root / "backend_cache_completed.csv", rows)
    chern_rows: list[dict[str, Any]] = []
    for row in rows:
        point_fields = {
            "u_index": row.get("u_index"),
            "theta_index": row.get("theta_index"),
            "u_D_meV": row.get("u_D"),
            "theta_deg": row.get("theta_deg"),
            "cache_hash": row.get("cache_hash"),
            "cache_path": row.get("cache_path"),
        }
        for chern in row.get("vp_hf_chern_rows", []) or []:
            chern_rows.append({**point_fields, **chern})
    _write_csv(output_root / "backend_cache_vp_chern_numbers.csv", chern_rows)
    return rows


def _plan_rows(args: argparse.Namespace, points: list[TaigeHysteresisPoint]) -> list[dict[str, Any]]:
    cache_root = _cache_root(args)
    interaction = _interaction(args)
    rows: list[dict[str, Any]] = []
    for flat_index, point in enumerate(_all_points(args)):
        model = _model_for_point(args, point)
        grid = ContinuumGridParams(n_k=args.n_k)
        signature = taige_backend_cache_signature(
            model=model,
            grid=grid,
            interaction=interaction,
        )
        selected = any(point == item for item in points)
        rows.append(
            {
                **point.model_dump(mode="json"),
                "task_id": int(flat_index),
                "selected": bool(selected),
                "cache_hash": taige_backend_cache_hash(signature),
                "cache_path": str(taige_backend_cache_path(cache_root, signature)),
            }
        )
    return rows


def run_point(args: argparse.Namespace, point: TaigeHysteresisPoint) -> dict[str, Any]:
    cache_root = _cache_root(args)
    model = _model_for_point(args, point)
    grid = ContinuumGridParams(n_k=args.n_k)
    interaction = _interaction(args)
    signature = taige_backend_cache_signature(
        model=model,
        grid=grid,
        interaction=interaction,
    )
    cache_path = taige_backend_cache_path(cache_root, signature)
    if args.skip_existing and cache_path.exists():
        try:
            load_taige_backend_cache(cache_path)
        except Exception as exc:
            print(
                f"Existing cache {cache_path} is unreadable; deleting and rebuilding: {exc}",
                file=sys.stderr,
            )
            cache_path.unlink(missing_ok=True)
            cache_path.with_suffix(".summary.json").unlink(missing_ok=True)
        else:
            print(f"Skipping existing validated cache {cache_path}")
            return {
                **point.model_dump(mode="json"),
                "cache_hash": taige_backend_cache_hash(signature),
                "cache_path": str(cache_path),
                "status": "skipped_existing",
            }
    print(
        "Building Taige backend cache "
        f"theta={point.theta_deg:.8g} u_D={point.u_D:.8g} n_k={args.n_k} "
        f"retention={args.density_vertex_retention} layout={args.density_vertex_layout} "
        f"exchange={args.exchange_representation} form_factor={args.form_factor_backend}"
    )
    start = time.perf_counter()
    bundle = build_continuum_bundle(model=model, grid=grid, interaction=interaction)
    vp_plus = None
    vp_minus = None
    if not args.no_vp_references:
        controls = _hf_params(args)
        constraint = ValleyU1Constraint(bundle.active)
        vp_plus = solve_reference_hf(bundle, "vp_plus", controls, constraint=constraint)
        vp_minus = solve_reference_hf(bundle, "vp_minus", controls, constraint=constraint)
    manifest = save_taige_backend_cache(
        cache_path,
        bundle=bundle,
        signature=signature,
        vp_plus=vp_plus,
        vp_minus=vp_minus,
        vp_reference_max_iter=args.max_iter,
        compute_vp_chern=not args.no_vp_chern,
    )
    seconds = time.perf_counter() - start
    summary = {
        **point.model_dump(mode="json"),
        "cache_hash": manifest.cache_hash,
        "cache_path": str(cache_path),
        "status": "built",
        "elapsed_seconds": float(seconds),
        "has_vp_references": manifest.has_vp_references,
        "vertex_layout": manifest.vertex_layout,
        "exchange_representation": manifest.exchange_representation,
        "density_vertex_retention": manifest.density_vertex_retention,
        "vp_hf_chern_rows": list(manifest.vp_hf_chern_rows),
        **manifest.vp_hf_chern_columns,
        **_reference_quality_fields(
            "vp_plus",
            vp_plus,
            n_blocks=bundle.backend.n_blocks,
            max_iter=args.max_iter,
        ),
        **_reference_quality_fields(
            "vp_minus",
            vp_minus,
            n_blocks=bundle.backend.n_blocks,
            max_iter=args.max_iter,
        ),
    }
    _write_json(cache_path.with_suffix(".summary.json"), summary | {"signature": signature})
    print(f"Wrote cache {cache_path} elapsed={seconds:.1f}s")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    points = _selected_points(args)
    plan_rows = _plan_rows(args, points)
    _write_csv(output_root / "backend_cache_plan.csv", plan_rows)
    _write_json(
        output_root / "backend_cache_plan.json",
        {
            "rows": plan_rows,
            "n_tasks": len(_all_points(args)),
            "n_selected": len(points),
            "args": {
                key: value
                for key, value in vars(args).items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
        },
    )
    if args.dry_run:
        print(f"Wrote backend-cache dry-run plan to {output_root}")
        return 0
    if args.merge_only:
        rows = _merge_cache_summaries(args)
        print(f"Merged {len(rows)} backend-cache summaries under {output_root}")
        return 0
    rows = [run_point(args, point) for point in points]
    _write_csv(output_root / "backend_cache_completed.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
