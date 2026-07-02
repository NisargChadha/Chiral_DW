#!/usr/bin/env python3
"""Run one Taige Q=0 IVC displacement-hysteresis branch at fixed twist."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from math import pi
from pathlib import Path
from typing import Any

import numpy as np

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
from chiral_dw.continuum import (  # noqa: E402
    TPrimeConstraint,
    TaigeHysteresisBranchRecord,
    TaigeHysteresisPoint,
    TaigeIvcSeedSpec,
    ValleyU1Constraint,
    active_basis_frames,
    build_branch_response_result,
    build_continuum_bundle,
    build_seed,
    load_taige_backend_cache,
    mix_projector_seeds,
    order_diagnostics,
    random_projector_like_seed,
    save_taige_backend_cache,
    select_lowest_energy_record,
    solve_hf,
    solve_reference_hf,
    taige_backend_cache_path,
    taige_backend_cache_signature,
    taige_interaction_params,
    taige_model_params,
    transport_projector_between_frames,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/taige_ivc_hysteresis")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--theta-index", type=int, default=None)
    parser.add_argument("--direction", choices=["up", "down"], default=None)
    parser.add_argument("--u-d-min", type=float, default=0.0)
    parser.add_argument("--u-d-max", type=float, default=20.0)
    parser.add_argument("--n-u-d", type=int, default=21)
    parser.add_argument("--theta-min-deg", type=float, default=2.0)
    parser.add_argument("--theta-max-deg", type=float, default=4.0)
    parser.add_argument("--n-twist", type=int, default=21)

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
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--min-iter", type=int, default=3)
    parser.add_argument("--mixing-method", choices=["linear", "oda"], default="oda")
    parser.add_argument("--mixing", type=float, default=0.45)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--energy-tolerance", type=float, default=1e-10)
    parser.add_argument("--final-residual-tolerance", type=float, default=1e-7)
    parser.add_argument("--random-seeds", default="1,7,13,29,53")
    parser.add_argument("--seed-ordered-weight", type=float, default=0.8)
    parser.add_argument("--seed-random-weight", type=float, default=0.2)
    parser.add_argument("--include-ordered-seed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--default-random-seed", type=int, default=7)

    parser.add_argument("--n-theta", type=int, default=41)
    parser.add_argument("--endpoint-eps", type=float, default=1e-5)
    parser.add_argument("--domain-radius", type=float, default=20.0)
    parser.add_argument("--domain-width", type=float, default=3.0)
    parser.add_argument("--domain-winding", type=int, default=1)
    parser.add_argument("--allow-texture-in-ivc-ground-state", action="store_true")
    parser.add_argument("--texture-energy-tie-atol", type=float, default=1e-9)
    parser.add_argument("--compute-invalid-texture-cg", action="store_true")

    parser.add_argument("--require-cache", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _parse_int_list(text: str | None) -> list[int]:
    if text is None:
        return []
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _axis_values(lower: float, upper: float, count: int, name: str) -> np.ndarray:
    if int(count) < 1:
        raise ValueError(f"{name} count must be positive")
    return np.linspace(float(lower), float(upper), int(count))


def _theta_values(args: argparse.Namespace) -> np.ndarray:
    return _axis_values(args.theta_min_deg, args.theta_max_deg, args.n_twist, "theta")


def _selected_tasks(args: argparse.Namespace) -> list[tuple[int, str]]:
    theta_values = _theta_values(args)
    tasks = [(it, direction) for it in range(len(theta_values)) for direction in ("up", "down")]
    if args.task_id is not None:
        task_id = int(args.task_id)
        if task_id < 0:
            raise ValueError("--task-id must be nonnegative")
        if task_id >= len(tasks):
            print(f"Task {task_id} is outside branch task count {len(tasks)}; exiting.")
            return []
        return [tasks[task_id]]
    if args.theta_index is not None or args.direction is not None:
        if args.theta_index is None or args.direction is None:
            raise ValueError("--theta-index and --direction must be supplied together")
        return [(int(args.theta_index), str(args.direction))]
    return tasks


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


def _point_for(args: argparse.Namespace, theta_index: int, u_index: int) -> TaigeHysteresisPoint:
    theta_values = _theta_values(args)
    u_values = _axis_values(args.u_d_min, args.u_d_max, args.n_u_d, "u_D")
    return TaigeHysteresisPoint(
        u_index=int(u_index),
        theta_index=int(theta_index),
        u_D=float(u_values[int(u_index)]),
        theta_deg=float(theta_values[int(theta_index)]),
    )


def _ordered_line(args: argparse.Namespace, theta_index: int, direction: str) -> list[TaigeHysteresisPoint]:
    points = [_point_for(args, theta_index, iu) for iu in range(int(args.n_u_d))]
    if direction == "down":
        points = list(reversed(points))
    return points


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


def _hf_params(args: argparse.Namespace, seed: TaigeIvcSeedSpec | None = None) -> ContinuumHFParams:
    ordered = 1.0 if seed is None else seed.ordered_weight
    random = 0.0 if seed is None else seed.random_weight
    random_seed = args.default_random_seed if seed is None or seed.random_seed is None else seed.random_seed
    return ContinuumHFParams(
        n_occ_per_k=args.n_occ_per_k,
        max_iter=args.max_iter,
        min_iter=args.min_iter,
        mixing_method=args.mixing_method,
        mixing=args.mixing,
        tolerance=args.tolerance,
        energy_tolerance=args.energy_tolerance,
        final_residual_tolerance=args.final_residual_tolerance,
        seed_ordered_weight=float(ordered),
        seed_random_weight=float(random),
        random_seed=int(random_seed),
        store_projector_snapshots=False,
    )


def _workflow_params(args: argparse.Namespace, point: TaigeHysteresisPoint, out_dir: Path) -> ContinuumWorkflowParams:
    return ContinuumWorkflowParams(
        model=_model_for_point(args, point),
        grid=ContinuumGridParams(n_k=args.n_k),
        interaction=_interaction(args),
        hf=_hf_params(args),
        response=ResponseParams(
            n_theta=args.n_theta,
            theta_min=float(args.endpoint_eps),
            theta_max=float(pi - args.endpoint_eps),
            endpoint_eps=float(args.endpoint_eps),
        ),
        domain_wall=DomainWallParams(
            radius=args.domain_radius,
            width=args.domain_width,
            winding=args.domain_winding,
        ),
        output_dir=str(out_dir),
    )


def _seed_specs(args: argparse.Namespace) -> list[TaigeIvcSeedSpec]:
    specs: list[TaigeIvcSeedSpec] = []
    if args.include_ordered_seed:
        specs.append(TaigeIvcSeedSpec(label="ordered", ordered_weight=1.0, random_weight=0.0))
    for seed in _parse_int_list(args.random_seeds):
        specs.append(
            TaigeIvcSeedSpec(
                label=f"mixed_seed_{seed}",
                ordered_weight=float(args.seed_ordered_weight),
                random_weight=float(args.seed_random_weight),
                random_seed=int(seed),
            )
        )
    if not specs:
        raise ValueError("at least one endpoint seed must be enabled")
    return specs


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _json_ready(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True))


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


def _branch_dir(args: argparse.Namespace, theta_index: int, direction: str) -> Path:
    theta = float(_theta_values(args)[theta_index])
    label = f"theta_{theta_index:03d}_{theta:.6g}".replace("-", "m").replace(".", "p")
    return _output_root(args) / "branches" / label / direction


def _point_dir(branch_dir: Path, point: TaigeHysteresisPoint) -> Path:
    return branch_dir / "points" / point.label


def _cache_path(args: argparse.Namespace, point: TaigeHysteresisPoint) -> tuple[Path, dict[str, Any]]:
    signature = taige_backend_cache_signature(
        model=_model_for_point(args, point),
        grid=ContinuumGridParams(n_k=args.n_k),
        interaction=_interaction(args),
    )
    return taige_backend_cache_path(_cache_root(args), signature), signature


def _load_or_build_cache(args: argparse.Namespace, point: TaigeHysteresisPoint):
    cache_path, signature = _cache_path(args, point)
    if cache_path.exists():
        loaded = load_taige_backend_cache(cache_path)
        return loaded, str(cache_path)
    if args.require_cache:
        raise FileNotFoundError(f"missing required backend cache {cache_path}")
    bundle = build_continuum_bundle(
        model=_model_for_point(args, point),
        grid=ContinuumGridParams(n_k=args.n_k),
        interaction=_interaction(args),
    )
    controls = _hf_params(args)
    vp_constraint = ValleyU1Constraint(bundle.active)
    vp_plus = solve_reference_hf(bundle, "vp_plus", controls, constraint=vp_constraint)
    vp_minus = solve_reference_hf(bundle, "vp_minus", controls, constraint=vp_constraint)
    save_taige_backend_cache(
        cache_path,
        bundle=bundle,
        signature=signature,
        vp_plus=vp_plus,
        vp_minus=vp_minus,
    )
    loaded = load_taige_backend_cache(cache_path)
    return loaded, str(cache_path)


def _initial_ivc_projector(bundle, args: argparse.Namespace, seed: TaigeIvcSeedSpec) -> np.ndarray:
    active = bundle.active
    P0 = build_seed(
        "ivc",
        active,
        n_occ_per_k=int(args.n_occ_per_k),
        random_seed_value=int(seed.random_seed or args.default_random_seed),
    )
    if seed.random_weight > 0.0:
        noise = random_projector_like_seed(P0, seed=int(seed.random_seed or args.default_random_seed))
        constraint = TPrimeConstraint(active)
        noise = constraint.project_density(noise)
        P0 = mix_projector_seeds(
            P0,
            noise,
            ordered_weight=float(seed.ordered_weight),
            random_weight=float(seed.random_weight),
        )
        P0 = constraint.project_density(P0)
    return P0


def _solve_ivc(
    *,
    args: argparse.Namespace,
    bundle,
    seed: TaigeIvcSeedSpec,
    initial_projector: np.ndarray | None,
) -> Any:
    constraint = TPrimeConstraint(bundle.active)
    P0 = _initial_ivc_projector(bundle, args, seed) if initial_projector is None else constraint.project_density(initial_projector)
    return solve_hf(
        bundle.backend,
        P0,
        _hf_params(args, seed),
        constraint=constraint,
        seed=seed.label,
    )


def _response_fields(
    *,
    args: argparse.Namespace,
    point: TaigeHysteresisPoint,
    point_dir: Path,
    bundle,
    vp_plus,
    vp_minus,
    ivc,
    direction: str,
) -> dict[str, Any]:
    params = _workflow_params(args, point, point_dir)
    physical = build_branch_response_result(
        params=params,
        bundle=bundle,
        vp_plus=vp_plus,
        vp_minus=vp_minus,
        ivc=ivc,
        branch_label=direction,
        suppress_texture_when_ivc_below_vp=not args.allow_texture_in_ivc_ground_state,
        texture_energy_tie_atol=args.texture_energy_tie_atol,
    )
    diagnostic_cg = None
    if args.compute_invalid_texture_cg and not bool(physical.branch_selection["texture_valid"]):
        diagnostic = build_branch_response_result(
            params=params,
            bundle=bundle,
            vp_plus=vp_plus,
            vp_minus=vp_minus,
            ivc=ivc,
            branch_label=direction,
            suppress_texture_when_ivc_below_vp=False,
            texture_energy_tie_atol=args.texture_energy_tie_atol,
        )
        diagnostic_cg = float(diagnostic.summary.cG)
    elif bool(physical.branch_selection["texture_valid"]):
        diagnostic_cg = float(physical.summary.cG)
    return {
        "cG": float(physical.summary.cG),
        "cG_diagnostic": diagnostic_cg,
        "K_min": float(physical.summary.kappa_min),
        "K_max": float(physical.summary.kappa_max),
        "texture_valid": bool(physical.branch_selection["texture_valid"]),
        "texture_invalid_reason": physical.branch_selection["texture_invalid_reason"],
        "vp_reference_name": physical.branch_selection["vp_reference_name"],
        "vp_reference_energy_per_cell": float(physical.branch_selection["vp_reference_energy_per_cell"]),
        "ivc_minus_vp_energy_per_cell": float(
            physical.branch_selection["selected_ivc_minus_vp_energy_per_cell"]
        ),
        "response_gap_min": physical.summary.gap_min,
    }


def _write_checkpoint(
    *,
    args: argparse.Namespace,
    point: TaigeHysteresisPoint,
    point_dir: Path,
    direction: str,
    run_id: str,
    seed: TaigeIvcSeedSpec,
    result,
    bundle,
    vp_plus,
    vp_minus,
    cache_path: str,
    warm_start_source: str,
    warm_start_from_run_id: str | None,
    transport_diag: Any | None,
    selection_pool: str | None,
) -> TaigeHysteresisBranchRecord:
    point_dir.mkdir(parents=True, exist_ok=True)
    projector_path = point_dir / "projector_final.npz"
    frames = active_basis_frames(bundle.active)
    np.savez_compressed(
        projector_path,
        final_projector=result.P,
        final_h_hf=result.H_hf,
        active_frames=frames,
    )
    energy = bundle.backend.energy(result.P)
    order = order_diagnostics(result.P, bundle.active, n_occ_per_k=args.n_occ_per_k)
    response = _response_fields(
        args=args,
        point=point,
        point_dir=point_dir,
        bundle=bundle,
        vp_plus=vp_plus,
        vp_minus=vp_minus,
        ivc=result,
        direction=direction,
    )
    record = TaigeHysteresisBranchRecord(
        u_index=point.u_index,
        theta_index=point.theta_index,
        u_D_meV=point.u_D,
        theta_deg=point.theta_deg,
        direction=direction,  # type: ignore[arg-type]
        point_label=point.label,
        run_id=run_id,
        seed_label=seed.label,
        warm_start_source=warm_start_source,
        warm_start_from_run_id=warm_start_from_run_id,
        energy_total_per_cell=float(energy.total / bundle.backend.n_blocks),
        direct_gap_min=float(result.diagnostics.direct_gap_min),
        indirect_gap=float(result.diagnostics.indirect_gap),
        ivc_amplitude_block=float(order.IVC_amplitude_block),
        c_ivc_block=float(order.C_IVC_block),
        aufbau_residual_norm=float(result.diagnostics.aufbau_residual_norm),
        warning_flag=bool(result.diagnostics.self_consistency_warning),
        converged=bool(result.converged),
        iteration_count=int(result.n_iter),
        transport_mean_retained_weight=(
            None if transport_diag is None else float(transport_diag.mean_retained_weight)
        ),
        transport_min_retained_weight=(
            None if transport_diag is None else float(transport_diag.min_retained_weight)
        ),
        projector_path=str(projector_path),
        cache_path=cache_path,
        endpoint_selection_pool=selection_pool,
        energy_one_body_per_cell=float(energy.one_body / bundle.backend.n_blocks),
        energy_hartree_per_cell=float(energy.hartree / bundle.backend.n_blocks),
        energy_fock_per_cell=float(energy.fock / bundle.backend.n_blocks),
        final_idempotency_error_fro=float(result.diagnostics.idempotency_error_fro),
        final_trace_error=float(result.diagnostics.trace_error),
        **response,
    )
    _write_json(point_dir / "point_record.json", record.model_dump(mode="json"))
    return record


def _load_checkpoint(point_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray] | None:
    record_path = point_dir / "point_record.json"
    projector_path = point_dir / "projector_final.npz"
    if not record_path.exists() or not projector_path.exists():
        return None
    record = json.loads(record_path.read_text())
    with np.load(projector_path, allow_pickle=False) as data:
        P = np.asarray(data["final_projector"], dtype=complex)
        frames = np.asarray(data["active_frames"], dtype=complex)
    return record, P, frames


def _write_branch_tables(branch_dir: Path, records: list[dict[str, Any] | TaigeHysteresisBranchRecord]) -> None:
    rows = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        for item in records
    ]
    rows.sort(key=lambda row: int(row["u_index"]))
    _write_csv(branch_dir / "branch_points.csv", rows)
    _write_json(
        branch_dir / "branch_summary.json",
        {
            "n_points": len(rows),
            "rows": rows,
            "branch_points_csv": str(branch_dir / "branch_points.csv"),
        },
    )


def _endpoint_seed_scan(
    *,
    args: argparse.Namespace,
    point: TaigeHysteresisPoint,
    point_dir: Path,
    direction: str,
    loaded,
    cache_path: str,
) -> tuple[TaigeHysteresisBranchRecord, np.ndarray, np.ndarray]:
    bundle = loaded.bundle
    if loaded.vp_plus is None or loaded.vp_minus is None:
        controls = _hf_params(args)
        vp_constraint = ValleyU1Constraint(bundle.active)
        loaded_vp_plus = solve_reference_hf(bundle, "vp_plus", controls, constraint=vp_constraint)
        loaded_vp_minus = solve_reference_hf(bundle, "vp_minus", controls, constraint=vp_constraint)
    else:
        loaded_vp_plus = loaded.vp_plus
        loaded_vp_minus = loaded.vp_minus
    seed_rows: list[dict[str, Any]] = []
    seed_results: dict[str, Any] = {}
    for seed in _seed_specs(args):
        run_id = f"{direction}_theta_{point.theta_index:03d}_u_{point.u_index:03d}_{seed.label}"
        print(f"Endpoint cold seed {run_id}")
        result = _solve_ivc(args=args, bundle=bundle, seed=seed, initial_projector=None)
        energy = bundle.backend.energy(result.P)
        row = {
            "run_id": run_id,
            "seed_label": seed.label,
            "energy_total_per_cell": float(energy.total / bundle.backend.n_blocks),
            "converged": bool(result.converged),
            "warning_flag": bool(result.diagnostics.self_consistency_warning),
            "direct_gap_min": float(result.diagnostics.direct_gap_min),
            "ivc_iteration_count": int(result.n_iter),
        }
        seed_rows.append(row)
        seed_results[run_id] = (seed, result)
    selected_row, selection_pool = select_lowest_energy_record(seed_rows)
    selected_seed, selected_result = seed_results[str(selected_row["run_id"])]
    _write_csv(point_dir / "endpoint_seed_scan.csv", seed_rows)
    _write_json(
        point_dir / "endpoint_seed_selection.json",
        {"selected": selected_row, "selection_pool": selection_pool, "candidates": seed_rows},
    )
    record = _write_checkpoint(
        args=args,
        point=point,
        point_dir=point_dir,
        direction=direction,
        run_id=str(selected_row["run_id"]),
        seed=selected_seed,
        result=selected_result,
        bundle=bundle,
        vp_plus=loaded_vp_plus,
        vp_minus=loaded_vp_minus,
        cache_path=cache_path,
        warm_start_source="endpoint_cold_seed_scan",
        warm_start_from_run_id=None,
        transport_diag=None,
        selection_pool=selection_pool,
    )
    with np.load(record.projector_path, allow_pickle=False) as data:
        frames = np.asarray(data["active_frames"], dtype=complex)
    return record, selected_result.P, frames


def _run_branch(args: argparse.Namespace, theta_index: int, direction: str) -> None:
    branch_dir = _branch_dir(args, theta_index, direction)
    branch_dir.mkdir(parents=True, exist_ok=True)
    points = _ordered_line(args, theta_index, direction)
    records: list[dict[str, Any] | TaigeHysteresisBranchRecord] = []
    previous_projector = None
    previous_frames = None
    previous_run_id = None
    for point in points:
        point_dir = _point_dir(branch_dir, point)
        if not args.rerun_existing:
            checkpoint = _load_checkpoint(point_dir)
            if checkpoint is not None:
                record, previous_projector, previous_frames = checkpoint
                previous_run_id = str(record["run_id"])
                records.append(record)
                print(f"Resumed checkpoint {point.label} run_id={previous_run_id}")
                continue
        loaded, cache_path = _load_or_build_cache(args, point)
        bundle = loaded.bundle
        if loaded.vp_plus is None or loaded.vp_minus is None:
            controls = _hf_params(args)
            vp_constraint = ValleyU1Constraint(bundle.active)
            vp_plus = solve_reference_hf(bundle, "vp_plus", controls, constraint=vp_constraint)
            vp_minus = solve_reference_hf(bundle, "vp_minus", controls, constraint=vp_constraint)
        else:
            vp_plus = loaded.vp_plus
            vp_minus = loaded.vp_minus
        if previous_projector is None:
            record, previous_projector, previous_frames = _endpoint_seed_scan(
                args=args,
                point=point,
                point_dir=point_dir,
                direction=direction,
                loaded=loaded,
                cache_path=cache_path,
            )
            previous_run_id = record.run_id
            records.append(record)
            _write_branch_tables(branch_dir, records)
            continue
        current_frames = active_basis_frames(bundle.active)
        transported, transport_diag = transport_projector_between_frames(
            previous_projector,
            previous_frames,
            current_frames,
            n_occ_per_k=args.n_occ_per_k,
        )
        seed = TaigeIvcSeedSpec(label="transported_warm_start", ordered_weight=1.0, random_weight=0.0)
        run_id = f"{direction}_theta_{point.theta_index:03d}_u_{point.u_index:03d}_warm"
        print(
            f"Warm branch {run_id} from={previous_run_id} "
            f"retained={transport_diag.mean_retained_weight:.8g}"
        )
        result = _solve_ivc(args=args, bundle=bundle, seed=seed, initial_projector=transported)
        record = _write_checkpoint(
            args=args,
            point=point,
            point_dir=point_dir,
            direction=direction,
            run_id=run_id,
            seed=seed,
            result=result,
            bundle=bundle,
            vp_plus=vp_plus,
            vp_minus=vp_minus,
            cache_path=cache_path,
            warm_start_source="active_frame_projected_largest_eigenvectors",
            warm_start_from_run_id=previous_run_id,
            transport_diag=transport_diag,
            selection_pool="warm_start",
        )
        previous_projector = result.P
        previous_frames = current_frames
        previous_run_id = run_id
        records.append(record)
        _write_branch_tables(branch_dir, records)
    _write_branch_tables(branch_dir, records)
    print(f"Wrote branch outputs to {branch_dir}")


def _dry_run_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    task_id = 0
    for theta_index, direction in [
        (it, direction)
        for it in range(int(args.n_twist))
        for direction in ("up", "down")
    ]:
        line = _ordered_line(args, theta_index, direction)
        rows.append(
            {
                "task_id": task_id,
                "theta_index": theta_index,
                "theta_deg": float(line[0].theta_deg),
                "direction": direction,
                "n_u_d": len(line),
                "first_u_D_meV": float(line[0].u_D),
                "last_u_D_meV": float(line[-1].u_D),
                "branch_dir": str(_branch_dir(args, theta_index, direction)),
            }
        )
        task_id += 1
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_root = _output_root(args)
    output_root.mkdir(parents=True, exist_ok=True)
    plan_rows = _dry_run_plan(args)
    _write_csv(output_root / "hysteresis_branch_plan.csv", plan_rows)
    _write_json(
        output_root / "hysteresis_branch_plan.json",
        {
            "rows": plan_rows,
            "n_tasks": len(plan_rows),
            "args": {
                key: value
                for key, value in vars(args).items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
        },
    )
    if args.dry_run:
        print(f"Wrote hysteresis branch dry-run plan to {output_root}")
        return 0
    tasks = _selected_tasks(args)
    for theta_index, direction in tasks:
        _run_branch(args, theta_index, direction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
