#!/usr/bin/env python3
"""Local diagnostics for constrained Q=0 Taige IVC Hartree-Fock branches."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
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
)
from chiral_dw.continuum import (  # noqa: E402
    TPrimeConstraint,
    ValleyU1Constraint,
    active_basis_frames,
    build_continuum_bundle,
    build_seed,
    mix_projector_seeds,
    order_diagnostics,
    projector_overlap_diagnostics_with_frames,
    random_projector_like_seed,
    solve_hf,
    solve_reference_hf,
    taige_interaction_params,
    taige_model_params,
)
from chiral_dw.continuum.ivc_diagnostics import (  # noqa: E402
    ProjectorOverlapDiagnostics,
    TaigeIvcDiagnosticPoint,
    TaigeIvcSeedSpec,
)


@dataclass(frozen=True)
class RunRecord:
    """In-memory record used to compute overlaps after HF solves."""

    run_id: str
    mode: str
    direction: str
    point: TaigeIvcDiagnosticPoint
    seed: TaigeIvcSeedSpec
    max_iter: int
    phase_key: str
    projector_key: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="results/local_ivc_branch_diagnostics",
        help="Root directory for local diagnostic artifacts.",
    )
    parser.add_argument("--run-label", default=None)
    parser.add_argument(
        "--preset",
        choices=["quick", "ivc_branch_linecuts", "custom"],
        default="quick",
    )
    parser.add_argument(
        "--diagnostic-mode",
        choices=["scan", "convergence", "hysteresis", "all"],
        default="scan",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")

    parser.add_argument("--n-k", type=int, default=18)
    parser.add_argument("--theta-deg-list", default=None)
    parser.add_argument("--u-d-list", default=None)
    parser.add_argument("--u-d-min", type=float, default=None)
    parser.add_argument("--u-d-max", type=float, default=None)
    parser.add_argument("--u-d-step", type=float, default=0.25)
    parser.add_argument(
        "--convergence-point-list",
        default=None,
        help="Semicolon-separated u_D:theta_deg pairs for convergence probes.",
    )
    parser.add_argument("--convergence-max-iters", default="100,200,400")

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
    parser.add_argument("--snapshot-interval", type=int, default=5)
    parser.add_argument("--first-iteration-snapshot", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--random-seeds", default=None)
    parser.add_argument("--seed-ordered-weight", type=float, default=0.8)
    parser.add_argument("--seed-random-weight", type=float, default=0.2)
    parser.add_argument("--include-ordered-seed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--default-random-seed", type=int, default=7)
    parser.add_argument("--solve-vp-baseline", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _parse_int_list(text: str | None) -> list[int]:
    if text is None:
        return []
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_float_list(text: str | None) -> list[float]:
    if text is None:
        return []
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _inclusive_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("step must be positive")
    n = int(np.floor((float(stop) - float(start)) / float(step) + 0.5))
    values = [float(start) + i * float(step) for i in range(n + 1)]
    if not values or values[-1] < float(stop) - 1e-9:
        values.append(float(stop))
    return [round(value, 10) for value in values]


def _float_key(value: float) -> str:
    return f"{float(value):.8f}"


def _float_label(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def _point_key(point: TaigeIvcDiagnosticPoint) -> str:
    return f"u={_float_key(point.u_D)}|theta={_float_key(point.theta_deg)}"


def _dedupe_points(points: list[TaigeIvcDiagnosticPoint]) -> list[TaigeIvcDiagnosticPoint]:
    out: list[TaigeIvcDiagnosticPoint] = []
    seen: set[tuple[str, str, str]] = set()
    for point in points:
        key = (point.group, _float_key(point.u_D), _float_key(point.theta_deg))
        if key not in seen:
            seen.add(key)
            out.append(point)
    return out


def _points_for_linecut(theta_deg: float, u_values: list[float], group: str) -> list[TaigeIvcDiagnosticPoint]:
    return [
        TaigeIvcDiagnosticPoint(
            u_index=iu,
            theta_index=0,
            u_D=float(u),
            theta_deg=float(theta_deg),
            group=group,
        )
        for iu, u in enumerate(u_values)
    ]


def _preset_points(args: argparse.Namespace) -> list[TaigeIvcDiagnosticPoint]:
    if args.preset == "quick":
        return _points_for_linecut(
            3.65,
            [5.25, 6.0, 6.75, 7.5, 8.25],
            "quick_theta_3p65",
        )
    if args.preset == "ivc_branch_linecuts":
        points: list[TaigeIvcDiagnosticPoint] = []
        specs = [
            (3.65, 4.5, 9.0, "theta_3p65"),
            (3.21, 3.0, 7.5, "theta_3p21"),
            (3.98, 6.0, 10.5, "theta_3p98"),
        ]
        for theta, lower, upper, group in specs:
            points.extend(_points_for_linecut(theta, _inclusive_range(lower, upper, 0.25), group))
        points.extend(
            [
                TaigeIvcDiagnosticPoint(
                    u_index=0,
                    theta_index=0,
                    u_D=2.25,
                    theta_deg=2.11,
                    group="stable_controls",
                ),
                TaigeIvcDiagnosticPoint(
                    u_index=1,
                    theta_index=1,
                    u_D=12.0,
                    theta_deg=2.33,
                    group="stable_controls",
                ),
            ]
        )
        return _dedupe_points(points)
    if args.preset != "custom":
        raise ValueError(f"unknown preset {args.preset!r}")
    theta_values = _parse_float_list(args.theta_deg_list)
    if not theta_values:
        raise ValueError("--theta-deg-list is required with --preset custom")
    if args.u_d_list is not None:
        u_values = _parse_float_list(args.u_d_list)
    else:
        if args.u_d_min is None or args.u_d_max is None:
            raise ValueError("--u-d-list or --u-d-min/--u-d-max is required with --preset custom")
        u_values = _inclusive_range(float(args.u_d_min), float(args.u_d_max), float(args.u_d_step))
    points = []
    for it, theta in enumerate(theta_values):
        for iu, u in enumerate(u_values):
            points.append(
                TaigeIvcDiagnosticPoint(
                    u_index=iu,
                    theta_index=it,
                    u_D=float(u),
                    theta_deg=float(theta),
                    group="custom",
                )
            )
    return points


def _convergence_points(args: argparse.Namespace) -> list[TaigeIvcDiagnosticPoint]:
    text = args.convergence_point_list
    pairs: list[tuple[float, float]]
    if text:
        pairs = []
        for item in str(text).split(";"):
            if not item.strip():
                continue
            left, right = item.replace(",", ":").split(":")[:2]
            pairs.append((float(left), float(right)))
    elif args.preset == "quick":
        pairs = [(7.5, 3.65), (6.75, 3.65), (8.25, 3.65)]
    else:
        pairs = [(7.5, 3.65), (4.5, 3.21), (9.0, 3.98)]
    return [
        TaigeIvcDiagnosticPoint(
            u_index=i,
            theta_index=i,
            u_D=u,
            theta_deg=theta,
            group="convergence_probe",
        )
        for i, (u, theta) in enumerate(pairs)
    ]


def _linecut_groups(points: list[TaigeIvcDiagnosticPoint]) -> dict[str, list[TaigeIvcDiagnosticPoint]]:
    groups: dict[str, list[TaigeIvcDiagnosticPoint]] = {}
    for point in points:
        if point.group == "stable_controls":
            continue
        groups.setdefault(point.group, []).append(point)
    for group, rows in groups.items():
        groups[group] = sorted(rows, key=lambda point: point.u_D)
    return groups


def _seed_specs(args: argparse.Namespace) -> list[TaigeIvcSeedSpec]:
    if args.random_seeds is None:
        random_seeds = [7, 13] if args.preset == "quick" else [1, 7, 13, 29, 53]
    else:
        random_seeds = _parse_int_list(args.random_seeds)
    specs: list[TaigeIvcSeedSpec] = []
    if args.include_ordered_seed:
        specs.append(
            TaigeIvcSeedSpec(
                label="ordered",
                ordered_weight=1.0,
                random_weight=0.0,
                random_seed=None,
            )
        )
    for seed in random_seeds:
        specs.append(
            TaigeIvcSeedSpec(
                label=f"mixed_seed_{seed}",
                ordered_weight=float(args.seed_ordered_weight),
                random_weight=float(args.seed_random_weight),
                random_seed=int(seed),
            )
        )
    if not specs:
        raise ValueError("at least one seed must be requested")
    return specs


def _hf_params(
    args: argparse.Namespace,
    *,
    max_iter: int,
    store_snapshots: bool,
    seed: TaigeIvcSeedSpec | None = None,
) -> ContinuumHFParams:
    ordered = args.seed_ordered_weight if seed is None else seed.ordered_weight
    random = args.seed_random_weight if seed is None else seed.random_weight
    random_seed = args.default_random_seed if seed is None or seed.random_seed is None else seed.random_seed
    return ContinuumHFParams(
        n_occ_per_k=int(args.n_occ_per_k),
        max_iter=int(max_iter),
        min_iter=int(args.min_iter),
        mixing_method=args.mixing_method,
        mixing=float(args.mixing),
        tolerance=float(args.tolerance),
        energy_tolerance=float(args.energy_tolerance),
        final_residual_tolerance=float(args.final_residual_tolerance),
        seed_ordered_weight=float(ordered),
        seed_random_weight=float(random),
        random_seed=int(random_seed),
        store_projector_snapshots=bool(store_snapshots),
        snapshot_interval=int(args.snapshot_interval),
        first_iteration_snapshot=bool(args.first_iteration_snapshot),
    )


def _build_bundle(args: argparse.Namespace, point: TaigeIvcDiagnosticPoint):
    model = taige_model_params(
        theta_deg=float(point.theta_deg),
        u_D=float(point.u_D),
        plane_wave_shell=int(args.plane_wave_shell),
        n_bands=int(args.n_bands),
        n_active_bands_per_valley=int(args.n_active_bands_per_valley),
    )
    interaction = taige_interaction_params(
        include_q0=not bool(args.omit_q0),
        q_mesh=args.q_mesh,
        q_shell=int(args.q_shell),
        local_field_cutoff=int(args.local_field_cutoff),
        epsilon=float(args.epsilon),
        gate_distance_nm=float(args.gate_distance_nm),
        smear_length_nm=float(args.smear_length_nm),
        interaction_strength_scale=float(args.v0),
        hartree_scale=float(args.hartree_scale),
        exchange_scale=float(args.exchange_scale),
        vertex_workers=int(args.vertex_workers),
        exchange_workers=int(args.exchange_workers),
        density_vertex_retention=args.density_vertex_retention,
        density_vertex_layout=args.density_vertex_layout,
        exchange_representation=args.exchange_representation,
        form_factor_backend=args.form_factor_backend,
    )
    return build_continuum_bundle(
        model=model,
        grid=ContinuumGridParams(n_k=int(args.n_k)),
        interaction=interaction,
    )


def _diagnostics_row(diagnostics, prefix: str = "") -> dict[str, Any]:
    return {
        f"{prefix}energy": float(diagnostics.energy),
        f"{prefix}delta_energy": float(diagnostics.delta_energy),
        f"{prefix}delta_P": float(diagnostics.delta_P),
        f"{prefix}idempotency_error_fro": float(diagnostics.idempotency_error_fro),
        f"{prefix}idempotency_error_max": float(diagnostics.idempotency_error_max),
        f"{prefix}constraint_error": float(diagnostics.constraint_error),
        f"{prefix}aufbau_residual_norm": float(diagnostics.aufbau_residual_norm),
        f"{prefix}commutator_norm": float(diagnostics.commutator_norm),
        f"{prefix}trace_error": float(diagnostics.trace_error),
        f"{prefix}direct_gap_min": float(diagnostics.direct_gap_min),
        f"{prefix}indirect_gap": float(diagnostics.indirect_gap),
        f"{prefix}iteration": int(diagnostics.iteration),
        f"{prefix}constraint_name": diagnostics.constraint_name,
        f"{prefix}lambda_value": diagnostics.lambda_value,
        f"{prefix}fallback_reason": diagnostics.fallback_reason,
        f"{prefix}density_kind": diagnostics.density_kind,
        f"{prefix}self_consistency_warning": bool(diagnostics.self_consistency_warning),
    }


def _order_row(order, prefix: str = "") -> dict[str, Any]:
    return {
        f"{prefix}Nz_block": float(order.Nz_block),
        f"{prefix}Nz_abs": float(order.Nz_abs),
        f"{prefix}c_ivc_block": float(order.C_IVC_block),
        f"{prefix}ivc_amplitude_block": float(order.IVC_amplitude_block),
        f"{prefix}c_ivc_scalar": order.C_IVC_scalar,
        f"{prefix}ivc_amplitude_scalar": order.IVC_amplitude_scalar,
    }


def _history_summary(result) -> dict[str, Any]:
    residuals = np.asarray([row.aufbau_residual_norm for row in result.history], dtype=float)
    energies = np.asarray([row.energy for row in result.history], dtype=float)
    lambdas = np.asarray(
        [np.nan if row.lambda_value is None else row.lambda_value for row in result.history],
        dtype=float,
    )
    if residuals.size:
        tail = residuals[-min(20, residuals.size) :]
        x = np.arange(tail.size, dtype=float)
        slope = float(np.polyfit(x, np.log10(np.maximum(tail, 1e-300)), 1)[0]) if tail.size >= 2 else float("nan")
    else:
        tail = np.asarray([], dtype=float)
        slope = float("nan")
    return {
        "history_length": int(residuals.size),
        "history_residual_min": float(np.min(residuals)) if residuals.size else float("nan"),
        "history_residual_last": float(residuals[-1]) if residuals.size else float("nan"),
        "history_residual_tail_min": float(np.min(tail)) if tail.size else float("nan"),
        "history_residual_tail_max": float(np.max(tail)) if tail.size else float("nan"),
        "history_log10_residual_slope_tail": slope,
        "history_energy_min": float(np.min(energies)) if energies.size else float("nan"),
        "history_energy_max": float(np.max(energies)) if energies.size else float("nan"),
        "history_energy_last": float(energies[-1]) if energies.size else float("nan"),
        "history_lambda_min": float(np.nanmin(lambdas)) if np.any(np.isfinite(lambdas)) else float("nan"),
        "history_lambda_max": float(np.nanmax(lambdas)) if np.any(np.isfinite(lambdas)) else float("nan"),
    }


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


def _base_fields(
    *,
    run_id: str,
    mode: str,
    direction: str,
    point: TaigeIvcDiagnosticPoint,
    seed: TaigeIvcSeedSpec,
    max_iter: int,
    warm_start_from_run_id: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "mode": mode,
        "direction": direction,
        "group": point.group,
        "point_label": point.label,
        "u_index": int(point.u_index),
        "theta_index": int(point.theta_index),
        "u_D_meV": float(point.u_D),
        "theta_deg": float(point.theta_deg),
        "seed_label": seed.label,
        "seed_ordered_weight": float(seed.ordered_weight),
        "seed_random_weight": float(seed.random_weight),
        "seed_random_seed": seed.random_seed,
        "max_iter": int(max_iter),
        "warm_start_from_run_id": warm_start_from_run_id,
    }


def _solve_ivc(
    *,
    args: argparse.Namespace,
    bundle,
    point: TaigeIvcDiagnosticPoint,
    seed: TaigeIvcSeedSpec,
    max_iter: int,
    run_id: str,
    mode: str,
    direction: str,
    initial_projector: np.ndarray | None,
    warm_start_from_run_id: str | None,
    iteration_rows: list[dict[str, Any]],
    snapshot_arrays: dict[str, np.ndarray],
    snapshot_rows: list[dict[str, Any]],
) -> tuple[Any, dict[str, Any]]:
    active = bundle.active
    controls = _hf_params(args, max_iter=max_iter, store_snapshots=True, seed=seed)
    constraint = TPrimeConstraint(active)
    P0 = _initial_ivc_projector(bundle, args, seed) if initial_projector is None else constraint.project_density(initial_projector)
    base = _base_fields(
        run_id=run_id,
        mode=mode,
        direction=direction,
        point=point,
        seed=seed,
        max_iter=max_iter,
        warm_start_from_run_id=warm_start_from_run_id,
    )

    def callback(iteration, P_iter, energy, diagnostics, is_snapshot):
        order = order_diagnostics(P_iter, active, n_occ_per_k=int(args.n_occ_per_k))
        iteration_rows.append(
            {
                **base,
                "iteration": int(iteration),
                "is_snapshot": bool(is_snapshot),
                "callback_energy": float(energy),
                **_diagnostics_row(diagnostics),
                **_order_row(order),
            }
        )

    result = solve_hf(
        bundle.backend,
        P0,
        controls,
        constraint=constraint,
        seed=seed.label,
        on_iteration=callback,
    )
    for snapshot in result.snapshots:
        key = f"{run_id}__iter_{snapshot.iteration:04d}"
        snapshot_arrays[key] = snapshot.P
        snapshot_rows.append(
            {
                **base,
                "array_key": key,
                "iteration": int(snapshot.iteration),
                "energy": float(snapshot.energy),
                **_diagnostics_row(snapshot.diagnostics),
            }
        )
    energy = bundle.backend.energy(result.P)
    order = order_diagnostics(result.P, active, n_occ_per_k=int(args.n_occ_per_k))
    row = {
        **base,
        "n_k": int(args.n_k),
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
        "hit_max_iter": bool(result.n_iter >= int(max_iter)),
        "energy_total_per_cell": float(energy.total / bundle.backend.n_blocks),
        "energy_one_body_per_cell": float(energy.one_body / bundle.backend.n_blocks),
        "energy_hartree_per_cell": float(energy.hartree / bundle.backend.n_blocks),
        "energy_fock_per_cell": float(energy.fock / bundle.backend.n_blocks),
        **_diagnostics_row(result.diagnostics, prefix="final_"),
        **_order_row(order, prefix="final_"),
        **_history_summary(result),
    }
    return result, row


def _solve_vp_baselines(args: argparse.Namespace, bundle) -> dict[str, Any]:
    if not args.solve_vp_baseline:
        return {
            "vp_plus_energy_per_cell": float("nan"),
            "vp_minus_energy_per_cell": float("nan"),
            "vp_reference_energy_per_cell": float("nan"),
            "vp_plus_converged": False,
            "vp_minus_converged": False,
            "vp_plus_warning": False,
            "vp_minus_warning": False,
        }
    controls = _hf_params(args, max_iter=int(args.max_iter), store_snapshots=False, seed=None)
    active = bundle.active
    vp_plus = solve_reference_hf(bundle, "vp_plus", controls, constraint=ValleyU1Constraint(active))
    vp_minus = solve_reference_hf(bundle, "vp_minus", controls, constraint=ValleyU1Constraint(active))
    norm = float(bundle.backend.n_blocks)
    return {
        "vp_plus_energy_per_cell": float(vp_plus.energy / norm),
        "vp_minus_energy_per_cell": float(vp_minus.energy / norm),
        "vp_reference_energy_per_cell": float(min(vp_plus.energy, vp_minus.energy) / norm),
        "vp_plus_converged": bool(vp_plus.converged),
        "vp_minus_converged": bool(vp_minus.converged),
        "vp_plus_warning": bool(vp_plus.diagnostics.self_consistency_warning),
        "vp_minus_warning": bool(vp_minus.diagnostics.self_consistency_warning),
        "vp_plus_direct_gap_min": float(vp_plus.diagnostics.direct_gap_min),
        "vp_minus_direct_gap_min": float(vp_minus.diagnostics.direct_gap_min),
        "vp_plus_aufbau_residual_norm": float(vp_plus.diagnostics.aufbau_residual_norm),
        "vp_minus_aufbau_residual_norm": float(vp_minus.diagnostics.aufbau_residual_norm),
    }


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


def _overlap_row(
    base: dict[str, Any],
    overlap: ProjectorOverlapDiagnostics,
) -> dict[str, Any]:
    return {**base, **overlap.model_dump(mode="json")}


def _compute_seed_overlaps(
    records: list[RunRecord],
    final_arrays: dict[str, np.ndarray],
    phase_frames: dict[str, np.ndarray],
    *,
    n_occ_per_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, int], list[RunRecord]] = {}
    for record in records:
        if record.mode not in {"scan", "convergence"}:
            continue
        groups.setdefault((record.mode, record.phase_key, record.max_iter), []).append(record)
    for (_mode, _phase_key, _max_iter), group in groups.items():
        for i, left in enumerate(group):
            for right in group[i + 1 :]:
                overlap = projector_overlap_diagnostics_with_frames(
                    final_arrays[left.projector_key],
                    final_arrays[right.projector_key],
                    phase_frames[left.phase_key],
                    phase_frames[right.phase_key],
                    n_occ_per_k=n_occ_per_k,
                )
                rows.append(
                    _overlap_row(
                        {
                            "mode": left.mode,
                            "u_D_meV": float(left.point.u_D),
                            "theta_deg": float(left.point.theta_deg),
                            "max_iter": int(left.max_iter),
                            "run_id_left": left.run_id,
                            "run_id_right": right.run_id,
                            "seed_label_left": left.seed.label,
                            "seed_label_right": right.seed.label,
                        },
                        overlap,
                    )
                )
    return rows


def _compute_neighbor_overlaps(
    records: list[RunRecord],
    final_arrays: dict[str, np.ndarray],
    phase_frames: dict[str, np.ndarray],
    *,
    n_occ_per_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str, float, int], list[RunRecord]] = {}
    for record in records:
        if record.mode != "scan":
            continue
        key = (record.mode, record.seed.label, round(record.point.theta_deg, 10), record.max_iter)
        groups.setdefault(key, []).append(record)
    for (_mode, _seed, _theta, _max_iter), group in groups.items():
        ordered = sorted(group, key=lambda record: record.point.u_D)
        for left, right in zip(ordered[:-1], ordered[1:]):
            overlap = projector_overlap_diagnostics_with_frames(
                final_arrays[left.projector_key],
                final_arrays[right.projector_key],
                phase_frames[left.phase_key],
                phase_frames[right.phase_key],
                n_occ_per_k=n_occ_per_k,
            )
            rows.append(
                _overlap_row(
                    {
                        "mode": left.mode,
                        "theta_deg": float(left.point.theta_deg),
                        "seed_label": left.seed.label,
                        "max_iter": int(left.max_iter),
                        "run_id_left": left.run_id,
                        "run_id_right": right.run_id,
                        "u_D_left_meV": float(left.point.u_D),
                        "u_D_right_meV": float(right.point.u_D),
                        "delta_u_D_meV": float(right.point.u_D - left.point.u_D),
                    },
                    overlap,
                )
            )
    return rows


def _compute_hysteresis_rows(
    records: list[RunRecord],
    final_arrays: dict[str, np.ndarray],
    phase_frames: dict[str, np.ndarray],
    run_rows_by_id: dict[str, dict[str, Any]],
    *,
    n_occ_per_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[float, float, int], dict[str, RunRecord]] = {}
    for record in records:
        if record.mode != "hysteresis":
            continue
        key = (round(record.point.theta_deg, 10), round(record.point.u_D, 10), record.max_iter)
        groups.setdefault(key, {})[record.direction] = record
    for (_theta, _u, _max_iter), item in groups.items():
        if "up" not in item or "down" not in item:
            continue
        up = item["up"]
        down = item["down"]
        overlap = projector_overlap_diagnostics_with_frames(
            final_arrays[up.projector_key],
            final_arrays[down.projector_key],
            phase_frames[up.phase_key],
            phase_frames[down.phase_key],
            n_occ_per_k=n_occ_per_k,
        )
        up_row = run_rows_by_id[up.run_id]
        down_row = run_rows_by_id[down.run_id]
        rows.append(
            _overlap_row(
                {
                    "theta_deg": float(up.point.theta_deg),
                    "u_D_meV": float(up.point.u_D),
                    "max_iter": int(up.max_iter),
                    "run_id_up": up.run_id,
                    "run_id_down": down.run_id,
                    "energy_total_per_cell_up": up_row["energy_total_per_cell"],
                    "energy_total_per_cell_down": down_row["energy_total_per_cell"],
                    "energy_total_per_cell_up_minus_down": float(
                        up_row["energy_total_per_cell"] - down_row["energy_total_per_cell"]
                    ),
                    "direct_gap_min_up": up_row["final_direct_gap_min"],
                    "direct_gap_min_down": down_row["final_direct_gap_min"],
                    "ivc_amplitude_block_up": up_row["final_ivc_amplitude_block"],
                    "ivc_amplitude_block_down": down_row["final_ivc_amplitude_block"],
                    "warning_up": up_row["final_self_consistency_warning"],
                    "warning_down": down_row["final_self_consistency_warning"],
                },
                overlap,
            )
        )
    return rows


def _write_plots(
    output_dir: Path,
    run_rows: list[dict[str, Any]],
    iteration_rows: list[dict[str, Any]],
    neighbor_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        _write_json(output_dir / "plot_warning.json", {"warning": f"plots skipped: {exc}"})
        return

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    scan_rows = [row for row in run_rows if row["mode"] == "scan"]
    if scan_rows:
        groups = sorted({(row["theta_deg"], row["seed_label"]) for row in scan_rows})
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        for theta, seed in groups:
            rows = sorted(
                [row for row in scan_rows if row["theta_deg"] == theta and row["seed_label"] == seed],
                key=lambda row: row["u_D_meV"],
            )
            if not rows:
                continue
            label = f"{theta:g} deg {seed}"
            x = [row["u_D_meV"] for row in rows]
            axes[0].plot(x, [row["energy_total_per_cell"] for row in rows], marker="o", ms=3, label=label)
            axes[1].plot(x, [row["final_direct_gap_min"] for row in rows], marker="o", ms=3, label=label)
            axes[2].plot(x, [row["final_ivc_amplitude_block"] for row in rows], marker="o", ms=3, label=label)
        axes[0].set_ylabel("E/cell")
        axes[1].set_ylabel("direct gap")
        axes[2].set_ylabel("IVC amplitude")
        axes[2].set_xlabel("u_D (meV)")
        axes[0].legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(plot_dir / "ivc_linecut_energy_gap_order.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        for theta, seed in groups:
            rows = sorted(
                [row for row in scan_rows if row["theta_deg"] == theta and row["seed_label"] == seed],
                key=lambda row: row["u_D_meV"],
            )
            if rows:
                ax.semilogy(
                    [row["u_D_meV"] for row in rows],
                    [row["final_aufbau_residual_norm"] for row in rows],
                    marker="o",
                    ms=3,
                    label=f"{theta:g} deg {seed}",
                )
        ax.axhline(1e-7, color="k", lw=1, ls="--")
        ax.set_xlabel("u_D (meV)")
        ax.set_ylabel("final Aufbau residual")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(plot_dir / "ivc_final_residuals.png", dpi=160)
        plt.close(fig)

    if neighbor_rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        for key in sorted({(row["theta_deg"], row["seed_label"]) for row in neighbor_rows}):
            theta, seed = key
            rows = sorted(
                [row for row in neighbor_rows if row["theta_deg"] == theta and row["seed_label"] == seed],
                key=lambda row: row["u_D_left_meV"],
            )
            x = [0.5 * (row["u_D_left_meV"] + row["u_D_right_meV"]) for row in rows]
            y = [row["one_minus_mean_overlap"] for row in rows]
            ax.plot(x, y, marker="o", ms=3, label=f"{theta:g} deg {seed}")
        ax.set_yscale("log")
        ax.set_xlabel("midpoint u_D (meV)")
        ax.set_ylabel("1 - projector overlap")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(plot_dir / "ivc_neighbor_projector_distance.png", dpi=160)
        plt.close(fig)

    if iteration_rows:
        final_by_run: dict[str, float] = {}
        for row in run_rows:
            final_by_run[row["run_id"]] = float(row["final_aufbau_residual_norm"])
        selected = sorted(final_by_run, key=final_by_run.get, reverse=True)[:12]
        fig, ax = plt.subplots(figsize=(8, 5))
        for run_id in selected:
            rows = sorted(
                [row for row in iteration_rows if row["run_id"] == run_id],
                key=lambda row: row["iteration"],
            )
            if rows:
                ax.semilogy(
                    [row["iteration"] for row in rows],
                    [row["aufbau_residual_norm"] for row in rows],
                    label=run_id,
                )
        ax.axhline(1e-7, color="k", lw=1, ls="--")
        ax.set_xlabel("HF iteration")
        ax.set_ylabel("mixed-density Aufbau residual")
        ax.legend(fontsize=5)
        fig.tight_layout()
        fig.savefig(plot_dir / "ivc_iteration_residuals_top_warnings.png", dpi=160)
        plt.close(fig)


def _run_scan_mode(
    args: argparse.Namespace,
    *,
    points: list[TaigeIvcDiagnosticPoint],
    seeds: list[TaigeIvcSeedSpec],
    run_rows: list[dict[str, Any]],
    iteration_rows: list[dict[str, Any]],
    snapshot_rows: list[dict[str, Any]],
    final_arrays: dict[str, np.ndarray],
    snapshot_arrays: dict[str, np.ndarray],
    phase_frames: dict[str, np.ndarray],
    records: list[RunRecord],
    run_counter: list[int],
) -> None:
    for point in points:
        print(f"Building bundle for u_D={point.u_D:g} meV theta={point.theta_deg:g} deg")
        bundle = _build_bundle(args, point)
        phase_key = _point_key(point)
        phase_frames[phase_key] = active_basis_frames(bundle.active)
        vp_fields = _solve_vp_baselines(args, bundle)
        for seed in seeds:
            run_id = f"run_{run_counter[0]:06d}"
            run_counter[0] += 1
            print(f"  IVC {run_id} seed={seed.label} max_iter={args.max_iter}")
            result, row = _solve_ivc(
                args=args,
                bundle=bundle,
                point=point,
                seed=seed,
                max_iter=int(args.max_iter),
                run_id=run_id,
                mode="scan",
                direction="none",
                initial_projector=None,
                warm_start_from_run_id=None,
                iteration_rows=iteration_rows,
                snapshot_arrays=snapshot_arrays,
                snapshot_rows=snapshot_rows,
            )
            row.update(vp_fields)
            row["ivc_minus_vp_reference_energy_per_cell"] = float(
                row["energy_total_per_cell"] - row["vp_reference_energy_per_cell"]
            )
            key = f"{run_id}__final"
            final_arrays[key] = result.P
            run_rows.append(row)
            records.append(
                RunRecord(
                    run_id=run_id,
                    mode="scan",
                    direction="none",
                    point=point,
                    seed=seed,
                    max_iter=int(args.max_iter),
                    phase_key=phase_key,
                    projector_key=key,
                )
            )


def _run_convergence_mode(
    args: argparse.Namespace,
    *,
    points: list[TaigeIvcDiagnosticPoint],
    seeds: list[TaigeIvcSeedSpec],
    run_rows: list[dict[str, Any]],
    iteration_rows: list[dict[str, Any]],
    snapshot_rows: list[dict[str, Any]],
    final_arrays: dict[str, np.ndarray],
    snapshot_arrays: dict[str, np.ndarray],
    phase_frames: dict[str, np.ndarray],
    records: list[RunRecord],
    run_counter: list[int],
) -> None:
    max_iters = _parse_int_list(args.convergence_max_iters)
    for point in points:
        bundle = _build_bundle(args, point)
        phase_key = _point_key(point)
        phase_frames[phase_key] = active_basis_frames(bundle.active)
        vp_fields = _solve_vp_baselines(args, bundle)
        for max_iter in max_iters:
            for seed in seeds:
                run_id = f"run_{run_counter[0]:06d}"
                run_counter[0] += 1
                print(
                    f"Convergence probe {run_id} u_D={point.u_D:g} theta={point.theta_deg:g} "
                    f"seed={seed.label} max_iter={max_iter}"
                )
                result, row = _solve_ivc(
                    args=args,
                    bundle=bundle,
                    point=point,
                    seed=seed,
                    max_iter=max_iter,
                    run_id=run_id,
                    mode="convergence",
                    direction="none",
                    initial_projector=None,
                    warm_start_from_run_id=None,
                    iteration_rows=iteration_rows,
                    snapshot_arrays=snapshot_arrays,
                    snapshot_rows=snapshot_rows,
                )
                row.update(vp_fields)
                row["ivc_minus_vp_reference_energy_per_cell"] = float(
                    row["energy_total_per_cell"] - row["vp_reference_energy_per_cell"]
                )
                key = f"{run_id}__final"
                final_arrays[key] = result.P
                run_rows.append(row)
                records.append(
                    RunRecord(
                        run_id=run_id,
                        mode="convergence",
                        direction="none",
                        point=point,
                        seed=seed,
                        max_iter=max_iter,
                        phase_key=phase_key,
                        projector_key=key,
                    )
                )


def _run_hysteresis_mode(
    args: argparse.Namespace,
    *,
    linecuts: dict[str, list[TaigeIvcDiagnosticPoint]],
    seed: TaigeIvcSeedSpec,
    run_rows: list[dict[str, Any]],
    iteration_rows: list[dict[str, Any]],
    snapshot_rows: list[dict[str, Any]],
    final_arrays: dict[str, np.ndarray],
    snapshot_arrays: dict[str, np.ndarray],
    phase_frames: dict[str, np.ndarray],
    records: list[RunRecord],
    run_counter: list[int],
) -> None:
    for group, points in linecuts.items():
        for direction, ordered_points in (
            ("up", sorted(points, key=lambda point: point.u_D)),
            ("down", sorted(points, key=lambda point: point.u_D, reverse=True)),
        ):
            previous_projector = None
            previous_run_id = None
            for point in ordered_points:
                bundle = _build_bundle(args, point)
                phase_key = _point_key(point)
                phase_frames.setdefault(phase_key, active_basis_frames(bundle.active))
                run_id = f"run_{run_counter[0]:06d}"
                run_counter[0] += 1
                print(
                    f"Hysteresis {group} {direction} {run_id} "
                    f"u_D={point.u_D:g} theta={point.theta_deg:g}"
                )
                result, row = _solve_ivc(
                    args=args,
                    bundle=bundle,
                    point=point,
                    seed=seed,
                    max_iter=int(args.max_iter),
                    run_id=run_id,
                    mode="hysteresis",
                    direction=direction,
                    initial_projector=previous_projector,
                    warm_start_from_run_id=previous_run_id,
                    iteration_rows=iteration_rows,
                    snapshot_arrays=snapshot_arrays,
                    snapshot_rows=snapshot_rows,
                )
                key = f"{run_id}__final"
                final_arrays[key] = result.P
                run_rows.append(row)
                records.append(
                    RunRecord(
                        run_id=run_id,
                        mode="hysteresis",
                        direction=direction,
                        point=point,
                        seed=seed,
                        max_iter=int(args.max_iter),
                        phase_key=phase_key,
                        projector_key=key,
                    )
                )
                previous_projector = result.P
                previous_run_id = run_id


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_label = args.run_label or f"{args.preset}_{args.diagnostic_mode}_nk{args.n_k}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = (Path(args.output_root) / run_label).resolve()
    points = _preset_points(args)
    seeds = _seed_specs(args)
    conv_points = _convergence_points(args)
    plan = {
        "args": vars(args),
        "run_label": run_label,
        "points": [point.model_dump(mode="json") for point in points],
        "convergence_points": [point.model_dump(mode="json") for point in conv_points],
        "seeds": [seed.model_dump(mode="json") for seed in seeds],
        "n_scan_hf_solves": len(points) * len(seeds),
    }
    _write_json(output_dir / "diagnostic_plan.json", plan)
    _write_csv(output_dir / "diagnostic_plan.csv", [point.model_dump(mode="json") for point in points])
    if args.dry_run:
        print(f"Wrote diagnostic dry-run plan to {output_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    final_arrays: dict[str, np.ndarray] = {}
    snapshot_arrays: dict[str, np.ndarray] = {}
    phase_frames: dict[str, np.ndarray] = {}
    records: list[RunRecord] = []
    run_counter = [0]

    if args.diagnostic_mode in {"scan", "all"}:
        _run_scan_mode(
            args,
            points=points,
            seeds=seeds,
            run_rows=run_rows,
            iteration_rows=iteration_rows,
            snapshot_rows=snapshot_rows,
            final_arrays=final_arrays,
            snapshot_arrays=snapshot_arrays,
            phase_frames=phase_frames,
            records=records,
            run_counter=run_counter,
        )
    if args.diagnostic_mode in {"convergence", "all"}:
        _run_convergence_mode(
            args,
            points=conv_points,
            seeds=seeds,
            run_rows=run_rows,
            iteration_rows=iteration_rows,
            snapshot_rows=snapshot_rows,
            final_arrays=final_arrays,
            snapshot_arrays=snapshot_arrays,
            phase_frames=phase_frames,
            records=records,
            run_counter=run_counter,
        )
    if args.diagnostic_mode in {"hysteresis", "all"}:
        linecuts = _linecut_groups(points)
        hysteresis_seed = seeds[0]
        _run_hysteresis_mode(
            args,
            linecuts=linecuts,
            seed=hysteresis_seed,
            run_rows=run_rows,
            iteration_rows=iteration_rows,
            snapshot_rows=snapshot_rows,
            final_arrays=final_arrays,
            snapshot_arrays=snapshot_arrays,
            phase_frames=phase_frames,
            records=records,
            run_counter=run_counter,
        )

    seed_overlap_rows = _compute_seed_overlaps(
        records,
        final_arrays,
        phase_frames,
        n_occ_per_k=int(args.n_occ_per_k),
    )
    neighbor_overlap_rows = _compute_neighbor_overlaps(
        records,
        final_arrays,
        phase_frames,
        n_occ_per_k=int(args.n_occ_per_k),
    )
    run_rows_by_id = {row["run_id"]: row for row in run_rows}
    hysteresis_rows = _compute_hysteresis_rows(
        records,
        final_arrays,
        phase_frames,
        run_rows_by_id,
        n_occ_per_k=int(args.n_occ_per_k),
    )

    _write_csv(output_dir / "runs.csv", run_rows)
    _write_csv(output_dir / "iteration_history.csv", iteration_rows)
    _write_csv(output_dir / "projector_snapshot_manifest.csv", snapshot_rows)
    _write_csv(output_dir / "projector_overlaps_seed_matrix.csv", seed_overlap_rows)
    _write_csv(output_dir / "projector_overlaps_neighbor.csv", neighbor_overlap_rows)
    _write_csv(output_dir / "hysteresis.csv", hysteresis_rows)
    np.savez_compressed(output_dir / "projectors_final.npz", **final_arrays)
    np.savez_compressed(output_dir / "projectors_snapshots.npz", **snapshot_arrays)
    _write_json(
        output_dir / "summary.json",
        {
            "run_label": run_label,
            "n_runs": len(run_rows),
            "n_iteration_rows": len(iteration_rows),
            "n_final_projectors": len(final_arrays),
            "n_snapshot_projectors": len(snapshot_arrays),
            "n_seed_overlap_rows": len(seed_overlap_rows),
            "n_neighbor_overlap_rows": len(neighbor_overlap_rows),
            "n_hysteresis_rows": len(hysteresis_rows),
            "output_dir": str(output_dir),
        },
    )
    if not args.skip_plots:
        _write_plots(output_dir, run_rows, iteration_rows, neighbor_overlap_rows)
    print(f"Wrote local IVC diagnostics to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
