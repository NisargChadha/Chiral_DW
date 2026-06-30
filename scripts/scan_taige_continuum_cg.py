#!/usr/bin/env python3
"""Sweep Taige-continuum symmetric-HF cG over displacement and twist angle."""

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/taige_continuum_cg_sweep")

    parser.add_argument("--u-d", type=float, default=None, help="Run one explicit displacement value in meV.")
    parser.add_argument("--theta-deg", type=float, default=None, help="Run one explicit twist angle in degrees.")
    parser.add_argument("--u-d-min", type=float, default=0.0)
    parser.add_argument("--u-d-max", type=float, default=20.0)
    parser.add_argument("--n-u-d", type=int, default=21)
    parser.add_argument("--theta-min-deg", type=float, default=2.0)
    parser.add_argument("--theta-max-deg", type=float, default=5.0)
    parser.add_argument("--n-twist", type=int, default=21)
    parser.add_argument("--task-id", type=int, default=None, help="SLURM-style flat grid index.")

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
    parser.add_argument(
        "--exchange-representation",
        choices=["auto", "dense", "valley_sector"],
        default="auto",
        help="Exchange representation; auto uses valley-sector exchange for compact Taige vertices.",
    )
    parser.add_argument(
        "--form-factor-backend",
        choices=["auto", "scalar", "cached_gather", "vectorized"],
        default="auto",
        help="Taige form-factor backend; auto uses vectorized form-factor construction.",
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
    parser.add_argument("--no-finite-q-ivc", action="store_true", help="Skip the Taige IVC- finite-Q diagnostic branch.")
    parser.add_argument(
        "--ivc-branch-policy",
        choices=["lower-energy", "q0"],
        default="lower-energy",
        help="Choose the IVC branch used for interpolation; lower-energy compares Q=0 and finite-Q IVC.",
    )
    parser.add_argument(
        "--ivc-branch-tie-atol",
        type=float,
        default=1e-9,
        help="Energy-per-cell tolerance for treating Q=0 and finite-Q IVC as tied; ties choose Q=0.",
    )
    parser.add_argument(
        "--allow-texture-in-ivc-ground-state",
        action="store_true",
        help="Keep cG/K(theta)/trial texture diagnostics even when IVC is below the VP reference.",
    )
    parser.add_argument(
        "--texture-energy-tie-atol",
        type=float,
        default=1e-9,
        help="Energy-per-cell tolerance for treating IVC and VP as tied; ties keep texture diagnostics.",
    )
    parser.add_argument("--write-hf-path-spectra", action="store_true", help="Write optional fixed-density HF path spectra.")
    parser.add_argument("--hf-path-n-per-segment", type=int, default=36)

    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write the selected scan plan without running HF.")
    parser.add_argument("--merge-only", action="store_true", help="Merge existing point summaries into sweep.csv/json.")
    return parser


def _linspace_points(args: argparse.Namespace) -> list[TaigeSweepPoint]:
    if (args.u_d is None) ^ (args.theta_deg is None):
        raise ValueError("--u-d and --theta-deg must be supplied together for an explicit single point")
    if args.u_d is not None and args.theta_deg is not None:
        return [
            TaigeSweepPoint(
                u_index=0,
                theta_index=0,
                u_D=float(args.u_d),
                theta_deg=float(args.theta_deg),
            )
        ]
    if args.n_u_d < 1 or args.n_twist < 1:
        raise ValueError("n-u-d and n-twist must both be positive")
    u_values = np.linspace(float(args.u_d_min), float(args.u_d_max), int(args.n_u_d))
    theta_values = np.linspace(float(args.theta_min_deg), float(args.theta_max_deg), int(args.n_twist))
    return [
        TaigeSweepPoint(u_index=iu, theta_index=it, u_D=float(u), theta_deg=float(theta))
        for iu, u in enumerate(u_values)
        for it, theta in enumerate(theta_values)
    ]


def _selected_points(args: argparse.Namespace) -> list[TaigeSweepPoint]:
    points = _linspace_points(args)
    if args.task_id is None:
        return points
    task_id = int(args.task_id)
    if task_id < 0:
        raise ValueError("--task-id must be nonnegative")
    if task_id >= len(points):
        print(f"Task {task_id} is outside mesh size {len(points)}; exiting.")
        return []
    return [points[task_id]]


def _params_for_point(args: argparse.Namespace, point: TaigeSweepPoint, point_dir: Path) -> ContinuumWorkflowParams:
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
            "exchange_representation": args.exchange_representation,
            "form_factor_backend": args.form_factor_backend,
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
        grid=ContinuumGridParams(n_k=args.n_k),
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


def _point_dir(output_root: Path, point: TaigeSweepPoint) -> Path:
    return output_root / "points" / point.label


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


def _stack_point_table(output_root: Path, filename: str, output_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((output_root / "points").glob(f"*/{filename}")):
        rows.extend(_read_csv(path))
    _write_csv(output_root / output_name, rows)
    return rows


def merge_point_summaries(output_root: Path) -> list[dict[str, Any]]:
    rows = [
        _load_point_summary(path)
        for path in sorted((output_root / "points").glob("*/point_summary.json"))
    ]
    rows.sort(key=lambda row: (int(row["u_index"]), int(row["theta_index"])))
    _write_csv(output_root / "sweep.csv", rows)
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
        output_root / "sweep.json",
        {
            "rows": rows,
            "n_points": len(rows),
            "stacked_counts": stacked_counts,
            "tables": {
                "sweep_csv": str(output_root / "sweep.csv"),
                "trial_theta_csv": str(output_root / "sweep_trial_theta.csv"),
                "reference_energies_csv": str(output_root / "sweep_reference_energies.csv"),
                "noninteracting_chern_numbers_csv": str(output_root / "sweep_noninteracting_chern_numbers.csv"),
                "hf_chern_numbers_csv": str(output_root / "sweep_hf_chern_numbers.csv"),
                "hf_path_spectra_csv": str(output_root / "sweep_hf_path_spectra.csv"),
            },
        },
    )
    return rows


def _write_plan(output_root: Path, points: list[TaigeSweepPoint], args: argparse.Namespace) -> None:
    rows = [
        point.model_dump(mode="json") | {"point_dir": str(_point_dir(output_root, point))}
        for point in points
    ]
    _write_csv(output_root / "sweep_plan.csv", rows)
    _write_json(
        output_root / "sweep_plan.json",
        {
            "points": rows,
            "n_points": len(rows),
            "args": {
                key: value
                for key, value in vars(args).items()
                if isinstance(value, (str, int, float, bool, type(None)))
            },
        },
    )


def _diagnostic_params(args: argparse.Namespace) -> TaigeSweepDiagnosticsParams:
    branch_policy = str(args.ivc_branch_policy).replace("-", "_")
    if args.no_finite_q_ivc:
        branch_policy = "q0"
    return TaigeSweepDiagnosticsParams(
        compute_chern_numbers=not args.no_chern,
        compute_finite_q_ivc=not args.no_finite_q_ivc,
        ivc_branch_policy=branch_policy,
        ivc_branch_tie_atol=float(args.ivc_branch_tie_atol),
        nan_texture_when_ivc_lower=not args.allow_texture_in_ivc_ground_state,
        texture_energy_tie_atol=float(args.texture_energy_tie_atol),
        write_hf_path_spectra=bool(args.write_hf_path_spectra),
        hf_path_n_per_segment=args.hf_path_n_per_segment,
    )


def _write_point_diagnostic_tables(point_dir: Path, diagnostics) -> None:
    _write_csv(point_dir / "trial_theta.csv", diagnostics.trial_theta_rows)
    _write_csv(point_dir / "reference_energies.csv", diagnostics.reference_energy_rows)
    if diagnostics.summary.chern_enabled:
        _write_csv(point_dir / "noninteracting_chern_numbers.csv", diagnostics.noninteracting_chern_rows)
        _write_csv(point_dir / "hf_chern_numbers.csv", diagnostics.hf_chern_rows)
    if diagnostics.summary.hf_path_spectra_csv is not None:
        _write_csv(point_dir / "hf_path_spectra.csv", diagnostics.hf_path_spectrum_rows)


def run_point(args: argparse.Namespace, output_root: Path, point: TaigeSweepPoint) -> dict[str, Any]:
    point_dir = _point_dir(output_root, point)
    point_summary = point_dir / "point_summary.json"
    if args.skip_existing and point_summary.exists():
        print(f"Skipping existing {point.label}: {point_summary}")
        return _load_point_summary(point_summary)

    params = _params_for_point(args, point, point_dir)
    _write_json(point_dir / "point_params.json", params.model_dump(mode="json"))
    print(
        "Running Taige continuum HF cG "
        f"u_D={point.u_D:.8g} meV theta={point.theta_deg:.8g} deg "
        f"n_k={args.n_k} q_mesh={args.q_mesh} q_shell={args.q_shell} "
        f"local_field_cutoff={args.local_field_cutoff} "
        f"vertex_workers={args.vertex_workers} "
        f"exchange_workers={args.exchange_workers} "
        f"exchange_representation={args.exchange_representation} "
        f"form_factor_backend={args.form_factor_backend}"
    )
    start = time.perf_counter()
    diagnostic_controls = _diagnostic_params(args)
    result = run_taige_branch_selected_symmetric_hf_workflow(
        params,
        finite_q_enabled=diagnostic_controls.compute_finite_q_ivc,
        ivc_branch_policy=diagnostic_controls.ivc_branch_policy,
        tie_atol=diagnostic_controls.ivc_branch_tie_atol,
        suppress_texture_when_ivc_below_vp=diagnostic_controls.nan_texture_when_ivc_lower,
        texture_energy_tie_atol=diagnostic_controls.texture_energy_tie_atol,
        write_outputs=True,
    )
    diagnostics = build_taige_sweep_diagnostics(
        point=point,
        workflow_result=result,
        controls=diagnostic_controls,
        elapsed_seconds=0.0,
        point_dir=point_dir,
    )
    seconds = time.perf_counter() - start
    _write_point_diagnostic_tables(point_dir, diagnostics)
    row = diagnostics.summary.model_copy(update={"elapsed_seconds": seconds}).as_csv_row()
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
        f"selected_ivc_branch={row['selected_ivc_branch']} elapsed={seconds:.1f}s"
    )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.no_finite_q_ivc and int(args.n_k) % 6:
        parser.error("finite-Q IVC diagnostics require --n-k divisible by 6; use --no-finite-q-ivc to skip them")
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        rows = merge_point_summaries(output_root)
        print(f"Merged {len(rows)} point summaries into {output_root / 'sweep.csv'}")
        return 0

    points = _selected_points(args)
    _write_plan(output_root, points, args)
    if args.dry_run:
        print(f"Wrote dry-run plan with {len(points)} selected point(s) to {output_root}")
        return 0
    if not points:
        return 0

    rows = [run_point(args, output_root, point) for point in points]
    if args.task_id is None:
        rows = merge_point_summaries(output_root)
        print(f"Wrote serial sweep summary to {output_root / 'sweep.csv'}")
    else:
        print("Single task complete. Run with --merge-only after the array finishes to build sweep.csv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
