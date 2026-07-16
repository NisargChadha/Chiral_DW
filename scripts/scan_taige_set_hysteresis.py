#!/usr/bin/env python3
"""Run and merge branch-resolved Taige scanning-SET hysteresis continuations."""

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

from chiral_dw.continuum import (  # noqa: E402
    SETBranchPointSummary,
    TaigeSETWorkflowParams,
    run_taige_set_hysteresis_branch_point,
    select_set_hysteresis_envelope,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="results/taige_set_nk18_theta3_u5_6_hysteresis20",
    )
    parser.add_argument("--direction", choices=["up", "down", "merge"], required=True)
    parser.add_argument("--seed-point-dir", default=None)
    parser.add_argument("--u-d-min", type=float, default=5.0)
    parser.add_argument("--u-d-max", type=float, default=6.0)
    parser.add_argument("--n-u-d", type=int, default=20)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument(
        "--filling-workers",
        type=int,
        default=None,
        help="Concurrent N-1, N, N+1 solves sharing one continuum backend.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Limit newly solved points for staged smoke tests.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _root(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _label(u_d: float) -> str:
    text = f"{float(u_d):08.4f}".replace("-", "m").replace(".", "p")
    return f"uD_{text}"


def _grid(args: argparse.Namespace) -> np.ndarray:
    if int(args.n_u_d) < 2:
        raise ValueError("--n-u-d must be at least two")
    if float(args.u_d_max) <= float(args.u_d_min):
        raise ValueError("--u-d-max must exceed --u-d-min")
    return np.linspace(float(args.u_d_min), float(args.u_d_max), int(args.n_u_d))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _point_dir(root: Path, direction: str, u_d: float) -> Path:
    return root / "branches" / direction / _label(u_d)


def _load_template(
    seed_point_dir: Path,
    max_iter: int | None,
    filling_workers: int | None,
) -> TaigeSETWorkflowParams:
    params_path = seed_point_dir / "point_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing seed parameters: {params_path}")
    params = TaigeSETWorkflowParams.model_validate_json(params_path.read_text())
    hf = (
        params.hf
        if max_iter is None
        else params.hf.model_copy(update={"max_iter": int(max_iter)})
    )
    workers = (
        int(params.filling_workers)
        if filling_workers is None
        else int(filling_workers)
    )
    if workers < 1:
        raise ValueError("--filling-workers must be at least one")
    return params.model_copy(
        update={
            "hf": hf,
            "particle_offsets": (-1, 0, 1),
            "filling_workers": workers,
        }
    )


def _params_at(template: TaigeSETWorkflowParams, u_d: float) -> TaigeSETWorkflowParams:
    model = template.model.model_copy(update={"displacement_mev": float(u_d)})
    return template.model_copy(update={"model": model})


def _load_projectors(path: Path, n_cells: int) -> dict[int, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing hysteresis seed states: {path}")
    archive = np.load(path)
    targets = (n_cells - 1, n_cells, n_cells + 1)
    missing = [target for target in targets if f"global_N{target}_P" not in archive]
    if missing:
        raise ValueError(f"Seed archive is missing projectors for N={missing}")
    return {
        target: np.asarray(archive[f"global_N{target}_P"], dtype=complex)
        for target in targets
    }


def _write_branch_point(
    point_dir: Path,
    result,
    *,
    elapsed_seconds: float,
) -> None:
    point_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "direction": result.summary.direction,
            "u_D_meV": result.summary.u_D_mev,
        }
        | row.model_dump(mode="json")
        for row in result.summary.filling_energy_rows
    ]
    _write_csv(point_dir / "filling_energies.csv", rows)
    arrays: dict[str, np.ndarray] = {}
    for n_particles, hf_result in sorted(result.filling_results.items()):
        arrays[f"global_N{n_particles}_P"] = np.asarray(hf_result.P)
        arrays[f"global_N{n_particles}_H_hf"] = np.asarray(hf_result.H_hf)
    np.savez_compressed(point_dir / "hf_states.npz", **arrays)
    _write_json(
        point_dir / "point_summary.json",
        {
            "schema": "taige_set_hysteresis_branch_point_v1",
            "elapsed_seconds": float(elapsed_seconds),
            "params": result.params.model_dump(mode="json"),
            "summary": result.summary.model_dump(mode="json"),
            "artifacts": {
                "filling_energies_csv": str(point_dir / "filling_energies.csv"),
                "hf_states_npz": str(point_dir / "hf_states.npz"),
            },
        },
    )


def run_branch(args: argparse.Namespace, root: Path) -> int:
    if args.seed_point_dir is None:
        raise ValueError("--seed-point-dir is required for up/down continuations")
    seed_dir = _root(args.seed_point_dir)
    template = _load_template(seed_dir, args.max_iter, args.filling_workers)
    values = _grid(args)
    if args.direction == "down":
        values = values[::-1]
    seed_u_d = float(template.model.displacement_mev)
    if not np.isclose(seed_u_d, float(values[0]), atol=1e-9, rtol=0.0):
        raise ValueError(
            f"seed displacement {seed_u_d:g} meV does not match the "
            f"{args.direction} endpoint {float(values[0]):g} meV"
        )

    n_cells = int(template.grid.n_k) ** 2
    projectors = _load_projectors(seed_dir / "hf_states.npz", n_cells)
    plan_rows = [
        {
            "sequence_index": index,
            "direction": args.direction,
            "u_D_meV": float(u_d),
            "point_dir": str(_point_dir(root, args.direction, float(u_d))),
        }
        for index, u_d in enumerate(values)
    ]
    _write_csv(root / f"hysteresis_plan_{args.direction}.csv", plan_rows)
    _write_json(
        root / f"hysteresis_plan_{args.direction}.json",
        {
            "schema": "taige_set_hysteresis_plan_v1",
            "direction": args.direction,
            "seed_point_dir": str(seed_dir),
            "template_params": template.model_dump(mode="json"),
            "points": plan_rows,
        },
    )
    if args.dry_run:
        print(
            f"Wrote {args.direction} hysteresis plan with {len(values)} points to {root}"
        )
        return 0

    newly_solved = 0
    for u_d in values:
        point_dir = _point_dir(root, args.direction, float(u_d))
        summary_path = point_dir / "point_summary.json"
        states_path = point_dir / "hf_states.npz"
        if args.skip_existing and summary_path.exists() and states_path.exists():
            projectors = _load_projectors(states_path, n_cells)
            print(f"Resumed {args.direction} branch at u_D={float(u_d):g} meV", flush=True)
            continue
        if args.max_points is not None and newly_solved >= int(args.max_points):
            break

        params = _params_at(template, float(u_d))
        print(
            f"Running SET hysteresis {args.direction} u_D={float(u_d):g} meV "
            f"n_k={params.grid.n_k} N={n_cells - 1},{n_cells},{n_cells + 1}",
            flush=True,
        )
        start = time.perf_counter()
        result = run_taige_set_hysteresis_branch_point(
            params,
            projectors,
            direction=args.direction,
        )
        elapsed = time.perf_counter() - start
        _write_branch_point(point_dir, result, elapsed_seconds=elapsed)
        projectors = {
            n_particles: np.asarray(hf_result.P)
            for n_particles, hf_result in result.filling_results.items()
        }
        topology = result.summary.neutral_topology
        print(
            f"Finished {args.direction} u_D={float(u_d):g}: "
            f"C={topology.hf_band_chern:.6g} "
            f"indirect={topology.band_validity.indirect_gap_mev:.6g} meV "
            f"converged={result.summary.all_fillings_converged} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        newly_solved += 1
    return 0


def _load_branch_points(root: Path, direction: str) -> dict[str, SETBranchPointSummary]:
    rows: dict[str, SETBranchPointSummary] = {}
    for path in sorted((root / "branches" / direction).glob("*/point_summary.json")):
        payload = json.loads(path.read_text())
        summary = SETBranchPointSummary.model_validate(payload["summary"])
        rows[path.parent.name] = summary
    return rows


def merge(args: argparse.Namespace, root: Path) -> int:
    up = _load_branch_points(root, "up")
    down = _load_branch_points(root, "down")
    expected = {_label(float(value)) for value in _grid(args)}
    missing_up = sorted(expected - set(up))
    missing_down = sorted(expected - set(down))
    if missing_up or missing_down:
        raise ValueError(
            f"Cannot merge incomplete hysteresis branches: "
            f"missing_up={missing_up}, missing_down={missing_down}"
        )

    selected_rows: list[dict[str, Any]] = []
    selected_energy_rows: list[dict[str, Any]] = []
    branch_energy_rows: list[dict[str, Any]] = []
    for label in sorted(expected, key=lambda item: up[item].u_D_mev):
        up_summary = up[label]
        down_summary = down[label]
        if not np.isclose(up_summary.u_D_mev, down_summary.u_D_mev):
            raise ValueError(f"up/down displacement mismatch at {label}")
        n_cells = int(up_summary.n_cells)
        envelope = select_set_hysteresis_envelope(
            up_summary.filling_energy_rows,
            down_summary.filling_energy_rows,
            n_particles_filling_one=n_cells,
        )
        direction_center = envelope.selected_direction_by_particles[n_cells]
        selected_topology = (
            up_summary.neutral_topology
            if direction_center == "up"
            else down_summary.neutral_topology
        )
        gap = envelope.set_gap
        row: dict[str, Any] = {
            "u_D_meV": float(up_summary.u_D_mev),
            "selected_direction_Nminus": envelope.selected_direction_by_particles[
                n_cells - 1
            ],
            "selected_direction_N0": direction_center,
            "selected_direction_Nplus": envelope.selected_direction_by_particles[
                n_cells + 1
            ],
            "down_minus_up_intrinsic_Nminus_meV": (
                envelope.down_minus_up_intrinsic_energy_mev[n_cells - 1]
            ),
            "down_minus_up_intrinsic_N0_meV": (
                envelope.down_minus_up_intrinsic_energy_mev[n_cells]
            ),
            "down_minus_up_intrinsic_Nplus_meV": (
                envelope.down_minus_up_intrinsic_energy_mev[n_cells + 1]
            ),
            "charge_gap_raw_meV": gap.charge_gap_raw_mev,
            "charge_gap_intrinsic_meV": gap.charge_gap_intrinsic_mev,
            "uniform_capacitance_contribution_meV": (
                gap.charge_gap_raw_mev - gap.charge_gap_intrinsic_mev
            ),
            "mu_minus_hole_raw_meV": gap.mu_minus_hole_raw_mev,
            "mu_plus_hole_raw_meV": gap.mu_plus_hole_raw_mev,
            "mu_minus_hole_intrinsic_meV": gap.mu_minus_hole_intrinsic_mev,
            "mu_plus_hole_intrinsic_meV": gap.mu_plus_hole_intrinsic_mev,
            "selected_hf_band_chern": selected_topology.hf_band_chern,
            "selected_direct_gap_meV": selected_topology.band_validity.direct_gap_mev,
            "selected_indirect_gap_meV": selected_topology.band_validity.indirect_gap_mev,
            "selected_fixed_per_k_valid_insulator": (
                selected_topology.band_validity.valid_fixed_per_k_insulator
            ),
            "selected_chern_physically_interpretable": (
                selected_topology.chern_physically_interpretable
            ),
            "up_hf_band_chern": up_summary.neutral_topology.hf_band_chern,
            "down_hf_band_chern": down_summary.neutral_topology.hf_band_chern,
            "up_all_fillings_converged": up_summary.all_fillings_converged,
            "down_all_fillings_converged": down_summary.all_fillings_converged,
        }
        selected_rows.append(row)
        for energy_row in envelope.selected_energy_rows:
            selected_energy_rows.append(
                {
                    "u_D_meV": float(up_summary.u_D_mev),
                    "selected_direction": envelope.selected_direction_by_particles[
                        energy_row.n_particles
                    ],
                }
                | energy_row.model_dump(mode="json")
            )
        for summary in (up_summary, down_summary):
            for energy_row in summary.filling_energy_rows:
                branch_energy_rows.append(
                    {
                        "u_D_meV": float(summary.u_D_mev),
                        "direction": summary.direction,
                        "hf_band_chern_N0": summary.neutral_topology.hf_band_chern,
                    }
                    | energy_row.model_dump(mode="json")
                )

    _write_csv(root / "set_hysteresis_selected.csv", selected_rows)
    _write_csv(root / "set_hysteresis_selected_filling_energies.csv", selected_energy_rows)
    _write_csv(root / "set_hysteresis_branch_filling_energies.csv", branch_energy_rows)
    _write_json(
        root / "set_hysteresis_manifest.json",
        {
            "schema": "taige_set_hysteresis_merged_v1",
            "n_displacements": len(selected_rows),
            "n_branch_energy_rows": len(branch_energy_rows),
            "n_selected_energy_rows": len(selected_energy_rows),
            "selection_rule": (
                "lowest converged intrinsic HF energy selected independently at each N; "
                "uniform Hartree capacitance then retained in raw SET differences"
            ),
            "artifacts": {
                "selected_csv": str(root / "set_hysteresis_selected.csv"),
                "selected_filling_energies_csv": str(
                    root / "set_hysteresis_selected_filling_energies.csv"
                ),
                "branch_filling_energies_csv": str(
                    root / "set_hysteresis_branch_filling_energies.csv"
                ),
            },
        },
    )
    print(f"Merged {len(selected_rows)} SET hysteresis points into {root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = _root(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    if args.direction == "merge":
        return merge(args, root)
    return run_branch(args, root)


if __name__ == "__main__":
    raise SystemExit(main())
