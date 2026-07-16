#!/usr/bin/env python3
"""Run a local Taige scanning-SET displacement/filling sweep."""

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
    TaigeSETWorkflowParams,
    run_taige_set_point,
    taige_interaction_params,
    taige_model_params,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="results/taige_set_nk18_theta3_u0_10")
    parser.add_argument("--u-d", type=float, default=None)
    parser.add_argument("--u-d-min", type=float, default=0.0)
    parser.add_argument("--u-d-max", type=float, default=10.0)
    parser.add_argument("--n-u-d", type=int, default=11)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--theta-deg", type=float, default=3.0)
    parser.add_argument("--n-k", type=int, default=18)
    parser.add_argument("--particle-offset-max", type=int, default=12)

    parser.add_argument("--plane-wave-shell", type=int, default=5)
    parser.add_argument("--n-bands", type=int, default=2)
    parser.add_argument("--n-active-bands-per-valley", type=int, default=2)
    parser.add_argument("--q-mesh", choices=["shell", "full"], default="full")
    parser.add_argument("--q-shell", type=int, default=0)
    parser.add_argument("--local-field-cutoff", type=int, default=4)
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

    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--min-iter", type=int, default=3)
    parser.add_argument("--mixing-method", choices=["linear", "oda"], default="oda")
    parser.add_argument("--mixing", type=float, default=0.45)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--energy-tolerance", type=float, default=1e-10)
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--gaussian-broadening-mev", type=float, default=0.1)
    parser.add_argument("--dos-energy-points", type=int, default=801)
    parser.add_argument("--direct-gap-tolerance-mev", type=float, default=1e-6)

    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--merge-only", action="store_true")
    return parser


def _output_root(args: argparse.Namespace) -> Path:
    path = Path(args.output_root)
    return path if path.is_absolute() else ROOT / path


def _all_displacements(args: argparse.Namespace) -> list[float]:
    if args.u_d is not None:
        return [float(args.u_d)]
    if int(args.n_u_d) < 1:
        raise ValueError("--n-u-d must be positive")
    return [
        float(value)
        for value in np.linspace(float(args.u_d_min), float(args.u_d_max), int(args.n_u_d))
    ]


def _selected_displacements(args: argparse.Namespace) -> list[float]:
    values = _all_displacements(args)
    if args.task_id is None:
        return values
    task_id = int(args.task_id)
    if task_id < 0:
        raise ValueError("--task-id must be nonnegative")
    return [] if task_id >= len(values) else [values[task_id]]


def _label(u_d: float) -> str:
    text = f"{float(u_d):08.4f}".replace("-", "m").replace(".", "p")
    return f"uD_{text}"


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


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _params(args: argparse.Namespace, u_d: float) -> TaigeSETWorkflowParams:
    model = taige_model_params(
        material="mote2",
        theta_deg=float(args.theta_deg),
        u_D=float(u_d),
        plane_wave_shell=int(args.plane_wave_shell),
        n_bands=int(args.n_bands),
        n_active_bands_per_valley=int(args.n_active_bands_per_valley),
    )
    interaction = taige_interaction_params(
        material="mote2",
        include_q0=True,
        q_mesh=args.q_mesh,
        q_shell=int(args.q_shell),
        local_field_cutoff=int(args.local_field_cutoff),
        epsilon=float(args.epsilon),
        gate_distance_nm=float(args.gate_distance_nm),
        smear_length_nm=float(args.smear_length_nm),
        interaction_strength_scale=float(args.v0),
        exchange_scale=float(args.exchange_scale),
        hartree_scale=float(args.hartree_scale),
        vertex_workers=int(args.vertex_workers),
        exchange_workers=int(args.exchange_workers),
        density_vertex_retention=args.density_vertex_retention,
        density_vertex_layout=args.density_vertex_layout,
        exchange_representation=args.exchange_representation,
        form_factor_backend=args.form_factor_backend,
    )
    hf = ContinuumHFParams(
        n_occ_per_k=1,
        max_iter=int(args.max_iter),
        min_iter=int(args.min_iter),
        mixing_method=args.mixing_method,
        mixing=float(args.mixing),
        tolerance=float(args.tolerance),
        energy_tolerance=float(args.energy_tolerance),
        random_seed=int(args.random_seed),
        seed_ordered_weight=1.0,
        seed_random_weight=0.0,
        store_projector_snapshots=False,
    )
    offset_max = int(args.particle_offset_max)
    if offset_max < 1:
        raise ValueError("--particle-offset-max must be at least one")
    return TaigeSETWorkflowParams(
        model=model,
        grid=ContinuumGridParams(n_k=int(args.n_k)),
        interaction=interaction,
        hf=hf,
        particle_offsets=tuple(range(-offset_max, offset_max + 1)),
        gaussian_broadening_mev=float(args.gaussian_broadening_mev),
        dos_energy_points=int(args.dos_energy_points),
        direct_gap_tolerance_mev=float(args.direct_gap_tolerance_mev),
    )


def _point_dir(root: Path, u_d: float) -> Path:
    return root / "points" / _label(u_d)


def _write_point_artifacts(point_dir: Path, result, elapsed_seconds: float) -> dict[str, Any]:
    params = result.params
    u_d = float(params.model.displacement_mev)
    filling_rows = [row.model_dump(mode="json") for row in result.filling_energy_rows]
    mu_rows = [row.model_dump(mode="json") for row in result.chemical_potential_rows]
    kappa_rows = [
        row.model_dump(mode="json") for row in result.inverse_compressibility_rows
    ]
    fixed_rows = [
        result.summary.fixed_vp_plus.model_dump(mode="json"),
        result.summary.fixed_vp_minus.model_dump(mode="json"),
    ]
    _write_csv(point_dir / "filling_energies.csv", filling_rows)
    _write_csv(point_dir / "chemical_potential.csv", mu_rows)
    _write_csv(point_dir / "inverse_compressibility.csv", kappa_rows)
    _write_csv(point_dir / "fixed_vp_references.csv", fixed_rows)
    _write_csv(point_dir / "hf_chern_numbers.csv", list(result.hf_chern_rows))
    _write_csv(point_dir / "dos.csv", list(result.dos_rows))

    arrays: dict[str, np.ndarray] = {
        "fixed_vp_plus_P": np.asarray(result.fixed_vp_plus.P),
        "fixed_vp_plus_H_hf": np.asarray(result.fixed_vp_plus.H_hf),
        "fixed_vp_minus_P": np.asarray(result.fixed_vp_minus.P),
        "fixed_vp_minus_H_hf": np.asarray(result.fixed_vp_minus.H_hf),
    }
    for n_particles, hf_result in sorted(result.filling_results.items()):
        arrays[f"global_N{n_particles}_P"] = np.asarray(hf_result.P)
        arrays[f"global_N{n_particles}_H_hf"] = np.asarray(hf_result.H_hf)
    np.savez_compressed(point_dir / "hf_states.npz", **arrays)

    artifacts = {
        "filling_energies_csv": str(point_dir / "filling_energies.csv"),
        "chemical_potential_csv": str(point_dir / "chemical_potential.csv"),
        "inverse_compressibility_csv": str(point_dir / "inverse_compressibility.csv"),
        "fixed_vp_references_csv": str(point_dir / "fixed_vp_references.csv"),
        "hf_chern_numbers_csv": str(point_dir / "hf_chern_numbers.csv"),
        "dos_csv": str(point_dir / "dos.csv"),
        "hf_states_npz": str(point_dir / "hf_states.npz"),
    }
    payload = {
        "schema": "taige_set_point_v1",
        "u_D_meV": u_d,
        "elapsed_seconds": float(elapsed_seconds),
        "params": params.model_dump(mode="json"),
        "summary": result.summary.model_dump(mode="json"),
        "row": result.summary.as_csv_row() | {"elapsed_seconds": float(elapsed_seconds)},
        "artifacts": artifacts,
    }
    _write_json(point_dir / "point_summary.json", payload)
    return payload["row"]


def run_point(args: argparse.Namespace, root: Path, u_d: float) -> dict[str, Any]:
    point_dir = _point_dir(root, u_d)
    summary_path = point_dir / "point_summary.json"
    if args.skip_existing and summary_path.exists():
        return dict(json.loads(summary_path.read_text())["row"])
    point_dir.mkdir(parents=True, exist_ok=True)
    params = _params(args, u_d)
    _write_json(point_dir / "point_params.json", params.model_dump(mode="json"))
    print(
        f"Running SET point u_D={u_d:g} meV theta={params.model.theta_deg:g} deg "
        f"n_k={params.grid.n_k} fillings={len(params.particle_offsets)}",
        flush=True,
    )
    start = time.perf_counter()
    result = run_taige_set_point(params)
    elapsed = time.perf_counter() - start
    row = _write_point_artifacts(point_dir, result, elapsed)
    print(
        f"Finished u_D={u_d:g}: C={row['hf_band_chern']:.6g} "
        f"fixed_indirect_gap={row['fixed_indirect_gap_meV']:.6g} meV "
        f"SET_gap_intrinsic={row['charge_gap_intrinsic_meV']:.6g} meV "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return row


def merge(root: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted((root / "points").glob("*/point_summary.json")):
        summaries.append(dict(json.loads(path.read_text())["row"]))
    summaries.sort(key=lambda row: float(row["u_D_meV"]))
    _write_csv(root / "set_sweep_summary.csv", summaries)

    tables = {
        "filling_energies.csv": "set_filling_energies.csv",
        "chemical_potential.csv": "set_chemical_potential.csv",
        "inverse_compressibility.csv": "set_inverse_compressibility.csv",
        "hf_chern_numbers.csv": "set_hf_chern_numbers.csv",
        "dos.csv": "set_dos.csv",
    }
    counts: dict[str, int] = {}
    for source_name, output_name in tables.items():
        stacked: list[dict[str, Any]] = []
        for point in sorted((root / "points").glob("*")):
            rows = _read_csv(point / source_name)
            if not rows:
                continue
            payload = json.loads((point / "point_summary.json").read_text())
            u_d = float(payload["u_D_meV"])
            stacked.extend({"u_D_meV": u_d} | row for row in rows)
        _write_csv(root / output_name, stacked)
        counts[output_name] = len(stacked)
    _write_json(
        root / "set_sweep_manifest.json",
        {
            "schema": "taige_set_sweep_v1",
            "n_points": len(summaries),
            "table_row_counts": counts,
        },
    )
    return summaries


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = _output_root(args)
    root.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        rows = merge(root)
        print(f"Merged {len(rows)} SET point(s) into {root}")
        return 0
    values = _selected_displacements(args)
    plan_rows = [
        {"task_id": index, "u_D_meV": value, "point_dir": str(_point_dir(root, value))}
        for index, value in enumerate(_all_displacements(args))
    ]
    _write_csv(root / "set_sweep_plan.csv", plan_rows)
    _write_json(root / "set_sweep_plan.json", {"points": plan_rows, "args": vars(args)})
    if args.dry_run:
        print(f"Wrote SET sweep plan with {len(plan_rows)} point(s) to {root}")
        return 0
    if not values:
        print("Selected task is outside the displacement grid; nothing to do.")
        return 0
    for value in values:
        run_point(args, root, value)
    if args.task_id is None:
        rows = merge(root)
        print(f"Merged {len(rows)} completed SET point(s) into {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
