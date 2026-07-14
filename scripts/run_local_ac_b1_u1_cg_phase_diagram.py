#!/usr/bin/env python3
"""Run the local conjugate-AC projected-HF b1-u1 cG phase-diagram test."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


ROOT = Path(__file__).resolve().parents[1]
SCAN_SCRIPT = ROOT / "scripts" / "scan_ac_projected_hf_b1_u1.py"
PLOT_SCRIPT = ROOT / "Plots" / "plot_ac_b1_u1_cg_phase_diagram.py"


class LocalACB1U1SweepParams(BaseModel):
    """Frozen controls for the laptop-scale conjugate-AC validation sweep."""

    model_config = ConfigDict(frozen=True)

    output_root: Path = Path("results/ac_b1_u1_cg_dual_gate_local_nk12_nll5_v0p1_grid11")
    plot_output: Path = Path("Plots/figures/ac_b1_u1_cg_dual_gate_local_nk12_nll5_v0p1_grid11.png")
    b1_min: float = -0.1
    b1_max: float = 0.1
    n_b1: int = Field(default=11, ge=2)
    u1_min: float = -0.1
    u1_max: float = 0.1
    n_u1: int = Field(default=11, ge=2)
    n_k: int = Field(default=12, ge=2)
    n_ll: int = Field(default=5, ge=1)
    active_band: int = Field(default=0, ge=0)
    v0_over_omega_c: float = Field(default=0.1, ge=0.0)
    gate_distance: float = Field(default=2.0, gt=0.0)
    q_shell: int = Field(default=1, ge=0)
    local_field_cutoff: int = Field(default=1, ge=0)
    band_diagnostics_n_k: int = Field(default=9, ge=2)
    max_iter: int = Field(default=800, ge=1)
    min_iter: int = Field(default=2, ge=0)
    n_theta: int = Field(default=81, ge=3)
    n_phi: int = Field(default=5, ge=2)
    phi_step: float = Field(default=0.2, gt=0.0)
    workers: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _ranges_and_active_band_are_valid(self) -> "LocalACB1U1SweepParams":
        if self.b1_max <= self.b1_min:
            raise ValueError("b1_max must exceed b1_min")
        if self.u1_max <= self.u1_min:
            raise ValueError("u1_max must exceed u1_min")
        if self.active_band >= self.n_ll:
            raise ValueError("active_band must be smaller than n_ll")
        return self

    def resolved_output_root(self) -> Path:
        return self.output_root if self.output_root.is_absolute() else ROOT / self.output_root

    def resolved_plot_output(self) -> Path:
        return self.plot_output if self.plot_output.is_absolute() else ROOT / self.plot_output

    def scan_command(
        self,
        *,
        dry_run: bool,
        task_id: int | None = None,
        no_write_plan: bool = False,
    ) -> list[str]:
        command = [
            sys.executable,
            str(SCAN_SCRIPT),
            "--output-root",
            str(self.resolved_output_root()),
            "--b1-min",
            str(self.b1_min),
            "--b1-max",
            str(self.b1_max),
            "--n-b1",
            str(self.n_b1),
            "--u1-min",
            str(self.u1_min),
            "--u1-max",
            str(self.u1_max),
            "--n-u1",
            str(self.n_u1),
            "--n-k",
            str(self.n_k),
            "--n-ll",
            str(self.n_ll),
            "--active-band",
            str(self.active_band),
            "--coulomb-kind",
            "dimensionless_dual_gate",
            "--v0",
            str(self.v0_over_omega_c),
            "--gate-distance",
            str(self.gate_distance),
            "--q-shell",
            str(self.q_shell),
            "--local-field-cutoff",
            str(self.local_field_cutoff),
            "--band-diagnostics-n-k",
            str(self.band_diagnostics_n_k),
            "--max-iter",
            str(self.max_iter),
            "--min-iter",
            str(self.min_iter),
            "--mixing-method",
            "oda",
            "--n-theta",
            str(self.n_theta),
            "--n-phi",
            str(self.n_phi),
            "--phi-step",
            str(self.phi_step),
            "--skip-existing",
        ]
        if dry_run:
            command.append("--dry-run")
        if task_id is not None:
            command.extend(["--task-id", str(task_id)])
        if no_write_plan:
            command.append("--no-write-plan")
        return command

    def merge_command(self) -> list[str]:
        return [
            sys.executable,
            str(SCAN_SCRIPT),
            "--output-root",
            str(self.resolved_output_root()),
            "--merge-only",
        ]

    def plot_command(self) -> list[str]:
        return [
            sys.executable,
            str(PLOT_SCRIPT),
            "--input-csv",
            str(self.resolved_output_root() / "sweep.csv"),
            "--output",
            str(self.resolved_plot_output()),
        ]


def _build_parser() -> argparse.ArgumentParser:
    defaults = LocalACB1U1SweepParams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument("--plot-output", type=Path, default=defaults.plot_output)
    parser.add_argument("--n-b1", type=int, default=defaults.n_b1)
    parser.add_argument("--n-u1", type=int, default=defaults.n_u1)
    parser.add_argument("--workers", type=int, default=defaults.workers)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser


def _write_run_config(params: LocalACB1U1SweepParams, *, dry_run: bool) -> Path:
    output_root = params.resolved_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "local_run_config.json"
    path.write_text(
        json.dumps(
            {
                "params": params.model_dump(mode="json"),
                "physics": {
                    "model": "conjugate finite-LL Aharonov-Casher bands",
                    "projection": "lowest AC band per valley",
                    "hf_references": "symmetry-constrained VP+, VP-, and T-prime IVC",
                    "variational_path": "convex interpolation of full HF Hamiltonians",
                    "interaction": "dimensionless dual-gate Coulomb",
                    "interaction_strength_convention": "V0/omega_c",
                },
                "scan_command": params.scan_command(dry_run=dry_run),
                "merge_command": params.merge_command(),
                "plot_command": params.plot_command(),
                "dry_run": bool(dry_run),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return path


def _worker_environment() -> dict[str, str]:
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def _run_parallel_sweep(params: LocalACB1U1SweepParams) -> None:
    subprocess.run(params.scan_command(dry_run=True), cwd=ROOT, check=True)
    log_dir = params.resolved_output_root() / "local_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    n_tasks = params.n_b1 * params.n_u1

    def run_task(task_id: int) -> tuple[int, Path]:
        log_path = log_dir / f"task_{task_id:03d}.log"
        command = params.scan_command(
            dry_run=False,
            task_id=task_id,
            no_write_plan=True,
        )
        with log_path.open("w") as handle:
            subprocess.run(
                command,
                cwd=ROOT,
                env=_worker_environment(),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return task_id, log_path

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=params.workers) as executor:
        futures = [executor.submit(run_task, task_id) for task_id in range(n_tasks)]
        for future in concurrent.futures.as_completed(futures):
            task_id, log_path = future.result()
            completed += 1
            print(
                f"Completed AC point {completed}/{n_tasks} (task {task_id}); log: {log_path}",
                flush=True,
            )
    subprocess.run(params.merge_command(), cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    params = LocalACB1U1SweepParams(
        output_root=args.output_root,
        plot_output=args.plot_output,
        n_b1=args.n_b1,
        n_u1=args.n_u1,
        workers=args.workers,
    )
    config_path = _write_run_config(params, dry_run=args.dry_run)
    print(f"Wrote local AC sweep configuration to {config_path}", flush=True)
    if args.dry_run:
        print("Running:", " ".join(params.scan_command(dry_run=True)), flush=True)
        subprocess.run(params.scan_command(dry_run=True), cwd=ROOT, check=True)
        print("Dry run complete; no HF points or phase diagram were generated.", flush=True)
        return 0
    if params.workers == 1:
        print("Running serial sweep:", " ".join(params.scan_command(dry_run=False)), flush=True)
        subprocess.run(
            params.scan_command(dry_run=False),
            cwd=ROOT,
            env=_worker_environment(),
            check=True,
        )
    else:
        print(f"Running local sweep with {params.workers} workers.", flush=True)
        _run_parallel_sweep(params)
    if not args.no_plot:
        print("Rendering:", " ".join(params.plot_command()), flush=True)
        subprocess.run(params.plot_command(), cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
