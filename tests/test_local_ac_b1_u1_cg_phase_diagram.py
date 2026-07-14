import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_local_ac_b1_u1_cg_phase_diagram.py"
PLOTTER = ROOT / "Plots" / "plot_ac_b1_u1_cg_phase_diagram.py"
DIAGNOSTIC_PLOTTER = ROOT / "Plots" / "plot_ac_b1_u1_cg_gaugefixed_diagnostics.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_local_ac_runner_defaults_match_requested_physics():
    module = _load_module(RUNNER, "local_ac_runner")
    params = module.LocalACB1U1SweepParams()
    command = params.scan_command(dry_run=False)

    assert params.b1_min == -0.1
    assert params.b1_max == 0.1
    assert params.u1_min == -0.1
    assert params.u1_max == 0.1
    assert params.n_b1 == 11
    assert params.n_u1 == 11
    assert params.n_k == 12
    assert params.n_ll == 5
    assert params.active_band == 0
    assert params.v0_over_omega_c == 0.1
    assert params.workers == 2
    assert command[command.index("--coulomb-kind") + 1] == "dimensionless_dual_gate"
    assert command[command.index("--mixing-method") + 1] == "oda"
    assert "--skip-existing" in command
    worker_command = params.scan_command(dry_run=False, task_id=7, no_write_plan=True)
    assert worker_command[worker_command.index("--task-id") + 1] == "7"
    assert "--no-write-plan" in worker_command


def test_local_ac_runner_dry_run_writes_121_point_plan(tmp_path: Path):
    output_root = tmp_path / "local_ac"
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        cwd=ROOT,
        check=True,
    )

    config = json.loads((output_root / "local_run_config.json").read_text())
    plan = json.loads((output_root / "sweep_plan.json").read_text())
    assert config["params"]["v0_over_omega_c"] == 0.1
    assert config["physics"]["interaction"] == "dimensionless dual-gate Coulomb"
    assert config["physics"]["projection"] == "lowest AC band per valley"
    assert plan["n_points"] == 121
    assert plan["args"]["n_k"] == 12
    assert plan["args"]["n_ll"] == 5
    assert plan["args"]["active_band"] == 0
    assert plan["args"]["v0"] == 0.1
    assert plan["args"]["coulomb_kind"] == "dimensionless_dual_gate"


def test_ac_cg_plotter_writes_png_pdf_csv_and_summary(tmp_path: Path):
    input_csv = tmp_path / "sweep.csv"
    output = tmp_path / "phase.png"
    fields = [
        "b_index",
        "u_index",
        "b1",
        "u1",
        "cG",
        "hf_all_converged",
        "response_status",
        "band_chern",
        "min_direct_gap",
        "ivc_minus_best_vp_energy_per_cell",
        "n_k",
        "n_ll",
        "active_band",
        "v0_over_omega_c",
    ]
    with input_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for ib, b1 in enumerate((-0.04, 0.0, 0.02)):
            for iu, u1 in enumerate((-0.04, 0.0, 0.02)):
                writer.writerow(
                    {
                        "b_index": ib,
                        "u_index": iu,
                        "b1": "-0.04000000000000001" if b1 == -0.04 and iu == 0 else b1,
                        "u1": "0.01999999999999999" if u1 == 0.02 and ib == 0 else u1,
                        "cG": -0.08 + 0.1 * b1 - 0.05 * u1,
                        "hf_all_converged": True,
                        "response_status": "ok",
                        "band_chern": 1.0,
                        "min_direct_gap": 0.8,
                        "ivc_minus_best_vp_energy_per_cell": 0.01,
                        "n_k": 12,
                        "n_ll": 5,
                        "active_band": 0,
                        "v0_over_omega_c": 0.1,
                    }
                )

    subprocess.run(
        [sys.executable, str(PLOTTER), "--input-csv", str(input_csv), "--output", str(output)],
        cwd=ROOT,
        check=True,
    )

    assert output.exists()
    assert output.with_suffix(".pdf").exists()
    assert output.with_suffix(".csv").exists()
    summary = json.loads(output.with_suffix(".json").read_text())
    assert summary["n_points"] == 9
    assert summary["n_valid"] == 9
    assert summary["n_b1"] == 3
    assert summary["n_u1"] == 3
    assert summary["n_k"] == 12
    assert summary["n_ll"] == 5
    assert summary["v0_over_omega_c"] == 0.1


def test_gaugefixed_diagnostic_plotter_writes_centered_outputs(tmp_path: Path):
    input_csv = tmp_path / "sweep.csv"
    output = tmp_path / "diagnostics.png"
    fields = [
        "b1",
        "u1",
        "cG",
        "status",
        "response_status",
        "min_direct_gap",
        "path_gap_min",
        "ivc_minus_best_vp_energy_per_cell",
        "n_k",
        "n_ll",
        "v0_over_omega_c",
    ]
    with input_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for b1 in (-0.1, 0.0, 0.1):
            for u1 in (-0.1, 0.0, 0.1):
                writer.writerow(
                    {
                        "b1": b1,
                        "u1": u1,
                        "cG": -0.072 + 0.001 * b1 + 0.002 * u1,
                        "status": "ok",
                        "response_status": "ok",
                        "min_direct_gap": 0.8,
                        "path_gap_min": 0.2,
                        "ivc_minus_best_vp_energy_per_cell": 0.01,
                        "n_k": 18,
                        "n_ll": 6,
                        "v0_over_omega_c": 0.1,
                    }
                )

    subprocess.run(
        [
            sys.executable,
            str(DIAGNOSTIC_PLOTTER),
            "--input-csv",
            str(input_csv),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads(output.with_suffix(".json").read_text())
    assert output.exists()
    assert output.with_suffix(".pdf").exists()
    assert output.with_suffix(".csv").exists()
    assert summary["center_cG"] == -0.072
    assert summary["n_points"] == 9
    assert summary["n_k"] == 18
    assert summary["n_ll"] == 6
