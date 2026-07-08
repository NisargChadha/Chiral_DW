from __future__ import annotations

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "scan_taige_parameter_convergence.py"
PLOT_SCRIPT = ROOT / "Plots" / "plot_taige_parameter_convergence.py"
SCAN_JOB = ROOT / "jobs" / "scan_taige_parameter_convergence_array.sh"
MERGE_JOB = ROOT / "jobs" / "merge_taige_parameter_convergence.sh"


def _run_script(args: list[str]) -> None:
    subprocess.run([sys.executable, str(SCRIPT), *args], check=True)


def test_parameter_convergence_plane_wave_dry_run_writes_default_plan(tmp_path: Path):
    output_root = tmp_path / "plane_wave"

    _run_script(["--output-root", str(output_root), "--dry-run"])

    plan = json.loads((output_root / "sweep_plan.json").read_text())
    assert plan["n_points"] == 6
    assert plan["scan_axis"] == "plane_wave_shell"
    assert [point["scan_value"] for point in plan["points"]] == [3, 4, 5, 6, 7, 8]
    assert [point["plane_wave_shell"] for point in plan["points"]] == [3, 4, 5, 6, 7, 8]
    assert {point["n_bands"] for point in plan["points"]} == {2}
    assert {point["n_active_bands_per_valley"] for point in plan["points"]} == {2}
    assert plan["points"][0]["label"] == "plane_wave_shell_003"
    assert "points/plane_wave_shell_003" in plan["points"][0]["point_dir"]
    assert plan["args"]["ivc_branch_policy"] == "q0"
    assert plan["args"]["compute_finite_q_ivc"] is False
    assert plan["args"]["density_vertex_retention"] == "hartree_only"
    assert (output_root / "sweep_plan.csv").exists()


def test_parameter_convergence_active_band_dry_run_sets_n_bands_equal_active(tmp_path: Path):
    output_root = tmp_path / "active_bands"

    _run_script(
        [
            "--scan-axis",
            "active-bands",
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )

    plan = json.loads((output_root / "sweep_plan.json").read_text())
    assert plan["n_points"] == 5
    assert plan["scan_axis"] == "active_bands"
    values = [1, 2, 3, 4, 5]
    assert [point["scan_value"] for point in plan["points"]] == values
    assert [point["n_active_bands_per_valley"] for point in plan["points"]] == values
    assert [point["n_bands"] for point in plan["points"]] == values
    assert {point["plane_wave_shell"] for point in plan["points"]} == {5}
    assert plan["points"][0]["label"] == "active_bands_001"


def test_parameter_convergence_task_id_selects_one_value(tmp_path: Path):
    output_root = tmp_path / "selected"

    _run_script(
        [
            "--output-root",
            str(output_root),
            "--value-list",
            "4,5,6",
            "--task-id",
            "1",
            "--dry-run",
        ]
    )

    plan = json.loads((output_root / "sweep_plan.json").read_text())
    assert plan["n_points"] == 1
    assert plan["points"][0]["scan_value"] == 5
    assert plan["points"][0]["value_index"] == 1


def test_parameter_convergence_merge_only_writes_delta_summary(tmp_path: Path):
    output_root = tmp_path / "merge"
    fixtures = [
        (0, 3, 1.2, -0.10, -0.04),
        (1, 4, 1.1, -0.11, -0.03),
    ]
    for value_index, shell, cG, vp_energy, delta in fixtures:
        point_dir = output_root / "points" / f"plane_wave_shell_{shell:03d}"
        point_dir.mkdir(parents=True)
        row = {
            "scan_axis": "plane_wave_shell",
            "scan_value": shell,
            "value_index": value_index,
            "convergence_point_label": f"plane_wave_shell_{shell:03d}",
            "theta_deg": 3.5,
            "u_D_meV": 0.0,
            "n_k": 24,
            "plane_wave_shell": shell,
            "n_plane_waves": 3 * shell * shell + 3 * shell + 1,
            "n_bands": 2,
            "n_active_bands_per_valley": 2,
            "cG": cG,
            "K_min": -0.5,
            "K_max": 0.5,
            "gap_min": 0.2,
            "valid_local_gap": True,
            "texture_valid": True,
            "texture_invalid_reason": None,
            "hf_ground_state": "VP",
            "ivc_branch_policy": "q0",
            "selected_ivc_branch": "q0",
            "finite_q_ivc_enabled": False,
            "finite_q_shift_policy": None,
            "finite_q_exact": None,
            "q0_ivc_energy_per_cell": vp_energy + delta,
            "finite_q_ivc_energy_per_cell": None,
            "finite_q_minus_q0_ivc_energy_per_cell": None,
            "selected_ivc_energy_per_cell": vp_energy + delta,
            "vp_plus_energy_per_cell": vp_energy,
            "vp_minus_energy_per_cell": vp_energy + 0.002,
            "vp_reference_name": "VP+",
            "vp_reference_energy_per_cell": vp_energy,
            "ivc_q0_energy_per_cell": vp_energy + delta,
            "ivc_q0_minus_vp_energy_per_cell": delta,
            "selected_ivc_minus_vp_energy_per_cell": delta,
            "vp_reference_direct_gap": 1.0,
            "vp_reference_indirect_gap": 0.8,
            "selected_ivc_direct_gap": 0.9,
            "selected_ivc_indirect_gap": 0.7,
            "vp_plus_self_consistency_warning": False,
            "vp_minus_self_consistency_warning": False,
            "ivc_self_consistency_warning": False,
            "ivc_finite_q_self_consistency_warning": None,
            "vp_reference_order_abs_nz": 1.0,
            "selected_ivc_order_abs_nz": 0.0,
            "selected_ivc_ivc_amplitude_block": 1.0,
            "ivc_q0_ivc_amplitude_block": 1.0,
            "elapsed_seconds": 12.0,
            "point_dir": str(point_dir),
        }
        (point_dir / "point_summary.json").write_text(json.dumps({"row": row}))
        (point_dir / "trial_theta.csv").write_text(
            "scan_axis,scan_value,value_index,theta,K_theta\n"
            f"plane_wave_shell,{shell},{value_index},0.1,{cG}\n"
        )
        (point_dir / "reference_energies.csv").write_text(
            "scan_axis,scan_value,value_index,quantity,value,reference\n"
            f"plane_wave_shell,{shell},{value_index},E_IVC_Q0_per_cell,{vp_energy + delta},IVC Q=0\n"
        )

    _run_script(["--output-root", str(output_root), "--merge-only"])

    merged = json.loads((output_root / "sweep.json").read_text())
    assert merged["n_points"] == 2
    assert merged["stacked_counts"]["trial_theta"] == 2
    summary_rows = list(csv.DictReader((output_root / "convergence_summary.csv").open()))
    assert len(summary_rows) == 2
    assert summary_rows[0]["scan_value"] == "3"
    assert summary_rows[1]["scan_value"] == "4"
    assert float(summary_rows[0]["delta_largest_cG"]) == pytest.approx(0.1)
    assert float(summary_rows[1]["delta_largest_cG"]) == pytest.approx(0.0)
    assert float(summary_rows[1]["delta_prev_selected_ivc_minus_vp_energy_per_cell"]) == pytest.approx(0.01)
    summary_json = json.loads((output_root / "convergence_summary.json").read_text())
    assert summary_json["delta_metrics"][0] == "cG"


def test_parameter_convergence_tiny_smoke_runs_two_plane_wave_values(tmp_path: Path):
    output_root = tmp_path / "smoke"

    _run_script(
        [
            "--output-root",
            str(output_root),
            "--value-list",
            "0,1",
            "--u-d",
            "0.0",
            "--theta-deg",
            "3.5",
            "--n-k",
            "3",
            "--n-bands",
            "1",
            "--n-active-bands-per-valley",
            "1",
            "--q-mesh",
            "shell",
            "--q-shell",
            "0",
            "--local-field-cutoff",
            "0",
            "--v0",
            "0.0",
            "--exchange-scale",
            "0.0",
            "--hartree-scale",
            "0.0",
            "--max-iter",
            "1",
            "--min-iter",
            "0",
            "--mixing-method",
            "linear",
            "--seed-ordered-weight",
            "1.0",
            "--seed-random-weight",
            "0.0",
            "--n-theta",
            "5",
            "--no-chern",
            "--no-finite-q-ivc",
        ]
    )

    summary = json.loads((output_root / "sweep.json").read_text())
    assert summary["n_points"] == 2
    rows = list(csv.DictReader((output_root / "convergence_summary.csv").open()))
    assert [int(row["plane_wave_shell"]) for row in rows] == [0, 1]
    assert all(row["finite_q_ivc_enabled"] == "False" for row in rows)
    assert all(row["selected_ivc_branch"] == "q0" for row in rows)
    assert "delta_largest_cG" in rows[0]


def _load_plot_module():
    pytest.importorskip("matplotlib")
    spec = importlib.util.spec_from_file_location("plot_taige_parameter_convergence", PLOT_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_taige_parameter_convergence_plot_writes_png_pdf_and_csv(tmp_path: Path):
    module = _load_plot_module()
    input_csv = tmp_path / "convergence_summary.csv"
    rows = [
        {
            "scan_axis": "plane_wave_shell",
            "scan_value": value,
            "theta_deg": 3.5,
            "u_D_meV": 0.0,
            "n_k": 24,
            "plane_wave_shell": value,
            "n_bands": 2,
            "n_active_bands_per_valley": 2,
            "cG": 1.0 / value,
            "vp_plus_energy_per_cell": -0.2 - 0.01 / value,
            "vp_minus_energy_per_cell": -0.199 - 0.01 / value,
            "vp_reference_energy_per_cell": -0.2 - 0.01 / value,
            "ivc_q0_energy_per_cell": -0.15 - 0.01 / value,
            "selected_ivc_energy_per_cell": -0.15 - 0.01 / value,
            "ivc_q0_minus_vp_energy_per_cell": 0.05,
            "selected_ivc_minus_vp_energy_per_cell": 0.05,
            "gap_min": 0.1 + 0.01 * value,
            "vp_reference_direct_gap": 1.0 + 0.01 * value,
            "vp_reference_indirect_gap": 0.8,
            "selected_ivc_direct_gap": 0.9 + 0.01 * value,
            "selected_ivc_indirect_gap": 0.7,
            "texture_valid": True,
            "hf_ground_state": "VP",
            "selected_ivc_branch": "q0",
        }
        for value in (3, 4, 5)
    ]
    with input_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    png_path, pdf_path, csv_path = module.render_taige_parameter_convergence_plot(
        module.TaigeConvergencePlotParams(
            input_csv=input_csv,
            output_dir=tmp_path,
            output_stem="synthetic_convergence",
            dpi=90,
        )
    )

    assert png_path == tmp_path / "synthetic_convergence.png"
    assert pdf_path == tmp_path / "synthetic_convergence.pdf"
    assert csv_path == tmp_path / "synthetic_convergence.csv"
    for path in (png_path, pdf_path, csv_path):
        assert path.exists()
        assert path.stat().st_size > 0

    image = module.plt.imread(png_path)
    assert image.size > 0


def test_parameter_convergence_job_scripts_expose_axis_and_merge_controls():
    scan_text = SCAN_JOB.read_text()
    assert "#SBATCH --array=0-5" in scan_text
    assert 'SCAN_AXIS=${SCAN_AXIS:-"plane-wave-shell"}' in scan_text
    assert 'VALUE_LIST=${VALUE_LIST:-"3,4,5,6,7,8"}' in scan_text
    assert 'OUTPUT_ROOT=${OUTPUT_ROOT:-"results/taige_convergence_plane_wave_shell_theta35_u0"}' in scan_text
    assert 'COMPUTE_FINITE_Q_IVC=${COMPUTE_FINITE_Q_IVC:-"0"}' in scan_text
    assert 'IVC_BRANCH_POLICY=${IVC_BRANCH_POLICY:-"q0"}' in scan_text
    assert "scripts/scan_taige_parameter_convergence.py" in scan_text
    assert "--scan-axis" in scan_text
    assert "--value-list" in scan_text
    assert "--compute-finite-q-ivc" in scan_text
    assert "--no-finite-q-ivc" in scan_text

    merge_text = MERGE_JOB.read_text()
    assert "scripts/scan_taige_parameter_convergence.py" in merge_text
    assert "--merge-only" in merge_text
    assert 'SCAN_AXIS=${SCAN_AXIS:-"plane-wave-shell"}' in merge_text
