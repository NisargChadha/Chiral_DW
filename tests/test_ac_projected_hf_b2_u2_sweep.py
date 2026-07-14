import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "scan_ac_projected_hf_b2_u2.py"
JOB = ROOT / "jobs" / "scan_ac_projected_hf_b2_u2_array.sh"
PLOTTER = ROOT / "Plots" / "plot_ac_b2_cg_linecut.py"


def test_ac_b2_u2_sweep_dry_run_writes_fixed_first_harmonics(tmp_path):
    output_root = tmp_path / "sweep"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(output_root),
            "--b1",
            "0.2",
            "--u1",
            "0.05",
            "--b2-min",
            "-0.3",
            "--b2-max",
            "0.3",
            "--n-b2",
            "3",
            "--u2-min",
            "-0.3",
            "--u2-max",
            "0.3",
            "--n-u2",
            "3",
            "--task-id",
            "4",
            "--dry-run",
        ],
        check=True,
    )

    plan = json.loads((output_root / "sweep_plan.json").read_text())
    assert plan["n_points"] == 1
    assert plan["args"]["coulomb_kind"] == "dimensionless_dual_gate"
    assert "b2/u2" in plan["sweep_convention"]
    point = plan["points"][0]
    assert point["b_index"] == 1
    assert point["u_index"] == 1
    assert point["b2_index"] == 1
    assert point["u2_index"] == 1
    assert point["b1"] == 0.2
    assert point["u1"] == 0.05
    assert point["b2"] == 0.0
    assert point["u2"] == 0.0
    assert "points/b2_001_u2_001" in point["point_dir"]


def test_ac_b2_u2_sweep_canonicalizes_linecut_coordinates(tmp_path):
    output_root = tmp_path / "sweep"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(output_root),
            "--b1",
            "0.0",
            "--u1",
            "0.0",
            "--b2-min",
            "-0.1",
            "--b2-max",
            "0.1",
            "--n-b2",
            "11",
            "--u2-min",
            "0.0",
            "--u2-max",
            "0.0",
            "--n-u2",
            "1",
            "--dry-run",
        ],
        check=True,
    )

    points = json.loads((output_root / "sweep_plan.json").read_text())["points"]
    assert len(points) == 11
    assert points[3]["b2"] == -0.04
    assert points[6]["b2"] == 0.02
    assert {point["u2"] for point in points} == {0.0}


def test_ac_b2_u2_sweep_no_write_plan_leaves_shared_plan_untouched(tmp_path):
    output_root = tmp_path / "sweep"
    output_root.mkdir()
    plan_path = output_root / "sweep_plan.json"
    plan_path.write_text('{"sentinel": true}')
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(output_root),
            "--b2",
            "0.0",
            "--u2",
            "0.0",
            "--dry-run",
            "--no-write-plan",
        ],
        check=True,
    )
    assert json.loads(plan_path.read_text()) == {"sentinel": True}


def test_ac_b2_cg_plotter_writes_linecut_artifacts(tmp_path):
    input_csv = tmp_path / "sweep.csv"
    output = tmp_path / "linecut.png"
    fields = [
        "b1",
        "u1",
        "b2",
        "u2",
        "cG",
        "hf_all_converged",
        "response_status",
        "band_chern",
        "chern_vp_plus",
        "chern_vp_minus",
        "chern_ivc",
        "min_direct_gap",
        "path_gap_min",
        "n_k",
        "n_ll",
        "active_band",
        "v0_over_omega_c",
    ]
    with input_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for b2 in (-0.1, 0.0, 0.1):
            writer.writerow(
                {
                    "b1": 0.0,
                    "u1": 0.0,
                    "b2": b2,
                    "u2": 0.0,
                    "cG": -0.072 + 0.01 * b2,
                    "hf_all_converged": True,
                    "response_status": "ok",
                    "band_chern": 1.0,
                    "chern_vp_plus": 1.0,
                    "chern_vp_minus": -1.0,
                    "chern_ivc": 0.0,
                    "min_direct_gap": 0.8,
                    "path_gap_min": 0.14,
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
    summary = json.loads(output.with_suffix(".json").read_text())
    assert output.exists()
    assert output.with_suffix(".pdf").exists()
    assert output.with_suffix(".csv").exists()
    assert summary["n_points"] == 3
    assert summary["n_valid"] == 3
    assert summary["b1"] == 0.0
    assert summary["u1"] == 0.0
    assert summary["u2"] == 0.0


def test_ac_b2_u2_sweep_tiny_point_runs_overlap_response(tmp_path):
    output_root = tmp_path / "sweep"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(output_root),
            "--b1",
            "0.0",
            "--u1",
            "0.0",
            "--b2",
            "0.0",
            "--u2",
            "0.0",
            "--n-k",
            "3",
            "--n-ll",
            "1",
            "--band-diagnostics-n-k",
            "5",
            "--q-shell",
            "1",
            "--local-field-cutoff",
            "1",
            "--max-iter",
            "220",
            "--min-iter",
            "2",
            "--mixing-method",
            "oda",
            "--n-theta",
            "8",
            "--n-phi",
            "2",
        ],
        check=True,
    )

    point_dir = output_root / "points" / "b2_000_u2_000"
    summary = json.loads((point_dir / "point_summary.json").read_text())
    row = summary["row"]
    assert row["b1"] == 0.0
    assert row["u1"] == 0.0
    assert row["b2"] == 0.0
    assert row["u2"] == 0.0
    assert row["n_active_bands_per_valley"] == 1
    assert row["active_band"] == 0
    assert row["n_ll"] == 1
    assert row["hf_all_converged"] is True
    assert row["reference_chern_valid"] is True
    assert row["response_status"] == "ok"
    assert abs(row["cG"]) > 1e-3
    assert abs(row["chern_vp_plus"] - 1.0) < 5e-3
    assert abs(row["chern_vp_minus"] + 1.0) < 5e-3
    assert abs(row["chern_ivc"]) < 5e-3
    assert (point_dir / "response.npz").exists()
    assert len(list(csv.DictReader((point_dir / "hf_chern_numbers.csv").open()))) == 3
    assert (output_root / "sweep.csv").exists()


def test_ac_b2_u2_sweep_job_uses_121_point_array_and_fixed_first_harmonics():
    text = JOB.read_text()
    assert "#SBATCH --array=0-120" in text
    assert "#SBATCH --mem=24G" in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "scripts/scan_ac_projected_hf_b2_u2.py" in text
    assert 'B1_FIXED=${B1_FIXED:-"0.0"}' in text
    assert 'U1_FIXED=${U1_FIXED:-"0.0"}' in text
    assert 'B2_MIN=${B2_MIN:-"-0.3"}' in text
    assert 'B2_MAX=${B2_MAX:-"0.3"}' in text
    assert 'N_B2=${N_B2:-"11"}' in text
    assert 'U2_MIN=${U2_MIN:-"-0.3"}' in text
    assert 'U2_MAX=${U2_MAX:-"0.3"}' in text
    assert 'N_U2=${N_U2:-"11"}' in text
    assert 'ACTIVE_BAND=${ACTIVE_BAND:-"0"}' in text
    assert 'COULOMB_KIND=${COULOMB_KIND:-"dimensionless_dual_gate"}' in text
