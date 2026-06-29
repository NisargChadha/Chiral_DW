import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "scan_taige_finite_size_cg.py"
SCAN_JOB = ROOT / "jobs" / "scan_taige_finite_size_cg_array.sh"
MERGE_JOB = ROOT / "jobs" / "merge_taige_finite_size_cg.sh"


def test_taige_finite_size_dry_run_writes_default_nk_plan(tmp_path):
    output_root = tmp_path / "finite_size"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(output_root),
            "--dry-run",
        ],
        check=True,
    )

    plan = json.loads((output_root / "sweep_plan.json").read_text())
    assert plan["n_points"] == 9
    assert [point["n_k"] for point in plan["points"]] == list(range(12, 21))
    assert plan["points"][0]["label"] == "nk_012_u_000_theta_000"
    assert "points/nk_012_u_000_theta_000" in plan["points"][0]["point_dir"]
    assert plan["args"]["finite_q_shift_policy"] == "nearest-half"
    assert plan["args"]["ivc_branch_policy"] == "q0"
    assert (output_root / "sweep_plan.csv").exists()


def test_taige_finite_size_merge_only_writes_fit_tables(tmp_path):
    output_root = tmp_path / "finite_size"
    for idx, (n_k, cG) in enumerate([(12, 1.2), (18, 1.1)]):
        point_dir = output_root / "points" / f"nk_{n_k:03d}_u_000_theta_000"
        point_dir.mkdir(parents=True)
        row = {
            "n_k_index": idx,
            "n_k": n_k,
            "inv_n_k": 1.0 / n_k,
            "finite_size_point_label": f"nk_{n_k:03d}_u_000_theta_000",
            "u_index": 0,
            "theta_index": 0,
            "u_D_meV": 0.0,
            "theta_deg": 3.5,
            "cG": cG,
            "finite_q_shift_policy": "nearest_half",
            "finite_q_exact": n_k in {12, 18},
            "point_dir": str(point_dir),
        }
        (point_dir / "point_summary.json").write_text(json.dumps({"row": row}))
        (point_dir / "trial_theta.csv").write_text(
            "n_k_index,n_k,inv_n_k,u_index,theta_index,u_D_meV,theta_deg,theta,K_theta\n"
            f"{idx},{n_k},{1.0 / n_k},0,0,0.0,3.5,0.1,{cG}\n"
        )
        (point_dir / "reference_energies.csv").write_text(
            "n_k_index,n_k,inv_n_k,u_index,theta_index,u_D_meV,theta_deg,quantity,value,reference\n"
            f"{idx},{n_k},{1.0 / n_k},0,0,0.0,3.5,E_IVC_Q0_per_cell,-1.0,IVC Q=0\n"
        )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(output_root),
            "--merge-only",
        ],
        check=True,
    )

    merged = json.loads((output_root / "sweep.json").read_text())
    assert merged["n_points"] == 2
    assert merged["stacked_counts"]["trial_theta"] == 2
    assert (output_root / "sweep.csv").exists()
    fit_rows = list(csv.DictReader((output_root / "finite_size_fits.csv").open()))
    assert len(fit_rows) == 1
    assert fit_rows[0]["n_fit"] == "2"
    assert "coefficients_descending_json" in fit_rows[0]
    fits = json.loads((output_root / "finite_size_fits.json").read_text())
    assert fits["fit_variable"] == "inv_n_k"


def test_taige_finite_size_smoke_runs_approximate_nk13_point(tmp_path):
    output_root = tmp_path / "finite_size"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(output_root),
            "--n-k-list",
            "13",
            "--u-d",
            "0.0",
            "--theta-deg",
            "3.5",
            "--plane-wave-shell",
            "0",
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
        ],
        check=True,
    )

    point_dir = output_root / "points" / "nk_013_u_000_theta_000"
    summary = json.loads((point_dir / "point_summary.json").read_text())
    row = summary["row"]
    assert row["n_k"] == 13
    assert row["selected_ivc_branch"] == "q0"
    assert row["finite_q_shift_policy"] == "nearest_half"
    assert row["finite_q_exact"] is False
    assert row["finite_q_q_coord"] == [4, 5]
    assert row["finite_q_half_shift_coord"] == [2, 9]
    assert row["finite_q_ivc_energy_per_cell"] is not None
    assert summary["finite_q_ivc"]["metadata"]["finite_q_exact"] is False
    merged = json.loads((output_root / "sweep.json").read_text())
    assert merged["n_points"] == 1


def test_taige_finite_size_job_scripts_expose_array_and_merge_controls():
    scan_text = SCAN_JOB.read_text()
    assert "#SBATCH --array=0-8" in scan_text
    assert 'N_K_LIST=${N_K_LIST:-"12,13,14,15,16,17,18,19,20"}' in scan_text
    assert 'COMPUTE_CHERN=${COMPUTE_CHERN:-"0"}' in scan_text
    assert 'COMPUTE_FINITE_Q_IVC=${COMPUTE_FINITE_Q_IVC:-"1"}' in scan_text
    assert 'FINITE_Q_SHIFT_POLICY=${FINITE_Q_SHIFT_POLICY:-"nearest-half"}' in scan_text
    assert 'IVC_BRANCH_POLICY=${IVC_BRANCH_POLICY:-"q0"}' in scan_text
    assert "scripts/scan_taige_finite_size_cg.py" in scan_text
    assert "--finite-q-shift-policy" in scan_text
    assert "--merge_taige_finite_size_cg.sh" not in scan_text

    merge_text = MERGE_JOB.read_text()
    assert "scripts/scan_taige_finite_size_cg.py" in merge_text
    assert "--merge-only" in merge_text
    assert 'OUTPUT_ROOT=${OUTPUT_ROOT:-"results/taige_cg_finite_size_nk12_20_u0_theta3p5"}' in merge_text
