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
    phase_plan = json.loads((output_root / "sweep_phase_plan.json").read_text())
    assert phase_plan["n_phase_points"] == 441
    assert plan["n_phase_points"] == 441
    assert plan["n_mesh_points"] == 1764
    assert plan["n_points"] == 1764
    assert [point["n_k"] for point in plan["points"][:4]] == [18, 20, 22, 24]
    assert plan["points"][0]["label"] == "u_000_theta_000_nk_018"
    assert "points/u_000_theta_000/nk_018" in plan["points"][0]["point_dir"]
    assert phase_plan["phase_points"][0]["label"] == "u_000_theta_000"
    assert plan["args"]["finite_q_shift_policy"] == "nearest-half"
    assert plan["args"]["ivc_branch_policy"] == "q0"
    assert plan["args"]["vertex_workers"] == 1
    assert (output_root / "sweep_plan.csv").exists()
    assert (output_root / "sweep_phase_plan.csv").exists()


def test_taige_finite_size_task_id_selects_one_phase_with_all_meshes(tmp_path):
    output_root = tmp_path / "finite_size"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(output_root),
            "--n-k-list",
            "18,20",
            "--n-u-d",
            "2",
            "--n-twist",
            "2",
            "--task-id",
            "2",
            "--dry-run",
        ],
        check=True,
    )

    plan = json.loads((output_root / "sweep_plan.json").read_text())
    assert plan["n_phase_points"] == 1
    assert plan["n_mesh_points"] == 2
    assert [point["n_k"] for point in plan["points"]] == [18, 20]
    assert {point["u_index"] for point in plan["points"]} == {1}
    assert {point["theta_index"] for point in plan["points"]} == {0}


def test_taige_finite_size_merge_only_writes_fit_tables(tmp_path):
    output_root = tmp_path / "finite_size"
    fixtures = [
        (0, 0, 0.0, 3.5, [(12, 1.2), (18, 1.1)]),
        (0, 1, 0.0, 3.6, [(12, float("nan")), (18, float("nan"))]),
    ]
    for u_index, theta_index, u_d, theta_deg, mesh_rows in fixtures:
        for idx, (n_k, cG) in enumerate(mesh_rows):
            phase_label = f"u_{u_index:03d}_theta_{theta_index:03d}"
            point_dir = output_root / "points" / phase_label / f"nk_{n_k:03d}"
            point_dir.mkdir(parents=True)
            row = {
                "n_k_index": idx,
                "n_k": n_k,
                "inv_n_k": 1.0 / n_k,
                "inv_n_k_squared": 1.0 / (n_k * n_k),
                "phase_point_label": phase_label,
                "finite_size_point_label": f"{phase_label}_nk_{n_k:03d}",
                "u_index": u_index,
                "theta_index": theta_index,
                "u_D_meV": u_d,
                "theta_deg": theta_deg,
                "cG": cG,
                "texture_valid": cG == cG,
                "texture_invalid_reason": None if cG == cG else "ivc_energy_below_vp_reference",
                "hf_ground_state": "VP" if cG == cG else "IVC_0",
                "gap_min": 0.01 * n_k if cG == cG else float("nan"),
                "vp_reference_name": "VP+",
                "vp_reference_energy_per_cell": -0.2,
                "vp_plus_energy_per_cell": -0.2,
                "vp_minus_energy_per_cell": -0.19,
                "ivc_q0_energy_per_cell": -0.1 if cG == cG else -0.3,
                "ivc_q0_minus_vp_energy_per_cell": 0.1 if cG == cG else -0.1,
                "selected_ivc_minus_vp_energy_per_cell": 0.1 if cG == cG else -0.1,
                "chern_hf_vp_plus_band_0": 1.0,
                "point_dir": str(point_dir),
            }
            (point_dir / "point_summary.json").write_text(json.dumps({"row": row}))
            (point_dir / "trial_theta.csv").write_text(
                "n_k_index,n_k,inv_n_k,u_index,theta_index,u_D_meV,theta_deg,theta,K_theta\n"
                f"{idx},{n_k},{1.0 / n_k},{u_index},{theta_index},{u_d},{theta_deg},0.1,{cG}\n"
            )
            (point_dir / "reference_energies.csv").write_text(
                "n_k_index,n_k,inv_n_k,u_index,theta_index,u_D_meV,theta_deg,quantity,value,reference\n"
                f"{idx},{n_k},{1.0 / n_k},{u_index},{theta_index},{u_d},{theta_deg},E_IVC_Q0_per_cell,-1.0,IVC Q=0\n"
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
    assert merged["n_points"] == 4
    assert merged["stacked_counts"]["trial_theta"] == 4
    assert (output_root / "sweep.csv").exists()
    fit_rows = list(csv.DictReader((output_root / "finite_size_fits.csv").open()))
    assert len(fit_rows) == 2
    assert fit_rows[0]["n_fit"] == "2"
    assert fit_rows[0]["largest_n_k"] == "18"
    assert fit_rows[0]["largest_nk_chern_hf_vp_plus_band_0"] == "1.0"
    assert fit_rows[1]["n_fit"] == "0"
    assert fit_rows[1]["hf_ground_state_largest_nk"] == "IVC_0"
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

    point_dir = output_root / "points" / "u_000_theta_000" / "nk_013"
    summary = json.loads((point_dir / "point_summary.json").read_text())
    row = summary["row"]
    assert row["n_k"] == 13
    assert row["selected_ivc_branch"] == "q0"
    assert row["finite_q_ivc_enabled"] is False
    assert row["finite_q_shift_policy"] is None
    assert row["finite_q_exact"] is None
    assert row["finite_q_q_coord"] is None
    assert row["finite_q_half_shift_coord"] is None
    assert row["finite_q_ivc_energy_per_cell"] is None
    assert summary["finite_q_ivc"] is None
    merged = json.loads((output_root / "sweep.json").read_text())
    assert merged["n_points"] == 1


def test_taige_finite_size_job_scripts_expose_array_and_merge_controls():
    script_text = SCRIPT.read_text()
    assert "--vertex-workers" in script_text

    scan_text = SCAN_JOB.read_text()
    assert "#SBATCH --array=0-440" in scan_text
    assert "#SBATCH -c 4" in scan_text
    assert 'N_K_LIST=${N_K_LIST:-"18,20,22,24"}' in scan_text
    assert 'U_D_MAX=${U_D_MAX:-"15.0"}' in scan_text
    assert 'N_U_D=${N_U_D:-"21"}' in scan_text
    assert 'THETA_MIN_DEG=${THETA_MIN_DEG:-"2.0"}' in scan_text
    assert 'THETA_MAX_DEG=${THETA_MAX_DEG:-"4.2"}' in scan_text
    assert 'N_TWIST=${N_TWIST:-"21"}' in scan_text
    assert 'COMPUTE_CHERN=${COMPUTE_CHERN:-"1"}' in scan_text
    assert 'COMPUTE_FINITE_Q_IVC=${COMPUTE_FINITE_Q_IVC:-"0"}' in scan_text
    assert 'VERTEX_WORKERS=${VERTEX_WORKERS:-"${SLURM_CPUS_PER_TASK:-1}"}' in scan_text
    assert "export OMP_NUM_THREADS=1" in scan_text
    assert "export MKL_NUM_THREADS=1" in scan_text
    assert "export OPENBLAS_NUM_THREADS=1" in scan_text
    assert "export NUMEXPR_NUM_THREADS=1" in scan_text
    assert 'FINITE_Q_SHIFT_POLICY=${FINITE_Q_SHIFT_POLICY:-"nearest-half"}' in scan_text
    assert 'IVC_BRANCH_POLICY=${IVC_BRANCH_POLICY:-"q0"}' in scan_text
    assert "TOTAL_TASKS=$((N_U_D * N_TWIST))" in scan_text
    assert "scripts/scan_taige_finite_size_cg.py" in scan_text
    assert "--vertex-workers" in scan_text
    assert "--compute-finite-q-ivc" in scan_text
    assert "--finite-q-shift-policy" in scan_text
    assert "--merge_taige_finite_size_cg.sh" not in scan_text

    merge_text = MERGE_JOB.read_text()
    assert "scripts/scan_taige_finite_size_cg.py" in merge_text
    assert "--merge-only" in merge_text
    assert 'OUTPUT_ROOT=${OUTPUT_ROOT:-"results/taige_cg_finite_size_nk18_24_u0_15_theta2_4p2"}' in merge_text
