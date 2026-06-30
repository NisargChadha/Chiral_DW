import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "scan_taige_continuum_cg.py"
JOB = ROOT / "jobs" / "scan_taige_continuum_cg_array.sh"


def test_taige_sweep_dry_run_writes_selected_plan(tmp_path):
    output_root = tmp_path / "sweep"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(output_root),
            "--u-d-min",
            "0",
            "--u-d-max",
            "10",
            "--n-u-d",
            "2",
            "--theta-min-deg",
            "3.0",
            "--theta-max-deg",
            "3.5",
            "--n-twist",
            "3",
            "--task-id",
            "4",
            "--dry-run",
        ],
        check=True,
    )

    plan = json.loads((output_root / "sweep_plan.json").read_text())
    assert plan["n_points"] == 1
    point = plan["points"][0]
    assert point["u_index"] == 1
    assert point["theta_index"] == 1
    assert point["u_D"] == 10.0
    assert point["theta_deg"] == 3.25
    assert "points/u_001_theta_001" in point["point_dir"]
    assert plan["args"]["ivc_branch_policy"] == "lower-energy"
    assert plan["args"]["vertex_workers"] == 1
    assert (output_root / "sweep_plan.csv").exists()


def test_taige_sweep_merge_only_collects_point_summaries(tmp_path):
    output_root = tmp_path / "sweep"
    point_dir = output_root / "points" / "u_000_theta_000"
    point_dir.mkdir(parents=True)
    (point_dir / "point_summary.json").write_text(
        json.dumps(
            {
                "row": {
                    "u_index": 0,
                    "theta_index": 0,
                    "u_D_meV": 0.0,
                    "theta_deg": 3.5,
                    "cG": 1.25,
                    "ivc_branch_policy": "lower_energy",
                    "selected_ivc_branch": "q0",
                    "selected_ivc_energy_per_cell": -1.0,
                    "q0_ivc_energy_per_cell": -1.0,
                    "finite_q_ivc_energy_per_cell": None,
                    "finite_q_minus_q0_ivc_energy_per_cell": None,
                    "chern_hf_vpplus_band_0": 1.0,
                    "point_dir": str(point_dir),
                }
            }
        )
    )
    (point_dir / "trial_theta.csv").write_text(
        "u_index,theta_index,u_D_meV,theta_deg,theta,K_theta,direct_gap\n"
        "0,0,0.0,3.5,0.1,1.2,0.3\n"
    )
    (point_dir / "reference_energies.csv").write_text(
        "u_index,theta_index,u_D_meV,theta_deg,quantity,value,reference\n"
        "0,0,0.0,3.5,E_IVC_Q0_per_cell,-1.0,IVC Q=0\n"
    )
    (point_dir / "noninteracting_chern_numbers.csv").write_text(
        "u_index,theta_index,u_D_meV,theta_deg,basis,valley,band,chern\n"
        "0,0,0.0,3.5,hole,K,0,1.0\n"
    )
    (point_dir / "hf_chern_numbers.csv").write_text(
        "u_index,theta_index,u_D_meV,theta_deg,reference,band,chern,energy_min,energy_max\n"
        "0,0,0.0,3.5,VP+,0,1.0,-1.0,1.0\n"
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
    assert merged["n_points"] == 1
    assert merged["rows"][0]["cG"] == 1.25
    assert merged["stacked_counts"]["trial_theta"] == 1
    assert merged["stacked_counts"]["hf_chern_numbers"] == 1
    csv_text = (output_root / "sweep.csv").read_text()
    assert "u_D_meV" in csv_text
    assert "chern_hf_vpplus_band_0" in csv_text
    assert "1.25" in csv_text
    assert "direct_gap" in (output_root / "sweep_trial_theta.csv").read_text()
    assert "E_IVC_Q0_per_cell" in (output_root / "sweep_reference_energies.csv").read_text()
    assert "hole,K,0,1.0" in (output_root / "sweep_noninteracting_chern_numbers.csv").read_text()
    assert "VP+,0,1.0" in (output_root / "sweep_hf_chern_numbers.csv").read_text()


def test_taige_sweep_point_writes_scalar_rich_diagnostics(tmp_path):
    output_root = tmp_path / "sweep"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(output_root),
            "--u-d",
            "0.0",
            "--theta-deg",
            "3.5",
            "--n-k",
            "3",
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
            "--no-finite-q-ivc",
        ],
        check=True,
    )

    point_dir = output_root / "points" / "u_000_theta_000"
    summary = json.loads((point_dir / "point_summary.json").read_text())
    row = summary["row"]
    assert row["cG"] == row["cG"]
    assert row["finite_q_ivc_enabled"] is False
    assert row["ivc_branch_policy"] == "q0"
    assert row["selected_ivc_branch"] == "q0"
    assert row["finite_q_ivc_energy_per_cell"] is None
    assert row["texture_valid"] is True
    assert row["texture_nan_policy"] is True
    assert row["texture_invalid_reason"] is None
    assert row["hf_ground_state"] == "VP"
    assert row["chern_enabled"] is True
    assert "ivc_q0_energy_per_cell" in row
    assert "q0_ivc_energy_per_cell" in row
    assert "vp_reference_order_abs_nz" in row
    assert "selected_ivc_ivc_amplitude_block" in row
    assert "ivc_q0_ivc_amplitude_block" in row
    assert "ivc_finite_q_ivc_amplitude_block" in row
    assert row["ivc_finite_q_ivc_amplitude_block"] is None
    assert "vp_reference_direct_gap" in row
    assert "selected_ivc_direct_gap" in row
    assert "ivc_q0_direct_gap" in row
    assert any(key.startswith("chern_nonint_hole_k_band_") for key in row)
    assert any(key.startswith("chern_hf_vpplus_band_") for key in row)

    trial_rows = list(csv.DictReader((point_dir / "trial_theta.csv").open()))
    assert len(trial_rows) == 5
    assert "energy_total_per_cell" in trial_rows[0]
    assert "direct_gap" in trial_rows[0]
    assert (point_dir / "reference_energies.csv").exists()
    assert (point_dir / "noninteracting_chern_numbers.csv").exists()
    assert (point_dir / "hf_chern_numbers.csv").exists()

    merged = json.loads((output_root / "sweep.json").read_text())
    assert merged["n_points"] == 1
    assert merged["stacked_counts"]["trial_theta"] == 5
    assert "sweep_hf_chern_numbers.csv" in merged["tables"]["hf_chern_numbers_csv"]


def test_taige_sweep_job_uses_array_task_and_results_root():
    script_text = SCRIPT.read_text()
    assert "--vertex-workers" in script_text

    text = JOB.read_text()
    assert "#SBATCH --array=0-440" in text
    assert "#SBATCH -c 4" in text
    assert "#SBATCH --mem=24G" in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "scripts/scan_taige_continuum_cg.py" in text
    assert 'OUTPUT_ROOT=${OUTPUT_ROOT:-"results/taige_cg_nk24_active2_shell5_vp_region"}' in text
    assert 'U_D_MAX=${U_D_MAX:-"20.0"}' in text
    assert 'N_U_D=${N_U_D:-"21"}' in text
    assert 'THETA_MIN_DEG=${THETA_MIN_DEG:-"2.0"}' in text
    assert 'THETA_MAX_DEG=${THETA_MAX_DEG:-"5.0"}' in text
    assert 'N_TWIST=${N_TWIST:-"21"}' in text
    assert 'N_K=${N_K:-"24"}' in text
    assert 'PLANE_WAVE_SHELL=${PLANE_WAVE_SHELL:-"5"}' in text
    assert 'N_ACTIVE_BANDS_PER_VALLEY=${N_ACTIVE_BANDS_PER_VALLEY:-"2"}' in text
    assert 'VERTEX_WORKERS=${VERTEX_WORKERS:-"${SLURM_CPUS_PER_TASK:-1}"}' in text
    assert "export OMP_NUM_THREADS=1" in text
    assert "export MKL_NUM_THREADS=1" in text
    assert "export OPENBLAS_NUM_THREADS=1" in text
    assert "export NUMEXPR_NUM_THREADS=1" in text
    assert 'COMPUTE_CHERN=${COMPUTE_CHERN:-"1"}' in text
    assert 'COMPUTE_FINITE_Q_IVC=${COMPUTE_FINITE_Q_IVC:-"1"}' in text
    assert 'IVC_BRANCH_POLICY=${IVC_BRANCH_POLICY:-"lower-energy"}' in text
    assert 'IVC_BRANCH_TIE_ATOL=${IVC_BRANCH_TIE_ATOL:-"1e-9"}' in text
    assert 'NAN_TEXTURE_WHEN_IVC_LOWER=${NAN_TEXTURE_WHEN_IVC_LOWER:-"1"}' in text
    assert 'TEXTURE_ENERGY_TIE_ATOL=${TEXTURE_ENERGY_TIE_ATOL:-"1e-9"}' in text
    assert 'WRITE_HF_PATH_SPECTRA=${WRITE_HF_PATH_SPECTRA:-"0"}' in text
    assert "--no-chern" in text
    assert "--no-finite-q-ivc" in text
    assert "--ivc-branch-policy" in text
    assert "--ivc-branch-tie-atol" in text
    assert "--allow-texture-in-ivc-ground-state" in text
    assert "--texture-energy-tie-atol" in text
    assert "--write-hf-path-spectra" in text
    assert "--vertex-workers" in text
    assert "--merge-only" in text
