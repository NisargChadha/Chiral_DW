import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "scan_ac_projected_hf_b1_u1.py"
JOB = ROOT / "jobs" / "scan_ac_projected_hf_b1_u1_array.sh"


def test_ac_b1_u1_sweep_dry_run_writes_selected_plan(tmp_path):
    output_root = tmp_path / "sweep"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(output_root),
            "--b1-min",
            "-0.3",
            "--b1-max",
            "0.3",
            "--n-b1",
            "3",
            "--u1-min",
            "-0.3",
            "--u1-max",
            "0.3",
            "--n-u1",
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
    assert np.isclose(plan["continuum_match"]["moire_length_nm"], 5.681350588613268)
    assert np.isclose(
        plan["continuum_match"]["landau_level_spacing_mev"], 27.625318938039467
    )
    assert plan["active_space_convention"].startswith("one active AC band per valley")
    point = plan["points"][0]
    assert point["b_index"] == 1
    assert point["u_index"] == 1
    assert point["b1"] == 0.0
    assert point["u1"] == 0.0
    assert "points/b_001_u_001" in point["point_dir"]


def test_ac_b1_u1_physical_dual_gate_plan_records_safe_continuum_match(tmp_path):
    output_root = tmp_path / "physical"
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
            "--coulomb-kind",
            "dual_gate",
            "--q-mesh",
            "full",
            "--v0",
            "0.1",
            "--dry-run",
        ],
        check=True,
    )

    plan = json.loads((output_root / "sweep_plan.json").read_text())
    match = plan["continuum_match"]
    assert plan["args"]["q_mesh"] == "full"
    assert np.isclose(match["theta_deg"], 3.5)
    assert np.isclose(match["a0_angstrom"], 3.47)
    assert np.isclose(match["m_eff"], 0.62)
    assert match["characteristic_coulomb_to_ll_ratio"] < 0.25
    assert np.isclose(match["characteristic_coulomb_to_ll_ratio"], 0.0549384545)


def test_ac_b1_u1_physical_dual_gate_rejects_interaction_above_ll_guard(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(tmp_path / "too_strong"),
            "--b1",
            "0.0",
            "--u1",
            "0.0",
            "--coulomb-kind",
            "dual_gate",
            "--v0",
            "1.0",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "interaction is too strong" in result.stderr


def test_ac_b1_u1_sweep_canonicalizes_linspace_roundoff(tmp_path):
    output_root = tmp_path / "sweep"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(output_root),
            "--b1-min",
            "-0.1",
            "--b1-max",
            "0.1",
            "--n-b1",
            "11",
            "--u1-min",
            "-0.1",
            "--u1-max",
            "0.1",
            "--n-u1",
            "11",
            "--dry-run",
        ],
        check=True,
    )

    plan = json.loads((output_root / "sweep_plan.json").read_text())
    point = next(
        row for row in plan["points"] if row["b_index"] == 3 and row["u_index"] == 3
    )
    assert point["b1"] == -0.04
    assert point["u1"] == -0.04

    explicit_root = tmp_path / "explicit"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(explicit_root),
            "--b1",
            "-0.04000000000000001",
            "--u1",
            "0.01999999999999999",
            "--dry-run",
        ],
        check=True,
    )
    explicit = json.loads((explicit_root / "sweep_plan.json").read_text())["points"][0]
    assert explicit["b1"] == -0.04
    assert explicit["u1"] == 0.02


def test_ac_b1_u1_sweep_merge_only_collects_point_summaries(tmp_path):
    output_root = tmp_path / "sweep"
    point_dir = output_root / "points" / "b_000_u_000"
    point_dir.mkdir(parents=True)
    (point_dir / "point_summary.json").write_text(
        json.dumps(
            {
                "row": {
                    "b_index": 0,
                    "u_index": 0,
                    "b1": -0.3,
                    "u1": -0.3,
                    "cG": -0.05,
                    "bandwidth": 0.1,
                    "min_direct_gap": 0.4,
                    "interaction_gap_ratio": 0.5,
                    "vp_plus_energy_per_cell": -0.2,
                    "vp_minus_energy_per_cell": -0.2,
                    "ivc_energy_per_cell": -0.1,
                    "ivc_minus_best_vp_energy_per_cell": 0.1,
                    "chern_vp_plus": 1.0,
                    "chern_vp_minus": -1.0,
                    "chern_ivc": 0.0,
                    "hf_all_converged": True,
                }
            }
        )
    )
    (point_dir / "reference_diagnostics.csv").write_text(
        "b_index,u_index,b1,u1,reference,energy_per_cell\n"
        "0,0,-0.3,-0.3,VP+,-0.2\n"
    )
    (point_dir / "hf_chern_numbers.csv").write_text(
        "b_index,u_index,b1,u1,reference,chern\n"
        "0,0,-0.3,-0.3,VP+,1.0\n"
    )
    (point_dir / "path_theta_edges.csv").write_text(
        "b_index,u_index,b1,u1,theta,energy_total_per_cell\n"
        "0,0,-0.3,-0.3,0.0,-0.2\n"
    )
    (point_dir / "response_K_theta.csv").write_text(
        "b_index,u_index,b1,u1,theta,K_theta,cG\n"
        "0,0,-0.3,-0.3,0.1,0.2,-0.05\n"
    )

    subprocess.run(
        [sys.executable, str(SCRIPT), "--output-root", str(output_root), "--merge-only"],
        check=True,
    )

    merged = json.loads((output_root / "sweep.json").read_text())
    assert merged["n_points"] == 1
    assert merged["rows"][0]["cG"] == -0.05
    assert merged["stacked_counts"]["hf_chern_numbers"] == 1
    assert "reference_diagnostics_csv" in merged["tables"]
    assert "VP+,-0.2" in (output_root / "sweep_reference_diagnostics.csv").read_text()
    arrays = np.load(output_root / "sweep_arrays.npz")
    assert arrays["cG"][0, 0] == -0.05
    assert arrays["chern_vp_plus"][0, 0] == 1.0
    assert "vp_plus_gap_mev" in arrays
    assert "ivc_indirect_gap_mev" in arrays
    assert "path_indirect_gap_min_mev" in arrays


def test_ac_b1_u1_sweep_tiny_point_runs_overlap_response(tmp_path):
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

    point_dir = output_root / "points" / "b_000_u_000"
    summary = json.loads((point_dir / "point_summary.json").read_text())
    row = summary["row"]
    assert row["n_active_bands_per_valley"] == 1
    assert row["active_band"] == 0
    assert row["n_ll"] == 1
    assert row["energy_unit"] == "hbar_omega_c"
    assert (
        row["density_form_factor_convention"]
        == "magnetic_bloch_source_to_target_folded_q_plus_g_v1"
    )
    assert np.isclose(
        row["vp_plus_gap_mev"],
        row["vp_plus_gap_over_omega_c"] * row["landau_level_spacing_mev"],
    )
    assert np.isclose(
        row["min_direct_gap_mev"],
        row["min_direct_gap_over_omega_c"] * row["landau_level_spacing_mev"],
    )
    assert np.isclose(
        row["interaction_gap_ratio"],
        row["characteristic_coulomb_to_active_band_gap_ratio"],
    )
    assert np.isclose(
        row["path_gap_min"],
        row["path_direct_gap_min_over_omega_c"],
    )
    assert np.isclose(
        row["path_direct_gap_min_mev"],
        row["path_direct_gap_min_over_omega_c"]
        * row["landau_level_spacing_mev"],
    )
    assert np.isclose(
        row["path_indirect_gap_min_mev"],
        row["path_indirect_gap_min_over_omega_c"]
        * row["landau_level_spacing_mev"],
    )
    assert row["hf_all_converged"] is True
    assert row["reference_chern_valid"] is True
    assert row["response_status"] == "ok"
    assert abs(row["cG"]) > 1e-3
    assert abs(row["chern_vp_plus"] - 1.0) < 5e-3
    assert abs(row["chern_vp_minus"] + 1.0) < 5e-3
    assert abs(row["chern_ivc"]) < 5e-3
    assert (point_dir / "response.npz").exists()
    assert (point_dir / "reference_states.npz").exists()
    assert len(list(csv.DictReader((point_dir / "hf_chern_numbers.csv").open()))) == 3
    assert (output_root / "sweep.csv").exists()


def test_ac_b1_u1_sweep_job_maps_four_meshes_and_uses_multicore_physical_dual_gate():
    text = JOB.read_text()
    assert "#SBATCH --array=0-483%24" in text
    assert "#SBATCH -c 4" in text
    assert "#SBATCH --mem=24G" in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "scripts/scan_ac_projected_hf_b1_u1.py" in text
    assert 'B1_MIN=${B1_MIN:-"-0.1"}' in text
    assert 'B1_MAX=${B1_MAX:-"0.1"}' in text
    assert 'N_B1=${N_B1:-"11"}' in text
    assert 'U1_MIN=${U1_MIN:-"-0.1"}' in text
    assert 'U1_MAX=${U1_MAX:-"0.1"}' in text
    assert 'N_U1=${N_U1:-"11"}' in text
    assert 'N_LL=${N_LL:-"8"}' in text
    assert 'ACTIVE_BAND=${ACTIVE_BAND:-"0"}' in text
    assert 'N_K_LIST=${N_K_LIST:-"15,18,21,24"}' in text
    assert 'COULOMB_KIND=${COULOMB_KIND:-"dual_gate"}' in text
    assert 'Q_MESH=${Q_MESH:-"full"}' in text
    assert 'V0=${V0:-"0.1"}' in text
    assert 'EPSILON=${EPSILON:-"4.0"}' in text
    assert 'CONTINUUM_THETA_DEG=${CONTINUUM_THETA_DEG:-"3.5"}' in text
    assert 'CONTINUUM_A0_ANGSTROM=${CONTINUUM_A0_ANGSTROM:-"3.47"}' in text
    assert 'CONTINUUM_M_EFF=${CONTINUUM_M_EFF:-"0.62"}' in text
    assert 'VERTEX_WORKERS=${VERTEX_WORKERS:-"${SLURM_CPUS_PER_TASK:-1}"}' in text
    assert 'EXCHANGE_WORKERS=${EXCHANGE_WORKERS:-"${SLURM_CPUS_PER_TASK:-1}"}' in text
    assert "export OMP_NUM_THREADS=1" in text
    assert "MESH_INDEX=$((GLOBAL_TASK_ID / POINTS_PER_MESH))" in text
    assert "POINT_TASK_ID=$((GLOBAL_TASK_ID % POINTS_PER_MESH))" in text
    assert "--no-write-plan" in text


def test_ac_b1_u1_sweep_no_write_plan_leaves_shared_plan_untouched(tmp_path):
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
            "--b1",
            "0.0",
            "--u1",
            "0.0",
            "--dry-run",
            "--no-write-plan",
        ],
        check=True,
    )
    assert json.loads(plan_path.read_text()) == {"sentinel": True}
