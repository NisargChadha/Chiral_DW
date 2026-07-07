import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from chiral_dw.continuum import (
    MomentumGrid,
    chern_number_table,
    compute_taige_bandstructure,
    taige_interaction_params,
    taige_model_params,
    taige_wse2_interaction_params,
    taige_wse2_model_params,
)


ROOT = Path(__file__).resolve().parents[1]
CG_SCRIPT = ROOT / "scripts" / "scan_wse2_continuum_cg.py"
CACHE_SCRIPT = ROOT / "scripts" / "precompute_wse2_backend_cache.py"
HYST_SCRIPT = ROOT / "scripts" / "scan_wse2_ivc_hysteresis_linecut.py"
CG_JOB = ROOT / "jobs" / "scan_wse2_continuum_cg_array.sh"
CACHE_JOB = ROOT / "jobs" / "precompute_wse2_backend_cache_array.sh"
HYST_JOB = ROOT / "jobs" / "scan_wse2_ivc_hysteresis_all_linecuts_array.sh"
BY_THETA_JOB = ROOT / "jobs" / "scan_wse2_ivc_hysteresis_by_theta.sh"
MERGE_JOB = ROOT / "jobs" / "merge_wse2_ivc_hysteresis_sweep.sh"
SUBMIT_JOB = ROOT / "jobs" / "submit_wse2_ivc_hysteresis_full_pipeline.sh"
FINITE_SUBMIT_JOB = ROOT / "jobs" / "submit_wse2_ivc_hysteresis_finite_size_pipeline.sh"
RECOMPUTE_SUBMIT_JOB = ROOT / "jobs" / "submit_recompute_hysteresis_cg_from_projectors_pipeline.sh"
FS_CG_SCRIPT = ROOT / "scripts" / "scan_wse2_finite_size_cg.py"
FS_HYST_MERGE_SCRIPT = ROOT / "scripts" / "merge_wse2_ivc_hysteresis_finite_size.py"
FS_CG_JOB = ROOT / "jobs" / "scan_wse2_finite_size_cg_array.sh"
FS_CG_MERGE_JOB = ROOT / "jobs" / "merge_wse2_finite_size_cg.sh"
FS_CG_SUBMIT_JOB = ROOT / "jobs" / "submit_wse2_finite_size_by_nk.sh"
FS_HYST_MERGE_JOB = ROOT / "jobs" / "merge_wse2_ivc_hysteresis_finite_size.sh"
WSE2_CLEANUP_JOB = ROOT / "jobs" / "cleanup_wse2_backend_cache.sh"
BACKEND_CLEANUP_SCRIPT = ROOT / "scripts" / "cleanup_taige_backend_cache.py"


def test_wse2_taige_presets_and_mote2_defaults():
    mote2 = taige_model_params(theta_deg=3.5, u_D=0.0)
    wse2 = taige_wse2_model_params(theta_deg=3.5, u_D=0.0)

    assert mote2.material == "MoTe2_Taige"
    assert mote2.a0_angstrom == 3.47
    assert mote2.phi_deg == 91.0
    assert wse2.material == "WSe2_Taige"
    assert wse2.a0_angstrom == 3.32
    assert wse2.m_eff == 0.43
    assert wse2.moire_potential_mev == 9.0
    assert wse2.phi_deg == -128.0
    assert wse2.tunneling_mev == 18.0
    assert wse2.active_model == "taige"

    mote2_interaction = taige_interaction_params()
    wse2_interaction = taige_wse2_interaction_params()
    assert np.isclose(mote2_interaction.smear_length_nm, 0.347)
    assert np.isclose(wse2_interaction.smear_length_nm, 0.332)
    assert wse2_interaction.epsilon == 16.7
    assert wse2_interaction.gate_distance_nm == 30.0


def test_wse2_noninteracting_chern_matches_paper_sign_convention():
    model = taige_wse2_model_params(
        theta_deg=3.5,
        u_D=0.0,
        plane_wave_shell=2,
        n_bands=2,
        n_active_bands_per_valley=2,
    )
    bands = compute_taige_bandstructure(model, MomentumGrid(15))
    rows = chern_number_table(bands, bases=("electron",), band_indices=(0,))
    k_row = [row for row in rows if row.valley == "K" and row.band == 0][0]

    assert np.isclose(k_row.chern, 1.0)


def test_wse2_cg_dry_run_writes_material_plan(tmp_path):
    output_root = tmp_path / "wse2_cg"
    subprocess.run(
        [
            sys.executable,
            str(CG_SCRIPT),
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
    assert plan["args"]["material"] == "wse2"
    assert plan["args"]["n_k"] == 24
    assert plan["args"]["theta_min_deg"] == 3.0
    assert plan["args"]["theta_max_deg"] == 3.5
    assert plan["points"][0]["u_index"] == 1
    assert plan["points"][0]["theta_index"] == 1
    assert "points/u_001_theta_001" in plan["points"][0]["point_dir"]


def test_wse2_cg_tiny_smoke_writes_material_summary(tmp_path):
    output_root = tmp_path / "wse2_cg"
    subprocess.run(
        [
            sys.executable,
            str(CG_SCRIPT),
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
    params = summary["params"]
    row = summary["row"]
    assert params["model"]["material"] == "WSe2_Taige"
    assert params["model"]["phi_deg"] == -128.0
    assert np.isclose(params["interaction"]["smear_length_nm"], 0.332)
    assert row["finite_q_ivc_enabled"] is False
    assert row["selected_ivc_branch"] == "q0"


def test_wse2_cache_and_hysteresis_dry_runs_use_wse2_roots(tmp_path):
    cache_root = tmp_path / "wse2_cache"
    subprocess.run(
        [
            sys.executable,
            str(CACHE_SCRIPT),
            "--output-root",
            str(cache_root),
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
    cache_plan = json.loads((cache_root / "backend_cache_plan.json").read_text())
    assert cache_plan["args"]["material"] == "wse2"
    assert cache_plan["n_selected"] == 1
    assert any(row["selected"] for row in cache_plan["rows"])

    hyst_root = tmp_path / "wse2_hyst"
    subprocess.run(
        [
            sys.executable,
            str(HYST_SCRIPT),
            "--output-root",
            str(hyst_root),
            "--sweep-axis",
            "both",
            "--n-u-d",
            "2",
            "--n-twist",
            "3",
            "--task-id",
            "4",
            "--dry-run",
        ],
        check=True,
    )
    hyst_plan = json.loads((hyst_root / "hysteresis_branch_plan.json").read_text())
    assert hyst_plan["args"]["material"] == "wse2"
    assert hyst_plan["n_tasks"] == 10
    assert "wse2_hyst" in hyst_plan["rows"][0]["branch_dir"]


def test_wse2_finite_size_cg_dry_run_writes_material_mesh_plan(tmp_path):
    output_root = tmp_path / "wse2_fs_cg"
    subprocess.run(
        [
            sys.executable,
            str(FS_CG_SCRIPT),
            "--output-root",
            str(output_root),
            "--n-k-list",
            "18,20",
            "--n-u-d",
            "2",
            "--n-twist",
            "2",
            "--task-id",
            "1",
            "--dry-run",
        ],
        check=True,
    )

    plan = json.loads((output_root / "sweep_plan.json").read_text())
    phase_plan = json.loads((output_root / "sweep_phase_plan.json").read_text())
    assert plan["args"]["material"] == "wse2"
    assert plan["args"]["output_root"] == str(output_root)
    assert phase_plan["n_phase_points"] == 1
    assert plan["n_mesh_points"] == 2
    assert [point["n_k"] for point in plan["points"]] == [18, 20]
    assert {point["theta_index"] for point in plan["points"]} == {1}
    assert "points/u_000_theta_001/nk_018" in plan["points"][0]["point_dir"]


def test_wse2_job_scripts_mirror_taige_controls():
    subprocess.run(
        [
            "bash",
            "-n",
            str(FS_CG_JOB),
            str(FS_CG_MERGE_JOB),
            str(FS_CG_SUBMIT_JOB),
            str(FS_HYST_MERGE_JOB),
            str(FINITE_SUBMIT_JOB),
            str(RECOMPUTE_SUBMIT_JOB),
            str(WSE2_CLEANUP_JOB),
        ],
        check=True,
    )

    cg_text = CG_JOB.read_text()
    assert "scripts/scan_wse2_continuum_cg.py" in cg_text
    assert 'OUTPUT_ROOT=${OUTPUT_ROOT:-"results/wse2_cg_nk24_active2_shell5_vp_region"}' in cg_text
    assert 'SMEAR_LENGTH_NM=${SMEAR_LENGTH_NM:-"0.332"}' in cg_text
    assert "--no-finite-q-ivc" in cg_text
    assert "--ivc-branch-policy" in cg_text
    assert "--density-vertex-layout" in cg_text

    cache_text = CACHE_JOB.read_text()
    assert "scripts/precompute_wse2_backend_cache.py" in cache_text
    assert 'OUTPUT_ROOT=${OUTPUT_ROOT:-"results/wse2_ivc_hysteresis_linear_interaction_nk24_active2_shell5_theta2_4_u0_20"}' in cache_text
    assert 'SMEAR_LENGTH_NM=${SMEAR_LENGTH_NM:-"0.332"}' in cache_text
    assert "--no-vp-references" in cache_text

    hyst_text = HYST_JOB.read_text()
    assert "scripts/scan_wse2_ivc_hysteresis_linecut.py" in hyst_text
    assert "--sweep-axis both" in hyst_text
    assert 'TRIAL_INTERPOLATION=${TRIAL_INTERPOLATION:-"linear_interaction"}' in hyst_text
    assert "--trial-interpolation \"$TRIAL_INTERPOLATION\"" in hyst_text
    assert 'SMEAR_LENGTH_NM=${SMEAR_LENGTH_NM:-"0.332"}' in hyst_text
    assert "--compute-invalid-texture-cg" in hyst_text

    assert "scripts/scan_wse2_ivc_hysteresis_linecut.py" in BY_THETA_JOB.read_text()
    assert "scripts/merge_wse2_ivc_hysteresis_sweep.py" in MERGE_JOB.read_text()
    assert "jobs/precompute_wse2_backend_cache_array.sh" in SUBMIT_JOB.read_text()
    assert "jobs/scan_wse2_ivc_hysteresis_all_linecuts_array.sh" in SUBMIT_JOB.read_text()
    assert "jobs/merge_wse2_ivc_hysteresis_sweep.sh" in SUBMIT_JOB.read_text()

    fs_cg_text = FS_CG_JOB.read_text()
    assert "scripts/scan_wse2_finite_size_cg.py" in fs_cg_text
    assert 'OUTPUT_ROOT=${OUTPUT_ROOT:-"results/wse2_cg_finite_size_nk18_24_u0_15_theta2_4p2"}' in fs_cg_text
    assert 'N_K_LIST=${N_K_LIST:-"18,20,22,24"}' in fs_cg_text
    assert 'SMEAR_LENGTH_NM=${SMEAR_LENGTH_NM:-"0.332"}' in fs_cg_text
    assert "jobs/merge_wse2_finite_size_cg.sh" in fs_cg_text
    assert "scripts/scan_wse2_finite_size_cg.py" in FS_CG_MERGE_JOB.read_text()
    assert "jobs/scan_wse2_finite_size_cg_array.sh" in FS_CG_SUBMIT_JOB.read_text()

    finite_submit_text = FINITE_SUBMIT_JOB.read_text()
    assert 'OUTPUT_ROOT=${OUTPUT_ROOT:-"results/wse2_ivc_hysteresis_linear_interaction_finite_size_nk18_22_grid41"}' in finite_submit_text
    assert 'N_K_LIST=${N_K_LIST:-"18,19,20,21,22"}' in finite_submit_text
    assert 'FINAL_N_K_LIST=${FINAL_N_K_LIST:-"$N_K_LIST"}' in finite_submit_text
    assert 'NK_MEMORY_GB_MAP=${NK_MEMORY_GB_MAP:-"18:12,19:14,20:16,21:18,22:20,23:22,24:24"}' in finite_submit_text
    assert 'N_U_D=${N_U_D:-"41"}' in finite_submit_text
    assert 'N_TWIST=${N_TWIST:-"41"}' in finite_submit_text
    assert 'SMEAR_LENGTH_NM=${SMEAR_LENGTH_NM:-"0.332"}' in finite_submit_text
    assert 'TRIAL_INTERPOLATION=${TRIAL_INTERPOLATION:-"linear_interaction"}' in finite_submit_text
    assert "jobs/precompute_wse2_backend_cache_array.sh" in finite_submit_text
    assert "jobs/scan_wse2_ivc_hysteresis_all_linecuts_array.sh" in finite_submit_text
    assert "jobs/merge_wse2_ivc_hysteresis_sweep.sh" in finite_submit_text
    assert 'CLEANUP_BACKEND_CACHE=${CLEANUP_BACKEND_CACHE:-"1"}' in finite_submit_text
    assert 'CLEANUP_TIME=${CLEANUP_TIME:-"01:00:00"}' in finite_submit_text
    assert 'CLEANUP_MEM_GB=${CLEANUP_MEM_GB:-"2"}' in finite_submit_text
    assert "CACHE_BASE_ROOT" in finite_submit_text
    assert "LAB_SCRATCH_ROOT" in finite_submit_text
    assert "SCRATCH" in finite_submit_text
    assert "jobs/cleanup_wse2_backend_cache.sh" in finite_submit_text
    assert "jobs/merge_wse2_ivc_hysteresis_finite_size.sh" in finite_submit_text
    assert "submit_recompute_hysteresis_cg_from_projectors_pipeline.sh" in str(RECOMPUTE_SUBMIT_JOB)
    fs_hyst_merge_text = FS_HYST_MERGE_JOB.read_text()
    assert "scripts/merge_wse2_ivc_hysteresis_finite_size.py" in fs_hyst_merge_text
    assert 'OUTPUT_ROOT=${OUTPUT_ROOT:-"results/wse2_ivc_hysteresis_linear_interaction_finite_size_nk18_24_grid41"}' in fs_hyst_merge_text
    assert "export OMP_NUM_THREADS=1" in fs_hyst_merge_text

    cleanup_text = WSE2_CLEANUP_JOB.read_text()
    assert "scripts/cleanup_taige_backend_cache.py" in cleanup_text
    assert "--allowed-cache-base-root" in cleanup_text


def test_wse2_finite_size_cg_submitter_dry_run_builds_by_nk_commands(tmp_path):
    output_root = tmp_path / "wse2_by_nk"
    proc = subprocess.run(
        [str(FS_CG_SUBMIT_JOB)],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "OUTPUT_ROOT": str(output_root),
            "N_K_LIST": "18,20",
            "N_U_D": "2",
            "N_TWIST": "3",
            "CPUS_PER_TASK": "4",
        },
    )
    out = proc.stdout

    assert "--array=0-5" in out
    assert "--mem=18G" in out
    assert "--mem=20G" in out
    assert "jobs/scan_wse2_finite_size_cg_array.sh" in out
    assert "N_K_LIST=18" in out
    assert "N_K_LIST=20" in out
    assert "SMEAR_LENGTH_NM=0.332" in out
    assert not (output_root / "slurm_jobs_finite_size_by_nk.csv").exists()


def test_wse2_finite_size_submitter_dry_run_builds_split_mesh_commands(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "OUTPUT_ROOT": str(tmp_path / "wse2_fs"),
            "N_K_LIST": "18,22",
            "N_U_D": "3",
            "N_TWIST": "2",
            "CACHE_BASE_ROOT": str(tmp_path / "cache"),
            "DRY_RUN": "1",
        }
    )
    proc = subprocess.run(
        [str(FINITE_SUBMIT_JOB)],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    out = proc.stdout

    assert "jobs/precompute_wse2_backend_cache_array.sh" in out
    assert "jobs/scan_wse2_ivc_hysteresis_all_linecuts_array.sh" in out
    assert "jobs/merge_wse2_ivc_hysteresis_sweep.sh" in out
    assert "jobs/cleanup_taige_backend_cache.sh" not in out
    assert "jobs/cleanup_wse2_backend_cache.sh" in out
    assert "jobs/merge_wse2_ivc_hysteresis_finite_size.sh" in out
    assert "--array=0-5" in out
    assert "--array=0-9" in out
    assert "--mem=12G" in out
    assert "--mem=20G" in out
    assert "SMEAR_LENGTH_NM=0.332" in out
    assert "TRIAL_INTERPOLATION=linear_interaction" in out
    assert f"CACHE_BASE_ROOT={tmp_path / 'cache'}" in out
    assert f"CACHE_ROOT={tmp_path / 'cache' / 'wse2_fs' / 'nk_018' / 'backend_cache'}" in out
    assert "N_K=18" in out
    assert "N_K=22" in out
    assert "--dependency=afterok:dry_merge_18" in out
    assert "--dependency=afterok:dry_cleanup_18" in out
    assert "--dependency=afterok:dry_cleanup_22" in out
    assert "Dry run task counts: cache=12 scan=20" in out


def test_wse2_recompute_submitter_dry_run_uses_material_defaults(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "MATERIAL": "wse2",
            "SOURCE_OUTPUT_ROOT": str(tmp_path / "wse2_source"),
            "OUTPUT_ROOT": str(tmp_path / "wse2_recomputed"),
            "N_K_LIST": "18,22",
            "CACHE_BASE_ROOT": str(tmp_path / "cache"),
            "POINT_TASKS_PER_MESH": "3",
            "MAX_CONCURRENT_RECOMPUTE": "2",
        }
    )
    proc = subprocess.run(
        [str(RECOMPUTE_SUBMIT_JOB)],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
    out = proc.stdout

    assert "MATERIAL=wse2" in out
    assert "SMEAR_LENGTH_NM=0.332" in out
    assert "--mem=12G" in out
    assert "--mem=20G" in out
    assert "--array=0-2%2" in out
    assert "POINT_TASKS_PER_MESH=3" in out
    assert "SKIP_MERGE=1" in out
    assert "TRIAL_INTERPOLATION=linear_interaction" in out
    assert "jobs/recompute_hysteresis_cg_from_projectors_by_mesh_array.sh" in out
    assert "jobs/merge_wse2_ivc_hysteresis_sweep.sh" in out
    assert "jobs/merge_wse2_ivc_hysteresis_finite_size.sh" in out
    assert "Dry run task counts: recompute_meshes=2 point_tasks_per_mesh=3 total_recompute_tasks=6 final_merge=1" in out


def test_wse2_cleanup_allows_declared_scratch_cache_after_merge_outputs(tmp_path):
    output_root = tmp_path / "out" / "nk_018"
    output_root.mkdir(parents=True)
    for name in (
        "hysteresis_sweep.csv",
        "hysteresis_comparison.csv",
        "hysteresis_all_branch_candidates.csv",
        "hysteresis_selected_trial_theta.csv",
        "hysteresis_vp_chern_numbers.csv",
    ):
        (output_root / name).write_text("ok\n")

    cache_base = tmp_path / "scratch_cache"
    cache_root = cache_base / "run" / "nk_018" / "backend_cache"
    cache_root.mkdir(parents=True)
    (cache_root / "cache_point.npz").write_bytes(b"cache")

    refused = subprocess.run(
        [
            sys.executable,
            str(BACKEND_CLEANUP_SCRIPT),
            "--output-root",
            str(output_root),
            "--cache-root",
            str(cache_root),
        ],
        capture_output=True,
        text=True,
    )
    assert refused.returncode == 2
    assert cache_root.exists()

    subprocess.run(
        [
            sys.executable,
            str(BACKEND_CLEANUP_SCRIPT),
            "--output-root",
            str(output_root),
            "--cache-root",
            str(cache_root),
            "--allowed-cache-base-root",
            str(cache_base),
        ],
        check=True,
    )
    assert not cache_root.exists()
    manifest = json.loads((output_root / "backend_cache_cleanup_manifest.json").read_text())
    assert manifest["status"] == "deleted"
    assert manifest["deleted"] is True
    assert manifest["file_count"] == 1
