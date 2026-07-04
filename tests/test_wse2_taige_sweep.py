import json
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


def test_wse2_job_scripts_mirror_taige_controls():
    cg_text = CG_JOB.read_text()
    assert "scripts/scan_wse2_continuum_cg.py" in cg_text
    assert 'OUTPUT_ROOT=${OUTPUT_ROOT:-"results/wse2_cg_nk24_active2_shell5_vp_region"}' in cg_text
    assert 'SMEAR_LENGTH_NM=${SMEAR_LENGTH_NM:-"0.332"}' in cg_text
    assert "--no-finite-q-ivc" in cg_text
    assert "--ivc-branch-policy" in cg_text
    assert "--density-vertex-layout" in cg_text

    cache_text = CACHE_JOB.read_text()
    assert "scripts/precompute_wse2_backend_cache.py" in cache_text
    assert 'OUTPUT_ROOT=${OUTPUT_ROOT:-"results/wse2_ivc_hysteresis_nk24_active2_shell5_theta2_4_u0_20"}' in cache_text
    assert 'SMEAR_LENGTH_NM=${SMEAR_LENGTH_NM:-"0.332"}' in cache_text
    assert "--no-vp-references" in cache_text

    hyst_text = HYST_JOB.read_text()
    assert "scripts/scan_wse2_ivc_hysteresis_linecut.py" in hyst_text
    assert "--sweep-axis both" in hyst_text
    assert 'SMEAR_LENGTH_NM=${SMEAR_LENGTH_NM:-"0.332"}' in hyst_text
    assert "--compute-invalid-texture-cg" in hyst_text

    assert "scripts/scan_wse2_ivc_hysteresis_linecut.py" in BY_THETA_JOB.read_text()
    assert "scripts/merge_wse2_ivc_hysteresis_sweep.py" in MERGE_JOB.read_text()
    assert "jobs/precompute_wse2_backend_cache_array.sh" in SUBMIT_JOB.read_text()
    assert "jobs/scan_wse2_ivc_hysteresis_all_linecuts_array.sh" in SUBMIT_JOB.read_text()
    assert "jobs/merge_wse2_ivc_hysteresis_sweep.sh" in SUBMIT_JOB.read_text()
