import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from chiral_dw.config import ContinuumFiniteQParams
from chiral_dw.continuum import TaigeSETWorkflowParams, build_continuum_bundle


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "jobs" / "submit_taige_set_hysteresis_nk24.sh"
SET_SCAN = ROOT / "scripts" / "scan_taige_set_spectrum.py"
HYSTERESIS_SCAN = ROOT / "scripts" / "scan_taige_set_hysteresis.py"


def test_taige_set_hysteresis_job_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(JOB)], check=True)


def test_taige_set_hysteresis_job_encodes_restartable_parallel_pipeline():
    text = JOB.read_text()
    assert 'N_K=${N_K:-"24"}' in text
    assert 'U_D_MIN=${U_D_MIN:-"0.0"}' in text
    assert 'U_D_MAX=${U_D_MAX:-"10.0"}' in text
    assert 'N_U_D=${N_U_D:-"21"}' in text
    assert 'EPSILON=${EPSILON:-"16.7"}' in text
    assert "taige_set_nk24_theta3_u0_10_hysteresis_step0p5" in text
    assert 'FILLING_WORKERS=${FILLING_WORKERS:-"3"}' in text
    assert "--array=0-1" in text
    assert "PIPELINE_STAGE=seed" in text
    assert "PIPELINE_STAGE=smoke" in text
    assert "PIPELINE_STAGE=branch" in text
    assert "PIPELINE_STAGE=merge" in text
    assert "--dependency=afterok:" in text
    assert "--max-points 1" in text
    assert "verify_smoke_artifacts" in text
    assert "--skip-existing" in text
    assert "--filling-workers" in text
    assert text.count("--projectors-only") == 2
    assert "hf_projectors.npz" in text
    assert "projectors only" in text
    assert "scripts/scan_taige_set_spectrum.py" in text
    assert "scripts/scan_taige_set_hysteresis.py" in text


def test_projector_only_seed_and_hysteresis_artifacts_are_reconstructible(tmp_path):
    seed_root = tmp_path / "seed"
    subprocess.run(
        [
            sys.executable,
            str(SET_SCAN),
            "--output-root",
            str(seed_root),
            "--u-d",
            "0",
            "--n-k",
            "2",
            "--particle-offset-max",
            "1",
            "--plane-wave-shell",
            "1",
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
            "0",
            "--max-iter",
            "3",
            "--min-iter",
            "1",
            "--dos-energy-points",
            "101",
            "--projectors-only",
        ],
        check=True,
        cwd=ROOT,
    )
    seed_point = seed_root / "points" / "uD_000p0000"
    seed_summary = json.loads((seed_point / "point_summary.json").read_text())
    assert seed_summary["state_storage"]["mode"] == "projectors_only"
    assert seed_summary["state_storage"]["hf_hamiltonian_reconstruction"] == (
        "backend.hf_hamiltonian(P)"
    )
    with np.load(seed_point / "hf_projectors.npz") as archive:
        assert set(archive.files) == {
            "fixed_vp_plus_P",
            "fixed_vp_minus_P",
            "global_N3_P",
            "global_N4_P",
            "global_N5_P",
        }
        neutral_projector = np.asarray(archive["global_N4_P"])

    params = TaigeSETWorkflowParams.model_validate(seed_summary["params"])
    bundle = build_continuum_bundle(
        model=params.model,
        grid=params.grid,
        interaction=params.interaction,
        finite_q=ContinuumFiniteQParams(enabled=False),
    )
    reconstructed = bundle.backend.hf_hamiltonian(neutral_projector)
    assert reconstructed.shape == neutral_projector.shape
    assert np.allclose(reconstructed, np.swapaxes(reconstructed.conj(), -1, -2))

    output_root = tmp_path / "hysteresis"
    subprocess.run(
        [
            sys.executable,
            str(HYSTERESIS_SCAN),
            "--output-root",
            str(output_root),
            "--direction",
            "up",
            "--seed-point-dir",
            str(seed_point),
            "--u-d-min",
            "0",
            "--u-d-max",
            "1",
            "--n-u-d",
            "2",
            "--max-iter",
            "3",
            "--max-points",
            "1",
            "--projectors-only",
        ],
        check=True,
        cwd=ROOT,
    )
    branch_point = output_root / "branches" / "up" / "uD_000p0000"
    branch_summary = json.loads((branch_point / "point_summary.json").read_text())
    assert branch_summary["state_storage"]["mode"] == "projectors_only"
    with np.load(branch_point / "hf_projectors.npz") as archive:
        assert set(archive.files) == {"global_N3_P", "global_N4_P", "global_N5_P"}
