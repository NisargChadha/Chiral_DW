import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "jobs" / "submit_taige_set_hysteresis_nk24.sh"


def test_taige_set_hysteresis_job_has_valid_shell_syntax():
    subprocess.run(["bash", "-n", str(JOB)], check=True)


def test_taige_set_hysteresis_job_encodes_restartable_parallel_pipeline():
    text = JOB.read_text()
    assert 'N_K=${N_K:-"24"}' in text
    assert 'U_D_MIN=${U_D_MIN:-"5.0"}' in text
    assert 'U_D_MAX=${U_D_MAX:-"6.0"}' in text
    assert 'N_U_D=${N_U_D:-"20"}' in text
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
    assert "scripts/scan_taige_set_spectrum.py" in text
    assert "scripts/scan_taige_set_hysteresis.py" in text
