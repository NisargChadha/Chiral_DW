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
                    "point_dir": str(point_dir),
                }
            }
        )
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
    csv_text = (output_root / "sweep.csv").read_text()
    assert "u_D_meV" in csv_text
    assert "1.25" in csv_text


def test_taige_sweep_job_uses_array_task_and_results_root():
    text = JOB.read_text()
    assert "#SBATCH --array=0-624" in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "scripts/scan_taige_continuum_cg.py" in text
    assert 'OUTPUT_ROOT=${OUTPUT_ROOT:-"results/taige_continuum_cg_sweep"}' in text
    assert "--merge-only" in text
