import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from chiral_dw.continuum.finite_q_memory_profile import (
    FINITE_Q_BUILD_VARIANTS,
    FiniteQBuildProfileParams,
    estimate_finite_q_build_arrays,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "profile_taige_finite_q_build_memory.py"


def test_finite_q_build_array_estimates_match_production_nk30():
    estimates = estimate_finite_q_build_arrays(FiniteQBuildProfileParams(n_k=30))

    assert estimates.n_blocks == 900
    assert estimates.n_q == 900
    assert estimates.n_g == 81
    assert estimates.compact_vertices_gib == pytest.approx(7.821321487426758)
    assert estimates.source_plus_rolled_vertices_gib == pytest.approx(
        15.642642974853516
    )
    assert estimates.valley_sector_exchange_gib == pytest.approx(
        0.7724761962890625
    )
    assert estimates.max_worker_result_slab_gib < 0.02


def test_finite_q_build_memory_profiler_uses_fresh_subprocesses(tmp_path):
    output_dir = tmp_path / "profile"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-dir",
            str(output_dir),
            "--n-k-list",
            "6",
            "--variants",
            ",".join(FINITE_Q_BUILD_VARIANTS),
            "--plane-wave-shell",
            "1",
            "--n-bands",
            "1",
            "--n-active-bands-per-valley",
            "1",
            "--local-field-cutoff",
            "0",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )

    payload = json.loads((output_dir / "finite_q_build_profile.json").read_text())
    results = payload["results"]
    assert len(results) == 3
    assert len({row["worker"]["worker_pid"] for row in results}) == 3
    assert all(row["worker"]["completed"] for row in results)
    assert all(row["returncode"] == 0 for row in results)
    assert all(row["worker"]["n_blocks"] == 36 for row in results)
    assert all(row["worker"]["exchange_representation"] == "valley_sector" for row in results)
    assert (output_dir / "finite_q_build_profile.csv").exists()
