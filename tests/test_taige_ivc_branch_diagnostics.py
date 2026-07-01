from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from chiral_dw.continuum.ivc_diagnostics import (
    projector_overlap_diagnostics,
    projector_overlap_diagnostics_with_frames,
)


def _projector(active_index: int, *, n_blocks: int = 2, dim: int = 2) -> np.ndarray:
    P = np.zeros((n_blocks, dim, dim), dtype=complex)
    P[:, active_index, active_index] = 1.0
    return P


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_projector_overlap_metrics_identify_equal_and_orthogonal_projectors():
    P = _projector(0)
    same = projector_overlap_diagnostics(P, P, n_occ_per_k=1)
    assert same.mean_overlap == pytest.approx(1.0)
    assert same.one_minus_mean_overlap == pytest.approx(0.0)
    assert same.frobenius_distance == pytest.approx(0.0)

    Q = _projector(1)
    orthogonal = projector_overlap_diagnostics(P, Q, n_occ_per_k=1)
    reverse = projector_overlap_diagnostics(Q, P, n_occ_per_k=1)
    assert orthogonal.mean_overlap == pytest.approx(0.0)
    assert orthogonal.frobenius_distance == pytest.approx(np.sqrt(2.0))
    assert reverse.mean_overlap == pytest.approx(orthogonal.mean_overlap)
    assert reverse.frobenius_distance == pytest.approx(orthogonal.frobenius_distance)


def test_projector_overlap_with_frames_compares_physical_embedded_subspaces():
    P = _projector(0, n_blocks=1)
    Q = _projector(1, n_blocks=1)
    frame_left = np.eye(2, dtype=complex)[None, :, :]
    frame_right = np.asarray([[[0.0, 1.0], [1.0, 0.0]]], dtype=complex)

    active_basis = projector_overlap_diagnostics(P, Q, n_occ_per_k=1)
    embedded = projector_overlap_diagnostics_with_frames(
        P,
        Q,
        frame_left,
        frame_right,
        n_occ_per_k=1,
    )

    assert active_basis.mean_overlap == pytest.approx(0.0)
    assert embedded.mean_overlap == pytest.approx(1.0)
    assert embedded.frobenius_distance == pytest.approx(0.0)


def test_local_ivc_branch_diagnostic_script_smoke(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_taige_ivc_branch_local.py"
    output_root = tmp_path / "diagnostics"
    cmd = [
        sys.executable,
        str(script),
        "--output-root",
        str(output_root),
        "--run-label",
        "smoke",
        "--preset",
        "custom",
        "--diagnostic-mode",
        "all",
        "--theta-deg-list",
        "3.5",
        "--u-d-list",
        "0.0,0.25",
        "--convergence-point-list",
        "0.0:3.5",
        "--convergence-max-iters",
        "1",
        "--n-k",
        "2",
        "--plane-wave-shell",
        "0",
        "--n-bands",
        "1",
        "--n-active-bands-per-valley",
        "1",
        "--q-mesh",
        "full",
        "--local-field-cutoff",
        "0",
        "--max-iter",
        "1",
        "--min-iter",
        "0",
        "--snapshot-interval",
        "1",
        "--random-seeds",
        "7",
        "--no-include-ordered-seed",
        "--no-solve-vp-baseline",
        "--skip-plots",
    ]
    subprocess.run(cmd, check=True, timeout=120)

    out_dir = output_root / "smoke"
    runs = _read_csv(out_dir / "runs.csv")
    history = _read_csv(out_dir / "iteration_history.csv")
    snapshots = _read_csv(out_dir / "projector_snapshot_manifest.csv")
    neighbor = _read_csv(out_dir / "projector_overlaps_neighbor.csv")
    hysteresis = _read_csv(out_dir / "hysteresis.csv")

    assert len(runs) == 7
    assert history
    assert snapshots
    assert neighbor
    assert hysteresis
    assert {"final_aufbau_residual_norm", "final_self_consistency_warning"} <= set(runs[0])
    assert {"aufbau_residual_norm", "iteration"} <= set(history[0])
    with np.load(out_dir / "projectors_final.npz") as arrays:
        assert len(arrays.files) == len(runs)
