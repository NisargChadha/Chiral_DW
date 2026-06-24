import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from chiral_dw.ac.bias_sweep import active_band_path
from chiral_dw.config import FirstShellACParams


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_lll_u1c3_sweep_smoke_writes_old_compatible_outputs(tmp_path: Path):
    out_dir = tmp_path / "lll_u1c3"
    cmd = [
        sys.executable,
        "scripts/sweep_conjugate_ac_lll_u1c3_cg.py",
        "--output-dir",
        str(out_dir),
        "--n-k",
        "3",
        "--n-theta",
        "5",
        "--n-u1-c3",
        "3",
        "--n-phi-check",
        "3",
        "--interaction-shell",
        "1",
        "--max-iter",
        "30",
        "--dispersion-points",
        "5",
        "--use-physical-coulomb",
        "--quiet",
    ]

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    expected = [
        "u1c3_cg_sweep.csv",
        "u1c3_cg_sweep.json",
        "u1c3_cg_sweep.npz",
        "cG_vs_u1c3.png",
        "Kappa_theta_vs_u1c3.png",
        "dispersion_split_vs_u1c3.png",
        "lll_band_structure_u1c3_min.png",
        "lll_band_structure_u1c3_max.png",
    ]
    for name in expected:
        assert (out_dir / name).exists(), name

    payload = json.loads((out_dir / "u1c3_cg_sweep.json").read_text())
    metadata = payload["metadata"]
    assert metadata["LL_gap"] == 1.0
    assert metadata["units"] == "omega_c"
    assert metadata["projection"] == "lowest_landau_level"
    assert metadata["n_ll"] == 1
    assert metadata["interaction_matching"] == "physical_coulomb"
    assert np.isclose(metadata["V0_dimensionless"], 0.267, rtol=5e-3)
    assert np.isclose(metadata["gate_distance_dimensionless"], 5.80, rtol=5e-3)
    assert "physical_coulomb_match" in payload

    rows = list(csv.DictReader((out_dir / "u1c3_cg_sweep.csv").open()))
    assert len(rows) == 3
    cG = np.asarray([float(row["cG"]) for row in rows])
    split = np.asarray([float(row["max_k_kprime_dispersion_split"]) for row in rows])
    assert np.all(np.isfinite(cG))
    assert split[0] < 1e-12
    assert split[-1] > 1e-3

    data = np.load(out_dir / "u1c3_cg_sweep.npz")
    assert data["Kappa"].shape[0] == 3
    assert np.all(np.isfinite(data["Kappa"]))
    assert data["lll_band_up"].shape == data["lll_band_down"].shape


def test_b1c3_sweep_smoke_writes_old_compatible_outputs(tmp_path: Path):
    out_dir = tmp_path / "b1c3"
    cmd = [
        sys.executable,
        "scripts/sweep_conjugate_ac_b1c3_cg.py",
        "--output-dir",
        str(out_dir),
        "--n-ll",
        "3",
        "--n-k",
        "3",
        "--n-theta",
        "5",
        "--n-b1-c3",
        "3",
        "--n-phi-check",
        "3",
        "--interaction-shell",
        "1",
        "--max-iter",
        "30",
        "--dispersion-points",
        "5",
        "--use-physical-coulomb",
        "--quiet",
    ]

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

    expected = [
        "b1c3_cg_sweep.csv",
        "b1c3_cg_sweep.json",
        "b1c3_cg_sweep.npz",
        "cG_vs_b1c3.png",
        "Kappa_theta_vs_b1c3.png",
        "dispersion_split_vs_b1c3.png",
        "lowest_band_structure_b1c3_min.png",
        "lowest_band_structure_b1c3_max.png",
    ]
    for name in expected:
        assert (out_dir / name).exists(), name

    payload = json.loads((out_dir / "b1c3_cg_sweep.json").read_text())
    metadata = payload["metadata"]
    assert metadata["units"] == "omega_c"
    assert metadata["projection"] == "finite_ll_lowest_active_band"
    assert metadata["sweep_parameter"] == "b1_c3"
    assert metadata["u1_c3_fixed"] == 0.0
    assert metadata["n_ll"] == 3
    assert metadata["b1_c3_strict_lll_matrix_element_zero"] is False
    assert metadata["interaction_matching"] == "physical_coulomb"
    assert np.isclose(metadata["V0_dimensionless"], 0.267, rtol=5e-3)
    assert np.isclose(metadata["gate_distance_dimensionless"], 5.80, rtol=5e-3)

    rows = list(csv.DictReader((out_dir / "b1c3_cg_sweep.csv").open()))
    assert len(rows) == 3
    cG = np.asarray([float(row["cG"]) for row in rows])
    split = np.asarray([float(row["max_k_kprime_dispersion_split"]) for row in rows])
    assert np.all(np.isfinite(cG))
    assert split[0] < 1e-12
    assert split[-1] > 1e-5

    data = np.load(out_dir / "b1c3_cg_sweep.npz")
    assert data["Kappa"].shape[0] == 3
    assert np.all(np.isfinite(data["Kappa"]))
    assert data["lowest_band_up"].shape == data["lowest_band_down"].shape


def test_b1c3_active_band_splitting_requires_finite_ll_mixing():
    strict_lll = active_band_path(
        FirstShellACParams(b1_c3=0.2, n_ll=1),
        n_segment=5,
    )
    finite_ll = active_band_path(
        FirstShellACParams(b1_c3=0.2, n_ll=3),
        n_segment=5,
    )

    assert strict_lll.max_split < 1e-12
    assert finite_ll.max_split > 1e-5
