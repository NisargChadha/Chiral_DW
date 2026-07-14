from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "Plots"
        / "plot_ac_b1_u1_cg_finite_size_scaling.py"
    )
    spec = importlib.util.spec_from_file_location("plot_ac_b1_u1_cg_finite_size_scaling", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_sweep(path: Path, n_k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "b1",
        "u1",
        "cG",
        "n_k",
        "n_ll",
        "v0_over_omega_c",
        "status",
        "response_status",
        "hf_all_converged",
        "reference_chern_valid",
        "band_chern",
        "chern_ivc",
        "min_direct_gap",
        "path_gap_min",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for b1 in (-0.1, 0.0, 0.1):
            for u1 in (-0.1, 0.0, 0.1):
                writer.writerow(
                    {
                        "b1": b1,
                        "u1": u1,
                        "cG": -0.073 + 0.002 * b1 - 0.001 * u1 + 0.05 / n_k**2,
                        "n_k": n_k,
                        "n_ll": 6,
                        "v0_over_omega_c": 0.1,
                        "status": "ok",
                        "response_status": "ok",
                        "hf_all_converged": True,
                        "reference_chern_valid": True,
                        "band_chern": 1.0,
                        "chern_ivc": 0.0,
                        "min_direct_gap": 0.5,
                        "path_gap_min": 0.18,
                    }
                )


def test_fit_model_recovers_inverse_square_intercept():
    module = _load_module()
    n_k = np.arange(18, 23)
    intercept = np.array([-0.073, -0.072])
    cG = intercept[None, :] + 0.05 / n_k[:, None] ** 2

    fit = module.fit_model(n_k, cG, (2,))

    assert np.allclose(fit.intercept, intercept)
    assert np.max(np.abs(fit.residuals)) < 1e-14
    assert np.max(np.abs(fit.loo_residuals)) < 1e-14


def test_complete_synthetic_sweeps_render_all_artifacts(tmp_path: Path):
    module = _load_module()
    for n_k in range(18, 23):
        _write_sweep(tmp_path / f"nk{n_k}" / "sweep.csv", n_k)
    output = tmp_path / "finite_size.png"
    params = module.ACB1U1CGFiniteSizeParams(
        input_template=str(tmp_path / "nk{n_k}" / "sweep.csv"),
        output=output,
        dpi=80,
    )

    png, pdf, table, summary_path, summary = module.render_finite_size_analysis(params)

    assert summary["all_meshes_passed_independent_audit"] is True
    assert summary["n_parameter_points"] == 9
    assert summary["scalar_fits"]["spatial_mean"]["inverse_n2"]["rmse"] < 1e-14
    assert all(
        item["spatial_map_status"] == "rejected"
        for item in summary["rejected_pointwise_extrapolation_diagnostics"].values()
    )
    exported_fields = next(csv.reader(table.open()))
    assert not any("infinity" in field for field in exported_fields)
    for path in (png, pdf, table, summary_path):
        assert path.exists()
        assert path.stat().st_size > 0


def test_audit_rejects_nonzero_ivc_chern(tmp_path: Path):
    module = _load_module()
    path = tmp_path / "sweep.csv"
    _write_sweep(path, 18)
    rows = list(csv.DictReader(path.open()))
    rows[0]["chern_ivc"] = "0.5"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="IVC C=0.5"):
        module._load_one_sweep(path, expected_n_k=18, expected_n_ll=6, expected_v0=0.1)
