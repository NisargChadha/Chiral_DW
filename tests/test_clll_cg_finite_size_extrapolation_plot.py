from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")


def _load_plot_module():
    path = Path(__file__).resolve().parents[1] / "Plots" / "clll_cg_finite_size_extrapolation.py"
    spec = importlib.util.spec_from_file_location("clll_cg_finite_size_extrapolation", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _small_plot_params(module, output: Path | None = None):
    return module.CLLLCGFiniteSizePlotParams(
        n_k_values=(3, 4),
        theta_count=17,
        n_r=5,
        output=output or Path("unused.png"),
        dpi=90,
    )


def test_default_sweep_matches_requested_values():
    module = _load_plot_module()
    params = module.CLLLCGFiniteSizePlotParams()

    assert params.n_k_values == (15, 16, 17, 18)
    assert params.theta_count == 641
    assert params.output == Path("Plots/figures/clll_cg_finite_size_extrapolation.png")


def test_nisarg_plot_style_uses_pt_serif_and_cm_mathtext():
    module = _load_plot_module()

    with module.matplotlib.rc_context():
        module.apply_nisarg_plot_style()

        assert module.plt.rcParams["font.family"] == ["serif"]
        assert module.plt.rcParams["font.serif"][:3] == [
            "PT Serif Caption",
            "PT Serif",
            "DejaVu Serif",
        ]
        assert module.plt.rcParams["mathtext.fontset"] == "cm"


def test_linear_fit_returns_intercept_at_zero_inverse_grid_size():
    module = _load_plot_module()
    x = np.array([0.25, 0.125, 0.0625])
    y = -0.08 + 0.015 * x

    slope, intercept = module.fit_linear_cg_extrapolation(x, y)

    assert slope == pytest.approx(0.015)
    assert intercept == pytest.approx(-0.08)


def test_small_cg_finite_size_data_is_finite():
    module = _load_plot_module()
    params = _small_plot_params(module)

    data = module.compute_cg_finite_size_data(params)

    assert data.n_k.tolist() == [3, 4]
    assert np.allclose(data.inverse_n_k, [1.0 / 3.0, 0.25])
    assert data.cG.shape == (2,)
    assert np.all(np.isfinite(data.cG))
    assert np.isfinite(data.slope)
    assert np.isfinite(data.intercept)


def test_clll_cg_finite_size_plot_writes_png_pdf_and_csv(tmp_path: Path):
    module = _load_plot_module()
    output = tmp_path / "clll_cg_finite_size_extrapolation.png"
    params = _small_plot_params(module, output)

    png_path, pdf_path, csv_path, data = module.render_clll_cg_finite_size_plot(params)

    assert png_path == output
    assert pdf_path == output.with_suffix(".pdf")
    assert csv_path == output.with_suffix(".csv")
    assert np.all(np.isfinite(data.cG))
    for path in (png_path, pdf_path, csv_path):
        assert path.exists()
        assert path.stat().st_size > 0
