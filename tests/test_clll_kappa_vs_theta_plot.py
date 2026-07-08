from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")


def _load_plot_module():
    path = Path(__file__).resolve().parents[1] / "Plots" / "clll_kappa_vs_theta.py"
    spec = importlib.util.spec_from_file_location("clll_kappa_vs_theta", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _small_plot_params(module, output: Path | None = None):
    return module.CLLLKappaThetaPlotParams(
        n_k=3,
        theta_count=21,
        n_r=5,
        output=output or Path("unused.png"),
        dpi=90,
    )


def test_default_output_goes_to_plots_figures():
    module = _load_plot_module()
    params = module.CLLLKappaThetaPlotParams()

    assert params.output == Path("Plots/figures/clll_kappa_vs_theta.png")
    assert params.n_k == 18
    assert params.theta_count == 321


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
        assert module.NISARG_FONTS["x_axis_label"] == 30
        assert module.NISARG_FONTS["x_tick_label"] == 25


def test_clll_kappa_response_is_antisymmetric():
    module = _load_plot_module()
    params = _small_plot_params(module)

    response = module.compute_clll_kappa_response(params)

    assert response.theta.shape == (params.theta_count - 1,)
    assert response.K.shape == response.theta.shape
    assert np.all(np.isfinite(response.K))
    assert np.isfinite(response.cG)
    assert np.allclose(response.K + response.K[::-1], 0.0, atol=1e-10)
    assert np.min(response.K) < 0.0 < np.max(response.K)


def test_clll_kappa_theta_plot_writes_png_and_pdf(tmp_path: Path):
    module = _load_plot_module()
    output = tmp_path / "clll_kappa_vs_theta.png"
    params = _small_plot_params(module, output)

    png_path, pdf_path = module.render_clll_kappa_theta_plot(params)

    assert png_path == output
    assert pdf_path == output.with_suffix(".pdf")
    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0
