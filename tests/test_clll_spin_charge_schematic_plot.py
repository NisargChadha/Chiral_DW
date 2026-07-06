from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")


def _load_plot_module():
    path = Path(__file__).resolve().parents[1] / "Plots" / "clll_spin_charge_schematic.py"
    spec = importlib.util.spec_from_file_location("clll_spin_charge_schematic", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _small_plot_params(module, output: Path | None = None):
    return module.CLLLSchematicPlotParams(
        n_k=3,
        n_r=7,
        radius_lB=2.2,
        width_lB=0.45,
        patch_length_lB=8.0,
        spin_stride=2,
        theta_count=9,
        output=output or Path("unused.png"),
        dpi=90,
    )


def test_clll_charge_density_grid_preserves_integrated_charge():
    module = _load_plot_module()
    result = module.compute_clll_response(_small_plot_params(module))
    charge = module.compute_charge_density_grid(result)

    assert np.allclose(charge.rho_density * charge.plaquette_area, result.rho_top)
    assert np.isclose(
        charge.integrated_charge_from_density,
        charge.integrated_charge_from_plaquettes,
    )


def test_origin_artifact_regularization_is_display_only():
    module = _load_plot_module()
    x, y = np.meshgrid(np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0]), indexing="ij")
    density = -2.0 * np.ones((3, 3), dtype=float)
    density[1, 1] = 100.0

    cleaned = module.regularize_origin_artifact(
        density,
        x,
        y,
        radius=0.25,
        transition_radius=1.0,
    )

    assert cleaned[1, 1] == -2.0
    assert density[1, 1] == 100.0


def test_default_wall_lengths_use_requested_values():
    module = _load_plot_module()
    params = module.CLLLSchematicPlotParams()
    magnetic_length = module.triangular_moire_magnetic_length()
    radius, width = module.clll_wall_lengths(params)

    assert np.isclose(radius / magnetic_length, 25.0)
    assert np.isclose(width / magnetic_length, 8.0)


def test_default_output_goes_to_plots_figures():
    module = _load_plot_module()
    params = module.CLLLSchematicPlotParams()

    assert params.output == Path("Plots/figures/clll_spin_charge_schematic.png")


def test_charge_display_grid_is_upsampled_from_numerical_density():
    module = _load_plot_module()
    params = _small_plot_params(module)
    result = module.compute_clll_response(params)
    charge = module.compute_charge_density_grid(result)
    display = module.compute_charge_display_grid(charge, params)

    assert display.rho_density.shape[0] == charge.rho_density.shape[0] * params.charge_upsample
    assert display.rho_density.shape[1] == charge.rho_density.shape[1] * params.charge_upsample
    assert display.x_edges.shape == tuple(size + 1 for size in display.rho_density.shape)
    assert display.y_edges.shape == display.x_edges.shape


def test_charge_display_grid_uses_plot_half_width():
    module = _load_plot_module()
    params = _small_plot_params(module).model_copy(update={"plot_half_width_lB": 3.0})
    result = module.compute_clll_response(params)
    charge = module.compute_charge_density_grid(result)
    display = module.compute_charge_display_grid(charge, params)
    half_width = module.plot_half_width(params)

    assert np.isclose(np.min(display.x_edges), -half_width)
    assert np.isclose(np.max(display.x_edges), half_width)
    assert np.isclose(np.min(display.y_edges), -half_width)
    assert np.isclose(np.max(display.y_edges), half_width)


def test_electron_charge_density_display_uses_negative_e_sign():
    module = _load_plot_module()
    number_density = np.array([-2.0, 0.0, 3.5])

    assert np.allclose(
        module.electron_charge_density_over_e(number_density),
        np.array([2.0, -0.0, -3.5]),
    )


def test_clll_spin_charge_schematic_writes_png_and_pdf(tmp_path: Path):
    module = _load_plot_module()
    output = tmp_path / "clll_spin_charge_schematic.png"
    params = _small_plot_params(module, output)

    png_path, pdf_path = module.render_clll_spin_charge_schematic(params)

    assert png_path == output
    assert pdf_path == output.with_suffix(".pdf")
    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0
