from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")


def _load_plot_module():
    path = Path(__file__).resolve().parents[1] / "Plots" / "energy_vs_radius_schematic.py"
    spec = importlib.util.spec_from_file_location("energy_vs_radius_schematic", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_energy_terms_match_surface_and_dipole_formulas():
    module = _load_plot_module()
    radius = np.array([0.25, 0.5])
    sigma = 1.7
    charge_strength = 0.21

    surface, dipole, total = module.energy_terms(
        radius,
        sigma=sigma,
        charge_strength=charge_strength,
    )

    assert np.allclose(surface, 2.0 * np.pi * sigma * radius)
    assert np.allclose(dipole, 2.0 * np.pi * charge_strength**2 / radius)
    assert np.allclose(total, surface + dipole)


def test_characteristic_radius_minimizes_total_energy():
    module = _load_plot_module()
    sigma = 1.3
    charge_strength = 0.2
    r_star = module.characteristic_radius(sigma=sigma, charge_strength=charge_strength)

    _, _, total_star = module.energy_terms(
        r_star,
        sigma=sigma,
        charge_strength=charge_strength,
    )
    _, _, total_left = module.energy_terms(
        0.9 * r_star,
        sigma=sigma,
        charge_strength=charge_strength,
    )
    _, _, total_right = module.energy_terms(
        1.1 * r_star,
        sigma=sigma,
        charge_strength=charge_strength,
    )

    assert np.isclose(r_star, charge_strength / np.sqrt(sigma))
    assert float(total_star) < float(total_left)
    assert float(total_star) < float(total_right)


def test_default_output_goes_to_plots_figures():
    module = _load_plot_module()
    params = module.EnergyRadiusSchematicParams()

    assert params.output == Path("Plots/figures/energy_vs_radius_schematic.png")


def test_default_figure_canvas_is_square():
    module = _load_plot_module()
    params = module.EnergyRadiusSchematicParams()

    assert params.figure_width == params.figure_height


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
        assert module.plt.rcParams["legend.fontsize"] == module.NISARG_FONTS["legend"]
        assert module.NISARG_FONTS["axis_label"] == 34
        assert module.NISARG_FONTS["tick_label"] == 26
        assert module.NISARG_FONTS["legend"] == 17


def test_boxed_axis_keeps_all_spines_visible():
    module = _load_plot_module()

    with module.matplotlib.rc_context():
        fig, ax = module.plt.subplots()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        module.style_boxed_axis(ax)

        assert all(spine.get_visible() for spine in ax.spines.values())
        module.plt.close(fig)


def test_energy_radius_schematic_writes_png_and_pdf(tmp_path: Path):
    module = _load_plot_module()
    output = tmp_path / "energy_vs_radius_schematic.png"
    params = module.EnergyRadiusSchematicParams(
        r_min=0.02,
        r_max=0.7,
        n_points=80,
        output=output,
        dpi=90,
    )

    png_path, pdf_path = module.render_energy_radius_schematic(params)

    assert png_path == output
    assert pdf_path == output.with_suffix(".pdf")
    assert png_path.exists()
    assert pdf_path.exists()
    assert png_path.stat().st_size > 0
    assert pdf_path.stat().st_size > 0

    image = module.plt.imread(png_path)
    assert image.shape[0] == image.shape[1]
