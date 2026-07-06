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
