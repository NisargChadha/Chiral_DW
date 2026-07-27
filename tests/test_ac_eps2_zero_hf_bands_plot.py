import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Plots" / "plot_ac_eps2_zero_hf_bands.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_ac_eps2_zero_hf_bands", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mesh_path_has_expected_endpoints_and_ticks():
    module = _load_module()
    flat, distance, ticks, labels, coords = module._mesh_path(15)

    assert coords[0] == (0, 0)
    assert coords[-1] == (0, 0)
    assert len(ticks) == len(labels) == 4
    assert flat.shape == distance.shape == (len(coords),)
    assert np.all(np.diff(distance) >= 0.0)
    assert labels[0] == labels[-1] == r"$\Gamma$"


def test_spectrum_diagnostics_distinguish_direct_and_indirect_gaps():
    module = _load_module()
    values = np.asarray(
        [
            [-2.0, 1.0],
            [0.0, 2.5],
            [-1.0, 4.0],
        ]
    )

    diagnostics = module._spectrum_diagnostics(values)

    assert diagnostics["direct_gap_min_mev"] == pytest.approx(2.5)
    assert diagnostics["indirect_gap_mev"] == pytest.approx(1.0)
    assert diagnostics["occupied_bandwidth_mev"] == pytest.approx(2.0)
    assert diagnostics["empty_bandwidth_mev"] == pytest.approx(3.0)


def test_default_plot_params_match_requested_strong_coupling_point():
    module = _load_module()
    params = module.ACHFPlotParams()

    assert params.n_k == 15
    assert params.n_ll == 6
    assert params.b1 == 0.0
    assert params.u1 == 0.0
    assert params.epsilon == 2.0
    assert params.characteristic_coulomb_mev / params.landau_level_spacing_mev > 0.25
