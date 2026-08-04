from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")


def _load_plot_module():
    path = Path(__file__).resolve().parents[1] / "Plots" / "plot_grid41_linear_interaction_single_cg_heatmap.py"
    spec = importlib.util.spec_from_file_location("plot_grid41_linear_interaction_single_cg_heatmap", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tmote2_publication_heatmap_defaults_to_all_seven_fit_points():
    module = _load_plot_module()

    assert module.FINITE_SIZE_FIT_KIND == "all7"
    assert module.CG_VALUE_COLUMN == "cG_all_plot"
    assert module.CG_GREY_MASK_COLUMN == "grey_mask_all"
    assert "all7" in module.OUTPUT_STEM


def test_hf_phase_diagram_has_requested_categorical_colors():
    module = _load_plot_module()

    assert module.COLORS["vp_c0"] == "#FD4C55"
    assert module.COLORS["vp_c1"] == "#0072B2"
    assert module.COLORS["ivc"] == "0.72"
    assert [module.LABELS[key] for key in ("vp_c0", "vp_c1", "ivc")] == [
        r"VP, $C=0$",
        r"VP, $C=1$",
        "IVC",
    ]


def test_hf_phase_labels_match_previous_tmote2_placement():
    module = _load_plot_module()

    assert module.FONTS["phase_label"] == module.FONTS["axis_label"] == 32
    assert module.PHASE_LABELS == (
        {"text": "VP\n$C=0$", "theta_deg": 2.50, "u_D_meV": 10.0},
        {"text": "VP\n$C=1$", "theta_deg": 3.50, "u_D_meV": 2.5},
        {"text": "IVC", "theta_deg": 3.62, "u_D_meV": 18.2},
    )


def test_hf_D_axis_variant_moves_ivc_label_northeast():
    module = _load_plot_module()

    original_ivc = module.PHASE_LABELS[-1]
    variant_ivc = module.PHASE_D_LABELS[-1]
    assert variant_ivc["theta_deg"] > original_ivc["theta_deg"]
    assert variant_ivc["u_D_meV"] > original_ivc["u_D_meV"]
    assert variant_ivc == {"text": "IVC", "theta_deg": 3.72, "u_D_meV": 18.7}
    assert module.LABELS["y_D"] == r"$D$ (meV)"
    assert module.PHASE_D_OUTPUT_STEM != module.PHASE_OUTPUT_STEM


def test_phase_codes_prioritize_ivc_over_vp_topology():
    module = _load_plot_module()

    codes = module._phase_codes(
        grey=np.array([[False, False, True]]),
        vp_topological=np.array([[False, True, True]]),
    )

    assert codes.tolist() == [[module.PHASE_CODES["vp_c0"], module.PHASE_CODES["vp_c1"], module.PHASE_CODES["ivc"]]]


def test_ivc_ground_state_uses_lowest_clean_hf_energy():
    module = _load_plot_module()
    heat = module.pd.DataFrame(
        {
            "theta_index": [0, 0],
            "u_index": [0, 1],
            "theta_deg": [2.0, 2.0],
            "u_D_meV": [0.0, 0.5],
        }
    )
    sweep = module.pd.DataFrame(
        {
            "theta_index": [0, 0, 0, 0, 0],
            "u_index": [0, 0, 1, 1, 1],
            "clean_branch": [True, True, True, True, False],
            "ivc_minus_vp_energy_per_cell": [0.2, 0.1, 0.3, -0.01, -1.0],
        }
    )

    result = module._with_hf_ground_state(heat, sweep)

    assert result["hf_ivc_ground_nk24"].tolist() == [False, True]
    assert result["lowest_ivc_minus_vp_energy_nk24"].tolist() == [0.1, -0.01]


def test_heatmap_legend_uses_short_labels_and_large_type():
    module = _load_plot_module()

    assert module.LABELS["vp_chern"] == "VP Chern"
    assert module.LABELS["vp_ivc"] == "IVC-VP"
    assert module.FONTS["legend"] == 22
    assert module.CG_BOUNDARY_KEYS == ("vp_chern", "vp_ivc")


def test_phase_and_cg_outputs_are_separate_and_all7():
    module = _load_plot_module()

    assert module.PHASE_OUTPUT_STEM != module.CG_OUTPUT_STEM
    assert "all7" in module.PHASE_OUTPUT_STEM
    assert "all7" in module.CG_OUTPUT_STEM
