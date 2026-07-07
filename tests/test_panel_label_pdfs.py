from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")


def _load_plot_module():
    path = Path(__file__).resolve().parents[1] / "Plots" / "panel_label_pdfs.py"
    spec = importlib.util.spec_from_file_location("panel_label_pdfs", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_labels_and_output_dir():
    module = _load_plot_module()
    params = module.PanelLabelPDFParams()

    assert params.labels == ("a", "b", "c", "d")
    assert params.output_dir == Path("Plots/figures/panel_labels")


def test_panel_label_format_and_output_names():
    module = _load_plot_module()

    assert module.format_panel_label("a") == "(a)"
    assert module.format_panel_label("(b)") == "(b)"
    assert module.panel_label_output_path(Path("out"), "(C)") == Path("out/panel_label_c.pdf")


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


def test_panel_label_pdfs_are_written(tmp_path: Path):
    module = _load_plot_module()
    params = module.PanelLabelPDFParams(output_dir=tmp_path)

    paths = module.render_panel_label_pdfs(params)

    assert [path.name for path in paths] == [
        "panel_label_a.pdf",
        "panel_label_b.pdf",
        "panel_label_c.pdf",
        "panel_label_d.pdf",
    ]
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0
