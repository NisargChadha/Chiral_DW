"""Export standalone panel-label PDFs for assembled figures."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Literal

import matplotlib
from pydantic import BaseModel, ConfigDict, Field, model_validator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NISARG_FONTS = {
    "base": 12,
    "panel_label": 28,
}

NISARG_COLORS = {
    "axis": "0.18",
}


class PanelLabelPDFParams(BaseModel):
    """User-facing controls for standalone panel-label PDFs."""

    model_config = ConfigDict(frozen=True)

    labels: tuple[str, ...] = ("a", "b", "c", "d")
    output_dir: Path = Path("Plots/figures/panel_labels")
    font_size: int = Field(default=NISARG_FONTS["panel_label"], ge=1)
    font_weight: Literal["normal", "bold"] = "bold"
    figure_width: float = Field(default=0.38, gt=0.0)
    figure_height: float = Field(default=0.30, gt=0.0)
    pad_inches: float = Field(default=0.01, ge=0.0)
    dpi: int = Field(default=320, ge=72)

    @model_validator(mode="after")
    def _labels_are_unique(self) -> "PanelLabelPDFParams":
        normalized = tuple(normalize_panel_label(label) for label in self.labels)
        if len(set(normalized)) != len(normalized):
            raise ValueError("panel labels must be unique")
        return self


def apply_nisarg_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["PT Serif Caption", "PT Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "font.size": NISARG_FONTS["base"],
            "savefig.transparent": True,
            "figure.facecolor": "none",
            "axes.facecolor": "none",
        }
    )


def normalize_panel_label(label: str) -> str:
    stripped = str(label).strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        stripped = stripped[1:-1].strip()
    if not stripped:
        raise ValueError("panel label cannot be empty")
    return stripped


def format_panel_label(label: str) -> str:
    return f"({normalize_panel_label(label)})"


def panel_label_output_path(output_dir: Path, label: str) -> Path:
    normalized = normalize_panel_label(label)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", normalized).strip("_").lower()
    if not slug:
        raise ValueError(f"could not make an output name for label {label!r}")
    return output_dir / f"panel_label_{slug}.pdf"


def render_panel_label_pdf(label: str, params: PanelLabelPDFParams) -> Path:
    apply_nisarg_plot_style()
    params.output_dir.mkdir(parents=True, exist_ok=True)
    output = panel_label_output_path(params.output_dir, label)

    fig = plt.figure(
        figsize=(params.figure_width, params.figure_height),
        facecolor="none",
    )
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        format_panel_label(label),
        ha="center",
        va="center",
        fontsize=params.font_size,
        fontweight=params.font_weight,
        color=NISARG_COLORS["axis"],
    )
    fig.savefig(
        output,
        dpi=params.dpi,
        transparent=True,
        bbox_inches="tight",
        pad_inches=params.pad_inches,
    )
    plt.close(fig)
    return output


def render_panel_label_pdfs(params: PanelLabelPDFParams) -> tuple[Path, ...]:
    return tuple(render_panel_label_pdf(label, params) for label in params.labels)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export standalone panel-label PDFs.")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(PanelLabelPDFParams.model_fields["labels"].default),
        help="Panel labels to export, with or without parentheses.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PanelLabelPDFParams.model_fields["output_dir"].default,
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=PanelLabelPDFParams.model_fields["font_size"].default,
    )
    parser.add_argument(
        "--font-weight",
        choices=["normal", "bold"],
        default=PanelLabelPDFParams.model_fields["font_weight"].default,
    )
    parser.add_argument("--dpi", type=int, default=PanelLabelPDFParams.model_fields["dpi"].default)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    params = PanelLabelPDFParams(
        labels=tuple(args.labels),
        output_dir=args.output_dir,
        font_size=args.font_size,
        font_weight=args.font_weight,
        dpi=args.dpi,
    )
    for path in render_panel_label_pdfs(params):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
