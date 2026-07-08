"""Render kappa(theta) for the ideal conjugate Landau-level response."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from chiral_dw.config import (
    IdealConjugateLLLChargeBenchmarkParams,
    MomentumGridParams,
    RealSpaceGridParams,
)
from chiral_dw.ideal_conjugate_lll import (
    IdealConjugateLLLBasis,
    k_theta_from_ideal_conjugate_projectors,
    uniform_conjugate_projector_path,
)
from chiral_dw.response import KThetaResult


NISARG_FONTS = {
    "base": 12,
    "title": 20,
    "axis_label": 24,
    "tick_label": 20,
    "x_axis_label": 30,
    "x_tick_label": 25,
    "legend": 13,
    "annotation": 13,
    "cg_annotation": 18,
}

NISARG_COLORS = {
    "teal": "#378d94",
    "teal_fill": "#77b5b6",
    "axis": "0.18",
    "grid": "0.70",
    "zero": "0.25",
}


class CLLLKappaThetaPlotParams(BaseModel):
    """User-facing controls for the ideal conjugate-LLL kappa plot."""

    model_config = ConfigDict(frozen=True)

    n_k: int = Field(default=18, ge=3)
    theta_count: int = Field(default=321, ge=3)
    phi_step: float = Field(default=0.2, gt=0.0)
    n_r: int = Field(default=7, ge=3)
    output: Path = Path("Plots/figures/clll_kappa_vs_theta.png")
    dpi: int = Field(default=320, ge=72)
    figure_width: float = Field(default=4.9, gt=0.0)
    figure_height: float = Field(default=3.6, gt=0.0)


def apply_nisarg_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["PT Serif Caption", "PT Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "font.size": NISARG_FONTS["base"],
            "axes.titlesize": NISARG_FONTS["title"],
            "axes.labelsize": NISARG_FONTS["axis_label"],
            "xtick.labelsize": NISARG_FONTS["tick_label"],
            "ytick.labelsize": NISARG_FONTS["tick_label"],
            "legend.fontsize": NISARG_FONTS["legend"],
            "axes.edgecolor": NISARG_COLORS["axis"],
            "axes.linewidth": 1.15,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def benchmark_params_from_plot_params(
    params: CLLLKappaThetaPlotParams,
) -> IdealConjugateLLLChargeBenchmarkParams:
    return IdealConjugateLLLChargeBenchmarkParams(
        grid=MomentumGridParams(n_k=params.n_k),
        real_space=RealSpaceGridParams(n_r=params.n_r),
        output_dir=str(params.output.parent),
    )


def compute_clll_kappa_response(params: CLLLKappaThetaPlotParams) -> KThetaResult:
    """Compute numerical kappa(theta) from the ideal conjugate-LLL projectors."""

    benchmark_params = benchmark_params_from_plot_params(params)
    basis = IdealConjugateLLLBasis(benchmark_params)
    theta_edges = np.linspace(0.0, np.pi, params.theta_count)
    projectors = uniform_conjugate_projector_path(theta_edges, benchmark_params.grid.n_k)
    return k_theta_from_ideal_conjugate_projectors(
        basis,
        projectors,
        theta_edges,
        np.array([0.0, params.phi_step], dtype=float),
    )


def render_clll_kappa_theta_plot(params: CLLLKappaThetaPlotParams) -> tuple[Path, Path]:
    apply_nisarg_plot_style()
    response = compute_clll_kappa_response(params)
    theta = np.asarray(response.theta, dtype=float)
    kappa = np.asarray(response.K, dtype=float)

    output = params.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output = output.with_suffix(".pdf")

    fig, ax = plt.subplots(figsize=(params.figure_width, params.figure_height))
    ax.set_box_aspect(3.0 / 5.0)
    ax.axhline(0.0, color=NISARG_COLORS["zero"], linestyle="--", linewidth=1.15, alpha=0.85, zorder=1)
    ax.fill_between(
        theta,
        0.0,
        kappa,
        color=NISARG_COLORS["teal_fill"],
        alpha=0.20,
        linewidth=0.0,
        zorder=2,
    )
    ax.plot(
        theta,
        kappa,
        color=NISARG_COLORS["teal"],
        linewidth=3.0,
        solid_capstyle="round",
        zorder=3,
    )

    pad = 0.12 * max(float(np.max(np.abs(kappa))), 1e-12)
    ax.set_xlim(0.0, np.pi)
    ax.set_ylim(float(np.min(kappa)) - pad, float(np.max(kappa)) + pad)
    ax.set_xlabel(r"$\theta$", fontsize=NISARG_FONTS["x_axis_label"])
    ax.set_ylabel(r"$K_{\phi}(\theta)$", labelpad=-3)
    ax.set_xticks([0.0, 0.5 * np.pi, np.pi])
    ax.set_xticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_yticks([-0.03, 0.0, 0.03])
    ax.set_yticklabels([r"$-0.03$", r"$0.00$", r"$0.03$"])
    ax.tick_params(axis="x", labelsize=NISARG_FONTS["x_tick_label"])
    ax.text(
        0.97,
        0.93,
        r"$c_G = -0.079$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=NISARG_FONTS["cg_annotation"],
        color=NISARG_COLORS["axis"],
    )

    for spine in ax.spines.values():
        spine.set_color(NISARG_COLORS["axis"])
        spine.set_linewidth(1.15)
    ax.margins(x=0.0)

    fig.tight_layout(pad=0.35)
    fig.savefig(output, dpi=params.dpi, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_output, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output, pdf_output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot kappa(theta) for ideal conjugate Landau levels."
    )
    parser.add_argument("--n-k", type=int, default=CLLLKappaThetaPlotParams.model_fields["n_k"].default)
    parser.add_argument(
        "--theta-count",
        type=int,
        default=CLLLKappaThetaPlotParams.model_fields["theta_count"].default,
    )
    parser.add_argument(
        "--phi-step",
        type=float,
        default=CLLLKappaThetaPlotParams.model_fields["phi_step"].default,
    )
    parser.add_argument("--n-r", type=int, default=CLLLKappaThetaPlotParams.model_fields["n_r"].default)
    parser.add_argument("--dpi", type=int, default=CLLLKappaThetaPlotParams.model_fields["dpi"].default)
    parser.add_argument(
        "--output",
        type=Path,
        default=CLLLKappaThetaPlotParams.model_fields["output"].default,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    params = CLLLKappaThetaPlotParams(
        n_k=args.n_k,
        theta_count=args.theta_count,
        phi_step=args.phi_step,
        n_r=args.n_r,
        output=args.output,
        dpi=args.dpi,
    )
    png_path, pdf_path = render_clll_kappa_theta_plot(params)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
