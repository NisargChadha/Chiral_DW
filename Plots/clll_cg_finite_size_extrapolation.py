"""Finite-size extrapolation of c_G for ideal conjugate Landau levels."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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


NISARG_FONTS = {
    "base": 12,
    "title": 20,
    "axis_label": 24,
    "tick_label": 20,
    "legend": 12,
    "annotation": 12,
}

NISARG_COLORS = {
    "red": "#FD4C55",
    "teal": "#378d94",
    "teal_fill": "#77b5b6",
    "grey": "0.25",
    "axis": "0.18",
    "grid": "0.70",
    "zero": "0.25",
}

EXACT_CG = -1.0 / (4.0 * np.pi)


class CLLLCGFiniteSizePlotParams(BaseModel):
    """User-facing controls for the cLLL finite-size c_G plot."""

    model_config = ConfigDict(frozen=True)

    n_k_values: tuple[int, ...] = (15, 16, 17, 18)
    theta_count: int = Field(default=641, ge=3)
    phi_step: float = Field(default=0.2, gt=0.0)
    n_r: int = Field(default=7, ge=3)
    output: Path = Path("Plots/figures/clll_cg_finite_size_extrapolation.png")
    dpi: int = Field(default=320, ge=72)
    figure_width: float = Field(default=6.6, gt=0.0)
    figure_height: float = Field(default=4.45, gt=0.0)

    @model_validator(mode="after")
    def _finite_size_values_are_valid(self) -> "CLLLCGFiniteSizePlotParams":
        if len(self.n_k_values) < 2:
            raise ValueError("at least two n_k values are required for a linear extrapolation")
        if len(set(self.n_k_values)) != len(self.n_k_values):
            raise ValueError("n_k values must be unique")
        if any(int(n_k) < 3 for n_k in self.n_k_values):
            raise ValueError("all n_k values must be at least 3")
        return self


@dataclass(frozen=True)
class CLLLCGFiniteSizeData:
    """Computed finite-size c_G data and linear 1/n_k extrapolation."""

    n_k: np.ndarray
    inverse_n_k: np.ndarray
    cG: np.ndarray
    slope: float
    intercept: float
    exact: float = EXACT_CG

    @property
    def cG_minus_exact(self) -> np.ndarray:
        return self.cG - self.exact

    @property
    def intercept_minus_exact(self) -> float:
        return float(self.intercept - self.exact)


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


def benchmark_params_for_n_k(
    params: CLLLCGFiniteSizePlotParams,
    n_k: int,
) -> IdealConjugateLLLChargeBenchmarkParams:
    return IdealConjugateLLLChargeBenchmarkParams(
        grid=MomentumGridParams(n_k=n_k),
        real_space=RealSpaceGridParams(n_r=params.n_r),
        output_dir=str(params.output.parent),
    )


def compute_cg_for_n_k(params: CLLLCGFiniteSizePlotParams, n_k: int) -> float:
    """Compute c_G from the numerical ideal conjugate-LLL projector response."""

    benchmark_params = benchmark_params_for_n_k(params, int(n_k))
    basis = IdealConjugateLLLBasis(benchmark_params)
    theta_edges = np.linspace(0.0, np.pi, params.theta_count)
    projectors = uniform_conjugate_projector_path(theta_edges, benchmark_params.grid.n_k)
    response = k_theta_from_ideal_conjugate_projectors(
        basis,
        projectors,
        theta_edges,
        np.array([0.0, params.phi_step], dtype=float),
    )
    return float(response.cG)


def fit_linear_cg_extrapolation(
    inverse_n_k: np.ndarray,
    cG: np.ndarray,
) -> tuple[float, float]:
    """Return slope and intercept for c_G = intercept + slope / n_k."""

    x = np.asarray(inverse_n_k, dtype=float)
    y = np.asarray(cG, dtype=float)
    if x.shape != y.shape:
        raise ValueError("inverse_n_k and cG must have matching shapes")
    if x.ndim != 1 or x.size < 2:
        raise ValueError("linear extrapolation requires at least two one-dimensional points")
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def compute_cg_finite_size_data(
    params: CLLLCGFiniteSizePlotParams,
) -> CLLLCGFiniteSizeData:
    n_k = np.array(params.n_k_values, dtype=int)
    inverse_n_k = 1.0 / n_k.astype(float)
    cG = np.array([compute_cg_for_n_k(params, int(value)) for value in n_k], dtype=float)
    slope, intercept = fit_linear_cg_extrapolation(inverse_n_k, cG)
    return CLLLCGFiniteSizeData(
        n_k=n_k,
        inverse_n_k=inverse_n_k,
        cG=cG,
        slope=slope,
        intercept=intercept,
    )


def write_cg_table_csv(output: Path, data: CLLLCGFiniteSizeData) -> Path:
    csv_output = output.with_suffix(".csv")
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n_k", "one_over_n_k", "cG", "cG_minus_exact"])
        for n_k, inverse_n_k, cG, err in zip(
            data.n_k,
            data.inverse_n_k,
            data.cG,
            data.cG_minus_exact,
            strict=True,
        ):
            writer.writerow([int(n_k), f"{inverse_n_k:.16g}", f"{cG:.16g}", f"{err:.16g}"])
        writer.writerow(
            [
                "infinity",
                "0",
                f"{data.intercept:.16g}",
                f"{data.intercept_minus_exact:.16g}",
            ]
        )
    return csv_output


def render_clll_cg_finite_size_plot(
    params: CLLLCGFiniteSizePlotParams,
) -> tuple[Path, Path, Path, CLLLCGFiniteSizeData]:
    apply_nisarg_plot_style()
    data = compute_cg_finite_size_data(params)

    output = params.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output = output.with_suffix(".pdf")
    csv_output = write_cg_table_csv(output, data)

    order = np.argsort(data.inverse_n_k)
    x_fit = np.linspace(0.0, float(np.max(data.inverse_n_k)) * 1.08, 256)
    y_fit = data.intercept + data.slope * x_fit

    y_values = np.concatenate([data.cG, y_fit, np.array([data.exact, data.intercept])])
    y_span = float(np.max(y_values) - np.min(y_values))
    y_pad = max(0.18 * y_span, 2.0e-5)

    fig, ax = plt.subplots(figsize=(params.figure_width, params.figure_height))
    ax.axhline(
        data.exact,
        color=NISARG_COLORS["grey"],
        linestyle=(0, (5.0, 3.2)),
        linewidth=1.4,
        alpha=0.75,
        label=r"$-1/(4\pi)$",
        zorder=1,
    )
    ax.plot(
        x_fit,
        y_fit,
        color=NISARG_COLORS["teal"],
        linewidth=2.6,
        label=r"linear fit in $1/n_k$",
        zorder=2,
    )
    ax.fill_between(
        x_fit,
        data.exact,
        y_fit,
        color=NISARG_COLORS["teal_fill"],
        alpha=0.16,
        linewidth=0.0,
        zorder=1,
    )
    ax.scatter(
        data.inverse_n_k[order],
        data.cG[order],
        s=72,
        color=NISARG_COLORS["teal"],
        edgecolor="white",
        linewidth=0.8,
        label=r"finite $n_k$",
        zorder=4,
    )
    ax.scatter(
        [0.0],
        [data.intercept],
        s=130,
        marker="*",
        color=NISARG_COLORS["red"],
        edgecolor="white",
        linewidth=0.7,
        label=r"$n_k\rightarrow\infty$",
        zorder=5,
    )

    ax.set_xlim(-0.0025, float(np.max(data.inverse_n_k)) * 1.08)
    ax.set_ylim(float(np.min(y_values)) - y_pad, float(np.max(y_values)) + y_pad)
    ax.set_xlabel(r"$1/n_k$")
    ax.set_ylabel(r"$c_G$")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)

    ax.text(
        0.04,
        0.08,
        rf"$c_G^{{\infty}}={data.intercept:.6f}$"
        "\n"
        rf"$(N_\theta={params.theta_count})$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=NISARG_FONTS["annotation"],
        color=NISARG_COLORS["axis"],
    )
    ax.legend(loc="upper right", frameon=True, framealpha=0.94, borderpad=0.55)

    for spine in ax.spines.values():
        spine.set_color(NISARG_COLORS["axis"])
        spine.set_linewidth(1.15)

    fig.tight_layout(pad=0.45)
    fig.savefig(output, dpi=params.dpi, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(pdf_output, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return output, pdf_output, csv_output, data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the finite-size extrapolation of ideal conjugate-LLL c_G."
    )
    parser.add_argument(
        "--n-k",
        type=int,
        nargs="+",
        default=list(CLLLCGFiniteSizePlotParams.model_fields["n_k_values"].default),
        help="Momentum-grid sizes to evaluate.",
    )
    parser.add_argument(
        "--theta-count",
        type=int,
        default=CLLLCGFiniteSizePlotParams.model_fields["theta_count"].default,
    )
    parser.add_argument(
        "--phi-step",
        type=float,
        default=CLLLCGFiniteSizePlotParams.model_fields["phi_step"].default,
    )
    parser.add_argument("--n-r", type=int, default=CLLLCGFiniteSizePlotParams.model_fields["n_r"].default)
    parser.add_argument("--dpi", type=int, default=CLLLCGFiniteSizePlotParams.model_fields["dpi"].default)
    parser.add_argument(
        "--output",
        type=Path,
        default=CLLLCGFiniteSizePlotParams.model_fields["output"].default,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    params = CLLLCGFiniteSizePlotParams(
        n_k_values=tuple(args.n_k),
        theta_count=args.theta_count,
        phi_step=args.phi_step,
        n_r=args.n_r,
        output=args.output,
        dpi=args.dpi,
    )
    png_path, pdf_path, csv_path, data = render_clll_cg_finite_size_plot(params)
    print("n_k, 1/n_k, c_G, c_G - (-1/(4*pi))")
    for n_k, inverse_n_k, cG, err in zip(
        data.n_k,
        data.inverse_n_k,
        data.cG,
        data.cG_minus_exact,
        strict=True,
    ):
        print(f"{int(n_k):2d}, {inverse_n_k:.8f}, {cG:.12f}, {err:+.3e}")
    print(f"inf, 0.00000000, {data.intercept:.12f}, {data.intercept_minus_exact:+.3e}")
    print(f"linear slope = {data.slope:.12e}")
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
