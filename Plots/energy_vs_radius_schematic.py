"""Render the analytic domain-wall energy schematic versus radius."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NISARG_FONTS = {
    "base": 12,
    "axis_label": 28,
    "tick_label": 20,
    "legend": 13,
}

NISARG_COLORS = {
    "surface": "#FD4C55",
    "dipole": "#378d94",
    "total": "0.05",
    "axis": "0.18",
    "grid": "0.70",
    "guide": "0.42",
}


class EnergyRadiusSchematicParams(BaseModel):
    """User-facing controls for the radius-energy schematic."""

    model_config = ConfigDict(frozen=True)

    sigma: float = Field(default=1.0, gt=0.0)
    charge_strength: float = Field(default=1.0 / (2.0 * np.pi), gt=0.0)
    r_min: float = Field(default=0.03, gt=0.0)
    r_max: float = Field(default=0.55, gt=0.0)
    n_points: int = Field(default=1200, ge=32)
    output: Path = Path("Plots/figures/energy_vs_radius_schematic.png")
    dpi: int = Field(default=320, ge=72)
    figure_width: float = Field(default=4.6, gt=0.0)
    figure_height: float = Field(default=4.6, gt=0.0)

    @model_validator(mode="after")
    def _radius_window_contains_optimum(self) -> "EnergyRadiusSchematicParams":
        if self.r_min >= self.r_max:
            raise ValueError("r_min must be smaller than r_max")
        r_star = characteristic_radius(self.sigma, self.charge_strength)
        if not (self.r_min < r_star < self.r_max):
            raise ValueError(
                "radius window must contain the optimal radius R*: "
                f"got r_min={self.r_min}, R*={r_star}, r_max={self.r_max}"
            )
        return self


def apply_nisarg_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["PT Serif Caption", "PT Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "font.size": NISARG_FONTS["base"],
            "axes.labelsize": NISARG_FONTS["axis_label"],
            "xtick.labelsize": NISARG_FONTS["tick_label"],
            "ytick.labelsize": NISARG_FONTS["tick_label"],
            "legend.fontsize": NISARG_FONTS["legend"],
            "axes.edgecolor": NISARG_COLORS["axis"],
            "axes.linewidth": 1.15,
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


def energy_terms(
    radius: np.ndarray | float,
    *,
    sigma: float = 1.0,
    charge_strength: float = 1.0 / (2.0 * np.pi),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return surface, dipolar, and total schematic energies."""

    radius_arr = np.asarray(radius, dtype=float)
    if np.any(radius_arr <= 0.0):
        raise ValueError("radius values must be positive")
    surface = 2.0 * np.pi * float(sigma) * radius_arr
    dipole = 2.0 * np.pi * float(charge_strength) ** 2 / radius_arr
    total = surface + dipole
    return surface, dipole, total


def characteristic_radius(
    sigma: float = 1.0,
    charge_strength: float = 1.0 / (2.0 * np.pi),
) -> float:
    """Return the energy-minimizing radius R*."""

    return float(charge_strength) / np.sqrt(float(sigma))


def energy_curve_data(
    params: EnergyRadiusSchematicParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    r_values = np.linspace(params.r_min, params.r_max, params.n_points)
    surface, dipole, total = energy_terms(
        r_values,
        sigma=params.sigma,
        charge_strength=params.charge_strength,
    )
    r_star = characteristic_radius(params.sigma, params.charge_strength)
    _, _, total_star = energy_terms(
        r_star,
        sigma=params.sigma,
        charge_strength=params.charge_strength,
    )
    return r_values, surface, dipole, total, r_star, float(total_star)


def render_energy_radius_schematic(params: EnergyRadiusSchematicParams) -> tuple[Path, Path]:
    apply_nisarg_plot_style()
    r_values, surface, dipole, total, r_star, energy_star = energy_curve_data(params)

    output = params.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output = output.with_suffix(".pdf")

    fig, ax = plt.subplots(figsize=(params.figure_width, params.figure_height))

    ax.plot(
        r_values,
        surface,
        linestyle="--",
        linewidth=2.6,
        color=NISARG_COLORS["surface"],
        label=r"$E_{\rm surf}=2\pi\sigma R$",
    )
    ax.plot(
        r_values,
        dipole,
        linestyle="--",
        linewidth=2.6,
        color=NISARG_COLORS["dipole"],
        label=r"$E_{\rm dip}=2\pi(N_w c_G)^2/R$",
    )
    ax.plot(
        r_values,
        total,
        linestyle="-",
        linewidth=3.0,
        color=NISARG_COLORS["total"],
        label=r"$E=E_{\rm surf}+E_{\rm dip}$",
    )

    ax.axvline(r_star, color=NISARG_COLORS["guide"], linestyle="--", linewidth=1.5)
    ax.axhline(energy_star, color=NISARG_COLORS["guide"], linestyle="--", linewidth=1.5)
    ax.plot(
        r_star,
        energy_star,
        marker="o",
        markersize=6.5,
        color=NISARG_COLORS["total"],
        zorder=5,
    )

    ax.set_xlabel(r"$R$", labelpad=-5)
    ax.set_ylabel(r"$E$")
    ax.set_xlim(params.r_min, params.r_max)
    ax.set_ylim(0.0, 1.08 * float(np.max(total)))
    ax.set_xticks([r_star])
    ax.set_xticklabels([r"$R^{\!\ast}$"])
    ax.set_yticks([])
    ax.set_box_aspect(1.0)
    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=2.1,
        handletextpad=0.55,
        labelspacing=0.45,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=4.5, width=1.0)
    ax.margins(x=0.0)

    fig.tight_layout(pad=0.45)
    fig.savefig(output, dpi=params.dpi)
    fig.savefig(pdf_output)
    plt.close(fig)
    return output, pdf_output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the analytic domain-wall energy schematic versus radius."
    )
    parser.add_argument("--sigma", type=float, default=EnergyRadiusSchematicParams.model_fields["sigma"].default)
    parser.add_argument(
        "--charge-strength",
        type=float,
        default=EnergyRadiusSchematicParams.model_fields["charge_strength"].default,
    )
    parser.add_argument("--r-min", type=float, default=EnergyRadiusSchematicParams.model_fields["r_min"].default)
    parser.add_argument("--r-max", type=float, default=EnergyRadiusSchematicParams.model_fields["r_max"].default)
    parser.add_argument(
        "--n-points",
        type=int,
        default=EnergyRadiusSchematicParams.model_fields["n_points"].default,
    )
    parser.add_argument("--dpi", type=int, default=EnergyRadiusSchematicParams.model_fields["dpi"].default)
    parser.add_argument(
        "--output",
        type=Path,
        default=EnergyRadiusSchematicParams.model_fields["output"].default,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    params = EnergyRadiusSchematicParams(
        sigma=args.sigma,
        charge_strength=args.charge_strength,
        r_min=args.r_min,
        r_max=args.r_max,
        n_points=args.n_points,
        output=args.output,
        dpi=args.dpi,
    )
    png_path, pdf_path = render_energy_radius_schematic(params)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
