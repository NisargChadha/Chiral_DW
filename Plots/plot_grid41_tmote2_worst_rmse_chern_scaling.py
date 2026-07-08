#!/usr/bin/env python3
"""Worst all-seven cG finite-size scaling annotated by VP+ band-0 Chern."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = (
    ROOT
    / "results/taige_ivc_hysteresis_finite_size_nk18_24_grid41_linear_interaction_recomputed_sharded"
)
FIT_SUMMARY_CSV = (
    RESULT_DIR / "analysis_plots/grid41_linear_interaction_lowest_clean_best5_cG_fits.csv"
)
BOUNDARY_CSV = (
    RESULT_DIR / "analysis_plots/grid41_linear_interaction_best5_cG_heatmap_with_boundaries.csv"
)
FIT_SOURCE_CSV = RESULT_DIR / "hysteresis_finite_size_fit_source.csv"
SELECTED_CSV = RESULT_DIR / "hysteresis_finite_size_selected.csv"
BRANCH_CANDIDATES_CSV = RESULT_DIR / "hysteresis_finite_size_branch_candidates.csv"
FIGURE_DIR = ROOT / "Plots/figures"

FIGURE = {
    "size": (8.8, 5.8),
    "dpi": 300,
}

FONTS = {
    "base": 12,
    "axis_label": 24,
    "tick_label": 20,
    "legend": 15,
    "annotation": 12,
}

COLORS = {
    "red": "#FD4C55",
    "axis": "0.18",
    "grid": "0.70",
    "zero": "0.25",
}


class WorstRMSEChernParams(BaseModel):
    """Controls for the worst-RMSE Chern-annotated finite-size plot."""

    model_config = ConfigDict(frozen=True)

    fit_summary_csv: Path = Field(default=FIT_SUMMARY_CSV)
    boundary_csv: Path = Field(default=BOUNDARY_CSV)
    fit_source_csv: Path = Field(default=FIT_SOURCE_CSV)
    selected_csv: Path = Field(default=SELECTED_CSV)
    branch_candidates_csv: Path = Field(default=BRANCH_CANDIDATES_CSV)
    output_dir: Path = Field(default=FIGURE_DIR)
    output_stem: str = Field(default="grid41_tmote2_worst_rmse_all7_cG_scaling_vpplus_chern")


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["PT Serif Caption", "PT Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "font.size": FONTS["base"],
            "axes.labelsize": FONTS["axis_label"],
            "xtick.labelsize": FONTS["tick_label"],
            "ytick.labelsize": FONTS["tick_label"],
            "legend.fontsize": FONTS["legend"],
            "axes.edgecolor": COLORS["axis"],
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if np.issubdtype(series.dtype, np.number):
        return series.fillna(0).astype(bool)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def box_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["axis"])
        spine.set_linewidth(1.15)


def load_worst_summary(params: WorstRMSEChernParams) -> pd.Series:
    summary = pd.read_csv(params.fit_summary_csv)
    boundaries = pd.read_csv(params.boundary_csv)
    summary = summary.merge(
        boundaries[["theta_deg", "u_D_meV", "grey_mask_all"]],
        on=["theta_deg", "u_D_meV"],
        how="left",
        validate="one_to_one",
    )
    summary["grey_mask_all"] = to_bool(summary["grey_mask_all"]).fillna(False)
    valid = (
        (summary["cG_all_status"] == "fit_ok")
        & np.isfinite(summary["cG_all_rmse"])
        & ~summary["grey_mask_all"]
    )
    if not valid.any():
        raise ValueError("No valid all-seven RMSE rows found.")
    return summary.loc[valid].sort_values("cG_all_rmse", ascending=False).iloc[0]


def load_worst_points(params: WorstRMSEChernParams, worst: pd.Series) -> pd.DataFrame:
    source = pd.read_csv(params.fit_source_csv)
    selected = pd.read_csv(params.selected_csv)
    branch_candidates = pd.read_csv(params.branch_candidates_csv)
    theta_deg = float(worst["theta_deg"])
    u_D_meV = float(worst["u_D_meV"])

    source_mask = (
        np.isclose(source["theta_deg"], theta_deg)
        & np.isclose(source["u_D_meV"], u_D_meV)
        & (source["trial_interpolation"] == "linear_interaction")
        & (source["branch_label"] == "lowest_energy_clean")
        & (source["selection_kind"] == "clean")
        & to_bool(source["clean"])
    )
    source_rows = source.loc[source_mask].copy()

    selected_mask = (
        np.isclose(selected["theta_deg"], theta_deg)
        & np.isclose(selected["u_D_meV"], u_D_meV)
        & (selected["trial_interpolation"] == "linear_interaction")
    )
    selected_rows = selected.loc[
        selected_mask,
        [
            "n_k",
            "lowest_energy_clean_branch",
            "lowest_energy_clean_cG",
            "chern_hf_vpplus_band_0",
            "chern_hf_vpminus_band_0",
        ],
    ].copy()
    branch_mask = (
        np.isclose(branch_candidates["theta_deg"], theta_deg)
        & np.isclose(branch_candidates["u_D_meV"], u_D_meV)
        & (branch_candidates["trial_interpolation"] == "linear_interaction")
    )
    gap_rows = (
        branch_candidates.loc[
            branch_mask,
            [
                "n_k",
                "vp_plus_direct_gap",
                "vp_minus_direct_gap",
                "vp_plus_clean",
                "vp_minus_clean",
            ],
        ]
        .sort_values("n_k")
        .drop_duplicates("n_k")
    )

    points = source_rows.merge(selected_rows, on="n_k", how="left", validate="one_to_one")
    points = points.merge(gap_rows, on="n_k", how="left", validate="one_to_one")
    points = points.sort_values("n_k")
    if len(points) != 7:
        raise ValueError(f"Expected seven worst-RMSE mesh points, found {len(points)}.")
    points["vpplus_band0_chern_rounded"] = points["chern_hf_vpplus_band_0"].round().astype(int)
    points["fit_cG"] = (
        float(worst["cG_all_intercept"])
        + float(worst["cG_all_slope"]) * points["inv_n_k"].astype(float)
    )
    return points


def annotate_meshes(ax: plt.Axes, points: pd.DataFrame) -> None:
    offsets = {
        18: (0.0012, 0.0040),
        19: (0.0016, 0.0037),
        20: (0.0002, 0.0045),
        21: (0.0001, 0.0040),
        22: (-0.0007, 0.0042),
        23: (-0.0019, 0.0040),
        24: (-0.0018, 0.0040),
    }
    for _, point in points.iterrows():
        n_k = int(point["n_k"])
        dx, dy = offsets[n_k]
        ax.text(
            float(point["inv_n_k"]) + dx,
            float(point["cG"]) + dy,
            f"{n_k}",
            ha="center",
            va="center",
            fontsize=FONTS["annotation"],
            color=COLORS["axis"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.6},
        )


def build_plot(params: WorstRMSEChernParams = WorstRMSEChernParams()) -> list[Path]:
    apply_style()
    params.output_dir.mkdir(parents=True, exist_ok=True)
    worst = load_worst_summary(params)
    points = load_worst_points(params, worst)

    intercept = float(worst["cG_all_intercept"])
    slope = float(worst["cG_all_slope"])
    x_fit = np.linspace(0.0, 1.0 / 18.0, 240)
    y_fit = intercept + slope * x_fit

    fig, ax = plt.subplots(figsize=FIGURE["size"])
    ax.plot(x_fit, y_fit, color=COLORS["red"], lw=2.5, ls="--", alpha=0.95)
    ax.plot(
        [0.0],
        [intercept],
        ls="none",
        marker="o",
        ms=9.0,
        mfc="white",
        mec=COLORS["red"],
        mew=2.2,
        color=COLORS["red"],
        zorder=4,
    )

    c1 = points["vpplus_band0_chern_rounded"] == 1
    c0 = points["vpplus_band0_chern_rounded"] == 0
    ax.plot(
        points.loc[c1, "inv_n_k"],
        points.loc[c1, "cG"],
        ls="none",
        marker="o",
        ms=9.5,
        mfc=COLORS["red"],
        mec="white",
        mew=1.1,
        color=COLORS["red"],
        zorder=5,
    )
    ax.plot(
        points.loc[c0, "inv_n_k"],
        points.loc[c0, "cG"],
        ls="none",
        marker="o",
        ms=9.5,
        mfc="white",
        mec=COLORS["red"],
        mew=2.2,
        color=COLORS["red"],
        zorder=6,
    )
    annotate_meshes(ax, points)

    y_values = np.r_[points["cG"].to_numpy(dtype=float), y_fit, intercept]
    y_pad = 0.14 * (float(y_values.max()) - float(y_values.min()))
    ax.set_xlim(-0.002, 1.0 / 18.0 + 0.002)
    ax.set_ylim(float(y_values.min()) - y_pad, float(y_values.max()) + y_pad)
    ax.set_xlabel(r"$1/n_k$")
    ax.set_ylabel(r"$c_G$")
    ax.set_xticks([0.0, 1.0 / 24.0, 1.0 / 21.0, 1.0 / 18.0])
    ax.set_xticklabels(["0", r"$1/24$", r"$1/21$", r"$1/18$"])
    ax.axhline(0.0, color=COLORS["zero"], lw=1.0, alpha=0.45)
    ax.axvline(0.0, color=COLORS["zero"], lw=1.0, alpha=0.45)
    box_axes(ax)

    handles = [
        Line2D([0], [0], color=COLORS["red"], lw=2.5, ls="--", label="all-seven fit"),
        Line2D(
            [0],
            [0],
            marker="o",
            color=COLORS["red"],
            markerfacecolor=COLORS["red"],
            markeredgecolor="white",
            markeredgewidth=1.1,
            lw=0,
            markersize=9,
            label=r"VP+ band 0: $C=1$ ($n_k=18,21,24$)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=COLORS["red"],
            markerfacecolor="white",
            markeredgecolor=COLORS["red"],
            markeredgewidth=2.2,
            lw=0,
            markersize=9,
            label=r"VP+ band 0: $C=0$ ($n_k=19,20,22,23$)",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        handlelength=2.2,
        borderaxespad=0.5,
    )
    ax.text(
        0.03,
        0.96,
        (
            rf"$\theta={float(worst['theta_deg']):.2f}^\circ$, "
            rf"$u_D={float(worst['u_D_meV']):.1f}\,\mathrm{{meV}}$"
            "\n"
            rf"RMSE$={float(worst['cG_all_rmse']):.3g}$"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONTS["annotation"],
        color=COLORS["axis"],
    )

    plot_rows = points[
        [
            "n_k",
            "inv_n_k",
            "theta_deg",
            "u_D_meV",
            "source_branch",
            "cG",
            "fit_cG",
            "chern_hf_vpplus_band_0",
            "vpplus_band0_chern_rounded",
            "chern_hf_vpminus_band_0",
            "vp_plus_direct_gap",
            "vp_minus_direct_gap",
            "vp_plus_clean",
            "vp_minus_clean",
        ]
    ].copy()
    plot_rows["cG_all_intercept"] = intercept
    plot_rows["cG_all_slope"] = slope
    plot_rows["cG_all_rmse"] = float(worst["cG_all_rmse"])

    csv_path = params.output_dir / f"{params.output_stem}.csv"
    png_path = params.output_dir / f"{params.output_stem}.png"
    pdf_path = params.output_dir / f"{params.output_stem}.pdf"
    svg_path = params.output_dir / f"{params.output_stem}.svg"
    plot_rows.to_csv(csv_path, index=False)
    for path in (png_path, pdf_path, svg_path):
        fig.savefig(path, dpi=FIGURE["dpi"], bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return [csv_path, png_path, pdf_path, svg_path]


def main() -> None:
    for path in build_plot():
        print(path)


if __name__ == "__main__":
    main()
