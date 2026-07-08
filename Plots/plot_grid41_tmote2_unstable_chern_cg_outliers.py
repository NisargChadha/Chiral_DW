#!/usr/bin/env python3
"""cG mesh sequences for points with unstable VP+ band-0 Chern numbers."""

from __future__ import annotations

from math import ceil
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
    "size": (11.0, 8.6),
    "dpi": 300,
}

FONTS = {
    "base": 12,
    "axis_label": 18,
    "tick_label": 13,
    "title": 13,
    "legend": 13,
}

COLORS = {
    "red": "#FD4C55",
    "axis": "0.18",
    "grid": "0.70",
    "zero": "0.25",
    "fit": "0.30",
}


class UnstableChernOutlierParams(BaseModel):
    """Controls for the unstable-Chern cG outlier panel."""

    model_config = ConfigDict(frozen=True)

    fit_summary_csv: Path = Field(default=FIT_SUMMARY_CSV)
    boundary_csv: Path = Field(default=BOUNDARY_CSV)
    fit_source_csv: Path = Field(default=FIT_SOURCE_CSV)
    selected_csv: Path = Field(default=SELECTED_CSV)
    branch_candidates_csv: Path = Field(default=BRANCH_CANDIDATES_CSV)
    output_dir: Path = Field(default=FIGURE_DIR)
    output_stem: str = Field(default="grid41_tmote2_unstable_vpplus_chern_cG_outliers")


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["PT Serif Caption", "PT Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "font.size": FONTS["base"],
            "axes.titlesize": FONTS["title"],
            "axes.labelsize": FONTS["axis_label"],
            "xtick.labelsize": FONTS["tick_label"],
            "ytick.labelsize": FONTS["tick_label"],
            "legend.fontsize": FONTS["legend"],
            "axes.edgecolor": COLORS["axis"],
            "axes.linewidth": 1.0,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
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
        spine.set_linewidth(1.0)


def load_data(params: UnstableChernOutlierParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(params.fit_summary_csv)
    boundaries = pd.read_csv(params.boundary_csv)
    selected = pd.read_csv(params.selected_csv)
    source = pd.read_csv(params.fit_source_csv)
    branch_candidates = pd.read_csv(params.branch_candidates_csv)

    summary = summary.merge(
        boundaries[["theta_deg", "u_D_meV", "grey_mask_all"]],
        on=["theta_deg", "u_D_meV"],
        how="left",
        validate="one_to_one",
    )
    summary["grey_mask_all"] = to_bool(summary["grey_mask_all"]).fillna(False)
    summary = summary.loc[
        (summary["cG_all_status"] == "fit_ok")
        & np.isfinite(summary["cG_all_rmse"])
        & ~summary["grey_mask_all"]
    ].copy()

    selected = selected.loc[selected["trial_interpolation"] == "linear_interaction"].copy()
    selected["vpplus_band0_chern_rounded"] = (
        selected["chern_hf_vpplus_band_0"].round().astype(int)
    )
    chern_stats = (
        selected.groupby(["theta_deg", "u_D_meV"])
        .agg(
            n_mesh=("n_k", "nunique"),
            chern_values=(
                "vpplus_band0_chern_rounded",
                lambda values: ",".join(str(v) for v in sorted(set(values))),
            ),
            n_chern_values=(
                "vpplus_band0_chern_rounded",
                lambda values: len(set(values)),
            ),
            n_C1=("vpplus_band0_chern_rounded", lambda values: int((values == 1).sum())),
            n_C0=("vpplus_band0_chern_rounded", lambda values: int((values == 0).sum())),
        )
        .reset_index()
    )
    unstable_summary = summary.merge(
        chern_stats, on=["theta_deg", "u_D_meV"], how="left", validate="one_to_one"
    )
    unstable_summary = unstable_summary.loc[
        (unstable_summary["n_mesh"] == 7) & (unstable_summary["n_chern_values"] > 1)
    ].copy()
    unstable_summary = unstable_summary.sort_values("cG_all_rmse", ascending=False)
    unstable_summary["unstable_rank"] = np.arange(1, len(unstable_summary) + 1)

    source = source.loc[
        (source["trial_interpolation"] == "linear_interaction")
        & (source["branch_label"] == "lowest_energy_clean")
        & (source["selection_kind"] == "clean")
        & to_bool(source["clean"])
    ].copy()
    chern_values = selected.loc[
        :,
        [
            "theta_deg",
            "u_D_meV",
            "n_k",
            "chern_hf_vpplus_band_0",
            "vpplus_band0_chern_rounded",
        ],
    ]
    gap_values = (
        branch_candidates.loc[
            branch_candidates["trial_interpolation"] == "linear_interaction",
            [
                "theta_deg",
                "u_D_meV",
                "n_k",
                "vp_plus_direct_gap",
                "vp_minus_direct_gap",
                "vp_plus_clean",
                "vp_minus_clean",
            ],
        ]
        .sort_values(["theta_deg", "u_D_meV", "n_k"])
        .drop_duplicates(["theta_deg", "u_D_meV", "n_k"])
    )
    points = source.merge(
        chern_values,
        on=["theta_deg", "u_D_meV", "n_k"],
        how="inner",
        validate="one_to_one",
    )
    points = points.merge(
        gap_values,
        on=["theta_deg", "u_D_meV", "n_k"],
        how="left",
        validate="one_to_one",
    )
    points = points.merge(
        unstable_summary[
            [
                "theta_deg",
                "u_D_meV",
                "unstable_rank",
                "cG_all_intercept",
                "cG_all_slope",
                "cG_all_rmse",
                "n_C1",
                "n_C0",
            ]
        ],
        on=["theta_deg", "u_D_meV"],
        how="inner",
        validate="many_to_one",
    )
    points["fit_cG"] = points["cG_all_intercept"] + points["cG_all_slope"] / points["n_k"]
    points = points.sort_values(["unstable_rank", "n_k"])
    return unstable_summary, points


def render_panel(
    unstable_summary: pd.DataFrame,
    points: pd.DataFrame,
    params: UnstableChernOutlierParams,
) -> list[Path]:
    n_panels = len(unstable_summary)
    if n_panels == 0:
        raise ValueError("No unstable VP+ band-0 Chern points found.")
    ncols = 3
    nrows = ceil(n_panels / ncols)

    y_values = points["cG"].to_numpy(dtype=float)
    y_pad = 0.13 * max(float(y_values.max() - y_values.min()), 1.0e-3)
    y_limits = (float(y_values.min()) - y_pad, float(y_values.max()) + y_pad)

    fig, axes = plt.subplots(nrows, ncols, figsize=FIGURE["size"], sharex=True, sharey=True)
    axes_arr = np.atleast_1d(axes).ravel()
    for ax in axes_arr[n_panels:]:
        ax.axis("off")

    n_fit = np.linspace(18, 24, 240)
    for ax, (_, row) in zip(axes_arr, unstable_summary.iterrows()):
        panel_points = points.loc[
            np.isclose(points["theta_deg"], row["theta_deg"])
            & np.isclose(points["u_D_meV"], row["u_D_meV"])
        ].sort_values("n_k")

        ax.plot(
            panel_points["n_k"],
            panel_points["cG"],
            color=COLORS["fit"],
            lw=1.1,
            alpha=0.45,
            zorder=1,
        )
        fit_y = float(row["cG_all_intercept"]) + float(row["cG_all_slope"]) / n_fit
        ax.plot(n_fit, fit_y, color=COLORS["red"], lw=1.5, ls="--", alpha=0.9, zorder=2)

        c1 = panel_points["vpplus_band0_chern_rounded"] == 1
        c0 = panel_points["vpplus_band0_chern_rounded"] == 0
        ax.plot(
            panel_points.loc[c1, "n_k"],
            panel_points.loc[c1, "cG"],
            ls="none",
            marker="o",
            ms=6.5,
            mfc=COLORS["red"],
            mec="white",
            mew=0.9,
            color=COLORS["red"],
            zorder=4,
        )
        ax.plot(
            panel_points.loc[c0, "n_k"],
            panel_points.loc[c0, "cG"],
            ls="none",
            marker="o",
            ms=6.5,
            mfc="white",
            mec=COLORS["red"],
            mew=1.7,
            color=COLORS["red"],
            zorder=5,
        )
        ax.axhline(0.0, color=COLORS["zero"], lw=0.9, alpha=0.45, zorder=0)
        ax.set_xlim(17.65, 24.35)
        ax.set_ylim(*y_limits)
        ax.set_xticks([18, 20, 22, 24])
        ax.set_title(
            rf"{int(row['unstable_rank'])}. $\theta={row['theta_deg']:.2f}^\circ$, "
            rf"$u_D={row['u_D_meV']:.1f}$"
            "\n"
            rf"RMSE$={row['cG_all_rmse']:.3g}$",
            pad=5,
        )
        box_axes(ax)

    for row_idx in range(nrows):
        axes_arr[row_idx * ncols].set_ylabel(r"$c_G$")
    for ax in axes_arr[(nrows - 1) * ncols : nrows * ncols]:
        if ax.has_data():
            ax.set_xlabel(r"$n_k$")

    handles = [
        Line2D([0], [0], color=COLORS["red"], lw=1.5, ls="--", label="all-seven fit"),
        Line2D(
            [0],
            [0],
            marker="o",
            color=COLORS["red"],
            markerfacecolor=COLORS["red"],
            markeredgecolor="white",
            markeredgewidth=0.9,
            lw=0,
            markersize=7,
            label=r"VP+ band 0: $C=1$",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=COLORS["red"],
            markerfacecolor="white",
            markeredgecolor=COLORS["red"],
            markeredgewidth=1.7,
            lw=0,
            markersize=7,
            label=r"VP+ band 0: $C=0$",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=(0.02, 0.07, 0.995, 0.995), h_pad=1.2, w_pad=0.75)

    params.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = params.output_dir / f"{params.output_stem}_summary.csv"
    points_csv = params.output_dir / f"{params.output_stem}_points.csv"
    png_path = params.output_dir / f"{params.output_stem}.png"
    pdf_path = params.output_dir / f"{params.output_stem}.pdf"
    svg_path = params.output_dir / f"{params.output_stem}.svg"

    unstable_summary.to_csv(summary_csv, index=False)
    points.to_csv(points_csv, index=False)
    for path in (png_path, pdf_path, svg_path):
        fig.savefig(path, dpi=FIGURE["dpi"], bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return [summary_csv, points_csv, png_path, pdf_path, svg_path]


def main() -> None:
    params = UnstableChernOutlierParams()
    apply_style()
    unstable_summary, points = load_data(params)
    print(
        f"Found {len(unstable_summary)} non-grey all-seven points with unstable "
        "VP+ band-0 Chern."
    )
    for path in render_panel(unstable_summary, points, params):
        print(path)


if __name__ == "__main__":
    main()
