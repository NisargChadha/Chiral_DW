#!/usr/bin/env python3
"""tMoTe2 linear-interaction finite-size cG-fit percentage-error heatmap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from pydantic import BaseModel, ConfigDict, Field, model_validator


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_BUNDLE = (
    REPO_ROOT
    / "results"
    / "taige_ivc_hysteresis_finite_size_nk18_24_grid41_linear_interaction_recomputed_sharded"
)
INPUT_CSV = RESULT_BUNDLE / "analysis_plots" / "grid41_linear_interaction_lowest_clean_best5_cG_fits.csv"
FIGURE_DIR = REPO_ROOT / "Plots" / "figures"
OUTPUT_STEM = "grid41_tmote2_linear_interaction_best5_cG_percent_error_cGinf_lt_m0p02_heatmap"


FIGURE = {
    "size": (7.4, 7.4),
    "dpi": 280,
    "subplots_adjust": {"left": 0.12, "right": 0.82, "bottom": 0.12, "top": 0.96},
}

FONTS = {
    "base": 12,
    "axis_label": 24,
    "tick_label": 20,
    "colorbar_label": 22,
    "colorbar_tick": 20,
}

COLORS = {
    "error_low": "#f7f7f7",
    "error_high": "#6a408d",
    "grey_mask": "0.72",
    "axis": "0.18",
}

LABELS = {
    "x": r"$\theta_M$ ($^\circ$)",
    "y": r"$u_D$ (meV)",
    "cbar": "Fit error\n(%)",
}

AXES = {
    "xticks": (2.0, 2.5, 3.0, 3.5, 4.0),
    "yticks": (0, 5, 10, 15, 20),
    "spine_linewidth": 1.15,
    "box_aspect": 1.0,
    "colorbar_width_fraction": 0.05,
    "colorbar_pad_fraction": 0.05,
}


class CGPercentErrorHeatmapParams(BaseModel):
    """User-facing controls for the tMoTe2 cG finite-size percentage-error heatmap."""

    model_config = ConfigDict(frozen=True)

    input_csv: Path = Field(default=INPUT_CSV)
    output_dir: Path = Field(default=FIGURE_DIR)
    output_stem: str = Field(default=OUTPUT_STEM)
    rmse_column: str = Field(default="cG_best5_rmse")
    intercept_column: str = Field(default="cG_best5_intercept")
    status_column: str = Field(default="cG_best5_status")
    valid_statuses: tuple[str, ...] = Field(default=("fit_ok_best5", "fit_ok_all_finite_fallback"))
    cG_inf_upper_threshold: float = Field(default=-0.02)
    vmin_percent: float = Field(default=0.0)
    vmax_percent: float = Field(default=10.0)

    @model_validator(mode="after")
    def _limits_are_valid(self) -> "CGPercentErrorHeatmapParams":
        if self.vmax_percent <= self.vmin_percent:
            raise ValueError("vmax_percent must be larger than vmin_percent")
        if self.cG_inf_upper_threshold >= 0.0:
            raise ValueError("cG_inf_upper_threshold should be negative for this masked plot")
        return self


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["PT Serif Caption", "PT Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "font.size": FONTS["base"],
            "axes.labelsize": FONTS["axis_label"],
            "xtick.labelsize": FONTS["tick_label"],
            "ytick.labelsize": FONTS["tick_label"],
            "axes.edgecolor": COLORS["axis"],
            "axes.linewidth": AXES["spine_linewidth"],
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


def _edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    steps = np.diff(values)
    step = float(np.median(steps)) if len(steps) else 1.0
    return np.r_[values[0] - step / 2, 0.5 * (values[:-1] + values[1:]), values[-1] + step / 2]


def _grid(df: pd.DataFrame, theta_vals: np.ndarray, u_vals: np.ndarray, col: str) -> np.ndarray:
    return (
        df.pivot(index="u_D_meV", columns="theta_deg", values=col)
        .reindex(index=u_vals, columns=theta_vals)
        .to_numpy()
    )


def _percent_error_cmap() -> LinearSegmentedColormap:
    cmap = LinearSegmentedColormap.from_list(
        "white_to_nisarg_purple",
        [COLORS["error_low"], COLORS["error_high"]],
        N=256,
    )
    cmap.set_over(COLORS["error_high"])
    return cmap


def _box_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["axis"])
        spine.set_linewidth(AXES["spine_linewidth"])


def _load_plot_data(params: CGPercentErrorHeatmapParams) -> pd.DataFrame:
    if not params.input_csv.exists():
        raise FileNotFoundError(f"Missing finite-size cG fit CSV: {params.input_csv}")
    required = [
        "theta_deg",
        "u_D_meV",
        params.rmse_column,
        params.intercept_column,
        params.status_column,
    ]
    df = pd.read_csv(params.input_csv)
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {params.input_csv}: {missing}")

    out = df.loc[:, required].copy()
    out["cG_inf"] = pd.to_numeric(out[params.intercept_column], errors="coerce")
    out["rmse_cG_fit"] = pd.to_numeric(out[params.rmse_column], errors="coerce")
    out["selected_cG_inf_lt_m0p02"] = (
        out[params.status_column].isin(params.valid_statuses)
        & np.isfinite(out["cG_inf"])
        & np.isfinite(out["rmse_cG_fit"])
        & (out["cG_inf"] < params.cG_inf_upper_threshold)
    )
    out["percent_error"] = np.nan
    selected = out["selected_cG_inf_lt_m0p02"]
    out.loc[selected, "percent_error"] = 100.0 * out.loc[selected, "rmse_cG_fit"] / out.loc[selected, "cG_inf"].abs()
    out = out.rename(columns={params.status_column: "fit_status"})
    return out.drop(columns=[params.rmse_column, params.intercept_column])


def render_percent_error_heatmap(
    params: CGPercentErrorHeatmapParams = CGPercentErrorHeatmapParams(),
) -> list[Path]:
    plot_data = _load_plot_data(params)
    theta_vals = np.array(sorted(plot_data["theta_deg"].unique()), dtype=float)
    u_vals = np.array(sorted(plot_data["u_D_meV"].unique()), dtype=float)
    theta_edges = _edges(theta_vals)
    u_edges = _edges(u_vals)

    percent_error = _grid(plot_data, theta_vals, u_vals, "percent_error").astype(float)
    selected = _grid(plot_data, theta_vals, u_vals, "selected_cG_inf_lt_m0p02").astype(bool)
    grey = ~selected | ~np.isfinite(percent_error)
    masked_percent_error = np.ma.masked_where(grey, percent_error)
    if not np.any(np.isfinite(percent_error)):
        raise ValueError("No selected finite percent-error values found to plot")

    fig, ax = plt.subplots(figsize=FIGURE["size"], constrained_layout=False)
    fig.subplots_adjust(**FIGURE["subplots_adjust"])
    mesh = ax.pcolormesh(
        theta_edges,
        u_edges,
        masked_percent_error,
        cmap=_percent_error_cmap(),
        norm=Normalize(vmin=params.vmin_percent, vmax=params.vmax_percent),
        shading="auto",
    )
    grey_cells = np.ma.masked_where(~grey, grey.astype(float))
    ax.pcolormesh(
        theta_edges,
        u_edges,
        grey_cells,
        cmap=ListedColormap([COLORS["grey_mask"]]),
        shading="auto",
        zorder=3,
    )

    ax.set_xlim(theta_vals.min(), theta_vals.max())
    ax.set_ylim(u_vals.min(), u_vals.max())
    ax.set_xticks(AXES["xticks"])
    ax.set_yticks(AXES["yticks"])
    ax.set_xlabel(LABELS["x"])
    ax.set_ylabel(LABELS["y"])
    ax.set_box_aspect(AXES["box_aspect"])
    _box_axes(ax)

    fig.canvas.draw()
    bbox = ax.get_position()
    cbar_width = bbox.width * AXES["colorbar_width_fraction"]
    cbar_pad = bbox.width * AXES["colorbar_pad_fraction"]
    cbar_ax = fig.add_axes([bbox.x1 + cbar_pad, bbox.y0, cbar_width, bbox.height])
    cbar = fig.colorbar(mesh, cax=cbar_ax, extend="max")
    ticks = np.arange(params.vmin_percent, params.vmax_percent + 0.1, 2.0)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{value:.0f}" for value in ticks])
    cbar.ax.set_title(LABELS["cbar"], fontsize=FONTS["colorbar_label"], pad=12)
    cbar.ax.tick_params(labelsize=FONTS["colorbar_tick"])

    params.output_dir.mkdir(parents=True, exist_ok=True)
    plot_csv = params.output_dir / f"{params.output_stem}.csv"
    plot_data.sort_values(["theta_deg", "u_D_meV"]).to_csv(plot_csv, index=False)

    paths = [plot_csv]
    for suffix in (".png", ".pdf"):
        path = params.output_dir / f"{params.output_stem}{suffix}"
        fig.savefig(path, dpi=FIGURE["dpi"], bbox_inches="tight", pad_inches=0.08)
        paths.append(path)
    plt.close(fig)
    return paths


def main() -> None:
    _apply_style()
    for path in render_percent_error_heatmap():
        print(path)


if __name__ == "__main__":
    main()
