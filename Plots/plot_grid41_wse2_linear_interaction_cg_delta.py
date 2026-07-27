#!/usr/bin/env python3
"""WSe2 linear-interaction cG maps and linear-minus-convex delta maps."""

from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm


REPO_ROOT = Path(__file__).resolve().parents[1]
LINEAR_ROOT = REPO_ROOT / "results" / "wse2_ivc_hysteresis_finite_size_nk18_24_grid41_linear_interaction_recomputed_sharded"
CONVEX_ROOT = REPO_ROOT / "results" / "wse2_ivc_hysteresis_finite_size_nk18_24_grid41"
CONVEX_ANALYSIS = CONVEX_ROOT / "analysis"
FIGURE_DIR = REPO_ROOT / "Plots" / "figures"

LINEAR_FIT_SOURCE = LINEAR_ROOT / "hysteresis_finite_size_fit_source.csv"
CONVEX_FIT_SOURCE = CONVEX_ROOT / "hysteresis_finite_size_fit_source.csv"
PHASE_CSV = CONVEX_ANALYSIS / "wse2_nk24_lowest_clean_phase_table.csv"
ENERGY_FITS_CSV = CONVEX_ANALYSIS / "wse2_lowest_clean_best5_fits.csv"
SUMMARY_JSON = CONVEX_ANALYSIS / "wse2_analysis_summary.json"

FIT_COMPARISON_CSV = FIGURE_DIR / "grid41_wse2_linear_vs_convex_cG_all_best5_fits.csv"
DELTA_CSV = FIGURE_DIR / "grid41_wse2_linear_minus_convex_delta_cG_all_best5.csv"

SINGLE_OUTPUT_STEM = "grid41_wse2_linear_interaction_all7_cG_heatmap"
SINGLE_CROPPED_OUTPUT_STEM = "grid41_wse2_linear_interaction_all7_cG_heatmap_theta_le_3p7"
CG_OUTPUT_STEM = "grid41_wse2_linear_interaction_all_vs_best5_cG_heatmaps"
DELTA_OUTPUT_STEM = "grid41_wse2_linear_minus_convex_delta_cG_all_best5_heatmaps"

MIN_ALL_FIT_POINTS = 3
BEST_FIT_POINTS = 5
SINGLE_CROPPED_THETA_MAX_DEG = 3.7


FIGURE = {
    "single_size": (7.4, 7.4),
    "two_panel_size": (13.2, 6.35),
    "dpi": 280,
    "single_adjust": {"left": 0.12, "right": 0.82, "bottom": 0.12, "top": 0.96},
}

FONTS = {
    "base": 12,
    "title": 18,
    "axis_label": 24,
    "tick_label": 20,
    "legend": 18,
    "phase_label": 20,
    "colorbar_label": 26,
    "colorbar_tick": 20,
}

COLORS = {
    "cG_negative": "#378d94",
    "cG_center": "#f7f7f7",
    "cG_positive": "#FD4C55",
    "grey_mask": "0.72",
    "ivc_ivc": "#0072B2",
    "vp_chern": "#7c3aed",
    "vp_ivc": "0.25",
    "axis": "0.18",
}

LABELS = {
    "x": r"$\theta_M$ ($^\circ$)",
    "y": r"$u_D$ (meV)",
    "cG": r"$c_G$",
    "delta_cG": r"$\Delta c_G$",
    "ivc_ivc": "IVC-IVC",
    "vp_chern": "VP Chern",
    "vp_ivc": "IVC-VP",
}

AXES = {
    "xticks": (2.0, 2.5, 3.0, 3.5, 4.0),
    "yticks": (0, 5, 10, 15, 20),
    "spine_linewidth": 1.15,
    "box_aspect": 1.0,
    "colorbar_width_fraction": 0.05,
    "colorbar_pad_fraction": 0.05,
}

LINE_STYLES = {
    "ivc_ivc": {"linestyle": "--", "linewidth": 2.35, "zorder": 8},
    "vp_chern": {"linestyle": "--", "linewidth": 2.25, "zorder": 9},
    "vp_ivc": {"linestyle": "-", "linewidth": 2.1, "zorder": 10},
}

CROPPED_PHASE_LABELS = (
    {"text": "VP\n$C=0$", "theta_deg": 2.30, "u_D_meV": 10.5},
    {"text": "VP\n$C=1$", "theta_deg": 3.38, "u_D_meV": 3.0},
    {"text": "IVC", "theta_deg": 3.40, "u_D_meV": 17.0},
)


def _apply_style() -> None:
    plt.rcParams.update(
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


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    steps = np.diff(values)
    step = float(np.median(steps)) if len(steps) else 1.0
    return np.r_[values[0] - step / 2, 0.5 * (values[:-1] + values[1:]), values[-1] + step / 2]


def _fit_line(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    coeff = np.polyfit(x, y, deg=1)
    pred = np.polyval(coeff, x)
    resid = y - pred
    return {
        "intercept": float(coeff[1]),
        "slope": float(coeff[0]),
        "rmse": float(np.sqrt(np.mean(resid**2))) if len(resid) else np.nan,
        "max_abs_resid": float(np.max(np.abs(resid))) if len(resid) else np.nan,
    }


def _all_fit(group: pd.DataFrame) -> dict[str, object]:
    clean = group[_as_bool(group["clean"])].copy()
    clean["cG"] = pd.to_numeric(clean["cG"], errors="coerce")
    clean = clean[np.isfinite(clean["cG"])].sort_values("n_k")
    if len(clean) < MIN_ALL_FIT_POINTS:
        return {
            "cG_all_status": "insufficient_clean_finite",
            "cG_all_n_finite": int(len(clean)),
            "cG_all_n_k_values": ",".join(str(int(v)) for v in clean["n_k"]),
            "cG_all_intercept": np.nan,
            "cG_all_slope": np.nan,
            "cG_all_rmse": np.nan,
            "cG_all_max_abs_resid": np.nan,
        }
    fit = _fit_line(clean["inv_n_k"].to_numpy(dtype=float), clean["cG"].to_numpy(dtype=float))
    return {
        "cG_all_status": "fit_ok",
        "cG_all_n_finite": int(len(clean)),
        "cG_all_n_k_values": ",".join(str(int(v)) for v in clean["n_k"]),
        "cG_all_intercept": fit["intercept"],
        "cG_all_slope": fit["slope"],
        "cG_all_rmse": fit["rmse"],
        "cG_all_max_abs_resid": fit["max_abs_resid"],
    }


def _best5_fit(group: pd.DataFrame) -> dict[str, object]:
    clean = group[_as_bool(group["clean"])].copy()
    clean["cG"] = pd.to_numeric(clean["cG"], errors="coerce")
    clean = clean[np.isfinite(clean["cG"])].sort_values("n_k")
    if len(clean) < BEST_FIT_POINTS:
        return {
            "cG_best5_status": "insufficient_clean_finite",
            "cG_best5_n_finite": int(len(clean)),
            "cG_best5_n_k_values": ",".join(str(int(v)) for v in clean["n_k"]),
            "cG_best5_best_n_k_values": "",
            "cG_best5_intercept": np.nan,
            "cG_best5_slope": np.nan,
            "cG_best5_rmse": np.nan,
            "cG_best5_max_abs_resid": np.nan,
        }

    best: dict[str, object] | None = None
    for indices in combinations(range(len(clean)), BEST_FIT_POINTS):
        subset = clean.iloc[list(indices)]
        fit = _fit_line(subset["inv_n_k"].to_numpy(dtype=float), subset["cG"].to_numpy(dtype=float))
        candidate = {
            "indices": indices,
            "intercept": fit["intercept"],
            "slope": fit["slope"],
            "rmse": fit["rmse"],
            "max_abs_resid": fit["max_abs_resid"],
        }
        key = (float(candidate["rmse"]), -float(np.mean(subset["n_k"].to_numpy(dtype=float))))
        if best is None or key < best["key"]:
            best = {"key": key, **candidate}

    assert best is not None
    subset = clean.iloc[list(best["indices"])]
    return {
        "cG_best5_status": "fit_ok_best5",
        "cG_best5_n_finite": int(len(clean)),
        "cG_best5_n_k_values": ",".join(str(int(v)) for v in clean["n_k"]),
        "cG_best5_best_n_k_values": ",".join(str(int(v)) for v in subset["n_k"]),
        "cG_best5_intercept": float(best["intercept"]),
        "cG_best5_slope": float(best["slope"]),
        "cG_best5_rmse": float(best["rmse"]),
        "cG_best5_max_abs_resid": float(best["max_abs_resid"]),
    }


def _load_fit_source(path: Path, interpolation: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    source = pd.read_csv(path)
    source = source[
        (source["branch_label"] == "lowest_energy_clean")
        & (source["selection_kind"] == "clean")
    ].copy()
    if "trial_interpolation" in source.columns:
        source = source[source["trial_interpolation"].fillna(interpolation).eq(interpolation)]
    return source


def _compute_fits(path: Path, interpolation: str, prefix: str) -> pd.DataFrame:
    source = _load_fit_source(path, interpolation)
    rows: list[dict[str, object]] = []
    group_cols = ["theta_index", "u_index", "theta_deg", "u_D_meV"]
    for keys, group in source.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys, strict=True))
        row.update(_all_fit(group))
        row.update(_best5_fit(group))
        rows.append(row)
    out = pd.DataFrame(rows)
    rename = {
        col: f"{prefix}_{col}"
        for col in out.columns
        if col not in group_cols
    }
    return out.rename(columns=rename)


def _load_gap_threshold() -> float | None:
    if not SUMMARY_JSON.exists():
        return None
    with SUMMARY_JSON.open() as handle:
        summary = json.load(handle)
    value = summary.get("gap_threshold_meV")
    return None if value is None else float(value)


def _build_comparison_table() -> pd.DataFrame:
    linear = _compute_fits(LINEAR_FIT_SOURCE, "linear_interaction", "linear")
    convex = _compute_fits(CONVEX_FIT_SOURCE, "convex_full_hf", "convex")
    merged = linear.merge(
        convex,
        on=["theta_index", "u_index", "theta_deg", "u_D_meV"],
        how="inner",
        validate="one_to_one",
    )

    phase = pd.read_csv(
        PHASE_CSV,
        usecols=[
            "theta_deg",
            "u_D_meV",
            "direct_gap_min",
            "gap_class",
            "chern_hf_vpplus_band_0",
            "delta_E_ivc_minus_vp",
        ],
    )
    energy = pd.read_csv(
        ENERGY_FITS_CSV,
        usecols=[
            "theta_deg",
            "u_D_meV",
            "delta_E_ivc_minus_vp_fit_status",
            "delta_E_ivc_minus_vp_intercept",
        ],
    )
    merged = merged.merge(phase, on=["theta_deg", "u_D_meV"], how="left", validate="one_to_one")
    merged = merged.merge(energy, on=["theta_deg", "u_D_meV"], how="left", validate="one_to_one")

    gap_threshold = _load_gap_threshold()
    if gap_threshold is None:
        merged["ivc_large_gap_region_nk24"] = merged["gap_class"].eq("large_gap")
    else:
        merged["ivc_large_gap_region_nk24"] = pd.to_numeric(merged["direct_gap_min"], errors="coerce") > gap_threshold
    merged["vp_topological_region_nk24"] = pd.to_numeric(merged["chern_hf_vpplus_band_0"], errors="coerce") < -0.5
    merged["ivc_below_vp_inf"] = pd.to_numeric(merged["delta_E_ivc_minus_vp_intercept"], errors="coerce") < 0.0

    for fit_kind in ("all", "best5"):
        merged[f"delta_cG_{fit_kind}_linear_minus_convex"] = (
            pd.to_numeric(merged[f"linear_cG_{fit_kind}_intercept"], errors="coerce")
            - pd.to_numeric(merged[f"convex_cG_{fit_kind}_intercept"], errors="coerce")
        )
        ok_status = "fit_ok" if fit_kind == "all" else "fit_ok_best5"
        merged[f"grey_mask_linear_{fit_kind}"] = (
            merged[f"linear_cG_{fit_kind}_status"].ne(ok_status)
            | ~np.isfinite(pd.to_numeric(merged[f"linear_cG_{fit_kind}_intercept"], errors="coerce"))
            | merged["ivc_below_vp_inf"]
        )
        merged[f"grey_mask_delta_{fit_kind}"] = (
            merged[f"linear_cG_{fit_kind}_status"].ne(ok_status)
            | merged[f"convex_cG_{fit_kind}_status"].ne(ok_status)
            | ~np.isfinite(pd.to_numeric(merged[f"delta_cG_{fit_kind}_linear_minus_convex"], errors="coerce"))
            | merged["ivc_below_vp_inf"]
        )
    merged.to_csv(FIT_COMPARISON_CSV, index=False)
    merged[
        [
            "theta_index",
            "u_index",
            "theta_deg",
            "u_D_meV",
            "delta_cG_all_linear_minus_convex",
            "delta_cG_best5_linear_minus_convex",
            "grey_mask_delta_all",
            "grey_mask_delta_best5",
            "ivc_below_vp_inf",
            "ivc_large_gap_region_nk24",
            "vp_topological_region_nk24",
        ]
    ].to_csv(DELTA_CSV, index=False)
    return merged


def _grid(df: pd.DataFrame, theta_vals: np.ndarray, u_vals: np.ndarray, col: str) -> np.ndarray:
    return (
        df.pivot(index="u_D_meV", columns="theta_deg", values=col)
        .reindex(index=u_vals, columns=theta_vals)
        .to_numpy()
    )


def _cell_edge_boundary_segments(mask: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> list[list[tuple[float, float]]]:
    values = np.asarray(mask, dtype=float)
    finite = np.isfinite(values)
    phase = values > 0.5
    segments: list[list[tuple[float, float]]] = []
    for row in range(values.shape[0]):
        for col in range(values.shape[1] - 1):
            if not (finite[row, col] and finite[row, col + 1]):
                continue
            if phase[row, col] == phase[row, col + 1]:
                continue
            x = float(x_edges[col + 1])
            segments.append([(x, float(y_edges[row])), (x, float(y_edges[row + 1]))])
    for row in range(values.shape[0] - 1):
        for col in range(values.shape[1]):
            if not (finite[row, col] and finite[row + 1, col]):
                continue
            if phase[row, col] == phase[row + 1, col]:
                continue
            y = float(y_edges[row + 1])
            segments.append([(float(x_edges[col]), y), (float(x_edges[col + 1]), y)])
    return segments


def _draw_boundary(ax: plt.Axes, mask: np.ndarray, theta_vals: np.ndarray, u_vals: np.ndarray, key: str) -> None:
    segments = _cell_edge_boundary_segments(mask, _edges(theta_vals), _edges(u_vals))
    if not segments:
        return
    style = LINE_STYLES[key]
    collection = LineCollection(
        segments,
        colors=[COLORS[key]],
        linewidths=style["linewidth"],
        linestyles=style["linestyle"],
        zorder=style["zorder"],
    )
    collection.set_capstyle("butt")
    collection.set_joinstyle("miter")
    ax.add_collection(collection)


def _colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "bootstrap_teal_red",
        [COLORS["cG_negative"], COLORS["cG_center"], COLORS["cG_positive"]],
        N=256,
    )


def _box_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["axis"])
        spine.set_linewidth(AXES["spine_linewidth"])


def _add_phase_labels(ax: plt.Axes, labels: tuple[dict[str, object], ...]) -> None:
    for label in labels:
        ax.text(
            float(label["theta_deg"]),
            float(label["u_D_meV"]),
            str(label["text"]),
            ha="center",
            va="center",
            fontsize=FONTS["phase_label"],
            fontweight="normal",
            linespacing=0.9,
            color="black",
            zorder=12,
        )


def _boundary_masks(heat: pd.DataFrame, theta_vals: np.ndarray, u_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ivc_large = _grid(heat, theta_vals, u_vals, "ivc_large_gap_region_nk24").astype(float)
    vp_topo = _grid(heat, theta_vals, u_vals, "vp_topological_region_nk24").astype(float)
    ivc_below_vp = _grid(heat, theta_vals, u_vals, "ivc_below_vp_inf").astype(float)
    vp_ground_region = np.isfinite(ivc_below_vp) & (ivc_below_vp < 0.5)
    vp_topo = np.where(vp_ground_region, vp_topo, np.nan)
    return ivc_large, vp_topo, ivc_below_vp


def _add_boundaries_and_legend(
    ax: plt.Axes,
    heat: pd.DataFrame,
    theta_vals: np.ndarray,
    u_vals: np.ndarray,
    *,
    include_legend: bool,
) -> None:
    ivc_large, vp_topo, ivc_below_vp = _boundary_masks(heat, theta_vals, u_vals)
    _draw_boundary(ax, ivc_large, theta_vals, u_vals, "ivc_ivc")
    _draw_boundary(ax, vp_topo, theta_vals, u_vals, "vp_chern")
    _draw_boundary(ax, ivc_below_vp, theta_vals, u_vals, "vp_ivc")
    if not include_legend:
        return
    for key in ("ivc_ivc", "vp_chern", "vp_ivc"):
        ax.plot(
            [],
            [],
            color=COLORS[key],
            lw=LINE_STYLES[key]["linewidth"],
            ls=LINE_STYLES[key]["linestyle"],
            label=LABELS[key],
        )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.20, 0.985),
        frameon=True,
        framealpha=0.78,
        facecolor="white",
        borderpad=0.35,
        handlelength=2.0,
        handletextpad=0.55,
        labelspacing=0.35,
    )


def _setup_axis(ax: plt.Axes, theta_vals: np.ndarray, u_vals: np.ndarray) -> None:
    ax.set_xlim(theta_vals.min(), theta_vals.max())
    ax.set_ylim(u_vals.min(), u_vals.max())
    ax.set_xticks(AXES["xticks"])
    ax.set_yticks(AXES["yticks"])
    ax.set_xlabel(LABELS["x"])
    ax.set_ylabel(LABELS["y"])
    ax.set_box_aspect(AXES["box_aspect"])
    _box_axes(ax)


def _draw_heatmap_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    heat: pd.DataFrame,
    theta_vals: np.ndarray,
    u_vals: np.ndarray,
    *,
    value_col: str,
    grey_col: str,
    title: str | None,
    cbar_label: str,
    norm: Normalize,
    cbar_extend: str = "neither",
    include_legend: bool = False,
) -> None:
    values = _grid(heat, theta_vals, u_vals, value_col).astype(float)
    grey = _grid(heat, theta_vals, u_vals, grey_col).astype(bool)
    masked = np.ma.masked_where(grey | ~np.isfinite(values), values)
    mesh = ax.pcolormesh(
        _edges(theta_vals),
        _edges(u_vals),
        masked,
        cmap=_colormap(),
        norm=norm,
        shading="auto",
    )
    grey_cells = np.ma.masked_where(~grey, grey.astype(float))
    ax.pcolormesh(
        _edges(theta_vals),
        _edges(u_vals),
        grey_cells,
        cmap=ListedColormap([COLORS["grey_mask"]]),
        shading="auto",
        zorder=3,
    )
    _add_boundaries_and_legend(ax, heat, theta_vals, u_vals, include_legend=include_legend)
    if title:
        ax.set_title(title, pad=10)
    _setup_axis(ax, theta_vals, u_vals)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.045, pad=0.035, extend=cbar_extend)
    cbar.set_label("")
    cbar.ax.set_title(cbar_label, fontsize=FONTS["axis_label"], pad=10)
    cbar.ax.tick_params(labelsize=FONTS["tick_label"] * 0.82)


def _save(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in (".png", ".pdf"):
        path = FIGURE_DIR / f"{stem}{suffix}"
        fig.savefig(path, dpi=FIGURE["dpi"], bbox_inches="tight", pad_inches=0.08)
        paths.append(path)
    plt.close(fig)
    return paths


def _plot_single_cg(heat: pd.DataFrame) -> list[Path]:
    theta_vals = np.array(sorted(heat["theta_deg"].unique()), dtype=float)
    u_vals = np.array(sorted(heat["u_D_meV"].unique()), dtype=float)
    values = pd.to_numeric(heat["linear_cG_all_intercept"], errors="coerce").to_numpy(dtype=float)
    vmax = max(float(np.nanpercentile(np.abs(values[np.isfinite(values)]), 99.0)), 0.05)

    fig, ax = plt.subplots(figsize=FIGURE["single_size"], constrained_layout=False)
    fig.subplots_adjust(**FIGURE["single_adjust"])
    _draw_heatmap_panel(
        fig,
        ax,
        heat,
        theta_vals,
        u_vals,
        value_col="linear_cG_all_intercept",
        grey_col="grey_mask_linear_all",
        title=None,
        cbar_label=LABELS["cG"],
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        include_legend=True,
    )
    return _save(fig, SINGLE_OUTPUT_STEM)


def _plot_single_cg_cropped(heat: pd.DataFrame) -> list[Path]:
    """Render the final all-7-fit WSe2 map only through theta_M=3.7 degrees."""

    theta_vals = np.array(sorted(heat["theta_deg"].unique()), dtype=float)
    u_vals = np.array(sorted(heat["u_D_meV"].unique()), dtype=float)
    values = pd.to_numeric(heat["linear_cG_all_intercept"], errors="coerce").to_numpy(dtype=float)
    vmax = max(float(np.nanpercentile(np.abs(values[np.isfinite(values)]), 99.0)), 0.05)

    if not theta_vals.min() < SINGLE_CROPPED_THETA_MAX_DEG <= theta_vals.max():
        raise ValueError(
            f"theta crop {SINGLE_CROPPED_THETA_MAX_DEG:g} degrees lies outside "
            f"[{theta_vals.min():g}, {theta_vals.max():g}]"
        )

    fig, ax = plt.subplots(figsize=FIGURE["single_size"], constrained_layout=False)
    fig.subplots_adjust(**FIGURE["single_adjust"])
    _draw_heatmap_panel(
        fig,
        ax,
        heat,
        theta_vals,
        u_vals,
        value_col="linear_cG_all_intercept",
        grey_col="grey_mask_linear_all",
        title=None,
        cbar_label=LABELS["cG"],
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        include_legend=True,
    )
    ax.set_xlim(theta_vals.min(), SINGLE_CROPPED_THETA_MAX_DEG)
    ax.set_xticks(
        [tick for tick in AXES["xticks"] if theta_vals.min() <= tick <= SINGLE_CROPPED_THETA_MAX_DEG]
    )
    _add_phase_labels(ax, CROPPED_PHASE_LABELS)
    return _save(fig, SINGLE_CROPPED_OUTPUT_STEM)


def _plot_cg_all_vs_best5(heat: pd.DataFrame) -> list[Path]:
    theta_vals = np.array(sorted(heat["theta_deg"].unique()), dtype=float)
    u_vals = np.array(sorted(heat["u_D_meV"].unique()), dtype=float)
    values = pd.to_numeric(
        pd.concat([heat["linear_cG_all_intercept"], heat["linear_cG_best5_intercept"]]),
        errors="coerce",
    ).to_numpy(dtype=float)
    vmax = max(float(np.nanpercentile(np.abs(values[np.isfinite(values)]), 99.0)), 0.05)

    fig, axes = plt.subplots(1, 2, figsize=FIGURE["two_panel_size"], constrained_layout=True)
    panels = [
        (axes[0], "linear_cG_all_intercept", "grey_mask_linear_all", r"$c_G$ all-7 fit"),
        (axes[1], "linear_cG_best5_intercept", "grey_mask_linear_best5", r"$c_G$ best-5 fit"),
    ]
    for i, (ax, value_col, grey_col, title) in enumerate(panels):
        _draw_heatmap_panel(
            fig,
            ax,
            heat,
            theta_vals,
            u_vals,
            value_col=value_col,
            grey_col=grey_col,
            title=title,
            cbar_label=LABELS["cG"],
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            include_legend=(i == 0),
        )
    return _save(fig, CG_OUTPUT_STEM)


def _plot_delta_all_vs_best5(heat: pd.DataFrame) -> list[Path]:
    theta_vals = np.array(sorted(heat["theta_deg"].unique()), dtype=float)
    u_vals = np.array(sorted(heat["u_D_meV"].unique()), dtype=float)
    values = pd.to_numeric(
        pd.concat([
            heat["delta_cG_all_linear_minus_convex"],
            heat["delta_cG_best5_linear_minus_convex"],
        ]),
        errors="coerce",
    ).to_numpy(dtype=float)
    vmax = max(float(np.nanpercentile(np.abs(values[np.isfinite(values)]), 99.0)), 0.002)

    fig, axes = plt.subplots(1, 2, figsize=FIGURE["two_panel_size"], constrained_layout=True)
    panels = [
        (axes[0], "delta_cG_all_linear_minus_convex", "grey_mask_delta_all", r"$\Delta c_G$ all-7 fit"),
        (axes[1], "delta_cG_best5_linear_minus_convex", "grey_mask_delta_best5", r"$\Delta c_G$ best-5 fit"),
    ]
    for i, (ax, value_col, grey_col, title) in enumerate(panels):
        _draw_heatmap_panel(
            fig,
            ax,
            heat,
            theta_vals,
            u_vals,
            value_col=value_col,
            grey_col=grey_col,
            title=title,
            cbar_label=LABELS["delta_cG"],
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            cbar_extend="both",
            include_legend=(i == 0),
        )
    return _save(fig, DELTA_OUTPUT_STEM)


def main() -> None:
    _apply_style()
    heat = _build_comparison_table()
    paths = []
    paths.extend(_plot_single_cg(heat))
    paths.extend(_plot_single_cg_cropped(heat))
    paths.extend(_plot_cg_all_vs_best5(heat))
    paths.extend(_plot_delta_all_vs_best5(heat))
    valid_best5 = heat[~heat["grey_mask_delta_best5"].astype(bool)]
    valid_all = heat[~heat["grey_mask_delta_all"].astype(bool)]
    summary = {
        "fit_comparison_csv": str(FIT_COMPARISON_CSV),
        "delta_csv": str(DELTA_CSV),
        "outputs": [str(path) for path in paths],
        "n_phase_points": int(len(heat)),
        "n_valid_all_delta": int(len(valid_all)),
        "n_valid_best5_delta": int(len(valid_best5)),
        "delta_all_min": float(np.nanmin(valid_all["delta_cG_all_linear_minus_convex"])) if len(valid_all) else np.nan,
        "delta_all_max": float(np.nanmax(valid_all["delta_cG_all_linear_minus_convex"])) if len(valid_all) else np.nan,
        "delta_best5_min": float(np.nanmin(valid_best5["delta_cG_best5_linear_minus_convex"])) if len(valid_best5) else np.nan,
        "delta_best5_max": float(np.nanmax(valid_best5["delta_cG_best5_linear_minus_convex"])) if len(valid_best5) else np.nan,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
