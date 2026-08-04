#!/usr/bin/env python3
"""Separate tMoTe2 all-7-fit Hartree-Fock phase and cG diagrams."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_BUNDLE = (
    REPO_ROOT
    / "results"
    / "taige_ivc_hysteresis_finite_size_nk18_24_grid41_linear_interaction_recomputed_sharded"
)
INPUT_CSV = RESULT_BUNDLE / "analysis_plots" / "grid41_linear_interaction_best5_cG_heatmap_with_boundaries.csv"
HF_SWEEP_CSV = RESULT_BUNDLE / "nk_024" / "hysteresis_sweep.csv"
FIGURE_DIR = REPO_ROOT / "Plots" / "figures"
PHASE_OUTPUT_STEM = "grid41_tmote2_linear_interaction_all7_hf_phase_diagram"
PHASE_D_OUTPUT_STEM = "grid41_tmote2_linear_interaction_all7_hf_phase_diagram_D_axis"
CG_OUTPUT_STEM = "grid41_tmote2_linear_interaction_all7_cG_ground_state_boundaries"
OUTPUT_STEM = CG_OUTPUT_STEM

# Publication default: use the established all-fit extrapolation.  This fits
# every clean finite mesh available from the nominal seven-mesh set and relies
# on the producer's grey_mask_all/status criterion for incomplete fits.
FINITE_SIZE_FIT_KIND = "all7"
CG_VALUE_COLUMN = "cG_all_plot"
CG_GREY_MASK_COLUMN = "grey_mask_all"


FIGURE = {
    "size": (7.4, 7.4),
    "dpi": 280,
    "cg_subplots_adjust": {"left": 0.12, "right": 0.82, "bottom": 0.12, "top": 0.96},
    "phase_subplots_adjust": {"left": 0.12, "right": 0.96, "bottom": 0.12, "top": 0.96},
}

FONTS = {
    "base": 12,
    "axis_label": 32,
    "tick_label": 22,
    "legend": 22,
    "phase_label": 32,
    "colorbar_label": 32,
    "colorbar_tick": 22,
}

COLORS = {
    "cG_negative": "#378d94",
    "cG_center": "#f7f7f7",
    "cG_positive": "#FD4C55",
    "vp_c0": "#FD4C55",
    "vp_c1": "#0072B2",
    "ivc": "0.72",
    "phase_boundary": "0.18",
    "grey_mask": "0.72",
    "vp_chern": "#7c3aed",
    "vp_ivc": "0.25",
    "axis": "0.18",
}

LABELS = {
    "x": r"$\theta_M$ ($^\circ$)",
    "y": r"$u_D$ (meV)",
    "y_D": r"$D$ (meV)",
    "cbar": r"$c_G$",
    "vp_c0": r"VP, $C=0$",
    "vp_c1": r"VP, $C=1$",
    "ivc": "IVC",
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
    "vp_chern": {"linestyle": "--", "linewidth": 2.25, "zorder": 9},
    "vp_ivc": {"linestyle": "-", "linewidth": 2.1, "zorder": 10},
}

CG_BOUNDARY_KEYS = ("vp_chern", "vp_ivc")
PHASE_CODES = {"vp_c0": 0, "vp_c1": 1, "ivc": 2}

PHASE_LABELS = (
    {"text": "VP\n$C=0$", "theta_deg": 2.50, "u_D_meV": 10.0},
    {"text": "VP\n$C=1$", "theta_deg": 3.50, "u_D_meV": 2.5},
    {"text": "IVC", "theta_deg": 3.62, "u_D_meV": 18.2},
)

PHASE_D_LABELS = (
    {"text": "VP\n$C=0$", "theta_deg": 2.50, "u_D_meV": 10.0},
    {"text": "VP\n$C=1$", "theta_deg": 3.50, "u_D_meV": 2.5},
    {"text": "IVC", "theta_deg": 3.72, "u_D_meV": 18.7},
)


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


def _cell_edge_class_boundary_segments(
    classes: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> list[list[tuple[float, float]]]:
    values = np.asarray(classes, dtype=float)
    finite = np.isfinite(values)
    segments: list[list[tuple[float, float]]] = []

    for row in range(values.shape[0]):
        for col in range(values.shape[1] - 1):
            if not (finite[row, col] and finite[row, col + 1]):
                continue
            if values[row, col] == values[row, col + 1]:
                continue
            x = float(x_edges[col + 1])
            segments.append([(x, float(y_edges[row])), (x, float(y_edges[row + 1]))])

    for row in range(values.shape[0] - 1):
        for col in range(values.shape[1]):
            if not (finite[row, col] and finite[row + 1, col]):
                continue
            if values[row, col] == values[row + 1, col]:
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


def _phase_codes(grey: np.ndarray, vp_topological: np.ndarray) -> np.ndarray:
    codes = np.full(np.asarray(grey).shape, PHASE_CODES["vp_c0"], dtype=int)
    codes[np.asarray(vp_topological, dtype=bool)] = PHASE_CODES["vp_c1"]
    codes[np.asarray(grey, dtype=bool)] = PHASE_CODES["ivc"]
    return codes


def _with_hf_ground_state(heat: pd.DataFrame, sweep: pd.DataFrame) -> pd.DataFrame:
    clean = sweep[sweep["clean_branch"].astype(bool)].copy()
    ivc_energy = (
        clean.groupby(["theta_index", "u_index"], as_index=False)["ivc_minus_vp_energy_per_cell"]
        .min()
        .rename(columns={"ivc_minus_vp_energy_per_cell": "lowest_ivc_minus_vp_energy_nk24"})
    )
    out = heat.merge(ivc_energy, on=["theta_index", "u_index"], how="left", validate="one_to_one")
    if out["lowest_ivc_minus_vp_energy_nk24"].isna().any():
        missing = out.loc[
            out["lowest_ivc_minus_vp_energy_nk24"].isna(), ["theta_index", "u_index"]
        ]
        raise ValueError(f"Missing clean nk=24 HF energies for {len(missing)} phase-diagram points")
    out["hf_ivc_ground_nk24"] = out["lowest_ivc_minus_vp_energy_nk24"] < 0.0
    return out


def _box_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["axis"])
        spine.set_linewidth(AXES["spine_linewidth"])


def _save(fig: plt.Figure, stem: str) -> list[Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in (".png", ".pdf"):
        path = FIGURE_DIR / f"{stem}{suffix}"
        fig.savefig(path, dpi=FIGURE["dpi"], bbox_inches="tight", pad_inches=0.08)
        paths.append(path)
    plt.close(fig)
    return paths


def _plot_phase_diagram(
    theta_vals: np.ndarray,
    u_vals: np.ndarray,
    hf_ivc: np.ndarray,
    vp_topo: np.ndarray,
    *,
    phase_labels: tuple[dict[str, object], ...] = PHASE_LABELS,
    ylabel: str = LABELS["y"],
    output_stem: str = PHASE_OUTPUT_STEM,
) -> list[Path]:
    theta_edges = _edges(theta_vals)
    u_edges = _edges(u_vals)

    phase_codes = _phase_codes(hf_ivc, vp_topo)
    phase_cmap = ListedColormap([COLORS["vp_c0"], COLORS["vp_c1"], COLORS["ivc"]])

    fig, ax = plt.subplots(figsize=FIGURE["size"], constrained_layout=False)
    fig.subplots_adjust(**FIGURE["phase_subplots_adjust"])
    ax.pcolormesh(
        theta_edges,
        u_edges,
        phase_codes,
        cmap=phase_cmap,
        vmin=-0.5,
        vmax=2.5,
        shading="auto",
    )
    segments = _cell_edge_class_boundary_segments(phase_codes, theta_edges, u_edges)
    if segments:
        boundary = LineCollection(
            segments,
            colors=[COLORS["phase_boundary"]],
            linewidths=2.1,
            linestyles="-",
            zorder=8,
        )
        boundary.set_capstyle("butt")
        boundary.set_joinstyle("miter")
        ax.add_collection(boundary)

    for phase_label in phase_labels:
        ax.text(
            phase_label["theta_deg"],
            phase_label["u_D_meV"],
            phase_label["text"],
            ha="center",
            va="center",
            color="black",
            fontsize=FONTS["phase_label"],
            linespacing=0.9,
            zorder=9,
        )
    _setup_axes(ax, theta_vals, u_vals, ylabel=ylabel)
    return _save(fig, output_stem)


def _plot_cg_diagram(
    heat: pd.DataFrame,
    theta_vals: np.ndarray,
    u_vals: np.ndarray,
    fit_grey: np.ndarray,
    hf_ivc: np.ndarray,
    vp_topo: np.ndarray,
) -> list[Path]:
    theta_edges = _edges(theta_vals)
    u_edges = _edges(u_vals)

    cG = _grid(heat, theta_vals, u_vals, CG_VALUE_COLUMN).astype(float)
    grey = fit_grey | hf_ivc
    vp_topo_visible = np.where(grey, np.nan, vp_topo)

    masked_cG = np.ma.masked_where(grey | ~np.isfinite(cG), cG)
    finite_cG = cG[np.isfinite(cG)]
    vmax = max(float(np.nanpercentile(np.abs(finite_cG), 99.0)), 0.05)

    fig, ax = plt.subplots(figsize=FIGURE["size"], constrained_layout=False)
    fig.subplots_adjust(**FIGURE["cg_subplots_adjust"])
    mesh = ax.pcolormesh(
        theta_edges,
        u_edges,
        masked_cG,
        cmap=_colormap(),
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
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

    _draw_boundary(ax, vp_topo_visible, theta_vals, u_vals, "vp_chern")
    _draw_boundary(ax, grey.astype(float), theta_vals, u_vals, "vp_ivc")

    for key in CG_BOUNDARY_KEYS:
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
        bbox_to_anchor=(0.03, 0.90),
        frameon=True,
        framealpha=0.78,
        facecolor="white",
        borderpad=0.35,
        handlelength=2.0,
        handletextpad=0.55,
        labelspacing=0.35,
    )

    _setup_axes(ax, theta_vals, u_vals)

    fig.canvas.draw()
    bbox = ax.get_position()
    cbar_width = bbox.width * AXES["colorbar_width_fraction"]
    cbar_pad = bbox.width * AXES["colorbar_pad_fraction"]
    cbar_ax = fig.add_axes([bbox.x1 + cbar_pad, bbox.y0, cbar_width, bbox.height])
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.set_label("")
    cbar.set_ticks([-0.10, -0.05, 0.00, 0.05, 0.10])
    cbar.set_ticklabels([f"{value:.2f}" for value in [-0.10, -0.05, 0.00, 0.05, 0.10]])
    cbar.ax.set_title(LABELS["cbar"], fontsize=FONTS["colorbar_label"], pad=12)
    cbar.ax.tick_params(labelsize=FONTS["colorbar_tick"])

    return _save(fig, CG_OUTPUT_STEM)


def _setup_axes(
    ax: plt.Axes,
    theta_vals: np.ndarray,
    u_vals: np.ndarray,
    *,
    ylabel: str = LABELS["y"],
) -> None:
    ax.set_xlim(theta_vals.min(), theta_vals.max())
    ax.set_ylim(u_vals.min(), u_vals.max())
    ax.set_xticks(AXES["xticks"])
    ax.set_yticks(AXES["yticks"])
    ax.set_xlabel(LABELS["x"])
    ax.set_ylabel(ylabel)
    ax.set_box_aspect(AXES["box_aspect"])
    _box_axes(ax)


def plot() -> list[Path]:
    missing = [path for path in (INPUT_CSV, HF_SWEEP_CSV) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing linear-interaction plot input(s): {missing}")

    heat = pd.read_csv(INPUT_CSV)
    sweep = pd.read_csv(
        HF_SWEEP_CSV,
        usecols=["theta_index", "u_index", "clean_branch", "ivc_minus_vp_energy_per_cell"],
    )
    heat = _with_hf_ground_state(heat, sweep)
    theta_vals = np.array(sorted(heat["theta_deg"].unique()), dtype=float)
    u_vals = np.array(sorted(heat["u_D_meV"].unique()), dtype=float)
    fit_grey = _grid(heat, theta_vals, u_vals, CG_GREY_MASK_COLUMN).astype(bool)
    hf_ivc = _grid(heat, theta_vals, u_vals, "hf_ivc_ground_nk24").astype(bool)
    vp_topo = _grid(heat, theta_vals, u_vals, "vp_topological_region_nk24").astype(float)

    return [
        *_plot_phase_diagram(theta_vals, u_vals, hf_ivc, vp_topo),
        *_plot_phase_diagram(
            theta_vals,
            u_vals,
            hf_ivc,
            vp_topo,
            phase_labels=PHASE_D_LABELS,
            ylabel=LABELS["y_D"],
            output_stem=PHASE_D_OUTPUT_STEM,
        ),
        *_plot_cg_diagram(heat, theta_vals, u_vals, fit_grey, hf_ivc, vp_topo),
    ]


def main() -> None:
    _apply_style()
    for path in plot():
        print(path)


if __name__ == "__main__":
    main()
