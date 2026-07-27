#!/usr/bin/env python3
"""Plot where the converged tWSe2 HF ground state has a negative indirect gap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pydantic import BaseModel, ConfigDict


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    REPO_ROOT
    / "results"
    / "wse2_ivc_hysteresis_finite_size_nk18_24_grid41_linear_interaction_recomputed_sharded"
)
INPUT_CSV = RESULT_ROOT / "nk_024" / "hysteresis_sweep.csv"
FIGURE_DIR = REPO_ROOT / "Plots" / "figures"
OUTPUT_STEM = "wse2_nk24_negative_indirect_gap_phase_diagram"

FIGURE = {
    "size": (7.4, 7.4),
    "dpi": 280,
    "subplots_adjust": {"left": 0.15, "right": 0.98, "bottom": 0.27, "top": 0.87},
}

FONTS = {
    "base": 12,
    "title": 20,
    "axis_label": 24,
    "tick_label": 20,
    "legend": 13,
}

COLORS = {
    "vp_positive": "#f7d9db",
    "ivc_positive": "#d7e9e9",
    "vp_negative": "#FD4C55",
    "ivc_negative": "#378d94",
    "vp_ivc_boundary": "0.18",
    "zero_gap_boundary": "#6a408d",
    "axis": "0.18",
}

LABELS = {
    "title": r"tWSe$_2$ HF: where is the indirect gap negative?",
    "subtitle": r"$n_k=24$; clean ground-state sector at each grid point",
    "x": r"$\theta_M$ ($^\circ$)",
    "y": r"$u_D$ (meV)",
}

AXES = {
    "xticks": (2.0, 2.5, 3.0, 3.5, 4.0),
    "yticks": (0, 5, 10, 15, 20),
    "box_aspect": 1.0,
    "spine_linewidth": 1.15,
}

PHASE_CODES = {
    "vp_nonnegative": 0,
    "ivc_nonnegative": 1,
    "vp_negative": 2,
    "ivc_negative": 3,
}


class GapPhasePlotConfig(BaseModel):
    """Recorded conventions for the phase/gap classification."""

    model_config = ConfigDict(frozen=True)

    material: str = "WSe2_Taige"
    n_k: int = 24
    negative_gap_threshold_meV: float = 0.0
    ivc_selection: str = "lowest_energy_clean_branch"
    ground_state_rule: str = "IVC if E_IVC < E_VP; ties prefer VP"
    vp_gap_rule: str = "indirect gap of the lower-energy VP+ or VP- reference"
    input_csv: str = str(INPUT_CSV.relative_to(REPO_ROOT))


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
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError("At least two grid coordinates are required.")
    return np.r_[
        values[0] - 0.5 * (values[1] - values[0]),
        0.5 * (values[:-1] + values[1:]),
        values[-1] + 0.5 * (values[-1] - values[-2]),
    ]


def _grid(
    frame: pd.DataFrame,
    theta_values: np.ndarray,
    displacement_values: np.ndarray,
    column: str,
    *,
    dtype: type = float,
) -> np.ndarray:
    return (
        frame.pivot(index="u_D_meV", columns="theta_deg", values=column)
        .reindex(index=displacement_values, columns=theta_values)
        .to_numpy(dtype=dtype)
    )


def _cell_edge_boundary_segments(
    values: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> list[list[tuple[float, float]]]:
    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    segments: list[list[tuple[float, float]]] = []

    for row in range(array.shape[0]):
        for col in range(array.shape[1] - 1):
            if not (finite[row, col] and finite[row, col + 1]):
                continue
            if array[row, col] == array[row, col + 1]:
                continue
            x_value = float(x_edges[col + 1])
            segments.append(
                [(x_value, float(y_edges[row])), (x_value, float(y_edges[row + 1]))]
            )

    for row in range(array.shape[0] - 1):
        for col in range(array.shape[1]):
            if not (finite[row, col] and finite[row + 1, col]):
                continue
            if array[row, col] == array[row + 1, col]:
                continue
            y_value = float(y_edges[row + 1])
            segments.append(
                [(float(x_edges[col]), y_value), (float(x_edges[col + 1]), y_value)]
            )
    return segments


def _draw_boundary(
    ax: plt.Axes,
    values: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    color: str,
    linewidth: float,
    linestyle: str,
    zorder: int,
) -> None:
    segments = _cell_edge_boundary_segments(values, x_edges, y_edges)
    if not segments:
        return
    collection = LineCollection(
        segments,
        colors=[color],
        linewidths=linewidth,
        linestyles=linestyle,
        zorder=zorder,
    )
    collection.set_capstyle("butt")
    collection.set_joinstyle("miter")
    ax.add_collection(collection)


def load_selected_ground_states(config: GapPhasePlotConfig) -> pd.DataFrame:
    """Select the clean IVC minimum and compare it with the clean VP reference."""

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing WSe2 HF sweep: {INPUT_CSV}")

    required = [
        "theta_deg",
        "u_D_meV",
        "branch_id",
        "gap_family_label",
        "run_id",
        "clean_branch",
        "converged",
        "energy_total_per_cell",
        "indirect_gap",
        "vp_reference_name",
        "vp_reference_energy_per_cell",
        "vp_plus_clean",
        "vp_minus_clean",
        "vp_plus_indirect_gap",
        "vp_minus_indirect_gap",
    ]
    sweep = pd.read_csv(INPUT_CSV, usecols=required, low_memory=False)
    for column in ("clean_branch", "converged", "vp_plus_clean", "vp_minus_clean"):
        sweep[column] = _as_bool(sweep[column])

    clean = sweep[
        sweep["clean_branch"]
        & sweep["converged"]
        & sweep["vp_plus_clean"]
        & sweep["vp_minus_clean"]
    ].copy()
    selected_indices = clean.groupby(["theta_deg", "u_D_meV"], sort=True)[
        "energy_total_per_cell"
    ].idxmin()
    selected = clean.loc[selected_indices].copy()

    expected_points = sweep[["theta_deg", "u_D_meV"]].drop_duplicates().shape[0]
    if len(selected) != expected_points:
        raise ValueError(
            f"Only {len(selected)} of {expected_points} WSe2 grid points have a clean "
            "converged IVC branch and clean VP references."
        )

    selected["delta_E_ivc_minus_vp_meV_per_cell"] = (
        selected["energy_total_per_cell"] - selected["vp_reference_energy_per_cell"]
    )
    selected["selected_phase"] = np.where(
        selected["delta_E_ivc_minus_vp_meV_per_cell"] < 0.0,
        "IVC",
        "VP",
    )
    selected["vp_reference_indirect_gap_meV"] = np.where(
        selected["vp_reference_name"].eq("VP+"),
        selected["vp_plus_indirect_gap"],
        selected["vp_minus_indirect_gap"],
    )
    selected["selected_indirect_gap_meV"] = np.where(
        selected["selected_phase"].eq("IVC"),
        selected["indirect_gap"],
        selected["vp_reference_indirect_gap_meV"],
    )
    selected["negative_indirect_gap"] = (
        selected["selected_indirect_gap_meV"] < config.negative_gap_threshold_meV
    )
    selected["phase_code"] = PHASE_CODES["vp_nonnegative"]
    selected.loc[
        selected["selected_phase"].eq("IVC") & ~selected["negative_indirect_gap"],
        "phase_code",
    ] = PHASE_CODES["ivc_nonnegative"]
    selected.loc[
        selected["selected_phase"].eq("VP") & selected["negative_indirect_gap"],
        "phase_code",
    ] = PHASE_CODES["vp_negative"]
    selected.loc[
        selected["selected_phase"].eq("IVC") & selected["negative_indirect_gap"],
        "phase_code",
    ] = PHASE_CODES["ivc_negative"]
    return selected.sort_values(["u_D_meV", "theta_deg"]).reset_index(drop=True)


def _write_tables(selected: pd.DataFrame, config: GapPhasePlotConfig) -> tuple[Path, Path]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data_path = FIGURE_DIR / f"{OUTPUT_STEM}.csv"
    metadata_path = FIGURE_DIR / f"{OUTPUT_STEM}_metadata.csv"

    output_columns = [
        "theta_deg",
        "u_D_meV",
        "selected_phase",
        "selected_indirect_gap_meV",
        "negative_indirect_gap",
        "delta_E_ivc_minus_vp_meV_per_cell",
        "energy_total_per_cell",
        "vp_reference_energy_per_cell",
        "branch_id",
        "gap_family_label",
        "run_id",
        "indirect_gap",
        "vp_reference_name",
        "vp_reference_indirect_gap_meV",
        "vp_plus_indirect_gap",
        "vp_minus_indirect_gap",
    ]
    selected[output_columns].to_csv(data_path, index=False)

    metadata = config.model_dump()
    metadata.update(
        {
            "n_grid_points": len(selected),
            "n_negative_total": int(selected["negative_indirect_gap"].sum()),
            "n_negative_vp": int(
                (
                    selected["negative_indirect_gap"]
                    & selected["selected_phase"].eq("VP")
                ).sum()
            ),
            "n_negative_ivc": int(
                (
                    selected["negative_indirect_gap"]
                    & selected["selected_phase"].eq("IVC")
                ).sum()
            ),
            "minimum_selected_indirect_gap_meV": float(
                selected["selected_indirect_gap_meV"].min()
            ),
            "maximum_selected_indirect_gap_meV": float(
                selected["selected_indirect_gap_meV"].max()
            ),
        }
    )
    pd.DataFrame([metadata]).to_csv(metadata_path, index=False)
    return data_path, metadata_path


def plot_phase_diagram(selected: pd.DataFrame) -> tuple[Path, Path]:
    theta_values = np.sort(selected["theta_deg"].unique())
    displacement_values = np.sort(selected["u_D_meV"].unique())
    theta_edges = _edges(theta_values)
    displacement_edges = _edges(displacement_values)
    phase_codes = _grid(selected, theta_values, displacement_values, "phase_code")
    phase_mask = _grid(
        selected.assign(ivc_phase=selected["selected_phase"].eq("IVC").astype(float)),
        theta_values,
        displacement_values,
        "ivc_phase",
    )
    negative_mask = _grid(
        selected.assign(negative=selected["negative_indirect_gap"].astype(float)),
        theta_values,
        displacement_values,
        "negative",
    )

    cmap = ListedColormap(
        [
            COLORS["vp_positive"],
            COLORS["ivc_positive"],
            COLORS["vp_negative"],
            COLORS["ivc_negative"],
        ]
    )
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1.0), cmap.N)

    fig, ax = plt.subplots(figsize=FIGURE["size"], constrained_layout=False)
    fig.subplots_adjust(**FIGURE["subplots_adjust"])
    ax.pcolormesh(
        theta_edges,
        displacement_edges,
        phase_codes,
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=True,
    )
    _draw_boundary(
        ax,
        negative_mask,
        theta_edges,
        displacement_edges,
        color=COLORS["zero_gap_boundary"],
        linewidth=2.2,
        linestyle="--",
        zorder=7,
    )
    _draw_boundary(
        ax,
        phase_mask,
        theta_edges,
        displacement_edges,
        color=COLORS["vp_ivc_boundary"],
        linewidth=2.2,
        linestyle="-",
        zorder=8,
    )

    ax.set_xlim(theta_values.min(), theta_values.max())
    ax.set_ylim(displacement_values.min(), displacement_values.max())
    ax.set_xticks(AXES["xticks"])
    ax.set_yticks(AXES["yticks"])
    ax.set_xlabel(LABELS["x"])
    ax.set_ylabel(LABELS["y"])
    ax.set_box_aspect(AXES["box_aspect"])
    ax.set_title(f"{LABELS['title']}\n{LABELS['subtitle']}", pad=13)

    handles = [
        Patch(facecolor=COLORS["vp_positive"], edgecolor="none", label=r"VP, $\Delta_{\rm ind}\geq0$"),
        Patch(facecolor=COLORS["ivc_positive"], edgecolor="none", label=r"IVC, $\Delta_{\rm ind}\geq0$"),
        Patch(facecolor=COLORS["vp_negative"], edgecolor="none", label=r"VP, $\Delta_{\rm ind}<0$"),
        Patch(facecolor=COLORS["ivc_negative"], edgecolor="none", label=r"IVC, $\Delta_{\rm ind}<0$"),
        Line2D(
            [0],
            [0],
            color=COLORS["vp_ivc_boundary"],
            lw=2.2,
            label="IVC–VP boundary",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["zero_gap_boundary"],
            lw=2.2,
            ls="--",
            label=r"$\Delta_{\rm ind}=0$ boundary",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
        frameon=False,
        columnspacing=1.7,
        handlelength=2.4,
        handletextpad=0.65,
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["axis"])
        spine.set_linewidth(AXES["spine_linewidth"])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{OUTPUT_STEM}.png"
    pdf_path = FIGURE_DIR / f"{OUTPUT_STEM}.pdf"
    fig.savefig(
        png_path,
        dpi=FIGURE["dpi"],
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    _apply_style()
    config = GapPhasePlotConfig()
    selected = load_selected_ground_states(config)
    png_path, pdf_path = plot_phase_diagram(selected)
    data_path, metadata_path = _write_tables(selected, config)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {data_path}")
    print(f"Wrote {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
