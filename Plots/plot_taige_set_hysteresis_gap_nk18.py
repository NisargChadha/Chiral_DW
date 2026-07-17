#!/usr/bin/env python3
"""Plot the branch-selected capacitance-added SET gap for the nk=18 sweep."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = (
    REPO_ROOT
    / "results"
    / "taige_set_nk18_theta3_u5_6_hysteresis20"
    / "set_hysteresis_selected.csv"
)
FIGURE_DIR = REPO_ROOT / "Plots" / "figures"
OUTPUT_STEM = "taige_set_nk18_theta3_hysteresis_selected_set_gap_u5_6"

FIGURE = {"size": (7.4, 7.4), "dpi": 280}
FONTS = {
    "base": 12,
    "title": 20,
    "axis_label": 24,
    "tick_label": 20,
    "phase_label": 18,
    "annotation": 13,
}
COLORS = {
    "red": "#FD4C55",
    "vp_chern": "#7c3aed",
    "grey": "0.25",
    "grey_span": "0.86",
    "axis": "0.18",
}
AXES = {"spine_linewidth": 1.15}


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
            "axes.edgecolor": COLORS["axis"],
            "axes.linewidth": AXES["spine_linewidth"],
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def _box_axes(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["axis"])
        spine.set_linewidth(AXES["spine_linewidth"])


def _load_rows() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing hysteresis merge: {SUMMARY_PATH}")
    rows = pd.read_csv(SUMMARY_PATH).sort_values("u_D_meV").reset_index(drop=True)
    if len(rows) != 20:
        raise ValueError(f"Expected 20 displacement-field points, found {len(rows)}")
    if not bool(rows["up_all_fillings_converged"].all()):
        raise ValueError("The upward continuation contains an unconverged filling")
    if not bool(rows["down_all_fillings_converged"].all()):
        raise ValueError("The downward continuation contains an unconverged filling")
    if not bool(rows["selected_fixed_per_k_valid_insulator"].all()):
        raise ValueError("A selected neutral state has a nonpositive indirect gap")
    return rows


def _transition_estimate(rows: pd.DataFrame) -> tuple[float, float, float]:
    """Bracket and linearly estimate the neutral-state energy crossing."""

    coexistence = rows[
        (rows["up_hf_band_chern"] - rows["down_hf_band_chern"]).abs() > 0.5
    ]
    delta = coexistence["down_minus_up_intrinsic_N0_meV"].to_numpy(float)
    field = coexistence["u_D_meV"].to_numpy(float)
    crossing_indices = np.flatnonzero(delta[:-1] * delta[1:] < 0.0)
    if crossing_indices.size != 1:
        raise ValueError(f"Expected one neutral branch crossing, found {crossing_indices.size}")
    left_index = int(crossing_indices[0])
    left, right = float(field[left_index]), float(field[left_index + 1])
    delta_left, delta_right = float(delta[left_index]), float(delta[left_index + 1])
    estimate = left - delta_left * (right - left) / (delta_right - delta_left)
    return left, right, estimate


def plot() -> list[Path]:
    rows = _load_rows()
    transition_left, transition_right, transition = _transition_estimate(rows)
    minimum = rows.loc[rows["charge_gap_raw_meV"].idxmin()]

    plot_rows = rows.copy()
    plot_rows["transition_bracket_left_meV"] = transition_left
    plot_rows["transition_bracket_right_meV"] = transition_right
    plot_rows["transition_linear_estimate_meV"] = transition

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FIGURE_DIR / f"{OUTPUT_STEM}.csv"
    plot_rows.to_csv(csv_path, index=False)

    _apply_style()
    fig, ax = plt.subplots(figsize=FIGURE["size"])
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.16, top=0.90)

    ax.axvspan(
        transition_left,
        transition_right,
        color=COLORS["grey_span"],
        alpha=0.72,
        zorder=0,
    )
    ax.axvline(
        transition,
        color=COLORS["vp_chern"],
        linestyle="--",
        linewidth=2.1,
        zorder=1,
    )
    ax.plot(
        rows["u_D_meV"],
        rows["charge_gap_raw_meV"],
        color=COLORS["red"],
        linewidth=2.7,
        marker="o",
        markersize=6.5,
        markerfacecolor="white",
        markeredgewidth=1.8,
        zorder=3,
    )

    ax.annotate(
        rf"minimum ${minimum['charge_gap_raw_meV']:.2f}$ meV",
        xy=(minimum["u_D_meV"], minimum["charge_gap_raw_meV"]),
        xytext=(5.15, 6.0),
        arrowprops={"arrowstyle": "->", "color": COLORS["grey"], "lw": 1.2},
        fontsize=FONTS["annotation"],
        color=COLORS["grey"],
        ha="left",
    )
    ax.text(
        5.20,
        13.0,
        r"$C=1$",
        color=COLORS["vp_chern"],
        fontsize=FONTS["phase_label"],
        ha="center",
    )
    ax.text(
        5.86,
        13.0,
        r"$C=0$",
        color=COLORS["grey"],
        fontsize=FONTS["phase_label"],
        ha="center",
    )

    ax.set_xlim(5.0, 6.0)
    ax.set_ylim(3.5, 14.5)
    ax.set_xticks(np.arange(5.0, 6.01, 0.2))
    ax.set_yticks(np.arange(4.0, 14.1, 2.0))
    ax.set_xlabel(r"displacement field $u_D$ (meV)")
    ax.set_ylabel(r"SET gap $\Delta_{\rm SET}$ (meV)")
    ax.set_title(r"Branch-selected scanning-SET gap at $\nu_h=1$")
    ax.set_box_aspect(1.0)
    _box_axes(ax)

    png_path = FIGURE_DIR / f"{OUTPUT_STEM}.png"
    pdf_path = FIGURE_DIR / f"{OUTPUT_STEM}.pdf"
    fig.savefig(png_path, dpi=FIGURE["dpi"], bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return [png_path, pdf_path, csv_path]


if __name__ == "__main__":
    for output in plot():
        print(output)
