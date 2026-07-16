#!/usr/bin/env python3
"""Plot the intrinsic scanning-SET charge gap versus displacement field."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = (
    REPO_ROOT
    / "results"
    / "taige_set_nk18_theta3_u0_10_full_filling"
    / "set_sweep_summary.csv"
)
REFINED_SUMMARY_PATH = (
    REPO_ROOT
    / "results"
    / "taige_set_nk18_theta3_u5p75_converged_smoke"
    / "points"
    / "uD_005p7500"
    / "point_summary.json"
)
FIGURE_DIR = REPO_ROOT / "Plots" / "figures"
OUTPUT_STEM = "taige_set_nk18_theta3_set_gap_vs_displacement"

FIGURE = {"size": (7.4, 7.4), "dpi": 280}
FONTS = {
    "base": 12,
    "title": 20,
    "axis_label": 24,
    "tick_label": 20,
    "annotation": 13,
    "phase_label": 18,
}
COLORS = {
    "red": "#FD4C55",
    "vp_chern": "#7c3aed",
    "grey": "0.25",
    "grey_span": "0.86",
    "axis": "0.18",
}
AXES = {
    "transition_min_meV": 5.75,
    "transition_max_meV": 6.0,
    "spine_linewidth": 1.15,
}


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


def _load_gap_rows() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing merged sweep summary: {SUMMARY_PATH}")
    if not REFINED_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing refined transition point: {REFINED_SUMMARY_PATH}")

    summary = pd.read_csv(SUMMARY_PATH).sort_values("u_D_meV")
    if not bool(summary["all_global_fillings_converged"].all()):
        raise ValueError("SET gap curve requires every global filling to converge")

    refined_payload = json.loads(REFINED_SUMMARY_PATH.read_text())
    refined = refined_payload["row"]
    rows = summary[["u_D_meV", "charge_gap_intrinsic_meV"]].assign(
        sampling="coarse"
    )
    rows = pd.concat(
        [
            rows,
            pd.DataFrame(
                [
                    {
                        "u_D_meV": float(refined["u_D_meV"]),
                        "charge_gap_intrinsic_meV": float(
                            refined["charge_gap_intrinsic_meV"]
                        ),
                        "sampling": "refined",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return rows.sort_values("u_D_meV").reset_index(drop=True)


def plot() -> list[Path]:
    rows = _load_gap_rows()
    coarse = rows[rows["sampling"] == "coarse"]
    refined = rows[rows["sampling"] == "refined"].iloc[0]

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FIGURE_DIR / f"{OUTPUT_STEM}.csv"
    rows.to_csv(csv_path, index=False)

    _apply_style()
    fig, ax = plt.subplots(figsize=FIGURE["size"])
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.16, top=0.90)

    transition_mid = 0.5 * (
        AXES["transition_min_meV"] + AXES["transition_max_meV"]
    )
    ax.axvspan(
        AXES["transition_min_meV"],
        AXES["transition_max_meV"],
        color=COLORS["grey_span"],
        alpha=0.72,
        zorder=0,
    )
    ax.axvline(
        transition_mid,
        color=COLORS["vp_chern"],
        linestyle="--",
        linewidth=2.1,
        zorder=1,
    )
    ax.axhline(0.0, color=COLORS["grey"], linestyle=":", linewidth=1.2, zorder=1)

    ax.plot(
        rows["u_D_meV"],
        rows["charge_gap_intrinsic_meV"],
        color=COLORS["red"],
        linewidth=2.6,
        zorder=3,
    )
    ax.scatter(
        coarse["u_D_meV"],
        coarse["charge_gap_intrinsic_meV"],
        color=COLORS["red"],
        marker="o",
        s=58,
        zorder=4,
    )
    ax.scatter(
        refined["u_D_meV"],
        refined["charge_gap_intrinsic_meV"],
        facecolor="white",
        edgecolor=COLORS["red"],
        linewidth=2.0,
        marker="o",
        s=88,
        zorder=5,
    )
    ax.annotate(
        r"refined point",
        xy=(refined["u_D_meV"], refined["charge_gap_intrinsic_meV"]),
        xytext=(4.0, 2.4),
        arrowprops={"arrowstyle": "->", "color": COLORS["grey"], "lw": 1.2},
        fontsize=FONTS["annotation"],
        color=COLORS["grey"],
        ha="center",
    )
    ax.text(
        2.1,
        21.4,
        r"$C=1$",
        color=COLORS["vp_chern"],
        fontsize=FONTS["phase_label"],
        ha="center",
    )
    ax.text(
        8.0,
        21.4,
        r"$C=0$",
        color=COLORS["grey"],
        fontsize=FONTS["phase_label"],
        ha="center",
    )

    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(-2.5, 23.0)
    ax.set_xticks(np.arange(0, 11, 2))
    ax.set_xlabel(r"displacement field $u_D$ (meV)")
    ax.set_ylabel(r"intrinsic SET gap $\Delta_{\rm SET}$ (meV)")
    ax.set_title(r"Scanning-SET charge gap at $\nu_h=1$")
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
