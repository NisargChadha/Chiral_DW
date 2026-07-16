#!/usr/bin/env python3
"""Plot the nk=18 theta=3-degree Taige scanning-SET spectrum."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "results" / "taige_set_nk18_theta3_u0_10_full_filling"
REFINED_SUMMARY = (
    REPO_ROOT
    / "results"
    / "taige_set_nk18_theta3_u5p75_converged_smoke"
    / "points"
    / "uD_005p7500"
    / "point_summary.json"
)
FIGURE_DIR = REPO_ROOT / "Plots" / "figures"
OUTPUT_STEM = "taige_set_nk18_theta3_scanning_set_spectrum"

FIGURE = {
    "size": (13.4, 6.8),
    "dpi": 280,
    "subplots_adjust": {
        "left": 0.075,
        "right": 0.975,
        "bottom": 0.23,
        "top": 0.91,
        "wspace": 0.38,
    },
}

FONTS = {
    "base": 12,
    "title": 18,
    "axis_label": 22,
    "tick_label": 17,
    "legend": 13,
    "annotation": 12,
    "colorbar_label": 19,
    "colorbar_tick": 15,
}

COLORS = {
    "red": "#FD4C55",
    "teal": "#378d94",
    "purple": "#6a408d",
    "vp_chern": "#7c3aed",
    "grey": "0.25",
    "grey_span": "0.86",
    "axis": "0.18",
    "white_center": "#f7f7f7",
}

AXES = {
    "spine_linewidth": 1.15,
    "heatmap_vmax": 800.0,
    "transition_min": 5.75,
    "transition_max": 6.0,
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


def _edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.asarray([values[0] - 0.5, values[0] + 0.5])
    return np.r_[
        values[0] - 0.5 * (values[1] - values[0]),
        0.5 * (values[:-1] + values[1:]),
        values[-1] + 0.5 * (values[-1] - values[-2]),
    ]


def _refined_row() -> dict[str, float]:
    payload = json.loads(REFINED_SUMMARY.read_text())
    row = payload["row"]
    return {
        "u_D_meV": float(row["u_D_meV"]),
        "hf_band_chern": float(row["hf_band_chern"]),
        "fixed_indirect_gap_meV": float(row["fixed_indirect_gap_meV"]),
        "charge_gap_intrinsic_meV": float(row["charge_gap_intrinsic_meV"]),
        "charge_gap_raw_meV": float(row["charge_gap_raw_meV"]),
        "source": "refined_three-filling smoke",
    }


def plot() -> list[Path]:
    summary_path = RESULT_ROOT / "set_sweep_summary.csv"
    kappa_path = RESULT_ROOT / "set_inverse_compressibility.csv"
    if not summary_path.exists() or not kappa_path.exists():
        raise FileNotFoundError("Missing merged Taige SET sweep tables")
    if not REFINED_SUMMARY.exists():
        raise FileNotFoundError(f"Missing refined transition summary: {REFINED_SUMMARY}")

    summary = pd.read_csv(summary_path).sort_values("u_D_meV")
    kappa = pd.read_csv(kappa_path)
    if not bool(summary["all_global_fillings_converged"].all()):
        raise ValueError("SET plot requires every global filling to be converged")
    if not bool(summary["fixed_per_k_valid_insulator"].all()):
        raise ValueError("SET plot contains an invalid fixed-per-k insulating point")

    fillings = np.asarray(sorted(kappa["filling_holes"].unique()), dtype=float)
    displacements = np.asarray(sorted(kappa["u_D_meV"].unique()), dtype=float)
    intrinsic = (
        kappa.pivot(
            index="u_D_meV",
            columns="filling_holes",
            values="dmu_dnu_intrinsic_mev",
        )
        .reindex(index=displacements, columns=fillings)
        .to_numpy(dtype=float)
    )

    refined = _refined_row()
    gap_rows = pd.concat(
        [
            summary[
                [
                    "u_D_meV",
                    "hf_band_chern",
                    "fixed_indirect_gap_meV",
                    "charge_gap_intrinsic_meV",
                    "charge_gap_raw_meV",
                ]
            ].assign(source="coarse 25-filling sweep"),
            pd.DataFrame([refined]),
        ],
        ignore_index=True,
    ).sort_values("u_D_meV")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data_path = FIGURE_DIR / f"{OUTPUT_STEM}_summary.csv"
    gap_rows.to_csv(data_path, index=False)

    _apply_style()
    fig, (ax_heat, ax_gap) = plt.subplots(1, 2, figsize=FIGURE["size"])
    fig.subplots_adjust(**FIGURE["subplots_adjust"])

    cmap = LinearSegmentedColormap.from_list(
        "bootstrap_teal_red",
        [COLORS["teal"], COLORS["white_center"], COLORS["red"]],
        N=256,
    )
    vmax = float(AXES["heatmap_vmax"])
    mesh = ax_heat.pcolormesh(
        _edges(fillings),
        _edges(displacements),
        intrinsic,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        shading="auto",
        rasterized=True,
    )
    transition_mid = 0.5 * (AXES["transition_min"] + AXES["transition_max"])
    ax_heat.axhspan(
        AXES["transition_min"],
        AXES["transition_max"],
        color=COLORS["grey_span"],
        alpha=0.55,
        zorder=3,
    )
    ax_heat.axhline(
        transition_mid,
        color=COLORS["vp_chern"],
        linestyle="--",
        linewidth=2.1,
        zorder=5,
    )
    ax_heat.axvline(1.0, color=COLORS["grey"], linestyle=":", linewidth=1.25)
    ax_heat.text(
        0.968,
        2.3,
        r"$C=1$",
        color=COLORS["vp_chern"],
        fontsize=15,
        ha="left",
        va="center",
    )
    ax_heat.text(
        0.968,
        8.2,
        r"$C=0$",
        color=COLORS["grey"],
        fontsize=15,
        ha="left",
        va="center",
    )
    ax_heat.set_xlabel(r"hole filling $\nu_h$")
    ax_heat.set_ylabel(r"$u_D$ (meV)")
    ax_heat.set_title(r"(a) intrinsic inverse compressibility", loc="left")
    ax_heat.set_box_aspect(1.0)
    _box_axes(ax_heat)
    cbar = fig.colorbar(mesh, ax=ax_heat, fraction=0.047, pad=0.045, extend="both")
    cbar.ax.tick_params(labelsize=FONTS["colorbar_tick"])
    cbar.ax.text(
        0.5,
        1.02,
        "$\\partial\\mu_h/\\partial\\nu_h$\n(meV)",
        transform=cbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=15,
        color=COLORS["axis"],
    )

    ax_gap.axvspan(
        AXES["transition_min"],
        AXES["transition_max"],
        color=COLORS["grey_span"],
        alpha=0.65,
        zorder=0,
    )
    ax_gap.axvline(
        transition_mid,
        color=COLORS["vp_chern"],
        linestyle="--",
        linewidth=2.1,
        zorder=1,
        label=r"VP Chern transition bracket",
    )
    ax_gap.axhline(0.0, color=COLORS["grey"], linewidth=1.0, linestyle=":")
    ax_gap.plot(
        gap_rows["u_D_meV"],
        gap_rows["fixed_indirect_gap_meV"],
        color=COLORS["purple"],
        marker="o",
        linewidth=2.0,
        markersize=5.5,
        label=r"HF indirect gap",
    )
    ax_gap.plot(
        gap_rows["u_D_meV"],
        gap_rows["charge_gap_intrinsic_meV"],
        color=COLORS["red"],
        marker="s",
        linewidth=2.0,
        markersize=5.2,
        label=r"intrinsic SET gap",
    )
    ax_gap.plot(
        gap_rows["u_D_meV"],
        gap_rows["charge_gap_raw_meV"],
        color=COLORS["teal"],
        marker="^",
        linewidth=1.9,
        markersize=5.6,
        label=r"with uniform capacitance",
    )
    ax_gap.scatter(
        [refined["u_D_meV"]] * 3,
        [
            refined["fixed_indirect_gap_meV"],
            refined["charge_gap_intrinsic_meV"],
            refined["charge_gap_raw_meV"],
        ],
        facecolors="white",
        edgecolors=[COLORS["purple"], COLORS["red"], COLORS["teal"]],
        linewidths=1.7,
        s=62,
        zorder=6,
    )
    ax_gap.annotate(
        r"refined $u_D=5.75$ meV",
        xy=(5.75, refined["charge_gap_raw_meV"]),
        xytext=(3.8, 3.3),
        arrowprops={"arrowstyle": "->", "color": COLORS["grey"], "lw": 1.1},
        fontsize=FONTS["annotation"],
        color=COLORS["grey"],
    )
    ax_gap.text(1.4, 21.2, r"$C=1$", color=COLORS["vp_chern"], fontsize=15)
    ax_gap.text(7.7, 21.2, r"$C=0$", color=COLORS["grey"], fontsize=15)
    ax_gap.set_xlim(0.0, 10.0)
    ax_gap.set_ylim(-2.2, 23.2)
    ax_gap.set_xticks(np.arange(0, 11, 2))
    ax_gap.set_xlabel(r"$u_D$ (meV)")
    ax_gap.set_ylabel("gap (meV)")
    ax_gap.set_title(r"(b) topology and charge gaps", loc="left")
    ax_gap.set_box_aspect(1.0)
    _box_axes(ax_gap)
    ax_gap.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
    )

    png = FIGURE_DIR / f"{OUTPUT_STEM}.png"
    pdf = FIGURE_DIR / f"{OUTPUT_STEM}.pdf"
    fig.savefig(png, dpi=FIGURE["dpi"], bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return [png, pdf, data_path]


if __name__ == "__main__":
    for output in plot():
        print(output)
