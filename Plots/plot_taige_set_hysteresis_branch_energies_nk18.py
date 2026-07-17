#!/usr/bin/env python3
"""Plot the coexisting neutral C=1 and C=0 Hartree-Fock energies."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BRANCH_ENERGIES_PATH = (
    REPO_ROOT
    / "results"
    / "taige_set_nk18_theta3_u5_6_hysteresis20"
    / "set_hysteresis_branch_filling_energies.csv"
)
FIGURE_DIR = REPO_ROOT / "Plots" / "figures"
OUTPUT_STEM = "taige_set_nk18_theta3_coexisting_hf_branch_energies_u5_6"

FIGURE = {"size": (7.4, 7.4), "dpi": 280}
FONTS = {
    "base": 12,
    "title": 20,
    "axis_label": 24,
    "tick_label": 20,
    "legend": 16,
    "annotation": 13,
}
COLORS = {
    "red": "#FD4C55",
    "teal": "#378d94",
    "purple": "#6a408d",
    "grey": "0.25",
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


def _load_coexisting_neutral_branches() -> pd.DataFrame:
    if not BRANCH_ENERGIES_PATH.exists():
        raise FileNotFoundError(f"Missing branch energies: {BRANCH_ENERGIES_PATH}")

    rows = pd.read_csv(BRANCH_ENERGIES_PATH)
    neutral = rows[rows["n_particles"] == 324].copy()
    if not bool(neutral["converged"].all()):
        raise ValueError("The neutral branch table contains an unconverged state")

    records: list[dict[str, float]] = []
    for field, pair in neutral.groupby("u_D_meV", sort=True):
        if len(pair) != 2:
            raise ValueError(f"Expected two continuations at u_D={field}, found {len(pair)}")
        c1 = pair[pair["hf_band_chern_N0"].abs() > 0.5]
        c0 = pair[pair["hf_band_chern_N0"].abs() < 0.5]
        if len(c1) == 1 and len(c0) == 1:
            records.append(
                {
                    "u_D_meV": float(field),
                    "c1_intrinsic_energy_total_meV": float(
                        c1.iloc[0]["intrinsic_energy_total_mev"]
                    ),
                    "c0_intrinsic_energy_total_meV": float(
                        c0.iloc[0]["intrinsic_energy_total_mev"]
                    ),
                    "c1_intrinsic_energy_per_cell_meV": float(
                        c1.iloc[0]["intrinsic_energy_per_cell_mev"]
                    ),
                    "c0_intrinsic_energy_per_cell_meV": float(
                        c0.iloc[0]["intrinsic_energy_per_cell_mev"]
                    ),
                }
            )

    coexistence = pd.DataFrame.from_records(records).sort_values("u_D_meV")
    if len(coexistence) != 6:
        raise ValueError(f"Expected six coexistence points, found {len(coexistence)}")

    reference = float(coexistence.iloc[0]["c1_intrinsic_energy_per_cell_meV"])
    coexistence["reference_energy_per_cell_meV"] = reference
    coexistence["c1_relative_energy_per_cell_meV"] = (
        coexistence["c1_intrinsic_energy_per_cell_meV"] - reference
    )
    coexistence["c0_relative_energy_per_cell_meV"] = (
        coexistence["c0_intrinsic_energy_per_cell_meV"] - reference
    )
    coexistence["c0_minus_c1_total_meV"] = (
        coexistence["c0_intrinsic_energy_total_meV"]
        - coexistence["c1_intrinsic_energy_total_meV"]
    )
    return coexistence.reset_index(drop=True)


def _crossing_estimate(rows: pd.DataFrame) -> float:
    field = rows["u_D_meV"].to_numpy(float)
    delta = rows["c0_minus_c1_total_meV"].to_numpy(float)
    indices = np.flatnonzero(delta[:-1] * delta[1:] < 0.0)
    if indices.size != 1:
        raise ValueError(f"Expected one branch crossing, found {indices.size}")
    index = int(indices[0])
    return float(
        field[index]
        - delta[index]
        * (field[index + 1] - field[index])
        / (delta[index + 1] - delta[index])
    )


def plot() -> list[Path]:
    rows = _load_coexisting_neutral_branches()
    crossing = _crossing_estimate(rows)
    rows["linear_crossing_estimate_meV"] = crossing

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FIGURE_DIR / f"{OUTPUT_STEM}.csv"
    rows.to_csv(csv_path, index=False)

    _apply_style()
    fig, ax = plt.subplots(figsize=FIGURE["size"])
    fig.subplots_adjust(left=0.20, right=0.96, bottom=0.20, top=0.90)

    ax.axvline(
        crossing,
        color=COLORS["purple"],
        linestyle="--",
        linewidth=2.1,
        zorder=1,
    )
    ax.plot(
        rows["u_D_meV"],
        rows["c1_relative_energy_per_cell_meV"],
        color=COLORS["red"],
        linewidth=2.7,
        marker="o",
        markersize=7.0,
        markerfacecolor="white",
        markeredgewidth=1.8,
        label=r"$C=1$ branch",
        zorder=3,
    )
    ax.plot(
        rows["u_D_meV"],
        rows["c0_relative_energy_per_cell_meV"],
        color=COLORS["teal"],
        linewidth=2.7,
        marker="s",
        markersize=6.5,
        markerfacecolor="white",
        markeredgewidth=1.8,
        label=r"$C=0$ branch",
        zorder=3,
    )
    ax.annotate(
        rf"crossing $u_D\simeq {crossing:.3f}$ meV",
        xy=(crossing, -0.071),
        xytext=(5.535, -0.112),
        arrowprops={"arrowstyle": "->", "color": COLORS["grey"], "lw": 1.2},
        fontsize=FONTS["annotation"],
        color=COLORS["grey"],
        ha="left",
    )

    ax.set_xlim(5.50, 5.82)
    ax.set_ylim(-0.18, 0.025)
    ax.set_xticks(np.arange(5.52, 5.81, 0.06))
    ax.set_yticks(np.arange(-0.16, 0.021, 0.04))
    ax.set_xlabel(r"displacement field $u_D$ (meV)")
    ax.set_ylabel(r"$(E_{\rm HF}-E_{\rm ref})/N_{\rm cell}$ (meV)")
    ax.set_title(r"Coexisting neutral Hartree--Fock branches")
    ax.set_box_aspect(1.0)
    _box_axes(ax)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
        handletextpad=0.5,
        columnspacing=1.3,
    )

    png_path = FIGURE_DIR / f"{OUTPUT_STEM}.png"
    pdf_path = FIGURE_DIR / f"{OUTPUT_STEM}.pdf"
    fig.savefig(png_path, dpi=FIGURE["dpi"], bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return [png_path, pdf_path, csv_path]


if __name__ == "__main__":
    for output in plot():
        print(output)
