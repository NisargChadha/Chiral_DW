#!/usr/bin/env python3
"""Plot frozen-remote and enlarged-HF orbital-magnetization convergence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "results" / "taige_orbital_magnetization_theta3p7_uD0_nk18"
DEFAULT_OUTPUT = ROOT / "Plots" / "figures" / "taige_orbital_magnetization_theta3p7_uD0_nk18_convergence"

RED = "#FD4C55"
TEAL = "#378d94"
PURPLE = "#6a408d"
GREEN = "#4f6f20"
GREY = "0.25"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["PT Serif Caption", "PT Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "axes.edgecolor": "0.18",
            "axes.linewidth": 1.15,
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


def make_figure(results_dir: Path, output_stem: Path) -> tuple[Path, Path]:
    _style()
    remote = _rows(results_dir / "remote_convergence.csv")
    hf = _rows(results_dir / "hf_active_space_convergence.csv")
    matched = _rows(results_dir / "matched_cutoff_comparison.csv")
    benchmarks = _rows(results_dir / "benchmarks.csv")

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 5.25))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.30, top=0.80, wspace=0.36)

    ax = axes[0]
    point_style = {
        "vbm": (RED, "o", "electron VBM"),
        "midgap": (TEAL, "s", "midgap"),
        "cbm": (PURPLE, "^", "electron CBM"),
    }
    for point, (color, marker, label) in point_style.items():
        selected = [row for row in remote if row["chemical_potential_point"] == point]
        selected.sort(key=lambda row: int(row["n_remote_bands_per_valley"]))
        x = np.asarray([int(row["n_remote_bands_per_valley"]) for row in selected])
        y = np.asarray([float(row["orbital_magnetization_mu_b_per_cell"]) for row in selected])
        ax.plot(x, y, color=color, marker=marker, lw=2.0, ms=5.5, label=label)
    ax.axhline(0.0, color=GREY, lw=0.9, alpha=0.65)
    ax.set_xlabel("frozen remote bands / valley")
    ax.set_ylabel(r"$M_{\rm orb}$ ($\mu_B$/cell)")
    ax.set_xticks(range(7))
    ax.set_title("Frozen completeness")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.29), ncol=3)

    ax = axes[1]
    matched.sort(key=lambda row: int(row["n_total_bands_per_valley"]))
    total = np.asarray([int(row["n_total_bands_per_valley"]) for row in matched])
    frozen_m = np.asarray([float(row["frozen_magnetization_mu_b_per_cell"]) for row in matched])
    hf_m = np.asarray([float(row["hf_magnetization_mu_b_per_cell"]) for row in matched])
    ax.plot(total, frozen_m, color=PURPLE, marker="o", ls="--", lw=2.0, label="frozen")
    ax.plot(total, hf_m, color=TEAL, marker="s", lw=2.0, label="self-consistent HF")
    ax.set_xlabel("total bands / valley")
    ax.set_ylabel(r"$M_{\rm orb}$ ($\mu_B$/cell)")
    ax.set_xticks(total)
    ax.set_title(r"Matched cutoff and $\mu$")
    ax.legend(frameon=False, loc="upper left")

    hf_mid = [row for row in hf if row["chemical_potential_point"] == "midgap"]
    hf_mid.sort(key=lambda row: int(row["n_hf_bands_per_valley"]))
    mixing = np.asarray([float(row["active_remote_mixing_lambda"]) for row in hf_mid])
    mix_ax = ax.twinx()
    mix_ax.grid(False)
    mix_ax.plot(total, mixing, color=GREY, marker="D", lw=1.35, ms=4.5, alpha=0.8)
    mix_ax.set_ylabel(r"$\lambda_{\rm mix}$", color=GREY, fontsize=14)
    mix_ax.tick_params(axis="y", colors=GREY, labelsize=11)
    mix_ax.spines["right"].set_color(GREY)

    ax = axes[2]
    stages = ("density_vertices", "exchange_backend", "vp_hf_solve")
    stage_style = {
        "density_vertices": (RED, "density vertices"),
        "exchange_backend": (TEAL, "exchange backend"),
        "vp_hf_solve": (PURPLE, "VP HF solve"),
    }
    width = 0.23
    for offset_index, stage in enumerate(stages):
        values = []
        for n_active in (2, 3, 4):
            candidates = [
                row
                for row in benchmarks
                if row["stage"] == stage
                and row["n_active_bands_per_valley"] == str(n_active)
            ]
            values.append(float(candidates[-1]["elapsed_seconds_median"]))
        color, label = stage_style[stage]
        ax.bar(
            np.asarray((2, 3, 4)) + (offset_index - 1) * width,
            values,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
        )
    ax.set_yscale("log")
    ax.set_xlabel("HF-active bands / valley")
    ax.set_ylabel("wall time (s)")
    ax.set_xticks((2, 3, 4))
    ax.set_title("Production stage cost")
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.29),
        ncol=3,
        fontsize=9.5,
    )

    for label, ax in zip(("a", "b", "c"), axes, strict=True):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=15, fontweight="bold")
    fig.suptitle(
        r"tMoTe$_2$ VP orbital magnetization: $\theta=3.7^\circ$, $u_D=0$, $n_k=18$, shell 5",
        y=0.94,
        fontsize=17,
    )
    fig.text(
        0.5,
        0.035,
        "Valence-continuum remote-band study; frozen and HF values in panel b share the same chemical potential.",
        ha="center",
        va="center",
        fontsize=10.5,
        color=GREY,
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=280, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return png, pdf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    png, pdf = make_figure(args.results_dir.resolve(), args.output_stem.resolve())
    print(png)
    print(pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
