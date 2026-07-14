#!/usr/bin/env python3
"""Plot the gauge-fixed AC cG correction and central b1/u1 linecuts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from pydantic import BaseModel, ConfigDict


ROOT = Path(__file__).resolve().parents[1]
FIGURE = {
    "size": (13.4, 6.2),
    "dpi": 280,
    "subplots_adjust": {
        "left": 0.085,
        "right": 0.965,
        "bottom": 0.17,
        "top": 0.86,
        "wspace": 0.38,
    },
}
FONTS = {
    "base": 12,
    "title": 18,
    "axis_label": 22,
    "tick_label": 17,
    "legend": 14,
    "colorbar_label": 20,
    "colorbar_tick": 16,
}
COLORS = {
    "red": "#FD4C55",
    "teal": "#378d94",
    "center": "#f7f7f7",
    "axis": "0.18",
    "grid": "0.70",
    "zero": "0.25",
}


class GaugeFixedACPlotParams(BaseModel):
    """Frozen controls for the gauge-fixed AC response diagnostics."""

    model_config = ConfigDict(frozen=True)

    input_csv: Path = Path(
        "results/ac_b1_u1_cg_nk18_nll6_v0p1_grid11_gaugefixed_v2/sweep.csv"
    )
    output: Path = Path(
        "Plots/figures/"
        "ac_b1_u1_cg_nk18_nll6_v0p1_grid11_gaugefixed_v2_diagnostics.png"
    )

    def resolved_input(self) -> Path:
        return self.input_csv if self.input_csv.is_absolute() else ROOT / self.input_csv

    def resolved_output(self) -> Path:
        return self.output if self.output.is_absolute() else ROOT / self.output


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
            "axes.linewidth": 1.15,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
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
    return np.r_[
        values[0] - 0.5 * (values[1] - values[0]),
        0.5 * (values[:-1] + values[1:]),
        values[-1] + 0.5 * (values[-1] - values[-2]),
    ]


def _load_grid(path: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty AC sweep: {path}")
    b_values = np.array(sorted({float(row["b1"]) for row in rows}))
    u_values = np.array(sorted({float(row["u1"]) for row in rows}))
    grid = np.full((len(u_values), len(b_values)), np.nan)
    b_lookup = {round(value, 14): index for index, value in enumerate(b_values)}
    u_lookup = {round(value, 14): index for index, value in enumerate(u_values)}
    for row in rows:
        ib = b_lookup[round(float(row["b1"]), 14)]
        iu = u_lookup[round(float(row["u1"]), 14)]
        if np.isfinite(grid[iu, ib]):
            raise ValueError(f"Duplicate point at b1={row['b1']}, u1={row['u1']}")
        if row.get("status", "ok") != "ok" or row.get("response_status", "ok") != "ok":
            continue
        grid[iu, ib] = float(row["cG"])
    if not np.all(np.isfinite(grid)):
        raise ValueError("Gauge-fixed diagnostic requires a complete valid cG grid")
    return rows, b_values, u_values, grid


def _write_data(
    output: Path,
    rows: list[dict[str, str]],
    center_cg: float,
) -> Path:
    path = output.with_suffix(".csv")
    fields = [
        "b1",
        "u1",
        "cG",
        "delta_cG_from_center",
        "min_direct_gap",
        "path_gap_min",
        "ivc_minus_best_vp_energy_per_cell",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key, "") for key in fields if key != "delta_cG_from_center"},
                    "delta_cG_from_center": float(row["cG"]) - center_cg,
                }
            )
    return path


def render_gaugefixed_diagnostics(
    params: GaugeFixedACPlotParams,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    _apply_style()
    input_csv = params.resolved_input()
    output = params.resolved_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, b_values, u_values, cG = _load_grid(input_csv)
    ib0 = int(np.argmin(np.abs(b_values)))
    iu0 = int(np.argmin(np.abs(u_values)))
    if abs(b_values[ib0]) > 1e-12 or abs(u_values[iu0]) > 1e-12:
        raise ValueError("b1=0 and u1=0 must both be present")
    center_cg = float(cG[iu0, ib0])
    delta_scaled = 1e4 * (cG - center_cg)
    vmax = float(np.max(np.abs(delta_scaled)))
    cmap = LinearSegmentedColormap.from_list(
        "bootstrap_teal_red",
        [COLORS["teal"], COLORS["center"], COLORS["red"]],
        N=256,
    )

    fig, (ax_map, ax_line) = plt.subplots(1, 2, figsize=FIGURE["size"])
    fig.subplots_adjust(**FIGURE["subplots_adjust"])
    mesh = ax_map.pcolormesh(
        _edges(b_values),
        _edges(u_values),
        delta_scaled,
        shading="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        rasterized=True,
    )
    ax_map.scatter([0.0], [0.0], s=38, facecolor="white", edgecolor=COLORS["axis"], zorder=4)
    ax_map.set_title(r"Center-subtracted response", pad=10)
    ax_map.set_xlabel(r"$b_1/\omega_c$")
    ax_map.set_ylabel(r"$u_1/\omega_c$")
    ax_map.set_box_aspect(1.0)
    ax_map.grid(False)

    fig.canvas.draw()
    bbox = ax_map.get_position()
    cax = fig.add_axes([bbox.x1 + 0.012, bbox.y0, 0.018, bbox.height])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.ax.tick_params(labelsize=FONTS["colorbar_tick"])
    cbar.ax.text(
        0.5,
        1.025,
        r"$10^4\,\Delta c_G$",
        transform=cbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=FONTS["colorbar_label"],
        color=COLORS["axis"],
    )

    ax_line.axhline(0.0, color=COLORS["zero"], linewidth=1.0, linestyle="--", zorder=1)
    ax_line.plot(
        b_values,
        delta_scaled[iu0],
        color=COLORS["red"],
        marker="o",
        markersize=5.8,
        linewidth=2.2,
        label=r"vary $b_1$ at $u_1=0$",
    )
    ax_line.plot(
        u_values,
        delta_scaled[:, ib0],
        color=COLORS["teal"],
        marker="s",
        markersize=5.4,
        linewidth=2.2,
        label=r"vary $u_1$ at $b_1=0$",
    )
    ax_line.set_title("Central linecuts", pad=10)
    ax_line.set_xlabel(r"varied parameter$/\omega_c$")
    ax_line.set_ylabel(r"$10^4[c_G-c_G(0,0)]$")
    ax_line.set_box_aspect(1.0)
    ax_line.legend(loc="upper left", frameon=False, handlelength=2.4)

    B, U = np.meshgrid(b_values, u_values)
    design = np.column_stack(
        [
            np.ones(cG.size),
            B.ravel(),
            U.ravel(),
            (B * B).ravel(),
            (B * U).ravel(),
            (U * U).ravel(),
        ]
    )
    coefficients = np.linalg.lstsq(design, cG.ravel(), rcond=None)[0]
    fitted = (design @ coefficients).reshape(cG.shape)
    minimum = np.unravel_index(int(np.argmin(cG)), cG.shape)
    maximum = np.unravel_index(int(np.argmax(cG)), cG.shape)
    first = rows[0]
    summary = {
        "input_csv": str(input_csv),
        "output_png": str(output),
        "center_cG": center_cg,
        "cG_min": float(cG[minimum]),
        "cG_min_coordinate": [float(b_values[minimum[1]]), float(u_values[minimum[0]])],
        "cG_max": float(cG[maximum]),
        "cG_max_coordinate": [float(b_values[maximum[1]]), float(u_values[maximum[0]])],
        "cG_range": float(np.ptp(cG)),
        "relative_range": float(np.ptp(cG) / abs(center_cg)),
        "quadratic_fit_coefficients_c_b_u_b2_bu_u2": coefficients.tolist(),
        "quadratic_fit_rmse": float(np.sqrt(np.mean((fitted - cG) ** 2))),
        "n_points": int(cG.size),
        "n_k": int(float(first["n_k"])),
        "n_ll": int(float(first["n_ll"])),
        "v0_over_omega_c": float(first["v0_over_omega_c"]),
    }
    pdf_output = output.with_suffix(".pdf")
    csv_output = _write_data(output, rows, center_cg)
    summary_output = output.with_suffix(".json")
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    fig.savefig(output, dpi=FIGURE["dpi"], bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_output, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output, pdf_output, csv_output, summary_output, summary


def _build_parser() -> argparse.ArgumentParser:
    defaults = GaugeFixedACPlotParams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=defaults.input_csv)
    parser.add_argument("--output", type=Path, default=defaults.output)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    outputs = render_gaugefixed_diagnostics(
        GaugeFixedACPlotParams(input_csv=args.input_csv, output=args.output)
    )
    print(f"Wrote gauge-fixed AC diagnostics to {outputs[0]}")
    print(json.dumps(outputs[-1], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
