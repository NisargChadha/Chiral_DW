#!/usr/bin/env python3
"""Render the conjugate-AC projected-HF cG phase diagram in the b1-u1 plane."""

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
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from pydantic import BaseModel, ConfigDict


ROOT = Path(__file__).resolve().parents[1]

FIGURE = {
    "size": (7.4, 7.4),
    "dpi": 280,
    "subplots_adjust": {"left": 0.17, "right": 0.80, "bottom": 0.15, "top": 0.82},
}
FONTS = {
    "base": 12,
    "title": 20,
    "axis_label": 24,
    "tick_label": 20,
    "colorbar_label": 24,
    "colorbar_tick": 20,
}
COLORS = {
    "negative": "#378d94",
    "center": "#f7f7f7",
    "positive": "#FD4C55",
    "invalid": "0.72",
    "axis": "0.18",
}


class ACB1U1CGPlotParams(BaseModel):
    """Frozen user-facing controls for the AC cG heatmap."""

    model_config = ConfigDict(frozen=True)

    input_csv: Path = Path("results/ac_b1_u1_cg_local_nk12_nll5_v0p1_grid11/sweep.csv")
    output: Path = Path("Plots/figures/ac_b1_u1_cg_local_nk12_nll5_v0p1_grid11.png")

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
            "axes.edgecolor": COLORS["axis"],
            "axes.linewidth": 1.15,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "savefig.transparent": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError("phase diagram requires at least two values on each axis")
    return np.r_[
        values[0] - 0.5 * (values[1] - values[0]),
        0.5 * (values[:-1] + values[1:]),
        values[-1] + 0.5 * (values[-1] - values[-2]),
    ]


def _load_grid(path: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing AC sweep table: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"AC sweep table is empty: {path}")

    canonical = lambda value: float(np.round(float(value), decimals=14))
    b_values = np.array(sorted({canonical(row["b1"]) for row in rows}), dtype=float)
    u_values = np.array(sorted({canonical(row["u1"]) for row in rows}), dtype=float)
    cG = np.full((len(u_values), len(b_values)), np.nan, dtype=float)
    valid = np.zeros(cG.shape, dtype=bool)
    b_lookup = {round(float(value), 14): index for index, value in enumerate(b_values)}
    u_lookup = {round(float(value), 14): index for index, value in enumerate(u_values)}
    seen: set[tuple[int, int]] = set()
    for row in rows:
        ib = b_lookup[canonical(row["b1"])]
        iu = u_lookup[canonical(row["u1"])]
        if (iu, ib) in seen:
            raise ValueError(f"duplicate AC sweep point at b1={row['b1']} u1={row['u1']}")
        seen.add((iu, ib))
        value = float(row["cG"])
        converged = _as_bool(row.get("hf_all_converged", False))
        response_ok = row.get("response_status", "ok") == "ok"
        cG[iu, ib] = value
        valid[iu, ib] = bool(converged and response_ok and np.isfinite(value))
    return rows, b_values, u_values, cG, valid


def _write_plot_data(output: Path, rows: list[dict[str, str]]) -> Path:
    path = output.with_suffix(".csv")
    fields = [
        "b_index",
        "u_index",
        "b1",
        "u1",
        "cG",
        "hf_all_converged",
        "response_status",
        "band_chern",
        "min_direct_gap",
        "ivc_minus_best_vp_energy_per_cell",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    return path


def render_ac_cg_phase_diagram(
    params: ACB1U1CGPlotParams,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    _apply_style()
    input_csv = params.resolved_input()
    output = params.resolved_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, b_values, u_values, cG, valid = _load_grid(input_csv)

    finite = cG[valid]
    if not finite.size:
        raise ValueError("AC sweep has no converged finite cG values to plot")
    vmax = max(float(np.max(np.abs(finite))), 1e-12)
    cmap = LinearSegmentedColormap.from_list(
        "bootstrap_teal_red",
        [COLORS["negative"], COLORS["center"], COLORS["positive"]],
        N=256,
    )

    fig, ax = plt.subplots(figsize=FIGURE["size"], constrained_layout=False)
    fig.subplots_adjust(**FIGURE["subplots_adjust"])
    masked = np.ma.masked_where(~valid, cG)
    mesh = ax.pcolormesh(
        _edges(b_values),
        _edges(u_values),
        masked,
        shading="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        rasterized=True,
    )
    if np.any(~valid):
        invalid = np.ma.masked_where(valid, np.ones_like(cG))
        ax.pcolormesh(
            _edges(b_values),
            _edges(u_values),
            invalid,
            shading="auto",
            cmap=ListedColormap([COLORS["invalid"]]),
            zorder=3,
        )

    first = rows[0]
    n_k = int(float(first["n_k"]))
    n_ll = int(float(first["n_ll"]))
    v0 = float(first["v0_over_omega_c"])
    ax.set_title(
        "Conjugate AC projected HF\n"
        + rf"$n_k={n_k}$, $N_{{\rm LL}}={n_ll}$, $V_0/\omega_c={v0:g}$",
        pad=12,
        linespacing=1.15,
    )
    ax.set_xlabel(r"$b_1/\omega_c$")
    ax.set_ylabel(r"$u_1/\omega_c$")
    ax.set_xlim(float(b_values.min()), float(b_values.max()))
    ax.set_ylim(float(u_values.min()), float(u_values.max()))
    ax.set_box_aspect(1.0)
    ax.grid(False)

    fig.canvas.draw()
    bbox = ax.get_position()
    cax = fig.add_axes([bbox.x1 + 0.035, bbox.y0, 0.035, bbox.height])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.ax.tick_params(labelsize=FONTS["colorbar_tick"])
    cbar.ax.text(
        1.0,
        1.015,
        r"$c_G$",
        transform=cbar.ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONTS["colorbar_label"],
        color=COLORS["axis"],
    )

    pdf_output = output.with_suffix(".pdf")
    csv_output = _write_plot_data(output, rows)
    summary_output = output.with_suffix(".json")
    summary = {
        "input_csv": str(input_csv),
        "output_png": str(output),
        "output_pdf": str(pdf_output),
        "plot_data_csv": str(csv_output),
        "n_b1": int(len(b_values)),
        "n_u1": int(len(u_values)),
        "n_points": int(len(rows)),
        "n_valid": int(np.count_nonzero(valid)),
        "n_invalid": int(valid.size - np.count_nonzero(valid)),
        "b1_range": [float(b_values.min()), float(b_values.max())],
        "u1_range": [float(u_values.min()), float(u_values.max())],
        "cG_min": float(np.min(finite)),
        "cG_max": float(np.max(finite)),
        "n_k": n_k,
        "n_ll": n_ll,
        "active_band": int(float(first["active_band"])),
        "v0_over_omega_c": v0,
    }
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    fig.savefig(output, dpi=FIGURE["dpi"], bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_output, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output, pdf_output, csv_output, summary_output, summary


def _build_parser() -> argparse.ArgumentParser:
    defaults = ACB1U1CGPlotParams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=defaults.input_csv)
    parser.add_argument("--output", type=Path, default=defaults.output)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    outputs = render_ac_cg_phase_diagram(
        ACB1U1CGPlotParams(input_csv=args.input_csv, output=args.output)
    )
    print(f"Wrote AC cG phase diagram to {outputs[0]}")
    print(f"Valid points: {outputs[-1]['n_valid']}/{outputs[-1]['n_points']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
