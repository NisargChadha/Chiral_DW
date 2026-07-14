#!/usr/bin/env python3
"""Render the conjugate-AC projected-HF cG phase diagram in the b2-u2 plane."""

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


class ACB2U2CGPlotParams(BaseModel):
    """Frozen user-facing controls for the second-harmonic AC cG heatmap."""

    model_config = ConfigDict(frozen=True)

    input_csv: Path = Path("results/ac_b2_u2_cg_local_nk12_nll5_v0p1_grid11/sweep.csv")
    output: Path = Path("Plots/figures/ac_b2_u2_cg_local_nk12_nll5_v0p1_grid11.png")

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
        raise FileNotFoundError(f"Missing AC b2-u2 sweep table: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"AC b2-u2 sweep table is empty: {path}")

    canonical = lambda value: float(np.round(float(value), decimals=14))
    b_values = np.array(sorted({canonical(row["b2"]) for row in rows}), dtype=float)
    u_values = np.array(sorted({canonical(row["u2"]) for row in rows}), dtype=float)
    cG = np.full((len(u_values), len(b_values)), np.nan, dtype=float)
    valid = np.zeros(cG.shape, dtype=bool)
    b_lookup = {round(float(value), 14): index for index, value in enumerate(b_values)}
    u_lookup = {round(float(value), 14): index for index, value in enumerate(u_values)}
    seen: set[tuple[int, int]] = set()
    for row in rows:
        ib = b_lookup[canonical(row["b2"])]
        iu = u_lookup[canonical(row["u2"])]
        if (iu, ib) in seen:
            raise ValueError(f"duplicate AC sweep point at b2={row['b2']} u2={row['u2']}")
        seen.add((iu, ib))
        value = float(row["cG"])
        converged = _as_bool(row.get("hf_all_converged", False))
        response_ok = row.get("response_status", "ok") == "ok"
        cG[iu, ib] = value
        valid[iu, ib] = bool(converged and response_ok and np.isfinite(value))
    return rows, b_values, u_values, cG, valid


def _write_plot_data(output: Path, rows: list[dict[str, str]], center_cG: float) -> Path:
    path = output.with_suffix(".csv")
    fields = [
        "b2_index",
        "u2_index",
        "b2",
        "u2",
        "cG",
        "delta_cG",
        "hf_all_converged",
        "response_status",
        "band_chern",
        "chern_vp_plus",
        "chern_vp_minus",
        "chern_ivc",
        "min_direct_gap",
        "path_gap_min",
        "ivc_minus_best_vp_energy_per_cell",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            plot_row = {key: row.get(key, "") for key in fields}
            plot_row["delta_cG"] = float(row["cG"]) - center_cG
            writer.writerow(plot_row)
    return path


def render_ac_b2_u2_cg_phase_diagram(
    params: ACB2U2CGPlotParams,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    _apply_style()
    input_csv = params.resolved_input()
    output = params.resolved_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, b_values, u_values, cG, valid = _load_grid(input_csv)

    finite = cG[valid]
    if not finite.size:
        raise ValueError("AC b2-u2 sweep has no converged finite cG values to plot")
    center_b = int(np.argmin(np.abs(b_values)))
    center_u = int(np.argmin(np.abs(u_values)))
    if abs(b_values[center_b]) > 1e-12 or abs(u_values[center_u]) > 1e-12:
        raise ValueError("AC b2-u2 sweep must contain b2=u2=0 for the delta-cG map")
    if not valid[center_u, center_b]:
        raise ValueError("AC b2-u2 center point is invalid")
    center_cG = float(cG[center_u, center_b])
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
    b1 = float(first["b1"])
    u1 = float(first["u1"])
    ax.set_title(
        "Second-harmonic AC projected HF\n"
        + rf"$n_k={n_k}$, $N_{{\rm LL}}={n_ll}$, $V_0/\omega_c={v0:g}$",
        pad=12,
        linespacing=1.15,
    )
    ax.set_xlabel(r"$b_2/\omega_c$")
    ax.set_ylabel(r"$u_2/\omega_c$")
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
    csv_output = _write_plot_data(output, rows, center_cG)
    delta_output = output.with_name(f"{output.stem}_delta{output.suffix}")
    delta_pdf_output = delta_output.with_suffix(".pdf")
    summary_output = output.with_suffix(".json")
    summary = {
        "input_csv": str(input_csv),
        "output_png": str(output),
        "output_pdf": str(pdf_output),
        "delta_output_png": str(delta_output),
        "delta_output_pdf": str(delta_pdf_output),
        "plot_data_csv": str(csv_output),
        "n_b2": int(len(b_values)),
        "n_u2": int(len(u_values)),
        "n_points": int(len(rows)),
        "n_valid": int(np.count_nonzero(valid)),
        "n_invalid": int(valid.size - np.count_nonzero(valid)),
        "b2_range": [float(b_values.min()), float(b_values.max())],
        "u2_range": [float(u_values.min()), float(u_values.max())],
        "cG_min": float(np.min(finite)),
        "cG_max": float(np.max(finite)),
        "cG_center": center_cG,
        "delta_cG_min": float(np.min(finite - center_cG)),
        "delta_cG_max": float(np.max(finite - center_cG)),
        "delta_plot_scale": 1.0e6,
        "b1": b1,
        "u1": u1,
        "n_k": n_k,
        "n_ll": n_ll,
        "active_band": int(float(first["active_band"])),
        "v0_over_omega_c": v0,
    }
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True))
    fig.savefig(output, dpi=FIGURE["dpi"], bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_output, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    delta_cG = cG - center_cG
    delta_plot_scale = summary["delta_plot_scale"]
    delta_plot = delta_plot_scale * delta_cG
    delta_vmax = max(float(np.max(np.abs(delta_plot[valid]))), 1e-6)
    delta_fig, delta_ax = plt.subplots(figsize=FIGURE["size"], constrained_layout=False)
    delta_fig.subplots_adjust(**FIGURE["subplots_adjust"])
    delta_mesh = delta_ax.pcolormesh(
        _edges(b_values),
        _edges(u_values),
        np.ma.masked_where(~valid, delta_plot),
        shading="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-delta_vmax, vcenter=0.0, vmax=delta_vmax),
        rasterized=True,
    )
    if np.any(~valid):
        invalid = np.ma.masked_where(valid, np.ones_like(cG))
        delta_ax.pcolormesh(
            _edges(b_values),
            _edges(u_values),
            invalid,
            shading="auto",
            cmap=ListedColormap([COLORS["invalid"]]),
            zorder=3,
        )
    delta_ax.set_title(
        "Second-harmonic AC response variation\n"
        + rf"$n_k={n_k}$, $N_{{\rm LL}}={n_ll}$, $V_0/\omega_c={v0:g}$",
        pad=12,
        linespacing=1.15,
    )
    delta_ax.set_xlabel(r"$b_2/\omega_c$")
    delta_ax.set_ylabel(r"$u_2/\omega_c$")
    delta_ax.set_xlim(float(b_values.min()), float(b_values.max()))
    delta_ax.set_ylim(float(u_values.min()), float(u_values.max()))
    delta_ax.set_box_aspect(1.0)
    delta_ax.grid(False)
    delta_fig.canvas.draw()
    delta_bbox = delta_ax.get_position()
    delta_cax = delta_fig.add_axes(
        [delta_bbox.x1 + 0.035, delta_bbox.y0, 0.035, delta_bbox.height]
    )
    delta_cbar = delta_fig.colorbar(delta_mesh, cax=delta_cax)
    delta_cbar.ax.tick_params(labelsize=FONTS["colorbar_tick"])
    delta_cbar.set_label(
        r"$10^6\Delta c_G$",
        fontsize=FONTS["colorbar_label"],
        color=COLORS["axis"],
        rotation=90,
        labelpad=18,
    )
    delta_fig.savefig(delta_output, dpi=FIGURE["dpi"], bbox_inches="tight", pad_inches=0.08)
    delta_fig.savefig(delta_pdf_output, bbox_inches="tight", pad_inches=0.08)
    plt.close(delta_fig)
    return output, pdf_output, csv_output, summary_output, summary


def _build_parser() -> argparse.ArgumentParser:
    defaults = ACB2U2CGPlotParams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=defaults.input_csv)
    parser.add_argument("--output", type=Path, default=defaults.output)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    outputs = render_ac_b2_u2_cg_phase_diagram(
        ACB2U2CGPlotParams(input_csv=args.input_csv, output=args.output)
    )
    print(f"Wrote AC b2-u2 cG phase diagram to {outputs[0]}")
    print(f"Valid points: {outputs[-1]['n_valid']}/{outputs[-1]['n_points']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
