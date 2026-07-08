#!/usr/bin/env python3
"""Plot Taige single-point parameter-convergence diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, ConfigDict


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_CSV = (
    REPO_ROOT
    / "results"
    / "taige_convergence_plane_wave_shell_theta35_u0"
    / "convergence_summary.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Plots" / "figures"

FIGURE = {
    "size": (10.2, 8.6),
    "dpi": 280,
}

FONTS = {
    "base": 12,
    "title": 20,
    "axis_label": 24,
    "tick_label": 18,
    "legend": 12,
}

COLORS = {
    "red": "#FD4C55",
    "teal": "#378d94",
    "purple": "#6a408d",
    "green": "#4f6f20",
    "grey": "0.25",
    "axis": "0.18",
    "grid": "0.70",
    "zero": "0.25",
}

AXES = {
    "spine_linewidth": 1.15,
    "marker_size": 6.0,
    "line_width": 2.2,
    "save_bbox_inches": "tight",
    "save_pad_inches": 0.08,
}

PLOT_COLUMNS = (
    "scan_axis",
    "scan_value",
    "theta_deg",
    "u_D_meV",
    "n_k",
    "plane_wave_shell",
    "n_bands",
    "n_active_bands_per_valley",
    "cG",
    "vp_plus_energy_per_cell",
    "vp_minus_energy_per_cell",
    "vp_reference_energy_per_cell",
    "ivc_q0_energy_per_cell",
    "selected_ivc_energy_per_cell",
    "ivc_q0_minus_vp_energy_per_cell",
    "selected_ivc_minus_vp_energy_per_cell",
    "gap_min",
    "vp_reference_direct_gap",
    "vp_reference_indirect_gap",
    "selected_ivc_direct_gap",
    "selected_ivc_indirect_gap",
    "texture_valid",
    "hf_ground_state",
    "selected_ivc_branch",
)


class TaigeConvergencePlotParams(BaseModel):
    """User-facing controls for the Taige convergence summary plot."""

    model_config = ConfigDict(frozen=True)

    input_csv: Path = DEFAULT_INPUT_CSV
    output_dir: Path = DEFAULT_OUTPUT_DIR
    output_stem: str | None = None
    dpi: int = FIGURE["dpi"]


def apply_nisarg_plot_style() -> None:
    """Apply the local Chiral_DW scientific plotting style."""

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


def style_boxed_axis(ax: plt.Axes) -> None:
    """Keep all axes boxed and visually aligned with nearby Chiral_DW plots."""

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["axis"])
        spine.set_linewidth(AXES["spine_linewidth"])


def _float_or_nan(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    rows.sort(key=lambda row: _float_or_nan(row.get("scan_value")))
    return rows


def _array(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_float_or_nan(row.get(key)) for row in rows], dtype=float)


def _first_text(rows: list[dict[str, Any]], key: str, default: str = "") -> str:
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _axis_label(scan_axis: str) -> str:
    if scan_axis == "active_bands":
        return r"active bands per valley"
    return r"plane-wave shell"


def _default_stem(rows: list[dict[str, Any]]) -> str:
    axis = _first_text(rows, "scan_axis", "parameter")
    theta = _float_or_nan(rows[0].get("theta_deg"))
    u_d = _float_or_nan(rows[0].get("u_D_meV"))
    theta_label = f"theta{theta:g}".replace(".", "p") if np.isfinite(theta) else "theta"
    ud_label = f"uD{u_d:g}".replace(".", "p").replace("-", "m") if np.isfinite(u_d) else "uD"
    return f"taige_{axis}_convergence_{theta_label}_{ud_label}"


def _finite_ylim(arrays: list[np.ndarray], *, pad_fraction: float = 0.08, min_pad: float = 1e-9) -> tuple[float, float] | None:
    finite_parts = [arr[np.isfinite(arr)] for arr in arrays if np.any(np.isfinite(arr))]
    if not finite_parts:
        return None
    finite = np.concatenate(finite_parts)
    low = float(np.min(finite))
    high = float(np.max(finite))
    pad = max((high - low) * pad_fraction, min_pad)
    return low - pad, high + pad


def _plot_series(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    label: str,
    color: str,
    marker: str = "o",
) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.any(finite):
        ax.plot(
            x[finite],
            y[finite],
            color=color,
            marker=marker,
            markersize=AXES["marker_size"],
            linewidth=AXES["line_width"],
            label=label,
        )


def _write_plot_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PLOT_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in PLOT_COLUMNS})
    return path


def render_taige_parameter_convergence_plot(params: TaigeConvergencePlotParams) -> tuple[Path, Path, Path]:
    """Render PNG/PDF convergence panels and an exported plot-data CSV."""

    rows = _load_rows(params.input_csv)
    output_stem = params.output_stem or _default_stem(rows)
    params.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = params.output_dir / f"{output_stem}.png"
    pdf_path = params.output_dir / f"{output_stem}.pdf"
    csv_path = params.output_dir / f"{output_stem}.csv"
    _write_plot_csv(rows, csv_path)

    apply_nisarg_plot_style()
    x = _array(rows, "scan_value")
    scan_axis = _first_text(rows, "scan_axis", "plane_wave_shell")
    xlabel = _axis_label(scan_axis)

    fig, axes = plt.subplots(2, 2, figsize=FIGURE["size"], constrained_layout=False)
    fig.subplots_adjust(left=0.11, right=0.985, top=0.965, bottom=0.10, wspace=0.28, hspace=0.34)
    ax_cg, ax_energy, ax_delta, ax_gap = axes.ravel()

    cG = _array(rows, "cG")
    _plot_series(ax_cg, x, cG, label=r"$c_G$", color=COLORS["red"])
    ax_cg.axhline(0.0, color=COLORS["zero"], linewidth=1.0, alpha=0.7)
    ax_cg.set_ylabel(r"$c_G$")

    vp = _array(rows, "vp_reference_energy_per_cell")
    ivc_q0 = _array(rows, "ivc_q0_energy_per_cell")
    selected_ivc = _array(rows, "selected_ivc_energy_per_cell")
    _plot_series(ax_energy, x, vp, label="VP ref.", color=COLORS["red"])
    _plot_series(ax_energy, x, ivc_q0, label="IVC Q=0", color=COLORS["teal"], marker="s")
    _plot_series(ax_energy, x, selected_ivc, label="selected IVC", color=COLORS["purple"], marker="^")
    ax_energy.set_ylabel("energy / cell (meV)")

    delta = _array(rows, "selected_ivc_minus_vp_energy_per_cell")
    delta_q0 = _array(rows, "ivc_q0_minus_vp_energy_per_cell")
    _plot_series(ax_delta, x, delta, label="selected IVC - VP", color=COLORS["purple"])
    _plot_series(ax_delta, x, delta_q0, label="Q=0 IVC - VP", color=COLORS["teal"], marker="s")
    ax_delta.axhline(0.0, color=COLORS["zero"], linewidth=1.0, alpha=0.7)
    ax_delta.set_ylabel(r"$\Delta E$ / cell (meV)")

    vp_gap = _array(rows, "vp_reference_direct_gap")
    ivc_gap = _array(rows, "selected_ivc_direct_gap")
    min_gap = _array(rows, "gap_min")
    _plot_series(ax_gap, x, vp_gap, label="VP direct", color=COLORS["red"])
    _plot_series(ax_gap, x, ivc_gap, label="IVC direct", color=COLORS["teal"], marker="s")
    _plot_series(ax_gap, x, min_gap, label="path min", color=COLORS["green"], marker="^")
    ax_gap.set_ylabel("gap (meV)")

    for ax in axes.ravel():
        ax.set_xlabel(xlabel)
        ax.set_xticks(x)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_box_aspect(1.0)
        style_boxed_axis(ax)
        legend = ax.legend(
            loc="best",
            frameon=True,
            facecolor="white",
            edgecolor="white",
            framealpha=0.94,
        )
        legend.get_frame().set_linewidth(0.0)

    y_limits = {
        ax_cg: _finite_ylim([cG], min_pad=1e-4),
        ax_energy: _finite_ylim([vp, ivc_q0, selected_ivc], min_pad=1e-3),
        ax_delta: _finite_ylim([delta, delta_q0, np.asarray([0.0])], min_pad=1e-3),
        ax_gap: _finite_ylim([vp_gap, ivc_gap, min_gap], min_pad=1e-3),
    }
    for ax, limits in y_limits.items():
        if limits is not None:
            ax.set_ylim(*limits)

    fig.savefig(png_path, dpi=int(params.dpi), bbox_inches=AXES["save_bbox_inches"], pad_inches=AXES["save_pad_inches"])
    fig.savefig(pdf_path, bbox_inches=AXES["save_bbox_inches"], pad_inches=AXES["save_pad_inches"])
    plt.close(fig)
    return png_path, pdf_path, csv_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-stem", default=None)
    parser.add_argument("--dpi", type=int, default=FIGURE["dpi"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    png_path, pdf_path, csv_path = render_taige_parameter_convergence_plot(
        TaigeConvergencePlotParams(
            input_csv=args.input_csv,
            output_dir=args.output_dir,
            output_stem=args.output_stem,
            dpi=args.dpi,
        )
    )
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
