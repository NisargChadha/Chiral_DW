#!/usr/bin/env python3
"""Plot cG versus the second magnetic AC harmonic b2 at fixed b1=u1=u2=0."""

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
from matplotlib.ticker import FormatStrFormatter
from pydantic import BaseModel, ConfigDict


ROOT = Path(__file__).resolve().parents[1]
FIGURE = {"size": (7.4, 7.4), "dpi": 280}
FONTS = {"base": 12, "title": 20, "axis_label": 24, "tick_label": 20}
COLORS = {
    "cG": "#1f4f9a",
    "invalid": "0.72",
    "axis": "0.18",
    "grid": "0.70",
}


class ACB2CGPlotParams(BaseModel):
    """Frozen controls for the second-harmonic cG linecut."""

    model_config = ConfigDict(frozen=True)

    input_csv: Path = Path("results/ac_b2_cg_local_nk12_nll5_v0p1_grid11/sweep.csv")
    output: Path = Path("Plots/figures/ac_b2_cg_local_nk12_nll5_v0p1_grid11.png")

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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing AC b2 sweep table: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"AC b2 sweep table is empty: {path}")
    rows.sort(key=lambda row: float(row["b2"]))
    for fixed in ("b1", "u1", "u2"):
        values = {round(float(row[fixed]), 14) for row in rows}
        if len(values) != 1:
            raise ValueError(f"{fixed} must be fixed for an AC b2 linecut; found {sorted(values)}")
    return rows


def _write_plot_data(output: Path, rows: list[dict[str, str]]) -> Path:
    path = output.with_suffix(".csv")
    fields = [
        "b2",
        "cG",
        "hf_all_converged",
        "response_status",
        "band_chern",
        "chern_vp_plus",
        "chern_vp_minus",
        "chern_ivc",
        "min_direct_gap",
        "path_gap_min",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    return path


def render_ac_b2_cg_linecut(
    params: ACB2CGPlotParams,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    _apply_style()
    input_csv = params.resolved_input()
    output = params.resolved_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(input_csv)

    b2 = np.asarray([float(row["b2"]) for row in rows], dtype=float)
    cG = np.asarray([float(row["cG"]) for row in rows], dtype=float)
    valid = np.asarray(
        [
            _as_bool(row.get("hf_all_converged", False))
            and row.get("response_status", "ok") == "ok"
            and np.isfinite(value)
            for row, value in zip(rows, cG, strict=True)
        ],
        dtype=bool,
    )
    if not np.any(valid):
        raise ValueError("AC b2 sweep has no converged finite cG points")

    fig, ax = plt.subplots(figsize=FIGURE["size"], constrained_layout=False)
    fig.subplots_adjust(left=0.23, right=0.96, bottom=0.16, top=0.82)
    ax.plot(
        b2[valid],
        cG[valid],
        color=COLORS["cG"],
        marker="o",
        markersize=7.0,
        markeredgecolor="white",
        markeredgewidth=0.9,
        linewidth=2.2,
        zorder=3,
    )
    if np.any(~valid):
        finite_valid = cG[valid]
        y_marker = float(np.mean(finite_valid))
        ax.scatter(
            b2[~valid],
            np.full(np.count_nonzero(~valid), y_marker),
            marker="x",
            s=70,
            linewidths=1.8,
            color=COLORS["invalid"],
            zorder=4,
        )

    first = rows[0]
    n_k = int(float(first["n_k"]))
    n_ll = int(float(first["n_ll"]))
    v0 = float(first["v0_over_omega_c"])
    ax.set_title(
        "Conjugate AC projected HF\n"
        + rf"$b_1=u_1=u_2=0$, $n_k={n_k}$, $N_{{\rm LL}}={n_ll}$, $V_0/\omega_c={v0:g}$",
        pad=12,
        linespacing=1.15,
    )
    ax.set_xlabel(r"$b_2/\omega_c$")
    ax.set_ylabel(r"$c_G$")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.7f"))
    ax.set_box_aspect(1.0)
    ax.margins(x=0.04, y=0.12)

    pdf_output = output.with_suffix(".pdf")
    csv_output = _write_plot_data(output, rows)
    summary_output = output.with_suffix(".json")
    finite = cG[valid]
    summary = {
        "input_csv": str(input_csv),
        "output_png": str(output),
        "output_pdf": str(pdf_output),
        "plot_data_csv": str(csv_output),
        "n_points": int(len(rows)),
        "n_valid": int(np.count_nonzero(valid)),
        "n_invalid": int(len(rows) - np.count_nonzero(valid)),
        "b2_range": [float(np.min(b2)), float(np.max(b2))],
        "cG_min": float(np.min(finite)),
        "cG_max": float(np.max(finite)),
        "b1": float(first["b1"]),
        "u1": float(first["u1"]),
        "u2": float(first["u2"]),
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
    defaults = ACB2CGPlotParams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=defaults.input_csv)
    parser.add_argument("--output", type=Path, default=defaults.output)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    outputs = render_ac_b2_cg_linecut(
        ACB2CGPlotParams(input_csv=args.input_csv, output=args.output)
    )
    print(f"Wrote AC b2 cG linecut to {outputs[0]}")
    print(f"Valid points: {outputs[-1]['n_valid']}/{outputs[-1]['n_points']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
