#!/usr/bin/env python3
"""Plot finite-size conjugate-AC cG and validity diagnostics in the b1-u1 plane."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from pydantic import BaseModel, ConfigDict


ROOT = Path(__file__).resolve().parents[1]

FIGURE = {
    "size": (19.2, 5.0),
    "dpi": 280,
    "subplots_adjust": {
        "left": 0.055,
        "right": 0.90,
        "bottom": 0.20,
        "top": 0.72,
        "wspace": 0.18,
    },
    "colorbar": [0.925, 0.20, 0.015, 0.52],
}
FONTS = {
    "base": 12,
    "suptitle": 19,
    "title": 16,
    "axis_label": 19,
    "tick_label": 14,
    "colorbar_label": 19,
    "colorbar_tick": 13,
    "legend": 13,
}
COLORS = {
    "red": "#FD4C55",
    "teal": "#378d94",
    "purple": "#6a408d",
    "axis": "0.18",
    "grey_mask": "0.72",
    "grey_span": "0.86",
    "white_center": "#f7f7f7",
}
MESHES = (18, 20, 21, 22, 24)


class ACFiniteSizeDiagnosticPlotParams(BaseModel):
    """Frozen controls for the finite-size AC phase-diagram comparison."""

    model_config = ConfigDict(frozen=True)

    input_root: Path = Path(
        "results/ac_b1_u1_cg_dual_gate_omega_v0p3_nll8_grid21_nk18_24"
    )
    output_stem: Path = Path(
        "Plots/figures/ac_b1_u1_cg_dual_gate_omega_v0p3_nll8_grid21_nk18_24"
    )
    meshes: tuple[int, ...] = MESHES
    chern_tolerance: float = 5.0e-3

    def resolved_input_root(self) -> Path:
        return self.input_root if self.input_root.is_absolute() else ROOT / self.input_root

    def resolved_output_stem(self) -> Path:
        return self.output_stem if self.output_stem.is_absolute() else ROOT / self.output_stem


class MeshDiagnosticSummary(BaseModel):
    """Per-mesh counts shown by the diagnostic phase diagrams."""

    model_config = ConfigDict(frozen=True)

    n_k: int
    n_points: int
    n_cg: int
    n_negative_path_indirect_gap: int
    n_path_gap_not_evaluated: int
    n_hf_nonconverged: int
    n_invalid_reference_chern: int
    n_invalid_chern_with_nonpositive_reference_direct_gap: int
    minimum_reference_direct_gap_mev_at_invalid_chern: float | None
    minimum_ivc_direct_gap_mev_at_invalid_chern: float | None


class DiagnosticPlotSummary(BaseModel):
    """Summary and artifact manifest for the rendered figures."""

    model_config = ConfigDict(frozen=True)

    input_root: str
    chern_tolerance: float
    direct_gap_overlap_definition: str
    meshes: tuple[MeshDiagnosticSummary, ...]
    outputs: dict[str, str]


@dataclass(frozen=True)
class MeshGrid:
    n_k: int
    rows: list[dict[str, str]]
    b_values: np.ndarray
    u_values: np.ndarray
    arrays: dict[str, np.ndarray]


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
            "axes.grid": False,
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


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _canonical(value: Any) -> float:
    return float(np.round(float(value), decimals=14))


def _edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError("phase diagram requires at least two points on each axis")
    return np.r_[
        values[0] - 0.5 * (values[1] - values[0]),
        0.5 * (values[:-1] + values[1:]),
        values[-1] + 0.5 * (values[-1] - values[-2]),
    ]


def _load_mesh(path: Path, n_k: int, chern_tolerance: float) -> MeshGrid:
    if not path.exists():
        raise FileNotFoundError(f"Missing merged AC sweep table: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Merged AC sweep table is empty: {path}")

    b_values = np.array(sorted({_canonical(row["b1"]) for row in rows}), dtype=float)
    u_values = np.array(sorted({_canonical(row["u1"]) for row in rows}), dtype=float)
    shape = (len(u_values), len(b_values))
    b_lookup = {value: index for index, value in enumerate(b_values)}
    u_lookup = {value: index for index, value in enumerate(u_values)}
    arrays = {
        "cG": np.full(shape, np.nan, dtype=float),
        "path_indirect_gap_min_mev": np.full(shape, np.nan, dtype=float),
        "hf_all_converged": np.zeros(shape, dtype=bool),
        "reference_chern_valid": np.zeros(shape, dtype=bool),
        "any_reference_direct_gap_nonpositive": np.zeros(shape, dtype=bool),
    }
    seen: set[tuple[int, int]] = set()
    for row in rows:
        ib = b_lookup[_canonical(row["b1"])]
        iu = u_lookup[_canonical(row["u1"])]
        if (iu, ib) in seen:
            raise ValueError(f"Duplicate sweep point at b1={row['b1']} u1={row['u1']}")
        seen.add((iu, ib))
        if int(float(row["n_k"])) != n_k:
            raise ValueError(f"Expected n_k={n_k}, found {row['n_k']} in {path}")

        arrays["cG"][iu, ib] = _as_float(row.get("cG"))
        arrays["path_indirect_gap_min_mev"][iu, ib] = _as_float(
            row.get("path_indirect_gap_min_mev")
        )
        arrays["hf_all_converged"][iu, ib] = _as_bool(row.get("hf_all_converged"))
        cherns = {
            "vp_plus": _as_float(row.get("chern_vp_plus")),
            "vp_minus": _as_float(row.get("chern_vp_minus")),
            "ivc": _as_float(row.get("chern_ivc")),
        }
        expected = {"vp_plus": 1.0, "vp_minus": -1.0, "ivc": 0.0}
        computed_chern_valid = all(
            np.isfinite(cherns[name])
            and abs(cherns[name] - target) <= chern_tolerance
            for name, target in expected.items()
        )
        stored_chern_valid = _as_bool(row.get("reference_chern_valid"))
        if computed_chern_valid != stored_chern_valid:
            raise ValueError(
                f"Stored and recomputed reference-Chern validity disagree at "
                f"n_k={n_k}, b1={row['b1']}, u1={row['u1']}"
            )
        arrays["reference_chern_valid"][iu, ib] = stored_chern_valid
        reference_direct_gaps = np.asarray(
            [
                _as_float(row.get("vp_plus_gap_mev")),
                _as_float(row.get("vp_minus_gap_mev")),
                _as_float(row.get("ivc_gap_mev")),
            ],
            dtype=float,
        )
        arrays["any_reference_direct_gap_nonpositive"][iu, ib] = bool(
            np.any(~np.isfinite(reference_direct_gaps))
            or np.any(reference_direct_gaps <= 0.0)
        )

    if len(seen) != len(rows) or len(rows) != shape[0] * shape[1]:
        raise ValueError(f"Expected a complete rectangular grid in {path}")
    return MeshGrid(n_k=n_k, rows=rows, b_values=b_values, u_values=u_values, arrays=arrays)


def _configure_axes(ax: plt.Axes, grid: MeshGrid, *, count_text: str) -> None:
    ax.set_title(rf"$n_k={grid.n_k}$" + "\n" + count_text, pad=8, linespacing=1.15)
    ax.set_xlabel(r"$b_1/\omega_c$")
    ax.set_xlim(float(grid.b_values.min()), float(grid.b_values.max()))
    ax.set_ylim(float(grid.u_values.min()), float(grid.u_values.max()))
    ax.set_xticks([-0.10, -0.05, 0.00, 0.05, 0.10])
    ax.set_yticks([-0.10, -0.05, 0.00, 0.05, 0.10])
    ax.set_box_aspect(1.0)


def _new_figure(title: str) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(1, 5, figsize=FIGURE["size"], sharex=True, sharey=True)
    fig.subplots_adjust(**FIGURE["subplots_adjust"])
    fig.suptitle(title, y=0.93, fontsize=FONTS["suptitle"])
    axes[0].set_ylabel(r"$u_1/\omega_c$")
    return fig, axes


def _save_figure(fig: plt.Figure, output: Path) -> tuple[Path, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = output.with_suffix(".pdf")
    fig.savefig(output, dpi=FIGURE["dpi"], bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output, pdf


def _render_cg(grids: list[MeshGrid], output: Path) -> tuple[Path, Path]:
    finite = np.concatenate(
        [grid.arrays["cG"][np.isfinite(grid.arrays["cG"])] for grid in grids]
    )
    if not finite.size:
        raise ValueError("No finite cG values are available")
    vmax = max(float(np.max(np.abs(finite))), 1.0e-12)
    cmap = LinearSegmentedColormap.from_list(
        "bootstrap_teal_red",
        [COLORS["teal"], COLORS["white_center"], COLORS["red"]],
        N=256,
    )
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    fig, axes = _new_figure(r"Conjugate AC response coefficient $c_G$")
    plotted = None
    for ax, grid in zip(axes, grids, strict=True):
        values = grid.arrays["cG"]
        valid = np.isfinite(values)
        plotted = ax.pcolormesh(
            _edges(grid.b_values),
            _edges(grid.u_values),
            np.ma.masked_where(~valid, values),
            shading="auto",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        if np.any(~valid):
            ax.pcolormesh(
                _edges(grid.b_values),
                _edges(grid.u_values),
                np.ma.masked_where(valid, np.ones_like(values)),
                shading="auto",
                cmap=ListedColormap([COLORS["grey_mask"]]),
                vmin=0.0,
                vmax=1.0,
            )
        _configure_axes(ax, grid, count_text=rf"{np.count_nonzero(valid)}/441 responses")
    cax = fig.add_axes(FIGURE["colorbar"])
    cbar = fig.colorbar(plotted, cax=cax)
    cbar.ax.tick_params(labelsize=FONTS["colorbar_tick"])
    cbar.ax.text(
        1.0,
        1.025,
        r"$c_G$",
        transform=cbar.ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONTS["colorbar_label"],
        color=COLORS["axis"],
    )
    fig.legend(
        handles=[Patch(facecolor=COLORS["grey_mask"], label="response not evaluated")],
        loc="lower right",
        bbox_to_anchor=(0.982, 0.075),
        frameon=False,
    )
    return _save_figure(fig, output)


def _render_discrete(
    grids: list[MeshGrid],
    output: Path,
    *,
    kind: Literal["negative_indirect_gap", "hf_nonconverged", "invalid_chern"],
) -> tuple[Path, Path]:
    if kind == "negative_indirect_gap":
        title = r"Negative minimum indirect gap along the interpolation"
        colors = [COLORS["grey_mask"], COLORS["white_center"], COLORS["teal"]]
        bounds = [-1.5, -0.5, 0.5, 1.5]
        ticks = [-1, 0, 1]
        ticklabels = ["not evaluated", r"$\Delta_{\rm ind}^{\min}\geq0$", r"$\Delta_{\rm ind}^{\min}<0$"]
    elif kind == "hf_nonconverged":
        title = "Self-consistent HF convergence"
        colors = [COLORS["white_center"], COLORS["red"]]
        bounds = [-0.5, 0.5, 1.5]
        ticks = [0, 1]
        ticklabels = ["all converged", "HF not converged"]
    else:
        title = "Self-consistent reference-Chern validity"
        colors = [COLORS["white_center"], COLORS["purple"]]
        bounds = [-0.5, 0.5, 1.5]
        ticks = [0, 1]
        ticklabels = ["valid", "invalid Chern"]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)
    fig, axes = _new_figure(title)
    plotted = None
    for ax, grid in zip(axes, grids, strict=True):
        if kind == "negative_indirect_gap":
            gap = grid.arrays["path_indirect_gap_min_mev"]
            evaluated = np.isfinite(gap)
            values = np.full(gap.shape, -1.0, dtype=float)
            values[evaluated] = (gap[evaluated] < 0.0).astype(float)
            count = int(np.count_nonzero(values == 1.0))
            count_text = rf"{count} negative"
        elif kind == "hf_nonconverged":
            values = (~grid.arrays["hf_all_converged"]).astype(float)
            count = int(np.count_nonzero(values))
            count_text = rf"{count} not converged"
        else:
            values = (~grid.arrays["reference_chern_valid"]).astype(float)
            count = int(np.count_nonzero(values))
            count_text = rf"{count} invalid"
        plotted = ax.pcolormesh(
            _edges(grid.b_values),
            _edges(grid.u_values),
            values,
            shading="auto",
            cmap=cmap,
            norm=norm,
            rasterized=True,
        )
        _configure_axes(ax, grid, count_text=count_text)

    cax = fig.add_axes(FIGURE["colorbar"])
    cbar = fig.colorbar(plotted, cax=cax, boundaries=bounds, ticks=ticks)
    cbar.ax.set_yticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=FONTS["colorbar_tick"], length=0, pad=7)
    return _save_figure(fig, output)


def _write_plot_data(output_stem: Path, grids: list[MeshGrid]) -> Path:
    output = output_stem.with_name(f"{output_stem.name}_diagnostic_plot_data.csv")
    fields = [
        "n_k",
        "b_index",
        "u_index",
        "b1",
        "u1",
        "status",
        "response_status",
        "cG",
        "path_indirect_gap_min_mev",
        "negative_path_indirect_gap",
        "hf_all_converged",
        "hf_nonconverged",
        "reference_chern_valid",
        "invalid_reference_chern",
        "chern_vp_plus",
        "chern_vp_minus",
        "chern_ivc",
        "vp_plus_gap_mev",
        "vp_minus_gap_mev",
        "ivc_gap_mev",
        "any_reference_direct_gap_nonpositive",
        "invalid_chern_with_nonpositive_reference_direct_gap",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for grid in grids:
            for row in grid.rows:
                gap = _as_float(row.get("path_indirect_gap_min_mev"))
                reference_gaps = [
                    _as_float(row.get("vp_plus_gap_mev")),
                    _as_float(row.get("vp_minus_gap_mev")),
                    _as_float(row.get("ivc_gap_mev")),
                ]
                nonpositive_direct = any(
                    not np.isfinite(value) or value <= 0.0 for value in reference_gaps
                )
                invalid_chern = not _as_bool(row.get("reference_chern_valid"))
                writer.writerow(
                    {
                        "n_k": grid.n_k,
                        "b_index": row.get("b_index", ""),
                        "u_index": row.get("u_index", ""),
                        "b1": row.get("b1", ""),
                        "u1": row.get("u1", ""),
                        "status": row.get("status", ""),
                        "response_status": row.get("response_status", ""),
                        "cG": row.get("cG", ""),
                        "path_indirect_gap_min_mev": row.get(
                            "path_indirect_gap_min_mev", ""
                        ),
                        "negative_path_indirect_gap": bool(np.isfinite(gap) and gap < 0.0),
                        "hf_all_converged": _as_bool(row.get("hf_all_converged")),
                        "hf_nonconverged": not _as_bool(row.get("hf_all_converged")),
                        "reference_chern_valid": not invalid_chern,
                        "invalid_reference_chern": invalid_chern,
                        "chern_vp_plus": row.get("chern_vp_plus", ""),
                        "chern_vp_minus": row.get("chern_vp_minus", ""),
                        "chern_ivc": row.get("chern_ivc", ""),
                        "vp_plus_gap_mev": row.get("vp_plus_gap_mev", ""),
                        "vp_minus_gap_mev": row.get("vp_minus_gap_mev", ""),
                        "ivc_gap_mev": row.get("ivc_gap_mev", ""),
                        "any_reference_direct_gap_nonpositive": nonpositive_direct,
                        "invalid_chern_with_nonpositive_reference_direct_gap": bool(
                            invalid_chern and nonpositive_direct
                        ),
                    }
                )
    return output


def _mesh_summary(grid: MeshGrid) -> MeshDiagnosticSummary:
    invalid = ~grid.arrays["reference_chern_valid"]
    bad_direct = grid.arrays["any_reference_direct_gap_nonpositive"]
    invalid_rows = [row for row in grid.rows if not _as_bool(row["reference_chern_valid"])]
    all_invalid_reference_gaps = [
        _as_float(row[key])
        for row in invalid_rows
        for key in ("vp_plus_gap_mev", "vp_minus_gap_mev", "ivc_gap_mev")
    ]
    invalid_ivc_gaps = [_as_float(row["ivc_gap_mev"]) for row in invalid_rows]
    path_gap = grid.arrays["path_indirect_gap_min_mev"]
    return MeshDiagnosticSummary(
        n_k=grid.n_k,
        n_points=len(grid.rows),
        n_cg=int(np.count_nonzero(np.isfinite(grid.arrays["cG"]))),
        n_negative_path_indirect_gap=int(
            np.count_nonzero(np.isfinite(path_gap) & (path_gap < 0.0))
        ),
        n_path_gap_not_evaluated=int(np.count_nonzero(~np.isfinite(path_gap))),
        n_hf_nonconverged=int(np.count_nonzero(~grid.arrays["hf_all_converged"])),
        n_invalid_reference_chern=int(np.count_nonzero(invalid)),
        n_invalid_chern_with_nonpositive_reference_direct_gap=int(
            np.count_nonzero(invalid & bad_direct)
        ),
        minimum_reference_direct_gap_mev_at_invalid_chern=(
            float(np.min(all_invalid_reference_gaps))
            if all_invalid_reference_gaps
            else None
        ),
        minimum_ivc_direct_gap_mev_at_invalid_chern=(
            float(np.min(invalid_ivc_gaps)) if invalid_ivc_gaps else None
        ),
    )


def render_all(params: ACFiniteSizeDiagnosticPlotParams) -> DiagnosticPlotSummary:
    _apply_style()
    input_root = params.resolved_input_root()
    output_stem = params.resolved_output_stem()
    grids = [
        _load_mesh(
            input_root / f"nk{n_k}" / "sweep.csv",
            n_k,
            params.chern_tolerance,
        )
        for n_k in params.meshes
    ]
    first_b = grids[0].b_values
    first_u = grids[0].u_values
    for grid in grids[1:]:
        if not np.array_equal(grid.b_values, first_b) or not np.array_equal(
            grid.u_values, first_u
        ):
            raise ValueError("All finite-size meshes must use the same b1-u1 scan grid")

    outputs: dict[str, str] = {}
    render_specs = {
        "cg": ("_cg_phase_diagrams.png", _render_cg),
        "negative_indirect_gap": (
            "_negative_indirect_gap_phase_diagrams.png",
            lambda g, o: _render_discrete(g, o, kind="negative_indirect_gap"),
        ),
        "hf_nonconverged": (
            "_hf_nonconverged_phase_diagrams.png",
            lambda g, o: _render_discrete(g, o, kind="hf_nonconverged"),
        ),
        "invalid_chern": (
            "_invalid_chern_phase_diagrams.png",
            lambda g, o: _render_discrete(g, o, kind="invalid_chern"),
        ),
    }
    for name, (suffix, renderer) in render_specs.items():
        png = output_stem.with_name(f"{output_stem.name}{suffix}")
        rendered_png, rendered_pdf = renderer(grids, png)
        outputs[f"{name}_png"] = str(rendered_png)
        outputs[f"{name}_pdf"] = str(rendered_pdf)

    plot_data = _write_plot_data(output_stem, grids)
    outputs["plot_data_csv"] = str(plot_data)
    summary_path = output_stem.with_name(f"{output_stem.name}_diagnostic_summary.json")
    outputs["summary_json"] = str(summary_path)
    summary = DiagnosticPlotSummary(
        input_root=str(input_root),
        chern_tolerance=params.chern_tolerance,
        direct_gap_overlap_definition=(
            "invalid reference Chern and any of VP+, VP-, or IVC self-consistent "
            "direct gaps is nonfinite or <= 0 meV"
        ),
        meshes=tuple(_mesh_summary(grid) for grid in grids),
        outputs=outputs,
    )
    summary_path.write_text(summary.model_dump_json(indent=2))
    return summary


def _build_parser() -> argparse.ArgumentParser:
    defaults = ACFiniteSizeDiagnosticPlotParams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=defaults.input_root)
    parser.add_argument("--output-stem", type=Path, default=defaults.output_stem)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = render_all(
        ACFiniteSizeDiagnosticPlotParams(
            input_root=args.input_root,
            output_stem=args.output_stem,
        )
    )
    print(json.dumps(summary.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
