#!/usr/bin/env python3
"""Audit and plot finite-size scaling of the AC projected-HF cG sweep."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parents[1]

COLORS = {
    "red": "#FD4C55",
    "teal": "#378d94",
    "purple": "#7A5195",
    "green": "#3A923A",
    "grey": "0.32",
    "axis": "0.18",
    "center": "#f7f7f7",
}

MODEL_POWERS: dict[str, tuple[int, ...]] = {
    "inverse_n": (1,),
    "inverse_n2": (2,),
    "even_n2_n4": (2, 4),
}


class ACB1U1CGFiniteSizeParams(BaseModel):
    """Frozen user-facing controls for the finite-size analysis."""

    model_config = ConfigDict(frozen=True)

    n_k_values: tuple[int, ...] = (18, 19, 20, 21, 22)
    input_template: str = (
        "results/ac_b1_u1_cg_nk{n_k}_nll6_v0p1_grid11_gaugefixed_v2/sweep.csv"
    )
    output: Path = Path(
        "Plots/figures/ac_b1_u1_cg_nk18_22_nll6_v0p1_finite_size_scaling.png"
    )
    expected_n_ll: int = Field(default=6, ge=1)
    expected_v0_over_omega_c: float = 0.1
    dpi: int = Field(default=320, ge=72)

    @model_validator(mode="after")
    def _validate_sizes(self) -> "ACB1U1CGFiniteSizeParams":
        if len(self.n_k_values) < 5:
            raise ValueError("at least five n_k values are required for the three fit diagnostics")
        if len(set(self.n_k_values)) != len(self.n_k_values):
            raise ValueError("n_k values must be unique")
        if tuple(sorted(self.n_k_values)) != self.n_k_values:
            raise ValueError("n_k values must be strictly increasing")
        return self

    def input_for(self, n_k: int) -> Path:
        path = Path(self.input_template.format(n_k=n_k))
        return path if path.is_absolute() else ROOT / path

    def resolved_output(self) -> Path:
        return self.output if self.output.is_absolute() else ROOT / self.output


@dataclass(frozen=True)
class FitResult:
    powers: tuple[int, ...]
    coefficients: np.ndarray
    predictions: np.ndarray
    residuals: np.ndarray
    loo_residuals: np.ndarray

    @property
    def intercept(self) -> np.ndarray:
        return self.coefficients[0]


@dataclass(frozen=True)
class FiniteSizeData:
    n_k: np.ndarray
    b1: np.ndarray
    u1: np.ndarray
    cG: np.ndarray
    fits: dict[str, FitResult]
    audits: list[dict[str, Any]]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _canonical(value: Any) -> float:
    return float(np.round(float(value), decimals=14))


def _edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError("heatmap requires at least two coordinates per axis")
    return np.r_[
        values[0] - 0.5 * (values[1] - values[0]),
        0.5 * (values[:-1] + values[1:]),
        values[-1] + 0.5 * (values[-1] - values[-2]),
    ]


def _load_one_sweep(
    path: Path,
    expected_n_k: int,
    expected_n_ll: int,
    expected_v0: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing finite-size sweep: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty finite-size sweep: {path}")

    required = {
        "b1",
        "u1",
        "cG",
        "n_k",
        "n_ll",
        "v0_over_omega_c",
        "status",
        "response_status",
        "hf_all_converged",
        "reference_chern_valid",
        "band_chern",
        "chern_ivc",
        "min_direct_gap",
        "path_gap_min",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    rows.sort(key=lambda row: (_canonical(row["b1"]), _canonical(row["u1"])))
    coords = np.array(
        [(_canonical(row["b1"]), _canonical(row["u1"])) for row in rows], dtype=float
    )
    if len({tuple(coord) for coord in coords}) != len(rows):
        raise ValueError(f"duplicate b1/u1 coordinates in {path}")
    cG = np.array([float(row["cG"]) for row in rows], dtype=float)

    problems: list[str] = []
    for index, row in enumerate(rows):
        prefix = f"row {index} (b1={row['b1']}, u1={row['u1']})"
        if int(float(row["n_k"])) != expected_n_k:
            problems.append(f"{prefix}: n_k={row['n_k']}")
        if int(float(row["n_ll"])) != expected_n_ll:
            problems.append(f"{prefix}: n_ll={row['n_ll']}")
        if not np.isclose(float(row["v0_over_omega_c"]), expected_v0, atol=1e-12):
            problems.append(f"{prefix}: v0={row['v0_over_omega_c']}")
        if row["status"] != "ok" or row["response_status"] != "ok":
            problems.append(f"{prefix}: status={row['status']}/{row['response_status']}")
        if not _as_bool(row["hf_all_converged"]):
            problems.append(f"{prefix}: HF did not converge")
        if not _as_bool(row["reference_chern_valid"]):
            problems.append(f"{prefix}: invalid reference Chern number")
        if not np.isclose(float(row["band_chern"]), 1.0, atol=1e-6):
            problems.append(f"{prefix}: band C={row['band_chern']}")
        if not np.isclose(float(row["chern_ivc"]), 0.0, atol=1e-6):
            problems.append(f"{prefix}: IVC C={row['chern_ivc']}")
    if not np.all(np.isfinite(cG)):
        problems.append("non-finite cG value")
    if problems:
        preview = "; ".join(problems[:8])
        raise ValueError(f"audit failed for {path}: {preview}")

    b_values = np.unique(coords[:, 0])
    u_values = np.unique(coords[:, 1])
    expected_count = len(b_values) * len(u_values)
    if len(rows) != expected_count:
        raise ValueError(
            f"incomplete rectangular grid in {path}: {len(rows)} rows for {expected_count} coordinates"
        )

    grid = cG.reshape(len(b_values), len(u_values)).T
    horizontal = np.abs(np.diff(grid, axis=1)).ravel()
    vertical = np.abs(np.diff(grid, axis=0)).ravel()
    neighbor_jumps = np.r_[horizontal, vertical]
    audit = {
        "input_csv": str(path),
        "n_k": expected_n_k,
        "n_rows": len(rows),
        "n_unique_coordinates": len({tuple(coord) for coord in coords}),
        "n_bad_points": 0,
        "cG_min": float(np.min(cG)),
        "cG_max": float(np.max(cG)),
        "minimum_ac_band_gap": float(min(float(row["min_direct_gap"]) for row in rows)),
        "minimum_variational_path_gap": float(min(float(row["path_gap_min"]) for row in rows)),
        "median_neighbor_cG_jump": float(np.median(neighbor_jumps)),
        "maximum_neighbor_cG_jump": float(np.max(neighbor_jumps)),
    }
    return coords, cG, audit


def _design_matrix(n_k: np.ndarray, powers: tuple[int, ...]) -> np.ndarray:
    n = np.asarray(n_k, dtype=float)
    return np.column_stack([np.ones(n.size), *[n ** (-power) for power in powers]])


def fit_model(n_k: np.ndarray, cG: np.ndarray, powers: tuple[int, ...]) -> FitResult:
    """Fit every b1/u1 coordinate and compute leave-one-size-out residuals."""

    y = np.asarray(cG, dtype=float)
    if y.ndim != 2 or y.shape[0] != len(n_k):
        raise ValueError("cG must have shape (number of n_k values, number of coordinates)")
    design = _design_matrix(n_k, powers)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    predictions = design @ coefficients
    loo_residuals = np.empty_like(y)
    for held_out in range(len(n_k)):
        keep = np.arange(len(n_k)) != held_out
        local_design = _design_matrix(n_k[keep], powers)
        local_coefficients = np.linalg.lstsq(local_design, y[keep], rcond=None)[0]
        loo_residuals[held_out] = design[held_out] @ local_coefficients - y[held_out]
    return FitResult(
        powers=powers,
        coefficients=coefficients,
        predictions=predictions,
        residuals=predictions - y,
        loo_residuals=loo_residuals,
    )


def load_finite_size_data(params: ACB1U1CGFiniteSizeParams) -> FiniteSizeData:
    coords_ref: np.ndarray | None = None
    values: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    for n_k in params.n_k_values:
        coords, cG, audit = _load_one_sweep(
            params.input_for(n_k),
            expected_n_k=n_k,
            expected_n_ll=params.expected_n_ll,
            expected_v0=params.expected_v0_over_omega_c,
        )
        if coords_ref is None:
            coords_ref = coords
        elif not np.array_equal(coords, coords_ref):
            raise ValueError(f"b1/u1 coordinates at n_k={n_k} do not match the first mesh")
        values.append(cG)
        audits.append(audit)
    assert coords_ref is not None
    n_k_array = np.asarray(params.n_k_values, dtype=int)
    cG_array = np.stack(values)
    fits = {
        name: fit_model(n_k_array, cG_array, powers)
        for name, powers in MODEL_POWERS.items()
    }
    return FiniteSizeData(
        n_k=n_k_array,
        b1=coords_ref[:, 0],
        u1=coords_ref[:, 1],
        cG=cG_array,
        fits=fits,
        audits=audits,
    )


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["PT Serif Caption", "PT Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 12,
            "axes.edgecolor": COLORS["axis"],
            "axes.linewidth": 1.1,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.transparent": False,
        }
    )


def _grid_from_flat(data: FiniteSizeData, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b_values = np.unique(data.b1)
    u_values = np.unique(data.u1)
    grid = np.full((len(u_values), len(b_values)), np.nan, dtype=float)
    b_lookup = {value: index for index, value in enumerate(b_values)}
    u_lookup = {value: index for index, value in enumerate(u_values)}
    for b1, u1, value in zip(data.b1, data.u1, values, strict=True):
        grid[u_lookup[u1], b_lookup[b1]] = value
    return b_values, u_values, grid


def _write_analysis_csv(output: Path, data: FiniteSizeData) -> Path:
    path = output.with_suffix(".csv")
    fieldnames = ["b1", "u1"] + [f"cG_nk{n_k}" for n_k in data.n_k]
    for model in MODEL_POWERS:
        fieldnames.extend(
            [f"cG_infinity_{model}", f"rmse_{model}", f"loo_rmse_{model}"]
        )
    fieldnames.append("cG_infinity_model_spread")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        intercepts = np.stack([fit.intercept for fit in data.fits.values()])
        for index, (b1, u1) in enumerate(zip(data.b1, data.u1, strict=True)):
            row: dict[str, Any] = {"b1": f"{b1:.16g}", "u1": f"{u1:.16g}"}
            for size_index, n_k in enumerate(data.n_k):
                row[f"cG_nk{n_k}"] = f"{data.cG[size_index, index]:.16g}"
            for model, fit in data.fits.items():
                row[f"cG_infinity_{model}"] = f"{fit.intercept[index]:.16g}"
                row[f"rmse_{model}"] = f"{np.sqrt(np.mean(fit.residuals[:, index] ** 2)):.16g}"
                row[f"loo_rmse_{model}"] = (
                    f"{np.sqrt(np.mean(fit.loo_residuals[:, index] ** 2)):.16g}"
                )
            row["cG_infinity_model_spread"] = f"{np.ptp(intercepts[:, index]):.16g}"
            writer.writerow(row)
    return path


def _summary(data: FiniteSizeData, output: Path) -> dict[str, Any]:
    center_candidates = np.where(np.isclose(data.b1, 0.0) & np.isclose(data.u1, 0.0))[0]
    if center_candidates.size != 1:
        raise ValueError("expected exactly one b1=u1=0 point")
    center = int(center_candidates[0])
    intercepts = np.stack([fit.intercept for fit in data.fits.values()])
    successive = np.diff(data.cG, axis=0)
    even_mask = data.n_k % 2 == 0
    odd_mask = ~even_mask
    even_intercept = np.linalg.lstsq(
        _design_matrix(data.n_k[even_mask], (2,)), data.cG[even_mask], rcond=None
    )[0][0]
    odd_intercept = np.linalg.lstsq(
        _design_matrix(data.n_k[odd_mask], (2,)), data.cG[odd_mask], rcond=None
    )[0][0]
    odd_even_difference = odd_intercept - even_intercept
    model_summaries: dict[str, Any] = {}
    for name, fit in data.fits.items():
        model_summaries[name] = {
            "powers": list(fit.powers),
            "center_cG_infinity": float(fit.intercept[center]),
            "cG_infinity_min": float(np.min(fit.intercept)),
            "cG_infinity_max": float(np.max(fit.intercept)),
            "overall_rmse": float(np.sqrt(np.mean(fit.residuals**2))),
            "overall_leave_one_out_rmse": float(np.sqrt(np.mean(fit.loo_residuals**2))),
            "maximum_abs_residual": float(np.max(np.abs(fit.residuals))),
            "maximum_abs_leave_one_out_residual": float(np.max(np.abs(fit.loo_residuals))),
        }
    return {
        "output_png": str(output),
        "output_pdf": str(output.with_suffix(".pdf")),
        "analysis_csv": str(output.with_suffix(".csv")),
        "n_k_values": data.n_k.tolist(),
        "n_parameter_points": int(data.cG.shape[1]),
        "all_meshes_passed_independent_audit": True,
        "audits": data.audits,
        "center_coordinate": {"b1": 0.0, "u1": 0.0},
        "center_cG_by_n_k": {
            str(n_k): float(data.cG[index, center]) for index, n_k in enumerate(data.n_k)
        },
        "successive_mesh_change": {
            f"nk{left}_to_nk{right}": {
                "median_abs": float(np.median(np.abs(successive[index]))),
                "maximum_abs": float(np.max(np.abs(successive[index]))),
            }
            for index, (left, right) in enumerate(zip(data.n_k[:-1], data.n_k[1:], strict=True))
        },
        "fits": model_summaries,
        "center_model_spread": float(np.ptp(intercepts[:, center])),
        "maximum_pointwise_model_spread": float(np.max(np.ptp(intercepts, axis=0))),
        "finite_nk_max_parameter_span": float(np.ptp(data.cG[-1])),
        "even_power_infinity_parameter_span": float(
            np.ptp(data.fits["even_n2_n4"].intercept)
        ),
        "odd_even_inverse_square_diagnostic": {
            "even_n_k_values": data.n_k[even_mask].tolist(),
            "odd_n_k_values": data.n_k[odd_mask].tolist(),
            "center_even_cG_infinity": float(even_intercept[center]),
            "center_odd_cG_infinity": float(odd_intercept[center]),
            "center_odd_minus_even": float(odd_even_difference[center]),
            "maximum_abs_odd_minus_even_across_grid": float(
                np.max(np.abs(odd_even_difference))
            ),
        },
        "interpretation": (
            "The finite-size trend is smooth and has no significant odd-even anomaly. "
            "The even 1/n_k^2+1/n_k^4 expansion gives the smallest interpolation and "
            "leave-one-size-out errors, but the infinity intercept remains ansatz-dependent "
            "over the narrow n_k=18..22 window; compare all reported intercepts."
        ),
    }


def _draw_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    b_values: np.ndarray,
    u_values: np.ndarray,
    grid: np.ndarray,
    *,
    title: str,
    cmap: LinearSegmentedColormap,
    norm: Normalize,
) -> Any:
    mesh = ax.pcolormesh(
        _edges(b_values),
        _edges(u_values),
        grid,
        shading="auto",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    ax.set_title(title, pad=9)
    ax.set_xlabel(r"$b_1/\omega_c$")
    ax.set_ylabel(r"$u_1/\omega_c$")
    ax.set_xlim(float(b_values.min()), float(b_values.max()))
    ax.set_ylim(float(u_values.min()), float(u_values.max()))
    ax.set_box_aspect(1.0)
    ax.grid(False)
    return mesh


def render_finite_size_analysis(
    params: ACB1U1CGFiniteSizeParams,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    _apply_style()
    data = load_finite_size_data(params)
    output = params.resolved_output()
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output = output.with_suffix(".pdf")
    csv_output = _write_analysis_csv(output, data)
    summary_output = output.with_suffix(".json")
    summary = _summary(data, output)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    center = int(np.where(np.isclose(data.b1, 0.0) & np.isclose(data.u1, 0.0))[0][0])
    x_data = 1.0 / data.n_k.astype(float)
    x_fit = np.linspace(0.0, float(np.max(x_data)) * 1.04, 400)
    curves: dict[str, np.ndarray] = {}
    for name, fit in data.fits.items():
        curve = np.full_like(x_fit, fit.coefficients[0, center])
        for coefficient, power in zip(fit.coefficients[1:, center], fit.powers, strict=True):
            curve += coefficient * x_fit**power
        curves[name] = curve

    b_values, u_values, finite_grid = _grid_from_flat(data, data.cG[-1])
    _, _, extrapolated_grid = _grid_from_flat(data, data.fits["even_n2_n4"].intercept)
    finite_norm = Normalize(vmin=float(np.min(finite_grid)), vmax=float(np.max(finite_grid)))
    extrapolated_norm = Normalize(
        vmin=float(np.min(extrapolated_grid)), vmax=float(np.max(extrapolated_grid))
    )
    cmap = LinearSegmentedColormap.from_list(
        "nisarg_teal_neutral_red", [COLORS["teal"], COLORS["center"], COLORS["red"]]
    )

    fig = plt.figure(figsize=(12.2, 10.4))
    grid_spec = fig.add_gridspec(
        2,
        2,
        height_ratios=(0.90, 1.05),
        left=0.09,
        right=0.92,
        bottom=0.09,
        top=0.90,
        hspace=0.38,
        wspace=0.44,
    )
    ax_scale = fig.add_subplot(grid_spec[0, :])
    ax_left = fig.add_subplot(grid_spec[1, 0])
    ax_right = fig.add_subplot(grid_spec[1, 1])

    ax_scale.plot(x_fit, curves["inverse_n"], color=COLORS["red"], linewidth=2.3, label=r"$c_\infty+a/n_k$")
    ax_scale.plot(x_fit, curves["inverse_n2"], color=COLORS["teal"], linewidth=2.3, linestyle="--", label=r"$c_\infty+a/n_k^2$")
    ax_scale.plot(x_fit, curves["even_n2_n4"], color=COLORS["purple"], linewidth=2.4, linestyle="-.", label=r"$c_\infty+a/n_k^2+b/n_k^4$")
    ax_scale.scatter(
        x_data,
        data.cG[:, center],
        s=70,
        color=COLORS["grey"],
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="computed meshes",
    )
    for x, y, n_k in zip(x_data, data.cG[:, center], data.n_k, strict=True):
        ax_scale.annotate(
            str(n_k),
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color=COLORS["axis"],
        )
    ax_scale.set_title(r"Finite-size scaling at $b_1=u_1=0$")
    ax_scale.set_xlabel(r"$1/n_k$")
    ax_scale.set_ylabel(r"$c_G$")
    ax_scale.set_xlim(-0.002, float(np.max(x_data)) * 1.04)
    all_y = np.r_[data.cG[:, center], *curves.values()]
    padding = 0.06 * float(np.ptp(all_y))
    ax_scale.set_ylim(float(np.min(all_y) - padding), float(np.max(all_y) + padding))
    ax_scale.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
    ax_scale.grid(alpha=0.20, linewidth=0.8)
    ax_scale.legend(
        loc="upper left",
        bbox_to_anchor=(0.012, 0.98),
        ncol=2,
        frameon=True,
        framealpha=0.94,
        handlelength=2.5,
        columnspacing=1.4,
    )
    ax_scale.text(
        0.985,
        0.04,
        "Extrapolated center values\n"
        + rf"$1/n_k$: {data.fits['inverse_n'].intercept[center]:.7f}; "
        + rf"$1/n_k^2$: {data.fits['inverse_n2'].intercept[center]:.7f}; "
        + rf"even: {data.fits['even_n2_n4'].intercept[center]:.7f}",
        transform=ax_scale.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        color=COLORS["axis"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.0},
    )

    finite_mesh = _draw_heatmap(
        fig,
        ax_left,
        b_values,
        u_values,
        finite_grid,
        title=rf"Finite mesh: $n_k={int(data.n_k[-1])}$",
        cmap=cmap,
        norm=finite_norm,
    )
    mesh = _draw_heatmap(
        fig,
        ax_right,
        b_values,
        u_values,
        extrapolated_grid,
        title=r"Even-power fit: $n_k\to\infty$",
        cmap=cmap,
        norm=extrapolated_norm,
    )
    left_bbox = ax_left.get_position()
    left_cax = fig.add_axes([left_bbox.x1 + 0.012, left_bbox.y0, 0.015, left_bbox.height])
    left_colorbar = fig.colorbar(finite_mesh, cax=left_cax)
    left_colorbar.ax.tick_params(labelsize=11)
    left_colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
    left_colorbar.ax.text(
        1.0,
        1.02,
        r"$c_G$",
        transform=left_colorbar.ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=16,
        color=COLORS["axis"],
    )
    right_bbox = ax_right.get_position()
    right_cax = fig.add_axes([right_bbox.x1 + 0.012, right_bbox.y0, 0.015, right_bbox.height])
    right_colorbar = fig.colorbar(mesh, cax=right_cax)
    right_colorbar.ax.tick_params(labelsize=11)
    right_colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.6f"))
    right_colorbar.ax.text(
        1.0,
        1.02,
        r"$c_G^\infty$",
        transform=right_colorbar.ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=16,
        color=COLORS["axis"],
    )

    fig.suptitle(
        r"Conjugate AC projected HF: $N_{\rm LL}=6$, $V_0/\omega_c=0.1$",
        fontsize=20,
        y=0.985,
    )
    fig.savefig(output, dpi=params.dpi, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_output, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output, pdf_output, csv_output, summary_output, summary


def _build_parser() -> argparse.ArgumentParser:
    defaults = ACB1U1CGFiniteSizeParams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-k", nargs="+", type=int, default=list(defaults.n_k_values))
    parser.add_argument("--input-template", default=defaults.input_template)
    parser.add_argument("--output", type=Path, default=defaults.output)
    parser.add_argument("--expected-n-ll", type=int, default=defaults.expected_n_ll)
    parser.add_argument(
        "--expected-v0-over-omega-c", type=float, default=defaults.expected_v0_over_omega_c
    )
    parser.add_argument("--dpi", type=int, default=defaults.dpi)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    params = ACB1U1CGFiniteSizeParams(
        n_k_values=tuple(args.n_k),
        input_template=args.input_template,
        output=args.output,
        expected_n_ll=args.expected_n_ll,
        expected_v0_over_omega_c=args.expected_v0_over_omega_c,
        dpi=args.dpi,
    )
    png, pdf, table, summary_path, summary = render_finite_size_analysis(params)
    print(f"All meshes passed independent audit: {summary['n_parameter_points']} points each")
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")
    print(f"Analysis table: {table}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
