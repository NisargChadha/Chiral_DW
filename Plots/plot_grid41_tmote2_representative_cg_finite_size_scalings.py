#!/usr/bin/env python3
"""Representative all-seven finite-size cG scalings for the grid41 tMoTe2 run."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = (
    ROOT
    / "results/taige_ivc_hysteresis_finite_size_nk18_24_grid41_linear_interaction_recomputed_sharded"
)
FIT_SUMMARY_CSV = (
    RESULT_DIR / "analysis_plots/grid41_linear_interaction_lowest_clean_best5_cG_fits.csv"
)
FIT_SOURCE_CSV = RESULT_DIR / "hysteresis_finite_size_fit_source.csv"
BOUNDARY_CSV = (
    RESULT_DIR / "analysis_plots/grid41_linear_interaction_best5_cG_heatmap_with_boundaries.csv"
)
FIGURE_DIR = ROOT / "Plots/figures"

COLORS = {
    "red": "#FD4C55",
    "teal": "#378d94",
    "purple": "#6a408d",
    "green": "#4D9221",
    "grid": "#D7D7D7",
    "axis": "#1A1A1A",
}

NONLINEAR_INV_N_TICKS = np.array([0.0] + [1.0 / n_k for n_k in range(24, 17, -1)])
NONLINEAR_X_TICKS = np.arange(len(NONLINEAR_INV_N_TICKS), dtype=float)
NONLINEAR_X_LABELS = ["0"] + [rf"$1/{n_k}$" for n_k in range(24, 17, -1)]


class RepresentativeRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str
    theta_deg: float
    u_D_meV: float
    color: str
    marker: str


class RepresentativeScalingParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    fit_summary_csv: Path = FIT_SUMMARY_CSV
    fit_source_csv: Path = FIT_SOURCE_CSV
    boundary_csv: Path = BOUNDARY_CSV
    output_dir: Path = FIGURE_DIR
    output_stem: str = "grid41_tmote2_representative_all7_cG_finite_size_scalings"
    requested_points: tuple[RepresentativeRequest, ...] = (
        RepresentativeRequest(
            role="good",
            theta_deg=3.0,
            u_D_meV=0.0,
            color="teal",
            marker="s",
        ),
        RepresentativeRequest(
            role="intermediate",
            theta_deg=3.5,
            u_D_meV=7.0,
            color="purple",
            marker="^",
        ),
        RepresentativeRequest(
            role="VP, C=0",
            theta_deg=3.7,
            u_D_meV=10.0,
            color="green",
            marker="D",
        ),
    )


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["PT Serif", "Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 9,
            "axes.linewidth": 1.1,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def nearest_row(df: pd.DataFrame, theta_deg: float, u_D_meV: float) -> pd.Series:
    distance = (df["theta_deg"] - theta_deg).abs() + (df["u_D_meV"] - u_D_meV).abs()
    idx = distance.idxmin()
    row = df.loc[idx]
    if not (
        np.isclose(row["theta_deg"], theta_deg, atol=1.0e-9)
        and np.isclose(row["u_D_meV"], u_D_meV, atol=1.0e-9)
    ):
        raise ValueError(
            f"Requested point ({theta_deg}, {u_D_meV}) not present; "
            f"nearest is ({row['theta_deg']}, {row['u_D_meV']})."
        )
    return row


def load_summary(params: RepresentativeScalingParams) -> pd.DataFrame:
    summary = pd.read_csv(params.fit_summary_csv)
    boundaries = pd.read_csv(params.boundary_csv)
    keep = ["theta_deg", "u_D_meV", "grey_mask_all"]
    summary = summary.merge(boundaries[keep], on=["theta_deg", "u_D_meV"], how="left")
    summary["grey_mask_all"] = to_bool(summary["grey_mask_all"]).fillna(False)
    return summary


def load_source(params: RepresentativeScalingParams) -> pd.DataFrame:
    source = pd.read_csv(params.fit_source_csv)
    mask = (
        (source["trial_interpolation"] == "linear_interaction")
        & (source["branch_label"] == "lowest_energy_clean")
        & (source["selection_kind"] == "clean")
        & to_bool(source["clean"])
    )
    return source.loc[mask].copy()


def select_representatives(
    summary: pd.DataFrame, params: RepresentativeScalingParams
) -> list[dict[str, object]]:
    valid = (
        (summary["cG_all_status"] == "fit_ok")
        & np.isfinite(summary["cG_all_rmse"])
        & ~summary["grey_mask_all"]
    )
    worst = summary.loc[valid].sort_values("cG_all_rmse", ascending=False).iloc[0]
    representatives: list[dict[str, object]] = [
        {"role": "worst RMSE", "color": "red", "marker": "o", "row": worst}
    ]
    for request in params.requested_points:
        representatives.append(
            {
                "role": request.role,
                "color": request.color,
                "marker": request.marker,
                "row": nearest_row(summary, request.theta_deg, request.u_D_meV),
            }
        )
    return representatives


def source_for_point(source: pd.DataFrame, theta_deg: float, u_D_meV: float) -> pd.DataFrame:
    rows = source.loc[
        np.isclose(source["theta_deg"], theta_deg, atol=1.0e-9)
        & np.isclose(source["u_D_meV"], u_D_meV, atol=1.0e-9)
    ].copy()
    rows = rows.sort_values("n_k")
    if len(rows) != 7:
        raise ValueError(
            f"Expected seven finite-size points at ({theta_deg}, {u_D_meV}), found {len(rows)}."
        )
    return rows


def display_x(inv_n_k: np.ndarray | pd.Series | float) -> np.ndarray:
    return np.interp(
        np.asarray(inv_n_k, dtype=float),
        NONLINEAR_INV_N_TICKS,
        NONLINEAR_X_TICKS,
    )


def build_plot(params: RepresentativeScalingParams) -> None:
    apply_style()
    params.output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(params)
    source = load_source(params)
    representatives = select_representatives(summary, params)

    fig, ax = plt.subplots(figsize=(8.3, 5.8))
    plotted_rows: list[dict[str, object]] = []
    y_values: list[float] = []
    x_fit = np.linspace(0.0, 1.0 / 18.0, 240)
    x_fit_display = display_x(x_fit)

    for representative in representatives:
        role = str(representative["role"])
        row = representative["row"]
        if not isinstance(row, pd.Series):
            raise TypeError("Representative row must be a pandas Series.")
        color = COLORS[str(representative["color"])]
        marker = str(representative["marker"])
        theta_deg = float(row["theta_deg"])
        u_D_meV = float(row["u_D_meV"])
        points = source_for_point(source, theta_deg, u_D_meV)

        intercept = float(row["cG_all_intercept"])
        slope = float(row["cG_all_slope"])
        fit_y = intercept + slope * x_fit
        point_x = display_x(points["inv_n_k"])
        y_values.extend(points["cG"].astype(float).tolist())
        y_values.extend(fit_y.tolist())
        y_values.append(intercept)

        ax.plot(x_fit_display, fit_y, color=color, lw=2.0, ls="--", alpha=0.95)
        ax.plot(
            point_x,
            points["cG"],
            ls="none",
            marker=marker,
            ms=6.3,
            mfc=color,
            mec="white",
            mew=0.8,
            color=color,
            zorder=3,
        )
        ax.plot(
            [display_x(0.0)[()]],
            [intercept],
            ls="none",
            marker=marker,
            ms=6.2,
            mfc="white",
            mec=color,
            mew=1.5,
            color=color,
            zorder=4,
        )

        for _, point in points.iterrows():
            point_display_x = float(display_x(point["inv_n_k"])[()])
            plotted_rows.append(
                {
                    "role": role,
                    "theta_deg": theta_deg,
                    "u_D_meV": u_D_meV,
                    "n_k": int(point["n_k"]),
                    "inv_n_k": float(point["inv_n_k"]),
                    "display_x": point_display_x,
                    "cG": float(point["cG"]),
                    "source_branch": point.get("source_branch", ""),
                    "cG_all_intercept": intercept,
                    "cG_all_slope": slope,
                    "cG_all_rmse": float(row["cG_all_rmse"]),
                    "fit_cG_at_inv_n_k": float(intercept + slope * point["inv_n_k"]),
                }
            )

    y_min = float(np.nanmin(y_values))
    y_max = float(np.nanmax(y_values))
    y_pad = 0.07 * max(y_max - y_min, 1.0e-3)
    ax.set_xlim(-0.16, NONLINEAR_X_TICKS[-1] + 0.16)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel(r"$1/n_k$")
    ax.set_ylabel(r"$c_G$")
    ax.set_xticks(NONLINEAR_X_TICKS)
    ax.set_xticklabels(NONLINEAR_X_LABELS)
    ax.axvline(display_x(0.0)[()], color=COLORS["axis"], lw=1.0, alpha=0.45)
    ax.grid(True, color=COLORS["grid"], lw=0.7, alpha=0.45)

    png_path = params.output_dir / f"{params.output_stem}.png"
    pdf_path = params.output_dir / f"{params.output_stem}.pdf"
    svg_path = params.output_dir / f"{params.output_stem}.svg"
    csv_path = params.output_dir / f"{params.output_stem}_points.csv"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    fig.savefig(svg_path)
    plt.close(fig)
    pd.DataFrame(plotted_rows).to_csv(csv_path, index=False)

    selected = [
        (
            representative["role"],
            float(representative["row"]["theta_deg"]),
            float(representative["row"]["u_D_meV"]),
            float(representative["row"]["cG_all_rmse"]),
        )
        for representative in representatives
    ]
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {csv_path}")
    print("Selected all-seven fit points:")
    for role, theta_deg, u_D_meV, rmse in selected:
        print(f"  {role}: theta={theta_deg:.2f}, u_D={u_D_meV:.1f}, RMSE={rmse:.6g}")


def main() -> None:
    build_plot(RepresentativeScalingParams())


if __name__ == "__main__":
    main()
