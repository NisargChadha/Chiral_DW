#!/usr/bin/env python3
"""Red log-scale all-seven cG finite-size RMSE heatmap for tMoTe2."""

from __future__ import annotations

from plot_grid41_tmote2_linear_cg_rmse_heatmap import (
    CGRMSEHeatmapParams,
    COLORS,
    HeatmapMarker,
    _apply_style,
    render_rmse_heatmap,
)


OUTPUT_STEM = "grid41_tmote2_linear_interaction_all7_cG_rmse_heatmap_red_log_boundaries"


if __name__ == "__main__":
    COLORS["rmse_high"] = "#FD4C55"
    _apply_style()
    for path in render_rmse_heatmap(
        CGRMSEHeatmapParams(
            output_stem=OUTPUT_STEM,
            rmse_column="cG_all_rmse",
            status_column="cG_all_status",
            valid_statuses=("fit_ok",),
            boundary_grey_column="grey_mask_all",
            log_scale=True,
            log_vmin=1.0e-7,
            show_boundaries=True,
            mask_boundary_grey=True,
            representative_markers=(
                HeatmapMarker(
                    theta_deg=3.05,
                    u_D_meV=6.0,
                    color="rmse",
                    marker="o",
                    size=16.0,
                    edge_width=0.65,
                ),
                HeatmapMarker(
                    theta_deg=3.0,
                    u_D_meV=0.0,
                    color="#378d94",
                    marker="s",
                    size=14.0,
                    edge_width=0.65,
                ),
                HeatmapMarker(
                    theta_deg=3.5,
                    u_D_meV=7.0,
                    color="#6a408d",
                    marker="^",
                    size=16.0,
                    edge_width=0.65,
                ),
                HeatmapMarker(
                    theta_deg=3.95,
                    u_D_meV=12.5,
                    color="#4D9221",
                    marker="D",
                    size=14.0,
                    edge_width=0.65,
                ),
            ),
            show_chern_instability_crosses=True,
        )
    ):
        print(path)
