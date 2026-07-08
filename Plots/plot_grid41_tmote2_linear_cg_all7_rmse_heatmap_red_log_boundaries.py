#!/usr/bin/env python3
"""Red log-scale all-seven cG finite-size RMSE heatmap for tMoTe2."""

from __future__ import annotations

from plot_grid41_tmote2_linear_cg_rmse_heatmap import (
    CGRMSEHeatmapParams,
    COLORS,
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
            show_boundaries=True,
            mask_boundary_grey=True,
        )
    ):
        print(path)
