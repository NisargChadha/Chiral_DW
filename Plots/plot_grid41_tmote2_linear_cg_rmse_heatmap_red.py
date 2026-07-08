#!/usr/bin/env python3
"""Red-ramp variant of the tMoTe2 linear-interaction cG-fit RMSE heatmap."""

from __future__ import annotations

from plot_grid41_tmote2_linear_cg_rmse_heatmap import (
    CGRMSEHeatmapParams,
    COLORS,
    _apply_style,
    render_rmse_heatmap,
)


OUTPUT_STEM = "grid41_tmote2_linear_interaction_best5_cG_rmse_heatmap_red"


def main() -> None:
    COLORS["rmse_high"] = "#FD4C55"
    _apply_style()
    params = CGRMSEHeatmapParams(output_stem=OUTPUT_STEM)
    for path in render_rmse_heatmap(params):
        print(path)


if __name__ == "__main__":
    main()
