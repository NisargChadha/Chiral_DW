"""Render a cLLL spin texture and its numerically computed charge density."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from chiral_dw.config import (
    IdealConjugateLLLChargeBenchmarkParams,
    MomentumGridParams,
    RealSpaceGridParams,
)
from chiral_dw.ideal_conjugate_lll import (
    ExplicitConjugateLLLTextureResponseResult,
    plaquette_average,
    run_explicit_chiral_domain_wall_texture_response,
    triangular_moire_magnetic_length,
)


class CLLLSchematicPlotParams(BaseModel):
    """User-facing controls for the cLLL schematic plot."""

    model_config = ConfigDict(frozen=True)

    n_k: int = Field(default=7, ge=1)
    n_r: int = Field(default=41, ge=3)
    radius_lB: float = Field(default=10.0, gt=0.0)
    width_lB: float = Field(default=3.5, gt=0.0)
    patch_length_lB: float = Field(default=56.0, gt=0.0)
    winding: int = 1
    helicity: float = 0.0
    spin_stride: int = Field(default=2, ge=1)
    theta_count: int = Field(default=42, ge=3)
    output: Path = Path("results/plots/clll_spin_charge_schematic.png")
    dpi: int = Field(default=300, ge=72)


@dataclass(frozen=True)
class ChargeDensityGrid:
    """Numerical cLLL charge density on real-space plaquette centers."""

    x_edges: np.ndarray
    y_edges: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    rho_per_plaquette: np.ndarray
    rho_density: np.ndarray
    plaquette_area: float

    @property
    def integrated_charge_from_density(self) -> float:
        return float(np.sum(self.rho_density) * self.plaquette_area)

    @property
    def integrated_charge_from_plaquettes(self) -> float:
        return float(np.sum(self.rho_per_plaquette))


def benchmark_params_from_plot_params(
    params: CLLLSchematicPlotParams,
) -> IdealConjugateLLLChargeBenchmarkParams:
    return IdealConjugateLLLChargeBenchmarkParams(
        grid=MomentumGridParams(n_k=params.n_k),
        real_space=RealSpaceGridParams(n_r=params.n_r),
        radius_lB=params.radius_lB,
        width_lB=params.width_lB,
        patch_length_lB=params.patch_length_lB,
        winding=params.winding,
        helicity=params.helicity,
        output_dir=str(params.output.parent),
    )


def compute_clll_response(
    params: CLLLSchematicPlotParams,
) -> ExplicitConjugateLLLTextureResponseResult:
    theta_edges = np.linspace(0.0, np.pi, params.theta_count)
    return run_explicit_chiral_domain_wall_texture_response(
        benchmark_params_from_plot_params(params),
        theta_edges=theta_edges,
        phi_nodes=np.array([0.0, 0.2], dtype=float),
    )


def compute_charge_density_grid(
    result: ExplicitConjugateLLLTextureResponseResult,
) -> ChargeDensityGrid:
    xy = np.asarray(result.solution.xy, dtype=float)
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError("solution.xy must have shape (n_r, n_r, 2)")
    x_edges = xy[..., 0]
    y_edges = xy[..., 1]
    x_step = _uniform_grid_step(x_edges[:, 0], "x")
    y_step = _uniform_grid_step(y_edges[0, :], "y")
    area = abs(x_step * y_step)
    if area <= 0.0:
        raise ValueError("real-space plaquette area must be positive")

    centers = plaquette_average(xy)
    rho_per_plaquette = np.asarray(result.rho_top, dtype=float)
    if rho_per_plaquette.shape != centers.shape[:2]:
        raise ValueError(
            "rho_top must live on the plaquette-center grid; "
            f"got {rho_per_plaquette.shape} and {centers.shape[:2]}"
        )
    return ChargeDensityGrid(
        x_edges=x_edges,
        y_edges=y_edges,
        x_centers=centers[..., 0],
        y_centers=centers[..., 1],
        rho_per_plaquette=rho_per_plaquette,
        rho_density=rho_per_plaquette / area,
        plaquette_area=area,
    )


def _uniform_grid_step(values: np.ndarray, label: str) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) < 2:
        raise ValueError(f"{label} grid must be one-dimensional with at least two points")
    steps = np.diff(arr)
    step = float(np.mean(steps))
    if not np.allclose(steps, step, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{label} grid must be uniformly spaced")
    return step


def clll_wall_lengths(
    params: CLLLSchematicPlotParams | IdealConjugateLLLChargeBenchmarkParams,
) -> tuple[float, float]:
    magnetic_length = triangular_moire_magnetic_length()
    return float(params.radius_lB) * magnetic_length, float(params.width_lB) * magnetic_length


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    arr = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(arr)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return arr / norm


def orthonormal_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = normalize_vector(direction)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(direction, reference))) > 0.92:
        reference = np.array([0.0, 1.0, 0.0])
    normal_1 = normalize_vector(np.cross(direction, reference))
    normal_2 = normalize_vector(np.cross(direction, normal_1))
    return normal_1, normal_2


def lighten_color(rgba: tuple[float, ...], factor: float) -> tuple[float, float, float, float]:
    rgb = np.clip(np.asarray(rgba[:3], dtype=float) * factor, 0.0, 1.0)
    alpha = float(rgba[3]) if len(rgba) == 4 else 1.0
    return float(rgb[0]), float(rgb[1]), float(rgb[2]), alpha


def compute_facecolor(
    base_color: tuple[float, ...],
    face_normal: np.ndarray,
    light_direction: np.ndarray,
) -> tuple[float, float, float, float]:
    normal = normalize_vector(face_normal)
    brightness = 0.40 + 0.75 * max(0.0, float(np.dot(normal, light_direction)))
    return lighten_color(base_color, brightness)


def build_cylinder_faces(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    segments: int,
    base_color: tuple[float, ...],
    light_direction: np.ndarray,
) -> tuple[list[np.ndarray], list[tuple[float, float, float, float]]]:
    axis = end - start
    axis_hat = normalize_vector(axis)
    n1, n2 = orthonormal_frame(axis_hat)
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring_start = []
    ring_end = []
    for angle in angles:
        radial = np.cos(angle) * n1 + np.sin(angle) * n2
        ring_start.append(start + radius * radial)
        ring_end.append(end + radius * radial)

    faces = []
    colors = []
    for idx in range(segments):
        nxt = (idx + 1) % segments
        face = np.array([ring_start[idx], ring_start[nxt], ring_end[nxt], ring_end[idx]])
        radial_mid = normalize_vector(
            0.5 * ((ring_start[idx] - start) + (ring_start[nxt] - start))
        )
        faces.append(face)
        colors.append(compute_facecolor(base_color, radial_mid, light_direction))
    faces.append(np.array(ring_start[::-1]))
    colors.append(compute_facecolor(base_color, -axis_hat, light_direction))
    return faces, colors


def build_cone_faces(
    base_center: np.ndarray,
    tip: np.ndarray,
    base_radius: float,
    segments: int,
    base_color: tuple[float, ...],
    light_direction: np.ndarray,
) -> tuple[list[np.ndarray], list[tuple[float, float, float, float]]]:
    axis_hat = normalize_vector(tip - base_center)
    n1, n2 = orthonormal_frame(axis_hat)
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = []
    for angle in angles:
        radial = np.cos(angle) * n1 + np.sin(angle) * n2
        ring.append(base_center + base_radius * radial)

    faces = []
    colors = []
    for idx in range(segments):
        nxt = (idx + 1) % segments
        face = np.array([ring[idx], ring[nxt], tip])
        face_normal = np.cross(face[1] - face[0], face[2] - face[0])
        faces.append(face)
        colors.append(compute_facecolor(base_color, face_normal, light_direction))
    return faces, colors


def build_arrow_mesh(
    position: np.ndarray,
    direction: np.ndarray,
    cmap,
    norm: Normalize,
    *,
    total_length: float,
    shaft_radius: float,
    cone_radius: float,
    cone_fraction: float = 0.36,
    segments: int = 16,
) -> tuple[list[np.ndarray], list[tuple[float, float, float, float]]]:
    direction = normalize_vector(direction)
    center = np.asarray(position, dtype=float)
    cone_length = cone_fraction * total_length
    shaft_length = total_length - cone_length
    shaft_start = center - 0.5 * total_length * direction
    shaft_end = shaft_start + shaft_length * direction
    tip = shaft_end + cone_length * direction
    base_color = cmap(norm(direction[2]))
    light_direction = normalize_vector(np.array([-0.55, -0.35, 1.0]))
    shaft_faces, shaft_colors = build_cylinder_faces(
        shaft_start,
        shaft_end,
        shaft_radius,
        segments,
        base_color,
        light_direction,
    )
    cone_faces, cone_colors = build_cone_faces(
        shaft_end,
        tip,
        cone_radius,
        segments,
        base_color,
        light_direction,
    )
    return shaft_faces + cone_faces, shaft_colors + cone_colors


def draw_spin_arrows_3d(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    *,
    total_length: float,
    shaft_radius: float,
    cone_radius: float,
    segments: int = 16,
) -> Poly3DCollection:
    cmap = matplotlib.colormaps["jet"]
    norm = Normalize(vmin=-1.0, vmax=1.0)
    all_faces = []
    all_colors = []
    for xi, yi, spin in zip(x.ravel(), y.ravel(), field.reshape(-1, 3), strict=True):
        faces, colors = build_arrow_mesh(
            np.array([xi, yi, 0.0]),
            spin,
            cmap,
            norm,
            total_length=total_length,
            shaft_radius=shaft_radius,
            cone_radius=cone_radius,
            segments=segments,
        )
        all_faces.extend(faces)
        all_colors.extend(colors)
    collection = Poly3DCollection(
        all_faces,
        facecolors=all_colors,
        edgecolors="none",
        linewidths=0.0,
        rasterized=True,
    )
    ax.add_collection3d(collection)
    return collection


def style_view_axis(
    ax,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    z_extent: float = 1.6,
    elev: float = 27.0,
    azim: float = -70.0,
    focal_length: float = 1.45,
    zoom: float = 1.65,
) -> None:
    x_min = float(np.min(x_edges))
    x_max = float(np.max(x_edges))
    y_min = float(np.min(y_edges))
    y_max = float(np.max(y_edges))
    span = max(x_max - x_min, y_max - y_min)
    margin = 0.04 * span
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(-z_extent, z_extent)
    ax.set_box_aspect(
        (x_max - x_min + 2.0 * margin, y_max - y_min + 2.0 * margin, 7.5),
        zoom=zoom,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("persp", focal_length=focal_length)
    ax.set_axis_off()


def project_plane_via_camera(ax3d, x: np.ndarray, y: np.ndarray, z_level: float = 0.0):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.full_like(x_arr, z_level, dtype=float)
    shape = x_arr.shape
    x_proj, y_proj, _ = proj3d.proj_transform(
        x_arr.ravel(),
        y_arr.ravel(),
        z_arr.ravel(),
        ax3d.get_proj(),
    )
    return x_proj.reshape(shape), y_proj.reshape(shape)


def draw_spin_guides_3d(ax, radii: tuple[float, float, float]) -> None:
    for idx, radius in enumerate(radii):
        if radius <= 0.0:
            continue
        theta = np.linspace(0.0, 2.0 * np.pi, 800)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = np.full_like(theta, -0.04)
        linewidth = 1.15 if idx == 1 else 0.8
        dash_pattern = (10, 4.5) if idx == 1 else (5, 4)
        ax.plot(
            x,
            y,
            z,
            color="black",
            linewidth=linewidth,
            linestyle="--",
            dashes=dash_pattern,
        )


def draw_projected_guides(ax, projection_ax, radii: tuple[float, float, float]) -> None:
    for idx, radius in enumerate(radii):
        if radius <= 0.0:
            continue
        theta = np.linspace(0.0, 2.0 * np.pi, 800)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        x_proj, y_proj = project_plane_via_camera(projection_ax, x, y)
        linewidth = 2.3 if idx == 1 else 1.9
        dash_pattern = (10, 4) if idx == 1 else (6, 4)
        ax.plot(
            x_proj,
            y_proj,
            color="black",
            linewidth=linewidth,
            linestyle="--",
            dashes=dash_pattern,
        )


def draw_projected_scale_arrows(ax, projection_ax, radius: float, width: float) -> None:
    theta_radius = -0.22 * np.pi
    theta_width = 0.13 * np.pi
    x_r = np.array([0.0, radius * np.cos(theta_radius)])
    y_r = np.array([0.0, radius * np.sin(theta_radius)])
    xr, yr = project_plane_via_camera(projection_ax, x_r, y_r)
    ax.annotate(
        "",
        xy=(xr[1], yr[1]),
        xytext=(xr[0], yr[0]),
        arrowprops={"arrowstyle": "<->", "color": "black", "linewidth": 1.8, "shrinkA": 0, "shrinkB": 0},
    )
    ax.text(
        0.53 * xr[1] + 0.47 * xr[0],
        0.53 * yr[1] + 0.47 * yr[0],
        r"$R$",
        fontsize=18,
        ha="left",
        va="center",
        color="black",
    )

    x_d = np.array(
        [
            radius * np.cos(theta_width),
            (radius + width) * np.cos(theta_width),
        ]
    )
    y_d = np.array(
        [
            radius * np.sin(theta_width),
            (radius + width) * np.sin(theta_width),
        ]
    )
    xd, yd = project_plane_via_camera(projection_ax, x_d, y_d)
    ax.annotate(
        "",
        xy=(xd[1], yd[1]),
        xytext=(xd[0], yd[0]),
        arrowprops={"arrowstyle": "<->", "color": "black", "linewidth": 1.8, "shrinkA": 0, "shrinkB": 0},
    )
    ax.text(
        xd[1],
        yd[1],
        r"$d_0$",
        fontsize=18,
        ha="left",
        va="center",
        color="black",
    )


def draw_charge_panel_projected(
    ax,
    projection_ax,
    charge: ChargeDensityGrid,
    radii: tuple[float, float, float],
):
    x_proj, y_proj = project_plane_via_camera(projection_ax, charge.x_edges, charge.y_edges)
    vmax = float(np.max(np.abs(charge.rho_density)))
    if not np.isfinite(vmax) or vmax <= 1e-15:
        vmax = 1.0
    mesh = ax.pcolormesh(
        x_proj,
        y_proj,
        charge.rho_density,
        cmap="RdBu_r",
        shading="flat",
        vmin=-vmax,
        vmax=vmax,
        rasterized=True,
    )
    draw_projected_guides(ax, projection_ax, radii)
    draw_projected_scale_arrows(ax, projection_ax, radii[1], radii[2] - radii[1])
    x_margin = 0.03 * max(float(np.ptp(x_proj)), 1e-12)
    y_margin = 0.03 * max(float(np.ptp(y_proj)), 1e-12)
    ax.set_xlim(float(np.min(x_proj)) - x_margin, float(np.max(x_proj)) + x_margin)
    ax.set_ylim(float(np.min(y_proj)) - y_margin, float(np.max(y_proj)) + y_margin)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return mesh


def render_clll_spin_charge_schematic(
    params: CLLLSchematicPlotParams,
) -> tuple[Path, Path]:
    result = compute_clll_response(params)
    charge = compute_charge_density_grid(result)
    radius, width = clll_wall_lengths(params)
    radii = (max(radius - width, 0.0), radius, radius + width)

    xy = np.asarray(result.solution.xy, dtype=float)
    field = np.asarray(result.solution.wall_field, dtype=float)
    spin_slice = (slice(None, None, params.spin_stride), slice(None, None, params.spin_stride))
    x_spin = xy[..., 0][spin_slice]
    y_spin = xy[..., 1][spin_slice]
    field_spin = field[spin_slice]
    grid_step = max(
        abs(_uniform_grid_step(xy[:, 0, 0], "x")),
        abs(_uniform_grid_step(xy[0, :, 1], "y")),
    )
    drawn_spacing = grid_step * float(params.spin_stride)
    arrow_length = 0.58 * drawn_spacing

    output = params.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output = output.with_suffix(".pdf")

    fig = plt.figure(figsize=(9.6, 8.8), facecolor="white")
    spin_ax = fig.add_axes([0.02, 0.53, 0.78, 0.43], projection="3d")
    charge_ax = fig.add_axes([0.02, 0.07, 0.78, 0.43], frameon=False)
    cbar_ax = fig.add_axes([0.84, 0.16, 0.026, 0.30])

    draw_spin_guides_3d(spin_ax, radii)
    draw_spin_arrows_3d(
        spin_ax,
        x_spin,
        y_spin,
        field_spin,
        total_length=arrow_length,
        shaft_radius=0.055 * drawn_spacing,
        cone_radius=0.11 * drawn_spacing,
    )
    style_view_axis(spin_ax, xy[..., 0], xy[..., 1])
    fig.canvas.draw()

    mesh = draw_charge_panel_projected(charge_ax, spin_ax, charge, radii)
    colorbar = fig.colorbar(mesh, cax=cbar_ax)
    colorbar.set_label(r"$\rho a_M^2$", rotation=90, labelpad=10)
    colorbar.ax.tick_params(labelsize=9)

    fig.savefig(output, dpi=params.dpi, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_output, dpi=params.dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output, pdf_output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a cLLL domain-wall spin texture and numerical charge density."
    )
    parser.add_argument("--n-k", type=int, default=CLLLSchematicPlotParams.model_fields["n_k"].default)
    parser.add_argument("--n-r", type=int, default=CLLLSchematicPlotParams.model_fields["n_r"].default)
    parser.add_argument(
        "--radius-lb",
        type=float,
        default=CLLLSchematicPlotParams.model_fields["radius_lB"].default,
    )
    parser.add_argument(
        "--width-lb",
        type=float,
        default=CLLLSchematicPlotParams.model_fields["width_lB"].default,
    )
    parser.add_argument(
        "--patch-length-lb",
        type=float,
        default=CLLLSchematicPlotParams.model_fields["patch_length_lB"].default,
    )
    parser.add_argument(
        "--winding",
        type=int,
        default=CLLLSchematicPlotParams.model_fields["winding"].default,
    )
    parser.add_argument(
        "--helicity",
        type=float,
        default=CLLLSchematicPlotParams.model_fields["helicity"].default,
    )
    parser.add_argument(
        "--spin-stride",
        type=int,
        default=CLLLSchematicPlotParams.model_fields["spin_stride"].default,
    )
    parser.add_argument(
        "--theta-count",
        type=int,
        default=CLLLSchematicPlotParams.model_fields["theta_count"].default,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CLLLSchematicPlotParams.model_fields["output"].default,
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    params = CLLLSchematicPlotParams(
        n_k=args.n_k,
        n_r=args.n_r,
        radius_lB=args.radius_lb,
        width_lB=args.width_lb,
        patch_length_lB=args.patch_length_lb,
        winding=args.winding,
        helicity=args.helicity,
        spin_stride=args.spin_stride,
        theta_count=args.theta_count,
        output=args.output,
    )
    png_path, pdf_path = render_clll_spin_charge_schematic(params)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
