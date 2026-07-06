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
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.mplot3d import proj3d

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

    n_k: int = Field(default=5, ge=1)
    n_r: int = Field(default=121, ge=3)
    radius_lB: float = Field(default=25.0, gt=0.0)
    width_lB: float = Field(default=8.0, gt=0.0)
    patch_length_lB: float = Field(default=110.0, gt=0.0)
    plot_half_width_lB: float = Field(default=41.0, gt=0.0)
    winding: int = 1
    helicity: float = 0.0
    spin_stride: int = Field(default=4, ge=1)
    theta_count: int = Field(default=42, ge=3)
    charge_upsample: int = Field(default=6, ge=1)
    origin_regularization_lB: float = Field(default=1.25, ge=0.0)
    origin_transition_lB: float = Field(default=2.5, ge=0.0)
    color_percentile: float = Field(default=99.5, gt=0.0, le=100.0)
    output: Path = Path("Plots/figures/clll_spin_charge_schematic.png")
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


@dataclass(frozen=True)
class ChargeDisplayGrid:
    """Interpolated display-only charge density for the schematic panel."""

    x_edges: np.ndarray
    y_edges: np.ndarray
    rho_density: np.ndarray


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


def origin_regularization_radius(params: CLLLSchematicPlotParams) -> float:
    return float(params.origin_regularization_lB) * triangular_moire_magnetic_length()


def origin_transition_radius(params: CLLLSchematicPlotParams) -> float:
    return float(params.origin_transition_lB) * triangular_moire_magnetic_length()


def plot_half_width(params: CLLLSchematicPlotParams) -> float:
    return float(params.plot_half_width_lB) * triangular_moire_magnetic_length()


def regularize_origin_artifact(
    density: np.ndarray,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    *,
    radius: float,
    transition_radius: float,
) -> np.ndarray:
    """Smoothly inpaint the display-only central plaquette artifact."""

    cleaned = np.asarray(density, dtype=float).copy()
    if radius <= 0.0:
        return cleaned
    rr = np.sqrt(np.asarray(x_centers, dtype=float) ** 2 + np.asarray(y_centers, dtype=float) ** 2)
    transition = max(float(transition_radius), 1e-14)
    annulus = (rr > float(radius)) & (rr <= float(radius) + transition)
    fallback = rr > float(radius)
    if np.any(annulus):
        fill_value = float(np.nanmedian(cleaned[annulus]))
    elif np.any(fallback):
        nearest_idx = int(np.nanargmin(np.where(fallback, rr, np.inf)))
        fill_value = float(cleaned.reshape(-1)[nearest_idx])
    else:
        fill_value = 0.0

    inner = rr <= float(radius)
    blend = (rr > float(radius)) & (rr < float(radius) + transition)
    cleaned[inner] = fill_value
    if np.any(blend):
        t = (rr[blend] - float(radius)) / transition
        weight = t * t * (3.0 - 2.0 * t)
        cleaned[blend] = (1.0 - weight) * fill_value + weight * cleaned[blend]
    return cleaned


def compute_charge_display_grid(
    charge: ChargeDensityGrid,
    params: CLLLSchematicPlotParams,
) -> ChargeDisplayGrid:
    """Return an interpolated charge-density grid for the tilted schematic plane."""

    density = regularize_origin_artifact(
        charge.rho_density,
        charge.x_centers,
        charge.y_centers,
        radius=origin_regularization_radius(params),
        transition_radius=origin_transition_radius(params),
    )
    upsample = int(params.charge_upsample)

    x_axis = np.asarray(charge.x_centers[:, 0], dtype=float)
    y_axis = np.asarray(charge.y_centers[0, :], dtype=float)
    n_x, n_y = density.shape
    half_width = plot_half_width(params)
    x_min = max(float(np.min(charge.x_edges)), -half_width)
    x_max = min(float(np.max(charge.x_edges)), half_width)
    y_min = max(float(np.min(charge.y_edges)), -half_width)
    y_max = min(float(np.max(charge.y_edges)), half_width)
    x_edge_axis = np.linspace(
        x_min,
        x_max,
        n_x * upsample + 1,
    )
    y_edge_axis = np.linspace(
        y_min,
        y_max,
        n_y * upsample + 1,
    )
    x_center_axis = 0.5 * (x_edge_axis[:-1] + x_edge_axis[1:])
    y_center_axis = 0.5 * (y_edge_axis[:-1] + y_edge_axis[1:])
    xx, yy = np.meshgrid(x_center_axis, y_center_axis, indexing="ij")
    interpolator = RegularGridInterpolator(
        (x_axis, y_axis),
        density,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    display_density = interpolator(np.stack([xx.ravel(), yy.ravel()], axis=-1)).reshape(
        xx.shape
    )
    x_edges, y_edges = np.meshgrid(x_edge_axis, y_edge_axis, indexing="ij")
    return ChargeDisplayGrid(
        x_edges=x_edges,
        y_edges=y_edges,
        rho_density=display_density,
    )


def electron_charge_density_over_e(number_density: np.ndarray) -> np.ndarray:
    """Convert carrier number density to electron charge density divided by e."""

    return -np.asarray(number_density, dtype=float)


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


def build_spin_arrow_faces(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    *,
    total_length: float,
    shaft_radius: float,
    cone_radius: float,
    segments: int = 16,
) -> tuple[list[np.ndarray], list[tuple[float, float, float, float]]]:
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
    return all_faces, all_colors


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
    margin_fraction: float = 0.13,
) -> None:
    x_min = float(np.min(x_edges))
    x_max = float(np.max(x_edges))
    y_min = float(np.min(y_edges))
    y_max = float(np.max(y_edges))
    span = max(x_max - x_min, y_max - y_min)
    margin = float(margin_fraction) * span
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


def project_vertices_via_camera(ax3d, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts = np.asarray(vertices, dtype=float)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must have shape (n, 3)")
    x_proj, y_proj, z_proj = proj3d.proj_transform(
        verts[:, 0],
        verts[:, 1],
        verts[:, 2],
        ax3d.get_proj(),
    )
    return np.asarray(x_proj), np.asarray(y_proj), np.asarray(z_proj)


def projected_face_limits(
    projection_ax,
    faces: list[np.ndarray],
    *,
    margin_fraction: float = 0.03,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if not faces:
        return ((-1.0, 1.0), (-1.0, 1.0))
    x_values = []
    y_values = []
    for face in faces:
        x_proj, y_proj, _ = project_vertices_via_camera(projection_ax, face)
        x_values.append(x_proj)
        y_values.append(y_proj)
    x_all = np.concatenate(x_values)
    y_all = np.concatenate(y_values)
    x_margin = float(margin_fraction) * max(float(np.ptp(x_all)), 1e-12)
    y_margin = float(margin_fraction) * max(float(np.ptp(y_all)), 1e-12)
    return (
        (float(np.min(x_all)) - x_margin, float(np.max(x_all)) + x_margin),
        (float(np.min(y_all)) - y_margin, float(np.max(y_all)) + y_margin),
    )


def combine_limits(
    first: tuple[tuple[float, float], tuple[float, float]],
    second: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (min(first[0][0], second[0][0]), max(first[0][1], second[0][1])),
        (min(first[1][0], second[1][0]), max(first[1][1], second[1][1])),
    )


def draw_spin_arrows_projected(
    ax,
    projection_ax,
    faces: list[np.ndarray],
    colors: list[tuple[float, float, float, float]],
) -> PolyCollection:
    projected = []
    sort_keys = []
    for face in faces:
        x_proj, y_proj, z_proj = project_vertices_via_camera(projection_ax, face)
        projected.append(np.column_stack([x_proj, y_proj]))
        sort_keys.append(float(np.mean(z_proj)))
    order = np.argsort(sort_keys)
    collection = PolyCollection(
        [projected[idx] for idx in order],
        facecolors=[colors[idx] for idx in order],
        edgecolors="none",
        linewidths=0.0,
        rasterized=True,
        clip_on=False,
    )
    ax.add_collection(collection)
    return collection


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


def projected_plane_limits(
    projection_ax,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    *,
    margin_fraction: float = 0.03,
) -> tuple[tuple[float, float], tuple[float, float]]:
    x_proj, y_proj = project_plane_via_camera(projection_ax, x_edges, y_edges)
    x_margin = float(margin_fraction) * max(float(np.ptp(x_proj)), 1e-12)
    y_margin = float(margin_fraction) * max(float(np.ptp(y_proj)), 1e-12)
    return (
        (float(np.min(x_proj)) - x_margin, float(np.max(x_proj)) + x_margin),
        (float(np.min(y_proj)) - y_margin, float(np.max(y_proj)) + y_margin),
    )


def style_projected_panel(
    ax,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_axis_off()


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
    ax.annotate(
        r"$R$",
        xy=(
            0.53 * xr[1] + 0.47 * xr[0],
            0.53 * yr[1] + 0.47 * yr[0],
        ),
        xytext=(12, 10),
        textcoords="offset points",
        fontsize=18,
        ha="left",
        va="bottom",
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
    charge: ChargeDisplayGrid,
    radii: tuple[float, float, float],
    *,
    color_percentile: float,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    x_proj, y_proj = project_plane_via_camera(projection_ax, charge.x_edges, charge.y_edges)
    charge_density = electron_charge_density_over_e(charge.rho_density)
    vmax = float(np.nanpercentile(np.abs(charge_density), color_percentile))
    if not np.isfinite(vmax) or vmax <= 1e-15:
        vmax = 1.0
    mesh = ax.pcolormesh(
        x_proj,
        y_proj,
        charge_density,
        cmap="RdBu_r",
        shading="flat",
        vmin=-vmax,
        vmax=vmax,
        rasterized=True,
    )
    draw_projected_guides(ax, projection_ax, radii)
    draw_projected_scale_arrows(ax, projection_ax, radii[1], radii[2] - radii[1])
    if xlim is None or ylim is None:
        xlim, ylim = projected_plane_limits(projection_ax, charge.x_edges, charge.y_edges)
    style_projected_panel(ax, xlim, ylim)
    return mesh


def render_clll_spin_charge_schematic(
    params: CLLLSchematicPlotParams,
) -> tuple[Path, Path]:
    result = compute_clll_response(params)
    charge = compute_charge_density_grid(result)
    display_charge = compute_charge_display_grid(charge, params)
    radius, width = clll_wall_lengths(params)
    radii = (max(radius - width, 0.0), radius, radius + width)

    xy = np.asarray(result.solution.xy, dtype=float)
    field = np.asarray(result.solution.wall_field, dtype=float)
    spin_slice = (slice(None, None, params.spin_stride), slice(None, None, params.spin_stride))
    x_spin_raw = xy[..., 0][spin_slice]
    y_spin_raw = xy[..., 1][spin_slice]
    field_spin_raw = field[spin_slice]
    visible_half_width = plot_half_width(params)
    visible_spin = (np.abs(x_spin_raw) <= visible_half_width) & (
        np.abs(y_spin_raw) <= visible_half_width
    )
    x_spin = x_spin_raw[visible_spin]
    y_spin = y_spin_raw[visible_spin]
    field_spin = field_spin_raw[visible_spin]
    grid_step = max(
        abs(_uniform_grid_step(xy[:, 0, 0], "x")),
        abs(_uniform_grid_step(xy[0, :, 1], "y")),
    )
    drawn_spacing = grid_step * float(params.spin_stride)
    arrow_length = 0.58 * drawn_spacing

    output = params.output
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output = output.with_suffix(".pdf")

    faces, colors = build_spin_arrow_faces(
        x_spin,
        y_spin,
        field_spin,
        total_length=arrow_length,
        shaft_radius=0.055 * drawn_spacing,
        cone_radius=0.11 * drawn_spacing,
    )

    fig = plt.figure(figsize=(9.8, 8.6), facecolor="white")
    panel_box = [0.01, 0.06, 0.79, 0.42]
    spin_panel_box = [panel_box[0], 0.54, panel_box[2], panel_box[3]]
    projection_ax = fig.add_axes([0.0, 0.0, 0.01, 0.01], projection="3d")
    spin_ax = fig.add_axes(spin_panel_box, frameon=False)
    charge_ax = fig.add_axes(panel_box, frameon=False)
    cbar_ax = fig.add_axes([0.84, 0.16, 0.028, 0.31])

    style_view_axis(projection_ax, xy[..., 0], xy[..., 1])
    projection_ax.set_visible(False)
    fig.canvas.draw()
    plane_limits = projected_plane_limits(projection_ax, display_charge.x_edges, display_charge.y_edges)
    arrow_limits = projected_face_limits(projection_ax, faces)
    xlim, ylim = combine_limits(plane_limits, arrow_limits)

    mesh = draw_charge_panel_projected(
        charge_ax,
        projection_ax,
        display_charge,
        radii,
        color_percentile=params.color_percentile,
        xlim=xlim,
        ylim=ylim,
    )
    draw_spin_arrows_projected(spin_ax, projection_ax, faces, colors)
    draw_projected_guides(spin_ax, projection_ax, radii)
    style_projected_panel(spin_ax, xlim, ylim)
    colorbar = fig.colorbar(mesh, cax=cbar_ax)
    colorbar.set_label(r"$\rho_Q a_M^2/e$", rotation=90, labelpad=10)
    colorbar.ax.tick_params(labelsize=9)

    fig.savefig(output, dpi=params.dpi)
    fig.savefig(pdf_output, dpi=params.dpi)
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
        "--plot-half-width-lb",
        type=float,
        default=CLLLSchematicPlotParams.model_fields["plot_half_width_lB"].default,
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
        "--charge-upsample",
        type=int,
        default=CLLLSchematicPlotParams.model_fields["charge_upsample"].default,
    )
    parser.add_argument(
        "--origin-regularization-lb",
        type=float,
        default=CLLLSchematicPlotParams.model_fields["origin_regularization_lB"].default,
    )
    parser.add_argument(
        "--origin-transition-lb",
        type=float,
        default=CLLLSchematicPlotParams.model_fields["origin_transition_lB"].default,
    )
    parser.add_argument(
        "--color-percentile",
        type=float,
        default=CLLLSchematicPlotParams.model_fields["color_percentile"].default,
    )
    parser.add_argument("--dpi", type=int, default=CLLLSchematicPlotParams.model_fields["dpi"].default)
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
        plot_half_width_lB=args.plot_half_width_lb,
        winding=args.winding,
        helicity=args.helicity,
        spin_stride=args.spin_stride,
        theta_count=args.theta_count,
        charge_upsample=args.charge_upsample,
        origin_regularization_lB=args.origin_regularization_lb,
        origin_transition_lB=args.origin_transition_lb,
        color_percentile=args.color_percentile,
        output=args.output,
        dpi=args.dpi,
    )
    png_path, pdf_path = render_clll_spin_charge_schematic(params)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
