"""Reciprocal-channel domains for triangular moire momentum grids."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

GridCoord = tuple[int, int]
C3_RADIAL_Q_PLUS_G_V2 = "c3_radial_q_plus_g_orbit_symmetrized_v2"


class _GridShape(Protocol):
    n1: int
    n2: int


def reciprocal_box(g_cutoff: int) -> tuple[GridCoord, ...]:
    """Return candidate reciprocal channels in a square integer box."""

    n = int(g_cutoff)
    coords = [(g1, g2) for g1 in range(-n, n + 1) for g2 in range(-n, n + 1)]
    coords.sort(key=lambda c: (c[0] ** 2 + c[1] ** 2 + c[0] * c[1], c[0], c[1]))
    return tuple(coords)


def hexagonal_q_shell(radius: int) -> tuple[GridCoord, ...]:
    """Return a C3-closed hexagonal shell of mesh momentum transfers."""

    n = int(radius)
    coords = [
        (i, j)
        for i in range(-n, n + 1)
        for j in range(-n, n + 1)
        if max(abs(i), abs(j), abs(i + j)) <= n
    ]
    coords.sort(key=lambda c: (c[0] ** 2 + c[1] ** 2 + c[0] * c[1], c[0], c[1]))
    return tuple(coords)


def c3_radial_channel_mask(
    grid: _GridShape,
    q_list: Sequence[GridCoord],
    g_channels: Sequence[GridCoord],
    local_field_cutoff: int,
    momentum_transfer_cutoff_km: float | None = None,
) -> np.ndarray:
    """Select a radial, C3-invariant domain in the combined momentum ``q+G``.

    For a triangular reciprocal basis with unit vectors separated by 60 degrees,
    ``|u b1 + v b2|^2 = u^2 + v^2 + u v``.  A positive integer cutoff ``L``
    retains channels inside radius ``L*sqrt(3)/2`` unless an explicit physical
    radius ``momentum_transfer_cutoff_km`` is supplied in units of ``|g_M|``.
    On square momentum meshes the comparison uses the integer quadratic form,
    so symmetry-related boundary points cannot be split by floating-point
    roundoff. The inequality is strict in either convention.

    ``L=0`` with no explicit radius preserves the historical single-G-channel
    behavior. ``local_field_cutoff`` still controls the candidate reciprocal
    box; callers must choose it large enough to contain the requested disk.
    """

    cutoff = int(local_field_cutoff)
    mask = np.ones((len(q_list), len(g_channels)), dtype=bool)
    physical_cutoff = (
        None
        if momentum_transfer_cutoff_km is None
        else float(momentum_transfer_cutoff_km)
    )
    if cutoff <= 0 and physical_cutoff is None:
        return mask

    n1 = int(grid.n1)
    n2 = int(grid.n2)
    if n1 == n2:
        n = n1
        if physical_cutoff is None:
            cutoff_rhs = 3 * (cutoff * n) ** 2
            lhs_scale = 4
        else:
            cutoff_rhs = (physical_cutoff * n) ** 2
            lhs_scale = 1
        for iq, (qi, qj) in enumerate(q_list):
            for ig, (g1, g2) in enumerate(g_channels):
                a = int(qi) + n * int(g1)
                b = int(qj) + n * int(g2)
                mask[iq, ig] = lhs_scale * (a * a + b * b + a * b) < cutoff_rhs
        return mask

    cutoff_squared = (
        0.75 * float(cutoff**2)
        if physical_cutoff is None
        else physical_cutoff**2
    )
    for iq, (qi, qj) in enumerate(q_list):
        for ig, (g1, g2) in enumerate(g_channels):
            u = float(qi) / n1 + int(g1)
            v = float(qj) / n2 + int(g2)
            mask[iq, ig] = u * u + v * v + u * v < cutoff_squared
    return mask


def c3_channel_index_map(
    grid: _GridShape,
    q_list: Sequence[GridCoord],
    g_channels: Sequence[GridCoord],
    channel_mask: np.ndarray,
) -> np.ndarray:
    """Map every retained channel to its C3-rotated retained channel.

    The returned array ends in ``(iq, ig)``.  Excluded entries contain ``-1``.
    This exact integer construction is defined for the square meshes used by
    the continuum and conjugate-AC calculations.
    """

    if int(grid.n1) != int(grid.n2):
        raise ValueError("C3 channel maps require n1 == n2")
    mask = np.asarray(channel_mask, dtype=bool)
    expected_shape = (len(q_list), len(g_channels))
    if mask.shape != expected_shape:
        raise ValueError(f"channel_mask must have shape {expected_shape}")

    n = int(grid.n1)
    lookup: dict[GridCoord, tuple[int, int]] = {}
    for iq, (qi, qj) in enumerate(q_list):
        for ig, (g1, g2) in enumerate(g_channels):
            if not mask[iq, ig]:
                continue
            key = (int(qi) + n * int(g1), int(qj) + n * int(g2))
            if key in lookup:
                raise ValueError(f"duplicate retained q+G channel {key}")
            lookup[key] = (iq, ig)

    partner = np.full(mask.shape + (2,), -1, dtype=int)
    missing: list[tuple[GridCoord, GridCoord]] = []
    for key, (iq, ig) in lookup.items():
        a, b = key
        rotated = (-a - b, a)
        target = lookup.get(rotated)
        if target is None:
            missing.append((key, rotated))
        else:
            partner[iq, ig] = target
    if missing:
        example = missing[0]
        raise ValueError(
            f"retained q+G domain is not C3 closed; "
            f"{len(missing)} channels are missing, including {example[0]} -> {example[1]}"
        )
    return partner


def inversion_channel_index_map(
    grid: _GridShape,
    q_list: Sequence[GridCoord],
    g_channels: Sequence[GridCoord],
    channel_mask: np.ndarray,
) -> np.ndarray:
    """Map every retained channel to the representative of ``-(q+G)``."""

    if int(grid.n1) != int(grid.n2):
        raise ValueError("inversion channel maps require n1 == n2")
    mask = np.asarray(channel_mask, dtype=bool)
    expected_shape = (len(q_list), len(g_channels))
    if mask.shape != expected_shape:
        raise ValueError(f"channel_mask must have shape {expected_shape}")

    n = int(grid.n1)
    lookup: dict[GridCoord, tuple[int, int]] = {}
    for iq, (qi, qj) in enumerate(q_list):
        for ig, (g1, g2) in enumerate(g_channels):
            if not mask[iq, ig]:
                continue
            key = (int(qi) + n * int(g1), int(qj) + n * int(g2))
            if key in lookup:
                raise ValueError(f"duplicate retained q+G channel {key}")
            lookup[key] = (iq, ig)

    partner = np.full(mask.shape + (2,), -1, dtype=int)
    missing: list[tuple[GridCoord, GridCoord]] = []
    for key, (iq, ig) in lookup.items():
        inverted = (-key[0], -key[1])
        target = lookup.get(inverted)
        if target is None:
            missing.append((key, inverted))
        else:
            partner[iq, ig] = target
    if missing:
        example = missing[0]
        raise ValueError(
            f"retained q+G domain is not inversion closed; "
            f"{len(missing)} channels are missing, including {example[0]} -> {example[1]}"
        )
    return partner


def c3_mesh_index_map(grid: _GridShape) -> np.ndarray:
    """Return the discrete C3 map ``(i,j) -> (-i-j,i)``."""

    if int(grid.n1) != int(grid.n2):
        raise ValueError("C3 mesh maps require n1 == n2")
    n = int(grid.n1)
    return np.asarray(
        [((-i - j) % n) * n + (i % n) for i in range(n) for j in range(n)],
        dtype=int,
    )


def c3_spectrum_residual(grid: _GridShape, blocks: np.ndarray) -> float:
    """Return the maximum C3 eigenvalue mismatch of mesh-local operators."""

    arr = np.asarray(blocks, dtype=complex)
    expected_blocks = int(grid.n1) * int(grid.n2)
    if arr.ndim != 3 or arr.shape[0] != expected_blocks or arr.shape[-1] != arr.shape[-2]:
        raise ValueError("blocks must have shape (n1*n2, dim, dim)")
    spectra = np.linalg.eigvalsh(arr)
    partner = c3_mesh_index_map(grid)
    return float(np.max(np.abs(spectra - spectra[partner])))


def c3_channel_value_residual(
    values: np.ndarray,
    partner: np.ndarray,
    channel_mask: np.ndarray,
) -> float:
    """Return the maximum C3 mismatch of scalar channel data."""

    arr = np.asarray(values)
    mask = np.asarray(channel_mask, dtype=bool)
    mapping = np.asarray(partner, dtype=int)
    if arr.shape != mask.shape or mapping.shape != mask.shape + (2,):
        raise ValueError("values, channel_mask, and partner shapes are incompatible")
    if not np.any(mask):
        return 0.0
    rotated = arr[mapping[..., 0], mapping[..., 1]]
    return float(np.max(np.abs(arr[mask] - rotated[mask])))
