"""Native optimized continuum Hartree-Fock backend and solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from chiral_dw.config import ContinuumHFParams, ContinuumInteractionParams
from chiral_dw.continuum.models import (
    ContinuumHFDiagnostics,
    ContinuumHFIterationSnapshot,
    ContinuumHFResult,
    DensityVertices,
    PhysicalDensityChannels,
    ReferenceHamiltonianDiagnostics,
    block_trace_product,
    dense_lambdas_from_compact,
    hermitize,
    projector_idempotency_errors,
)
from chiral_dw.continuum.symmetry import _fixed_per_k_aufbau, _global_aufbau

HFIterationCallback = Callable[[int, np.ndarray, float, ContinuumHFDiagnostics, bool], None]


@dataclass(frozen=True)
class EnergyComponents:
    """One-body, Hartree, Fock, and total HF energy."""

    total: float
    one_body: float
    hartree: float
    fock: float


@dataclass
class ValleySectorExchange:
    """Exchange superoperator stored by independent valley-pair density sectors."""

    sectors: np.ndarray
    n_blocks: int
    n_active: int

    def __post_init__(self) -> None:
        self.sectors = np.asarray(self.sectors, dtype=complex)
        self.n_blocks = int(self.n_blocks)
        self.n_active = int(self.n_active)
        sector_dim = self.n_blocks * self.n_active * self.n_active
        if self.sectors.shape != (2, 2, sector_dim, sector_dim):
            raise ValueError(
                "valley-sector exchange must have shape "
                f"(2, 2, {sector_dim}, {sector_dim})"
            )

    @property
    def sector_block_dim(self) -> int:
        return self.n_active * self.n_active

    @property
    def dim(self) -> int:
        return 2 * self.n_active

    def matvec_sector(self, iv: int, jv: int, sector_density: np.ndarray) -> np.ndarray:
        flat = np.reshape(sector_density, (-1,))
        return np.reshape(
            self.sectors[int(iv), int(jv)] @ flat,
            (self.n_blocks, self.n_active, self.n_active),
        )

    def to_dense(self) -> np.ndarray:
        """Reconstruct the full dense exchange matrix for small debug comparisons."""

        block_dim = self.dim * self.dim
        sector_dim = self.sector_block_dim
        dense = np.zeros((self.n_blocks * block_dim, self.n_blocks * block_dim), dtype=complex)
        sector_local = np.arange(sector_dim)
        sector_rows = np.arange(self.n_blocks)[:, None] * sector_dim + sector_local[None, :]
        sector_flat = np.reshape(sector_rows, (-1,))
        for iv in range(2):
            for jv in range(2):
                full_local = np.asarray(
                    [
                        (iv * self.n_active + a) * self.dim + (jv * self.n_active + b)
                        for a in range(self.n_active)
                        for b in range(self.n_active)
                    ],
                    dtype=int,
                )
                full_rows = np.arange(self.n_blocks)[:, None] * block_dim + full_local[None, :]
                full_flat = np.reshape(full_rows, (-1,))
                dense[np.ix_(full_flat, full_flat)] = self.sectors[iv, jv][
                    np.ix_(sector_flat, sector_flat)
                ]
        return dense


_MAX_EXCHANGE_Q_POINTS_PER_SLAB = 8


def _exchange_q_slab_ranges(n_q: int, exchange_workers: int) -> tuple[tuple[int, int], ...]:
    del exchange_workers
    stop = max(0, int(n_q))
    slab = _MAX_EXCHANGE_Q_POINTS_PER_SLAB
    return tuple((start, min(start + slab, stop)) for start in range(0, stop, slab))


def _exchange_tve_q_slab(
    *,
    q_start: int,
    q_stop: int,
    lambda_blocks: np.ndarray,
    v_over_a: np.ndarray,
    exchange_scale: float,
) -> tuple[tuple[int, np.ndarray, np.ndarray], ...]:
    """Return dense block-exchange contributions for one q-slab."""

    rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    scale = float(exchange_scale)
    for iq in range(int(q_start), int(q_stop)):
        v = 0.5 * scale * v_over_a[iq]
        if not np.any(v):
            continue
        lam_q = lambda_blocks[iq]
        forward = np.einsum(
            "g,gkac,gkbd->kabcd",
            v,
            lam_q,
            np.conj(lam_q),
            optimize=True,
        )
        reverse = np.einsum(
            "g,gkca,gkdb->kabcd",
            v,
            np.conj(lam_q),
            lam_q,
            optimize=True,
        )
        rows.append((iq, forward, reverse))
    return tuple(rows)


def _exchange_tve_q_slab_compact(
    *,
    q_start: int,
    q_stop: int,
    lambda_compact: np.ndarray,
    v_over_a: np.ndarray,
    exchange_scale: float,
) -> tuple[tuple[int, np.ndarray, np.ndarray], ...]:
    """Return dense exchange contributions from valley-compact vertices."""

    rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    scale = float(exchange_scale)
    compact = np.asarray(lambda_compact)
    n_active = int(compact.shape[-1])
    dim = 2 * n_active
    for iq in range(int(q_start), int(q_stop)):
        v = 0.5 * scale * v_over_a[iq]
        if not np.any(v):
            continue
        lam_q = compact[iq]
        n_blocks = int(lam_q.shape[1])
        forward = np.zeros((n_blocks, dim, dim, dim, dim), dtype=complex)
        reverse = np.zeros_like(forward)
        for iv in range(2):
            a_slice = slice(iv * n_active, (iv + 1) * n_active)
            lam_left = lam_q[:, :, iv]
            for jv in range(2):
                b_slice = slice(jv * n_active, (jv + 1) * n_active)
                lam_right = lam_q[:, :, jv]
                forward[:, a_slice, b_slice, a_slice, b_slice] = np.einsum(
                    "g,gkac,gkbd->kabcd",
                    v,
                    lam_left,
                    np.conj(lam_right),
                    optimize=True,
                )
                reverse[:, a_slice, b_slice, a_slice, b_slice] = np.einsum(
                    "g,gkca,gkdb->kabcd",
                    v,
                    np.conj(lam_left),
                    lam_right,
                    optimize=True,
                )
        rows.append((iq, forward, reverse))
    return tuple(rows)


def _exchange_sector_tve_q_slab_compact(
    *,
    q_start: int,
    q_stop: int,
    lambda_compact: np.ndarray,
    v_over_a: np.ndarray,
    exchange_scale: float,
) -> tuple[tuple[int, int, int, np.ndarray, np.ndarray], ...]:
    """Return valley-sector exchange contributions from compact vertices."""

    rows: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
    scale = float(exchange_scale)
    compact = np.asarray(lambda_compact)
    for iq in range(int(q_start), int(q_stop)):
        v = 0.5 * scale * v_over_a[iq]
        if not np.any(v):
            continue
        lam_q = compact[iq]
        for iv in range(2):
            lam_left = lam_q[:, :, iv]
            for jv in range(2):
                lam_right = lam_q[:, :, jv]
                forward = np.einsum(
                    "g,gkac,gkbd->kabcd",
                    v,
                    lam_left,
                    np.conj(lam_right),
                    optimize=True,
                )
                reverse = np.einsum(
                    "g,gkca,gkdb->kabcd",
                    v,
                    np.conj(lam_left),
                    lam_right,
                    optimize=True,
                )
                rows.append((iq, iv, jv, forward, reverse))
    return tuple(rows)


def _hermitize_dense_in_place(matrix: np.ndarray, *, tile_size: int = 1024) -> np.ndarray:
    """Replace ``matrix`` by ``0.5 * (matrix + matrix.conj().T)`` using tiles."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    n = matrix.shape[0]
    tile = max(1, int(tile_size))
    for row_start in range(0, n, tile):
        row_stop = min(row_start + tile, n)
        diagonal = matrix[row_start:row_stop, row_start:row_stop]
        diagonal[...] = 0.5 * (diagonal + diagonal.conj().T)
        for col_start in range(row_stop, n, tile):
            col_stop = min(col_start + tile, n)
            upper = matrix[row_start:row_stop, col_start:col_stop].copy()
            lower = matrix[col_start:col_stop, row_start:row_stop]
            sym = 0.5 * (upper + lower.conj().T)
            matrix[row_start:row_stop, col_start:col_stop] = sym
            matrix[col_start:col_stop, row_start:row_stop] = sym.conj().T
    return matrix


class ContinuumHFBackend:
    """TMD_HF-style optimized block HF backend for native continuum vertices."""

    def __init__(
        self,
        h0: np.ndarray,
        vertices: DensityVertices,
        interaction: ContinuumInteractionParams | None = None,
    ) -> None:
        self.h0 = hermitize(np.asarray(h0, dtype=complex))
        if self.h0.ndim != 3 or self.h0.shape[-1] != self.h0.shape[-2]:
            raise ValueError("h0 must have shape (n_blocks, dim, dim)")
        self.n_blocks, self.dim, _ = self.h0.shape
        self.n_total = self.n_blocks * self.dim
        self.vertices = vertices
        self.interaction = interaction or ContinuumInteractionParams()
        self.vertex_layout = str(getattr(vertices, "vertex_layout", "dense"))
        self.physical_channels = getattr(vertices, "physical_channels", None)
        if self.physical_channels is not None:
            channels = self.physical_channels
            if self.vertex_layout != "valley_compact":
                raise ValueError(
                    "flat physical channels require valley_compact vertex layout"
                )
            if channels.n_blocks != self.n_blocks:
                raise ValueError(
                    "physical channels and h0 have incompatible block counts"
                )
            self.n_active = int(channels.n_active)
            if 2 * self.n_active != self.dim:
                raise ValueError(
                    "physical-channel active dimension is incompatible with h0"
                )
            self.lambda_compact = channels.compact_form_factor[:, None]
            self.lambda_blocks = np.asarray(vertices.lambda_blocks, dtype=complex)
            self.n_q, self.n_g = channels.n_channels, 1
        elif self.vertex_layout == "valley_compact":
            if vertices.lambda_compact is None:
                raise ValueError("valley_compact DensityVertices require lambda_compact")
            self.lambda_compact = np.asarray(vertices.lambda_compact, dtype=complex)
            if self.lambda_compact.ndim != 6:
                raise ValueError(
                    "lambda_compact must have shape "
                    "(n_q, n_g, n_blocks, 2, n_active, n_active)"
                )
            if self.lambda_compact.shape[2] != self.n_blocks:
                raise ValueError("density vertices and h0 have incompatible block counts")
            if self.lambda_compact.shape[3] != 2:
                raise ValueError("lambda_compact must have exactly two valley blocks")
            if self.lambda_compact.shape[-1] != self.lambda_compact.shape[-2]:
                raise ValueError("lambda_compact active-band blocks must be square")
            self.n_active = int(self.lambda_compact.shape[-1])
            if 2 * self.n_active != self.dim:
                raise ValueError("lambda_compact active-band dimension is incompatible with h0")
            self.lambda_blocks = np.asarray(vertices.lambda_blocks, dtype=complex)
            self.n_q, self.n_g = self.lambda_compact.shape[:2]
        elif self.vertex_layout == "dense":
            self.lambda_blocks = np.asarray(vertices.lambda_blocks, dtype=complex)
            if self.lambda_blocks.ndim != 5:
                raise ValueError("lambda_blocks must have shape (n_q, n_g, n_blocks, dim, dim)")
            if self.lambda_blocks.shape[2:] != self.h0.shape:
                raise ValueError("density vertices and h0 have incompatible shapes")
            self.lambda_compact = None
            self.n_active = self.dim // 2
            self.n_q, self.n_g = self.lambda_blocks.shape[:2]
        else:
            raise ValueError("density vertex layout must be 'dense' or 'valley_compact'")
        self.exchange_representation = self._resolve_exchange_representation()
        self.target_minus_q = np.asarray(
            (
                self.physical_channels.momentum_permutation
                if self.physical_channels is not None
                else vertices.target_minus_q
            ),
            dtype=int,
        )
        if self.target_minus_q.shape != (self.n_q, self.n_blocks):
            raise ValueError("target_minus_q must have shape (n_q, n_blocks)")
        self.q_is_zero = np.asarray(
            (
                np.all(self.physical_channels.mesh_transfer == 0, axis=1)
                if self.physical_channels is not None
                else vertices.q_is_zero
            ),
            dtype=bool,
        )
        if self.q_is_zero.shape != (self.n_q,):
            raise ValueError("q_is_zero must have shape (n_q,)")
        self.v_over_a = np.asarray(
            (
                self.physical_channels.weight[:, None]
                if self.physical_channels is not None
                else vertices.v_over_a
            ),
            dtype=float,
        )
        if self.v_over_a.shape != (self.n_q, self.n_g):
            raise ValueError("v_over_a must have shape (n_q, n_g)")
        self.p_ref = np.zeros_like(self.h0, dtype=complex)
        self.hartree_channels = self._find_hartree_channels()
        self.full_hartree_channels = tuple(self.hartree_channels)
        self._hartree_channel_lookup = {
            (int(iq), int(ig)): (int(iq), int(ig), float(v))
            for iq, ig, v in self.hartree_channels
        }
        self._uniform_hartree_records = self._capture_uniform_hartree_records()
        self.tVE: np.ndarray | None = None
        self.valley_sector_exchange: ValleySectorExchange | None = None
        if self.exchange_representation == "valley_sector":
            prebuilt_sectors = getattr(vertices, "prebuilt_exchange_sectors", None)
            if prebuilt_sectors is None:
                self.valley_sector_exchange = self._build_valley_sector_exchange()
            else:
                self.valley_sector_exchange = ValleySectorExchange(
                    sectors=np.asarray(prebuilt_sectors, dtype=complex),
                    n_blocks=self.n_blocks,
                    n_active=self.n_active,
                )
        else:
            self.tVE = self._build_exchange_tve()
        self._apply_density_vertex_retention()

    def as_block_density(self, P: np.ndarray) -> np.ndarray:
        arr = np.asarray(P, dtype=complex)
        if arr.shape == self.h0.shape:
            return hermitize(arr)
        if arr.shape == (self.n_total, self.n_total):
            out = np.empty_like(self.h0)
            for ik in range(self.n_blocks):
                start = ik * self.dim
                out[ik] = arr[start : start + self.dim, start : start + self.dim]
            return hermitize(out)
        raise ValueError(
            f"density must have shape {self.h0.shape} or {(self.n_total, self.n_total)}; "
            f"got {arr.shape}"
        )

    def _resolve_exchange_representation(self) -> str:
        requested = str(getattr(self.interaction, "exchange_representation", "auto"))
        if requested == "auto":
            return "valley_sector" if self.vertex_layout == "valley_compact" else "dense"
        if requested == "dense":
            return "dense"
        if requested == "valley_sector":
            if self.vertex_layout != "valley_compact":
                raise ValueError("valley_sector exchange requires valley_compact density vertices")
            return "valley_sector"
        raise ValueError("exchange_representation must be 'auto', 'dense', or 'valley_sector'")

    def _find_hartree_channels(self) -> list[tuple[int, int, float]]:
        channels: list[tuple[int, int, float]] = []
        scale = float(self.interaction.hartree_scale)
        if scale == 0.0:
            return channels
        if self.physical_channels is not None:
            return [
                (int(index), 0, scale * float(self.v_over_a[int(index), 0]))
                for index in np.flatnonzero(self.physical_channels.hartree)
            ]
        for iq in range(self.n_q):
            if not bool(self.q_is_zero[iq]):
                continue
            for ig in range(self.n_g):
                if (
                    self.interaction.q0_hartree == "omit_uniform"
                    and self._is_uniform_channel(iq, ig)
                ):
                    continue
                v = scale * float(self.v_over_a[iq, ig])
                if v != 0.0:
                    channels.append((iq, ig, v))
        return channels

    def _capture_uniform_hartree_records(self) -> tuple[tuple[float, np.ndarray], ...]:
        """Copy uniform density vertices before optional Hartree-only retention."""

        records: list[tuple[float, np.ndarray]] = []
        scale = float(self.interaction.hartree_scale)
        for iq in range(self.n_q):
            for ig in range(self.n_g):
                if not self._is_uniform_channel(iq, ig):
                    continue
                v = scale * float(self.v_over_a[iq, ig])
                if v == 0.0:
                    continue
                lam = (
                    self.lambda_compact[iq, ig]
                    if self.vertex_layout == "valley_compact"
                    else self.lambda_blocks[iq, ig]
                )
                records.append((v, np.asarray(lam, dtype=complex).copy()))
        return tuple(records)

    def _is_uniform_channel(self, iq: int, ig: int) -> bool:
        if self.physical_channels is not None:
            transfer = self.physical_channels.physical_transfer_nm_inv[int(iq)]
            return bool(float(np.linalg.norm(transfer)) < 1e-12)
        if self.vertices.q_norm_nm_inv is not None:
            return bool(float(self.vertices.q_norm_nm_inv[int(iq), int(ig)]) < 1e-12)
        q = self.vertices.q_shifts[int(iq)]
        g = self.vertices.g_channels[int(ig)]
        return int(q[0]) + int(g[0]) == 0 and int(q[1]) + int(g[1]) == 0

    def _build_exchange_tve(self) -> np.ndarray:
        block_dim = self.dim * self.dim
        size = self.n_blocks * block_dim
        tVE = np.zeros((size, size), dtype=complex)
        scale = float(self.interaction.exchange_scale)
        if scale == 0.0:
            return tVE
        local = np.arange(block_dim)
        block_rows = np.arange(self.n_blocks)[:, None] * block_dim + local[None, :]
        if self.interaction.exchange_workers <= 1:
            self._build_exchange_tve_serial(tVE, block_rows, block_dim, scale)
        else:
            self._build_exchange_tve_parallel(tVE, block_rows, block_dim, scale)
        return _hermitize_dense_in_place(tVE)

    def _scatter_exchange_tve_q_contribution(
        self,
        tVE: np.ndarray,
        block_rows: np.ndarray,
        block_dim: int,
        iq: int,
        forward: np.ndarray,
        reverse: np.ndarray,
    ) -> None:
        local = np.arange(block_dim)
        target_rows = self.target_minus_q[int(iq), :, None] * block_dim + local[None, :]
        tVE[block_rows[:, :, None], target_rows[:, None, :]] += forward.reshape(
            self.n_blocks,
            block_dim,
            block_dim,
        )
        tVE[target_rows[:, :, None], block_rows[:, None, :]] += reverse.reshape(
            self.n_blocks,
            block_dim,
            block_dim,
        )

    def _build_exchange_tve_serial(
        self,
        tVE: np.ndarray,
        block_rows: np.ndarray,
        block_dim: int,
        scale: float,
    ) -> None:
        for iq in range(self.n_q):
            if self.vertex_layout == "valley_compact":
                rows = _exchange_tve_q_slab_compact(
                    q_start=iq,
                    q_stop=iq + 1,
                    lambda_compact=self.lambda_compact,
                    v_over_a=self.v_over_a,
                    exchange_scale=scale,
                )
            else:
                rows = _exchange_tve_q_slab(
                    q_start=iq,
                    q_stop=iq + 1,
                    lambda_blocks=self.lambda_blocks,
                    v_over_a=self.v_over_a,
                    exchange_scale=scale,
                )
            for q_index, forward, reverse in rows:
                self._scatter_exchange_tve_q_contribution(
                    tVE,
                    block_rows,
                    block_dim,
                    q_index,
                    forward,
                    reverse,
                )

    def _build_exchange_tve_parallel(
        self,
        tVE: np.ndarray,
        block_rows: np.ndarray,
        block_dim: int,
        scale: float,
    ) -> None:
        from joblib import Parallel, delayed

        n_jobs = max(1, min(int(self.interaction.exchange_workers), self.n_q))
        ranges = _exchange_q_slab_ranges(self.n_q, n_jobs)
        if self.vertex_layout == "valley_compact":
            tasks = (
                delayed(_exchange_tve_q_slab_compact)(
                    q_start=start,
                    q_stop=stop,
                    lambda_compact=self.lambda_compact,
                    v_over_a=self.v_over_a,
                    exchange_scale=scale,
                )
                for start, stop in ranges
            )
        else:
            tasks = (
                delayed(_exchange_tve_q_slab)(
                    q_start=start,
                    q_stop=stop,
                    lambda_blocks=self.lambda_blocks,
                    v_over_a=self.v_over_a,
                    exchange_scale=scale,
                )
                for start, stop in ranges
            )
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            return_as="generator",
            mmap_mode="r",
            max_nbytes="32M",
            batch_size=1,
            pre_dispatch=n_jobs,
        )(tasks)
        for slab_rows in results:
            for q_index, forward, reverse in slab_rows:
                self._scatter_exchange_tve_q_contribution(
                    tVE,
                    block_rows,
                    block_dim,
                    q_index,
                    forward,
                    reverse,
                )

    def _build_valley_sector_exchange(self) -> ValleySectorExchange:
        if self.lambda_compact is None:
            raise ValueError("valley-sector exchange requires compact density vertices")
        block_dim = self.n_active * self.n_active
        size = self.n_blocks * block_dim
        sectors = np.zeros((2, 2, size, size), dtype=complex)
        scale = float(self.interaction.exchange_scale)
        if scale == 0.0:
            return ValleySectorExchange(sectors=sectors, n_blocks=self.n_blocks, n_active=self.n_active)
        local = np.arange(block_dim)
        block_rows = np.arange(self.n_blocks)[:, None] * block_dim + local[None, :]
        if self.interaction.exchange_workers <= 1:
            self._build_valley_sector_exchange_serial(sectors, block_rows, block_dim, scale)
        else:
            self._build_valley_sector_exchange_parallel(sectors, block_rows, block_dim, scale)
        for iv in range(2):
            for jv in range(2):
                _hermitize_dense_in_place(sectors[iv, jv])
        return ValleySectorExchange(sectors=sectors, n_blocks=self.n_blocks, n_active=self.n_active)

    def _scatter_valley_sector_exchange_q_contribution(
        self,
        sectors: np.ndarray,
        block_rows: np.ndarray,
        block_dim: int,
        iq: int,
        iv: int,
        jv: int,
        forward: np.ndarray,
        reverse: np.ndarray,
    ) -> None:
        local = np.arange(block_dim)
        target_rows = self.target_minus_q[int(iq), :, None] * block_dim + local[None, :]
        sectors[int(iv), int(jv)][block_rows[:, :, None], target_rows[:, None, :]] += (
            forward.reshape(self.n_blocks, block_dim, block_dim)
        )
        sectors[int(iv), int(jv)][target_rows[:, :, None], block_rows[:, None, :]] += (
            reverse.reshape(self.n_blocks, block_dim, block_dim)
        )

    def _build_valley_sector_exchange_serial(
        self,
        sectors: np.ndarray,
        block_rows: np.ndarray,
        block_dim: int,
        scale: float,
    ) -> None:
        for iq in range(self.n_q):
            rows = _exchange_sector_tve_q_slab_compact(
                q_start=iq,
                q_stop=iq + 1,
                lambda_compact=self.lambda_compact,
                v_over_a=self.v_over_a,
                exchange_scale=scale,
            )
            for q_index, iv, jv, forward, reverse in rows:
                self._scatter_valley_sector_exchange_q_contribution(
                    sectors,
                    block_rows,
                    block_dim,
                    q_index,
                    iv,
                    jv,
                    forward,
                    reverse,
                )

    def _build_valley_sector_exchange_parallel(
        self,
        sectors: np.ndarray,
        block_rows: np.ndarray,
        block_dim: int,
        scale: float,
    ) -> None:
        from joblib import Parallel, delayed

        n_jobs = max(1, min(int(self.interaction.exchange_workers), self.n_q))
        ranges = _exchange_q_slab_ranges(self.n_q, n_jobs)
        tasks = (
            delayed(_exchange_sector_tve_q_slab_compact)(
                q_start=start,
                q_stop=stop,
                lambda_compact=self.lambda_compact,
                v_over_a=self.v_over_a,
                exchange_scale=scale,
            )
            for start, stop in ranges
        )
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            return_as="generator",
            mmap_mode="r",
            max_nbytes="32M",
            batch_size=1,
            pre_dispatch=n_jobs,
        )(tasks)
        for slab_rows in results:
            for q_index, iv, jv, forward, reverse in slab_rows:
                self._scatter_valley_sector_exchange_q_contribution(
                    sectors,
                    block_rows,
                    block_dim,
                    q_index,
                    iv,
                    jv,
                    forward,
                    reverse,
                )

    def dense_exchange_tve_for_debug(self) -> np.ndarray:
        """Return the dense exchange superoperator for tests and diagnostics."""

        if self.exchange_representation == "valley_sector":
            if self.valley_sector_exchange is None:
                raise RuntimeError("missing valley-sector exchange object")
            return self.valley_sector_exchange.to_dense()
        if self.tVE is None:
            raise RuntimeError("missing dense exchange matrix")
        return self.tVE

    def _empty_retained_lambdas(self) -> np.ndarray:
        return np.zeros((0, 0, self.n_blocks, self.dim, self.dim), dtype=complex)

    def _empty_retained_lambda_compact(self) -> np.ndarray:
        return np.zeros(
            (0, 0, self.n_blocks, 2, self.n_active, self.n_active),
            dtype=complex,
        )

    def _vertices_without_lambda_blocks(
        self,
        *,
        physical_channels=None,
    ) -> DensityVertices:
        return DensityVertices(
            q_shifts=self.vertices.q_shifts,
            target_minus_q=np.asarray(self.vertices.target_minus_q, dtype=int).copy(),
            q_is_zero=np.asarray(self.vertices.q_is_zero, dtype=bool).copy(),
            lambda_blocks=self._empty_retained_lambdas(),
            v_over_a=np.asarray(self.vertices.v_over_a, dtype=float).copy(),
            g_channels=self.vertices.g_channels,
            vertex_layout=self.vertex_layout,
            lambda_compact=(
                self._empty_retained_lambda_compact()
                if self.vertex_layout == "valley_compact"
                else None
            ),
            channel_in_disk=(
                None
                if self.vertices.channel_in_disk is None
                else np.asarray(self.vertices.channel_in_disk, dtype=bool).copy()
            ),
            q_vectors_nm_inv=(
                None
                if self.vertices.q_vectors_nm_inv is None
                else np.asarray(self.vertices.q_vectors_nm_inv, dtype=float).copy()
            ),
            q_norm_nm_inv=(
                None
                if self.vertices.q_norm_nm_inv is None
                else np.asarray(self.vertices.q_norm_nm_inv, dtype=float).copy()
            ),
            v_q=(
                None
                if self.vertices.v_q is None
                else np.asarray(self.vertices.v_q, dtype=float).copy()
            ),
            physical_channels=physical_channels,
        )

    def _apply_density_vertex_retention(self) -> None:
        policy = str(self.interaction.density_vertex_retention)
        if policy == "full":
            return
        if policy != "hartree_only":
            raise ValueError("density_vertex_retention must be 'full' or 'hartree_only'")

        original_lambdas = self.lambda_blocks
        original_compact = self.lambda_compact
        original_targets = self.target_minus_q
        original_v_over_a = self.v_over_a
        retained_count = len(self.full_hartree_channels)
        if retained_count == 0:
            self.lambda_blocks = self._empty_retained_lambdas()
            if self.vertex_layout == "valley_compact":
                self.lambda_compact = self._empty_retained_lambda_compact()
            self.target_minus_q = np.zeros((0, self.n_blocks), dtype=int)
            self.q_is_zero = np.zeros(0, dtype=bool)
            self.v_over_a = np.zeros((0, 0), dtype=float)
            self.hartree_channels = []
            self._hartree_channel_lookup = {}
            self.n_q, self.n_g = 0, 0
            retained_physical = None
            if self.physical_channels is not None:
                retained_physical = PhysicalDensityChannels(
                    physical_transfer_nm_inv=np.zeros((0, 2), dtype=float),
                    momentum_permutation=np.zeros(
                        (0, self.n_blocks),
                        dtype=int,
                    ),
                    weight=np.zeros(0, dtype=float),
                    compact_form_factor=np.zeros(
                        (0, self.n_blocks, 2, self.n_active, self.n_active),
                        dtype=complex,
                    ),
                    hartree=np.zeros(0, dtype=bool),
                    mesh_transfer=np.zeros((0, 2), dtype=int),
                    candidate_index=np.zeros((0, 2), dtype=int),
                )
                self.physical_channels = retained_physical
            self.vertices = self._vertices_without_lambda_blocks(
                physical_channels=retained_physical,
            )
            return

        retained_lambdas = (
            None
            if self.vertex_layout == "valley_compact"
            else np.empty(
                (retained_count, 1, self.n_blocks, self.dim, self.dim),
                dtype=complex,
            )
        )
        retained_compact = (
            np.empty(
                (retained_count, 1, self.n_blocks, 2, self.n_active, self.n_active),
                dtype=complex,
            )
            if self.vertex_layout == "valley_compact"
            else None
        )
        retained_targets = np.empty((retained_count, self.n_blocks), dtype=int)
        retained_v_over_a = np.empty((retained_count, 1), dtype=float)
        retained_q_is_zero = np.ones(retained_count, dtype=bool)
        retained_channels: list[tuple[int, int, float]] = []
        lookup: dict[tuple[int, int], tuple[int, int, float]] = {}
        for new_iq, (old_iq, old_ig, v) in enumerate(self.full_hartree_channels):
            if self.vertex_layout == "valley_compact":
                if original_compact is None or retained_compact is None:
                    raise ValueError("missing compact vertices for hartree_only retention")
                retained_compact[new_iq, 0] = original_compact[int(old_iq), int(old_ig)]
            else:
                if retained_lambdas is None:
                    raise ValueError("missing dense vertices for hartree_only retention")
                retained_lambdas[new_iq, 0] = original_lambdas[int(old_iq), int(old_ig)]
            retained_targets[new_iq] = original_targets[int(old_iq)]
            retained_v_over_a[new_iq, 0] = original_v_over_a[int(old_iq), int(old_ig)]
            retained_channels.append((new_iq, 0, float(v)))
            lookup[(int(old_iq), int(old_ig))] = (new_iq, 0, float(v))

        if self.vertex_layout == "valley_compact":
            self.lambda_blocks = self._empty_retained_lambdas()
            self.lambda_compact = retained_compact
        else:
            if retained_lambdas is None:
                raise ValueError("missing dense vertices for hartree_only retention")
            self.lambda_blocks = retained_lambdas
        self.target_minus_q = retained_targets
        self.q_is_zero = retained_q_is_zero
        self.v_over_a = retained_v_over_a
        self.hartree_channels = retained_channels
        self._hartree_channel_lookup = lookup
        if self.vertex_layout == "valley_compact":
            self.n_q, self.n_g = self.lambda_compact.shape[:2]
        else:
            self.n_q, self.n_g = self.lambda_blocks.shape[:2]
        retained_physical = None
        if self.physical_channels is not None:
            old_indices = np.asarray(
                [old_iq for old_iq, _old_ig, _v in self.full_hartree_channels],
                dtype=int,
            )
            retained_physical = PhysicalDensityChannels(
                physical_transfer_nm_inv=self.physical_channels.physical_transfer_nm_inv[
                    old_indices
                ].copy(),
                momentum_permutation=retained_targets.copy(),
                weight=retained_v_over_a[:, 0].copy(),
                compact_form_factor=np.asarray(retained_compact[:, 0]).copy(),
                hartree=np.ones(retained_count, dtype=bool),
                mesh_transfer=self.physical_channels.mesh_transfer[
                    old_indices
                ].copy(),
                candidate_index=self.physical_channels.candidate_index[
                    old_indices
                ].copy(),
            )
            self.physical_channels = retained_physical
        self.vertices = self._vertices_without_lambda_blocks(
            physical_channels=retained_physical,
        )

    def hartree_lambda_for_channel(self, iq: int, ig: int) -> np.ndarray | None:
        retained = self._hartree_channel_lookup.get((int(iq), int(ig)))
        if retained is None:
            return None
        retained_iq, retained_ig, _v = retained
        if self.vertex_layout == "valley_compact":
            return dense_lambdas_from_compact(
                self.lambda_compact[int(retained_iq), int(retained_ig)]
            )
        return self.lambda_blocks[int(retained_iq), int(retained_ig)]

    def hartree_weight_for_channel(self, iq: int, ig: int) -> float | None:
        retained = self._hartree_channel_lookup.get((int(iq), int(ig)))
        if retained is None:
            return None
        return float(retained[2])

    def _compact_channel_density_trace(self, lam: np.ndarray, density: np.ndarray) -> complex:
        rho = 0.0 + 0.0j
        for iv in range(2):
            start = iv * self.n_active
            stop = start + self.n_active
            rho += np.einsum(
                "kab,kba->",
                lam[:, iv],
                density[:, start:stop, start:stop],
                optimize=True,
            )
        return rho

    def _add_compact_hartree_channel(
        self,
        out: np.ndarray,
        lam: np.ndarray,
        rho: complex,
        v: float,
    ) -> None:
        for iv in range(2):
            start = iv * self.n_active
            stop = start + self.n_active
            lam_v = lam[:, iv]
            out[:, start:stop, start:stop] += 0.5 * v * (
                np.conj(rho) * lam_v + rho * np.swapaxes(lam_v.conj(), -1, -2)
            )

    def hartree_hamiltonian(self, Q: np.ndarray) -> np.ndarray:
        density = self.as_block_density(Q)
        out = np.zeros_like(density, dtype=complex)
        for iq, ig, v in self.hartree_channels:
            if self.vertex_layout == "valley_compact":
                lam = self.lambda_compact[iq, ig]
                rho = self._compact_channel_density_trace(lam, density)
                self._add_compact_hartree_channel(out, lam, rho, v)
            else:
                lam = self.lambda_blocks[iq, ig]
                rho = np.einsum("kab,kba->", lam, density, optimize=True)
                out += 0.5 * v * (
                    np.conj(rho) * lam + rho * np.swapaxes(lam.conj(), -1, -2)
                )
        return hermitize(out)

    def fock_hamiltonian(self, Q: np.ndarray) -> np.ndarray:
        density = self.as_block_density(Q)
        if self.exchange_representation == "valley_sector":
            if self.valley_sector_exchange is None:
                raise RuntimeError("missing valley-sector exchange object")
            out = np.zeros_like(density, dtype=complex)
            for iv in range(2):
                a_slice = slice(iv * self.n_active, (iv + 1) * self.n_active)
                for jv in range(2):
                    b_slice = slice(jv * self.n_active, (jv + 1) * self.n_active)
                    out[:, a_slice, b_slice] = -self.valley_sector_exchange.matvec_sector(
                        iv,
                        jv,
                        density[:, a_slice, b_slice],
                    )
        else:
            if self.tVE is None:
                raise RuntimeError("missing dense exchange matrix")
            out = -np.reshape(self.tVE @ np.reshape(density, (-1,)), density.shape)
        return hermitize(out)

    def self_energy(self, Q: np.ndarray) -> np.ndarray:
        """Return the linear Hartree-plus-Fock map evaluated on ``Q``."""

        density = self.as_block_density(Q)
        return hermitize(
            self.hartree_hamiltonian(density) + self.fock_hamiltonian(density)
        )

    def hf_hamiltonian(self, P: np.ndarray) -> np.ndarray:
        density = self.as_block_density(P)
        Q = density - self.p_ref
        return hermitize(self.h0 + self.self_energy(Q))

    def hartree_energy(self, Q: np.ndarray) -> float:
        """Evaluate the direct-channel quadratic form without rebuilding Fock."""

        density = self.as_block_density(Q)
        hartree = 0.0
        for iq, ig, v in self.hartree_channels:
            if self.vertex_layout == "valley_compact":
                lam = self.lambda_compact[iq, ig]
                rho = self._compact_channel_density_trace(lam, density)
            else:
                lam = self.lambda_blocks[iq, ig]
                rho = np.einsum("kab,kba->", lam, density, optimize=True)
            hartree += 0.5 * v * float(np.real(rho * np.conj(rho)))
        return float(hartree)

    def total_energy_from_fields(
        self,
        P: np.ndarray,
        hartree_field: np.ndarray,
        fock_field: np.ndarray,
    ) -> float:
        """Evaluate the HF energy using fields already matched to ``P``.

        The direct-channel energy is evaluated from the same explicit channel
        amplitudes as the full functional. The Fock quadratic form satisfies
        ``E_F=Tr(H_F Q)/2``. Keeping the cached fields separate therefore
        avoids a second expensive Fock application without depending on an
        additional Hartree trace identity.
        """

        density = self.as_block_density(P)
        direct_field = hermitize(np.asarray(hartree_field, dtype=complex))
        exchange_field = hermitize(np.asarray(fock_field, dtype=complex))
        if direct_field.shape != self.h0.shape:
            raise ValueError("hartree_field has the wrong shape")
        if exchange_field.shape != self.h0.shape:
            raise ValueError("fock_field has the wrong shape")
        relative_density = density - self.p_ref
        one_body = block_trace_product(self.h0, density)
        hartree = self.hartree_energy(relative_density)
        fock = 0.5 * block_trace_product(exchange_field, relative_density)
        return float(one_body + hartree + fock)

    def interaction_components(self, Q: np.ndarray) -> tuple[float, float]:
        density = self.as_block_density(Q)
        hartree = self.hartree_energy(density)
        fock = 0.5 * block_trace_product(self.fock_hamiltonian(density), density)
        return hartree, fock

    def interaction_energy(self, Q: np.ndarray) -> float:
        hartree, fock = self.interaction_components(Q)
        return float(hartree + fock)

    def uniform_hartree_energy(self, Q: np.ndarray) -> float:
        """Reconstruct the omitted uniform q=0 Hartree charging contribution.

        Native HF uses ``q0_hartree='omit_uniform'``.  This helper evaluates
        that excluded device-capacitance term with the same vertices and
        ``V(q->0)/A`` normalization for explicit SET postprocessing.
        """

        density = self.as_block_density(Q)
        energy = 0.0
        for v, lam in self._uniform_hartree_records:
            if self.vertex_layout == "valley_compact":
                rho = self._compact_channel_density_trace(lam, density)
            else:
                rho = np.einsum("kab,kba->", lam, density, optimize=True)
            energy += 0.5 * v * float(np.real(rho * np.conj(rho)))
        return float(energy)

    def energy(self, P: np.ndarray) -> EnergyComponents:
        density = self.as_block_density(P)
        one_body = block_trace_product(self.h0, density)
        hartree, fock = self.interaction_components(density - self.p_ref)
        return EnergyComponents(
            total=float(one_body + hartree + fock),
            one_body=float(one_body),
            hartree=float(hartree),
            fock=float(fock),
        )

    def total_energy(self, P: np.ndarray) -> float:
        return self.energy(P).total

    def update_density(
        self,
        H: np.ndarray,
        n_particles: float,
        constraint=None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        blocks = hermitize(np.asarray(H, dtype=complex))
        if blocks.shape != self.h0.shape:
            raise ValueError("H has the wrong shape")
        if constraint is not None and hasattr(constraint, "update_density_global"):
            return constraint.update_density_global(blocks, n_particles)
        if constraint is not None:
            blocks = constraint.project_operator(blocks)
        P, evals, direct, indirect = _global_aufbau(blocks, n_particles)
        if constraint is not None:
            P = constraint.project_density(P)
        return P, evals, direct, indirect

    def update_density_per_k(
        self,
        H: np.ndarray,
        n_occ_per_k: int,
        constraint=None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        blocks = hermitize(np.asarray(H, dtype=complex))
        if blocks.shape != self.h0.shape:
            raise ValueError("H has the wrong shape")
        if constraint is not None and hasattr(constraint, "update_density"):
            return constraint.update_density(blocks, n_occ_per_k)
        if constraint is not None:
            blocks = constraint.project_operator(blocks)
        return _fixed_per_k_aufbau(blocks, n_occ_per_k)


def _commutator_norm(H: np.ndarray, P: np.ndarray) -> float:
    comm = np.einsum("kab,kbc->kac", H, P, optimize=True) - np.einsum(
        "kab,kbc->kac", P, H, optimize=True
    )
    return float(np.linalg.norm(comm))


def _expected_trace(backend: ContinuumHFBackend, params: ContinuumHFParams) -> float:
    return float(backend.n_blocks * params.n_occ_per_k)


def _choose_oda_lambda(s: float, c: float, lambda_min: float) -> tuple[float, str | None]:
    """Choose the ODA damping lambda for E(lambda)=E0+s lambda+c lambda^2/2."""

    if c <= 0.0:
        return 1.0, "nonpositive_curvature_full_step"
    lam = float(np.clip(-float(s) / float(c), 0.0, 1.0))
    if lam <= float(lambda_min):
        return 1.0, "zero_lambda_full_step"
    return lam, None


def compute_hf_diagnostics(
    backend: ContinuumHFBackend,
    P: np.ndarray,
    params: ContinuumHFParams,
    *,
    constraint=None,
    P_prev: np.ndarray | None = None,
    energy_prev: float | None = None,
    iteration: int = 0,
    density_kind: Literal["mixed", "final_idempotent"] = "mixed",
    lambda_value: float | None = None,
    fallback_reason: str | None = None,
    working_hamiltonian: np.ndarray | None = None,
    working_energy: float | None = None,
) -> ContinuumHFDiagnostics:
    """Compute scalar diagnostics for one density."""

    density = backend.as_block_density(P)
    H = (
        backend.hf_hamiltonian(density)
        if working_hamiltonian is None
        else hermitize(np.asarray(working_hamiltonian, dtype=complex))
    )
    if H.shape != backend.h0.shape:
        raise ValueError("working_hamiltonian has the wrong shape")
    H_projected = constraint.project_operator(H) if constraint is not None else H
    expected_trace = _expected_trace(backend, params)
    P_aufbau, _evals, direct, indirect = backend.update_density_per_k(
        H_projected,
        params.n_occ_per_k,
        constraint,
    )
    energy = (
        backend.energy(density).total
        if working_energy is None
        else float(working_energy)
    )
    idem_fro, idem_max = projector_idempotency_errors(density)
    trace = np.trace(density, axis1=-2, axis2=-1)
    residual = float(np.linalg.norm(density - P_aufbau))
    constraint_error = (
        float(constraint.symmetry_error(density)) if constraint is not None else 0.0
    )
    return ContinuumHFDiagnostics(
        energy=float(energy),
        delta_energy=float("nan") if energy_prev is None else float(energy - energy_prev),
        delta_P=float("nan")
        if P_prev is None
        else float(np.linalg.norm(density - backend.as_block_density(P_prev))),
        idempotency_error_fro=idem_fro,
        idempotency_error_max=idem_max,
        constraint_error=constraint_error,
        aufbau_residual_norm=residual,
        commutator_norm=_commutator_norm(H_projected, density),
        trace_error=float(abs(np.real(np.sum(trace)) - expected_trace)),
        direct_gap_min=float(direct),
        indirect_gap=float(indirect),
        iteration=int(iteration),
        constraint_name=getattr(constraint, "name", None) if constraint is not None else None,
        lambda_value=lambda_value,
        fallback_reason=fallback_reason,
        density_kind=density_kind,
        self_consistency_warning=bool(residual > params.final_residual_tolerance),
    )


def compute_global_hf_diagnostics(
    backend: ContinuumHFBackend,
    P: np.ndarray,
    n_particles: float,
    params: ContinuumHFParams,
    *,
    constraint=None,
    P_prev: np.ndarray | None = None,
    energy_prev: float | None = None,
    iteration: int = 0,
    density_kind: Literal["mixed", "final_idempotent"] = "mixed",
    lambda_value: float | None = None,
    fallback_reason: str | None = None,
    working_hamiltonian: np.ndarray | None = None,
    working_energy: float | None = None,
) -> ContinuumHFDiagnostics:
    """Compute diagnostics for a globally filled zero-temperature density."""

    density = backend.as_block_density(P)
    H = (
        backend.hf_hamiltonian(density)
        if working_hamiltonian is None
        else hermitize(np.asarray(working_hamiltonian, dtype=complex))
    )
    if H.shape != backend.h0.shape:
        raise ValueError("working_hamiltonian has the wrong shape")
    H_projected = constraint.project_operator(H) if constraint is not None else H
    P_aufbau, _evals, direct, indirect = backend.update_density(
        H_projected,
        n_particles,
        constraint,
    )
    energy = (
        backend.energy(density).total
        if working_energy is None
        else float(working_energy)
    )
    idem_fro, idem_max = projector_idempotency_errors(density)
    trace = np.trace(density, axis1=-2, axis2=-1)
    residual = float(np.linalg.norm(density - P_aufbau))
    constraint_error = (
        float(constraint.symmetry_error(density)) if constraint is not None else 0.0
    )
    return ContinuumHFDiagnostics(
        energy=float(energy),
        delta_energy=float("nan") if energy_prev is None else float(energy - energy_prev),
        delta_P=float("nan")
        if P_prev is None
        else float(np.linalg.norm(density - backend.as_block_density(P_prev))),
        idempotency_error_fro=idem_fro,
        idempotency_error_max=idem_max,
        constraint_error=constraint_error,
        aufbau_residual_norm=residual,
        commutator_norm=_commutator_norm(H_projected, density),
        trace_error=float(abs(np.real(np.sum(trace)) - float(n_particles))),
        direct_gap_min=float(direct),
        indirect_gap=float(indirect),
        iteration=int(iteration),
        constraint_name=getattr(constraint, "name", None) if constraint is not None else None,
        lambda_value=lambda_value,
        fallback_reason=fallback_reason,
        occupation_mode="global",
        density_kind=density_kind,
        self_consistency_warning=bool(residual > params.final_residual_tolerance),
    )


def retarget_global_density(
    backend: ContinuumHFBackend,
    P_reference: np.ndarray,
    n_particles: float,
    *,
    constraint=None,
) -> np.ndarray:
    """Build a global-Aufbau seed at a new particle number from a reference HF field."""

    H = backend.hf_hamiltonian(backend.as_block_density(P_reference))
    H_projected = constraint.project_operator(H) if constraint is not None else H
    P, _evals, _direct, _indirect = backend.update_density(
        H_projected,
        n_particles,
        constraint,
    )
    return P


def _working_hf_state(
    backend: ContinuumHFBackend,
    P: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return a density and its matched Hartree/Fock fields, Hamiltonian, and energy."""

    density = backend.as_block_density(P)
    relative_density = density - backend.p_ref
    hartree_field = backend.hartree_hamiltonian(relative_density)
    fock_field = backend.fock_hamiltonian(relative_density)
    hamiltonian = hermitize(backend.h0 + hartree_field + fock_field)
    energy = backend.total_energy_from_fields(
        density,
        hartree_field,
        fock_field,
    )
    return density, hartree_field, fock_field, hamiltonian, energy


def solve_hf(
    backend: ContinuumHFBackend,
    P_init: np.ndarray,
    params: ContinuumHFParams | None = None,
    *,
    constraint=None,
    seed: str = "",
    on_iteration: HFIterationCallback | None = None,
) -> ContinuumHFResult:
    """Run zero-temperature fixed-per-k HF with TMD_HF-style ODA mixing."""

    controls = params or ContinuumHFParams()
    energy_tol = controls.energy_tolerance
    P = backend.as_block_density(P_init)
    if constraint is not None:
        P = constraint.project_density(P)
    (
        P,
        working_hartree,
        working_fock,
        working_hamiltonian,
        energy,
    ) = _working_hf_state(
        backend,
        P,
    )
    history: list[ContinuumHFDiagnostics] = []
    snapshots: list[ContinuumHFIterationSnapshot] = []
    converged = False
    diagnostics = compute_hf_diagnostics(
        backend,
        P,
        controls,
        constraint=constraint,
        iteration=0,
        working_hamiltonian=working_hamiltonian,
        working_energy=energy,
    )
    n_iter = 0
    for iteration in range(1, controls.max_iter + 1):
        n_iter = iteration
        P_prev = P
        hartree_prev = working_hartree
        fock_prev = working_fock
        energy_prev = energy
        H_prev = working_hamiltonian
        H_projected = constraint.project_operator(H_prev) if constraint is not None else H_prev
        P_aufbau, _evals, _direct, _indirect = backend.update_density_per_k(
            H_projected,
            controls.n_occ_per_k,
            constraint,
        )
        delta = hermitize(P_aufbau - P_prev)
        trial_relative_density = P_aufbau - backend.p_ref
        trial_hartree = backend.hartree_hamiltonian(trial_relative_density)
        trial_fock = backend.fock_hamiltonian(trial_relative_density)
        delta_hartree = hermitize(trial_hartree - hartree_prev)
        delta_fock = hermitize(trial_fock - fock_prev)
        fallback_reason = None
        if controls.mixing_method == "oda":
            s = block_trace_product(H_projected, delta)
            c = (
                2.0 * backend.hartree_energy(delta)
                + block_trace_product(delta_fock, delta)
            )
            mix, fallback_reason = _choose_oda_lambda(s, c, controls.oda_lambda_min)
        else:
            mix = float(controls.mixing)
        P = hermitize(P_prev + mix * delta)
        working_hartree = hermitize(hartree_prev + mix * delta_hartree)
        working_fock = hermitize(fock_prev + mix * delta_fock)
        if constraint is not None:
            P = constraint.project_density(P)
        working_hamiltonian = hermitize(
            backend.h0 + working_hartree + working_fock
        )
        energy = backend.total_energy_from_fields(
            P,
            working_hartree,
            working_fock,
        )
        diagnostics = compute_hf_diagnostics(
            backend,
            P,
            controls,
            constraint=constraint,
            P_prev=P_prev,
            energy_prev=energy_prev,
            iteration=iteration,
            lambda_value=float(mix),
            fallback_reason=fallback_reason,
            working_hamiltonian=working_hamiltonian,
            working_energy=energy,
        )
        history.append(diagnostics)
        should_snapshot = controls.store_projector_snapshots and (
            (controls.first_iteration_snapshot and iteration == 1)
            or (iteration % controls.snapshot_interval == 0)
        )
        if should_snapshot:
            snapshots.append(
                ContinuumHFIterationSnapshot(
                    iteration=iteration,
                    P=P.copy(),
                    energy=float(diagnostics.energy),
                    diagnostics=diagnostics,
                )
            )
        if on_iteration is not None:
            on_iteration(
                iteration,
                P.copy(),
                float(diagnostics.energy),
                diagnostics,
                bool(should_snapshot),
            )
        if (
            iteration >= controls.min_iter
            and diagnostics.commutator_norm < controls.tolerance
            and diagnostics.aufbau_residual_norm < controls.tolerance
            and diagnostics.constraint_error < controls.tolerance
            and diagnostics.trace_error < controls.tolerance
            and abs(diagnostics.delta_energy) < energy_tol
        ):
            converged = True
            break

    H_mixed = working_hamiltonian
    H_projected = constraint.project_operator(H_mixed) if constraint is not None else H_mixed
    P_final, _evals, _direct, _indirect = backend.update_density_per_k(
        H_projected,
        controls.n_occ_per_k,
        constraint,
    )
    (
        P_final,
        _final_hartree,
        _final_fock,
        final_H_raw,
        final_energy,
    ) = _working_hf_state(backend, P_final)
    final_H = constraint.project_operator(final_H_raw) if constraint is not None else final_H_raw
    final_diagnostics = compute_hf_diagnostics(
        backend,
        P_final,
        controls,
        constraint=constraint,
        P_prev=P,
        energy_prev=diagnostics.energy,
        iteration=n_iter,
        density_kind="final_idempotent",
        working_hamiltonian=final_H_raw,
        working_energy=final_energy,
    )
    if final_diagnostics.self_consistency_warning:
        converged = False
    return ContinuumHFResult(
        P=P_final,
        H_hf=final_H,
        energy=final_energy,
        converged=converged,
        n_iter=n_iter,
        diagnostics=final_diagnostics,
        history=tuple(history),
        snapshots=tuple(snapshots),
        seed=seed,
        constraint_name=getattr(constraint, "name", None) if constraint is not None else None,
    )


def solve_global_hf(
    backend: ContinuumHFBackend,
    P_init: np.ndarray,
    n_particles: float,
    params: ContinuumHFParams | None = None,
    *,
    constraint=None,
    seed: str = "",
    on_iteration: HFIterationCallback | None = None,
) -> ContinuumHFResult:
    """Run zero-temperature HF with global filling across momentum blocks."""

    controls = params or ContinuumHFParams()
    target = float(n_particles)
    if target < 0.0 or target > backend.n_total:
        raise ValueError(f"n_particles must be in [0, {backend.n_total}]")
    energy_tol = controls.energy_tolerance
    P = backend.as_block_density(P_init)
    if constraint is not None:
        P = constraint.project_density(P)
    trace = float(np.real(np.trace(P, axis1=-2, axis2=-1).sum()))
    if abs(trace - target) > controls.tolerance:
        P = retarget_global_density(
            backend,
            P,
            target,
            constraint=constraint,
        )
    (
        P,
        working_hartree,
        working_fock,
        working_hamiltonian,
        energy,
    ) = _working_hf_state(
        backend,
        P,
    )
    history: list[ContinuumHFDiagnostics] = []
    snapshots: list[ContinuumHFIterationSnapshot] = []
    converged = False
    diagnostics = compute_global_hf_diagnostics(
        backend,
        P,
        target,
        controls,
        constraint=constraint,
        iteration=0,
        working_hamiltonian=working_hamiltonian,
        working_energy=energy,
    )
    n_iter = 0
    for iteration in range(1, controls.max_iter + 1):
        n_iter = iteration
        P_prev = P
        hartree_prev = working_hartree
        fock_prev = working_fock
        energy_prev = energy
        H_prev = working_hamiltonian
        H_projected = constraint.project_operator(H_prev) if constraint is not None else H_prev
        P_aufbau, _evals, _direct, _indirect = backend.update_density(
            H_projected,
            target,
            constraint,
        )
        delta = hermitize(P_aufbau - P_prev)
        trial_relative_density = P_aufbau - backend.p_ref
        trial_hartree = backend.hartree_hamiltonian(trial_relative_density)
        trial_fock = backend.fock_hamiltonian(trial_relative_density)
        delta_hartree = hermitize(trial_hartree - hartree_prev)
        delta_fock = hermitize(trial_fock - fock_prev)
        fallback_reason = None
        if controls.mixing_method == "oda":
            s = block_trace_product(H_projected, delta)
            c = (
                2.0 * backend.hartree_energy(delta)
                + block_trace_product(delta_fock, delta)
            )
            mix, fallback_reason = _choose_oda_lambda(s, c, controls.oda_lambda_min)
        else:
            mix = float(controls.mixing)
        P = hermitize(P_prev + mix * delta)
        working_hartree = hermitize(hartree_prev + mix * delta_hartree)
        working_fock = hermitize(fock_prev + mix * delta_fock)
        if constraint is not None:
            P = constraint.project_density(P)
        working_hamiltonian = hermitize(
            backend.h0 + working_hartree + working_fock
        )
        energy = backend.total_energy_from_fields(
            P,
            working_hartree,
            working_fock,
        )
        diagnostics = compute_global_hf_diagnostics(
            backend,
            P,
            target,
            controls,
            constraint=constraint,
            P_prev=P_prev,
            energy_prev=energy_prev,
            iteration=iteration,
            lambda_value=float(mix),
            fallback_reason=fallback_reason,
            working_hamiltonian=working_hamiltonian,
            working_energy=energy,
        )
        history.append(diagnostics)
        should_snapshot = controls.store_projector_snapshots and (
            (controls.first_iteration_snapshot and iteration == 1)
            or (iteration % controls.snapshot_interval == 0)
        )
        if should_snapshot:
            snapshots.append(
                ContinuumHFIterationSnapshot(
                    iteration=iteration,
                    P=P.copy(),
                    energy=float(diagnostics.energy),
                    diagnostics=diagnostics,
                )
            )
        if on_iteration is not None:
            on_iteration(
                iteration,
                P.copy(),
                float(diagnostics.energy),
                diagnostics,
                bool(should_snapshot),
            )
        if (
            iteration >= controls.min_iter
            and diagnostics.commutator_norm < controls.tolerance
            and diagnostics.aufbau_residual_norm < controls.tolerance
            and diagnostics.constraint_error < controls.tolerance
            and diagnostics.trace_error < controls.tolerance
            and abs(diagnostics.delta_energy) < energy_tol
        ):
            converged = True
            break

    H_mixed = working_hamiltonian
    H_projected = constraint.project_operator(H_mixed) if constraint is not None else H_mixed
    P_final, _evals, _direct, _indirect = backend.update_density(
        H_projected,
        target,
        constraint,
    )
    (
        P_final,
        _final_hartree,
        _final_fock,
        final_H_raw,
        final_energy,
    ) = _working_hf_state(backend, P_final)
    final_H = constraint.project_operator(final_H_raw) if constraint is not None else final_H_raw
    final_diagnostics = compute_global_hf_diagnostics(
        backend,
        P_final,
        target,
        controls,
        constraint=constraint,
        P_prev=P,
        energy_prev=diagnostics.energy,
        iteration=n_iter,
        density_kind="final_idempotent",
        working_hamiltonian=final_H_raw,
        working_energy=final_energy,
    )
    if final_diagnostics.self_consistency_warning:
        converged = False
    return ContinuumHFResult(
        P=P_final,
        H_hf=final_H,
        energy=final_energy,
        converged=converged,
        n_iter=n_iter,
        diagnostics=final_diagnostics,
        history=tuple(history),
        snapshots=tuple(snapshots),
        seed=seed,
        constraint_name=getattr(constraint, "name", None) if constraint is not None else None,
    )


def scalar_channel(blocks: np.ndarray) -> np.ndarray:
    arr = np.asarray(blocks, dtype=complex)
    dim = arr.shape[-1]
    trace = np.trace(arr, axis1=-2, axis2=-1) / float(dim)
    return trace[..., None, None] * np.eye(dim, dtype=complex)


def intervalley_channel(blocks: np.ndarray) -> np.ndarray:
    arr = np.asarray(blocks, dtype=complex)
    dim = arr.shape[-1]
    if dim % 2:
        raise ValueError("active-space dimension must be even")
    n = dim // 2
    out = np.zeros_like(arr)
    out[..., :n, n:] = arr[..., :n, n:]
    out[..., n:, :n] = arr[..., n:, :n]
    return hermitize(out)


def valley_diagonal_channel(blocks: np.ndarray) -> np.ndarray:
    arr = np.asarray(blocks, dtype=complex)
    return hermitize(arr - intervalley_channel(arr))


def reference_hamiltonian_diagnostics(H: np.ndarray) -> ReferenceHamiltonianDiagnostics:
    arr = np.asarray(H, dtype=complex)
    scalar = scalar_channel(arr)
    return ReferenceHamiltonianDiagnostics(
        scalar_norm=float(np.linalg.norm(scalar)),
        traceless_norm=float(np.linalg.norm(arr - scalar)),
        valley_diagonal_norm=float(np.linalg.norm(valley_diagonal_channel(arr))),
        intervalley_norm=float(np.linalg.norm(intervalley_channel(arr))),
        hermiticity_error=float(np.max(np.abs(arr - arr.conj().swapaxes(-1, -2)))),
    )
