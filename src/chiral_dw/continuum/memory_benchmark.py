"""Local Taige memory-backend prototypes and benchmark helpers.

The helpers in this module are intentionally opt-in.  They are used to compare
candidate HF memory representations against the production dense backend
without changing the default sweep path.
"""

from __future__ import annotations

import gc
import csv
import json
import os
import resource
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.linalg import blas

from chiral_dw.config import (
    ContinuumGridParams,
    ContinuumHFParams,
    ContinuumInteractionParams,
    ContinuumModelParams,
)
from chiral_dw.continuum.builder import build_active_space
from chiral_dw.continuum.hf import (
    ContinuumHFBackend,
    EnergyComponents,
    _exchange_tve_q_slab,
    _hermitize_dense_in_place,
)
from chiral_dw.continuum.models import (
    ContinuumActiveSpace,
    ContinuumBundle,
    DensityVertices,
    MomentumGrid,
    block_trace_product,
    hermitize,
)
from chiral_dw.continuum.references import build_symmetric_hf_references
from chiral_dw.continuum.symmetry import _fixed_per_k_aufbau
from chiral_dw.continuum.taige import (
    _channel_mask,
    _dimensionless_v_over_a,
    _physical_v_over_a,
    _taige_density_vertex_q_slab,
    build_taige_density_vertices,
    q_transfers,
    reciprocal_box,
    taige_model_params,
)

BackendVariant = Literal[
    "baseline",
    "hartree_only",
    "fused",
    "compact",
    "fused_compact",
    "packed",
    "matrix_free",
    "complex64",
]

_ALL_VARIANTS: tuple[BackendVariant, ...] = (
    "baseline",
    "hartree_only",
    "fused",
    "compact",
    "fused_compact",
    "packed",
    "matrix_free",
    "complex64",
)


class TaigeMemoryBenchmarkInput(BaseModel):
    """Inputs for one local Taige backend-memory benchmark worker."""

    model_config = ConfigDict(frozen=True)

    n_k: int = Field(default=6, ge=1)
    u_D: float = 0.0
    theta_deg: float = 3.5
    plane_wave_shell: int = Field(default=5, ge=0)
    n_bands: int = Field(default=2, ge=1)
    n_active_bands_per_valley: int = Field(default=2, ge=1)
    q_mesh: Literal["shell", "full"] = "full"
    q_shell: int = Field(default=0, ge=0)
    local_field_cutoff: int = Field(default=4, ge=0)
    vertex_workers: int = Field(default=1, ge=0)
    exchange_workers: int = Field(default=1, ge=0)
    fock_repeats: int = Field(default=25, ge=1)
    run_hf_smoke: bool = False
    hf_max_iter: int = Field(default=6, ge=1)
    max_rss_gb: float | None = Field(default=None, gt=0.0)

    def model_params(self) -> ContinuumModelParams:
        return taige_model_params(
            theta_deg=self.theta_deg,
            u_D=self.u_D,
            plane_wave_shell=self.plane_wave_shell,
            n_bands=self.n_bands,
            n_active_bands_per_valley=self.n_active_bands_per_valley,
        )

    def grid_params(self) -> ContinuumGridParams:
        return ContinuumGridParams(n_k=self.n_k)

    def interaction_params(
        self,
        *,
        exchange_scale: float = 1.0,
        dtype_variant: BackendVariant | None = None,
    ) -> ContinuumInteractionParams:
        _unused = dtype_variant
        return ContinuumInteractionParams(
            coulomb_kind="dual_gate",
            q_mesh=self.q_mesh,
            q_shell=self.q_shell,
            local_field_cutoff=self.local_field_cutoff,
            include_q0=True,
            vertex_workers=self.vertex_workers,
            exchange_workers=self.exchange_workers,
            exchange_scale=exchange_scale,
        )


class TaigeBackendVariantSpec(BaseModel):
    """One backend variant selected for benchmarking."""

    model_config = ConfigDict(frozen=True)

    name: BackendVariant
    description: str
    keeps_full_lambda_blocks: bool
    keeps_dense_tve: bool
    uses_packed_tve: bool = False
    uses_matrix_free_fock: bool = False
    dtype: Literal["complex128", "complex64"] = "complex128"


class TaigeStageMeasurement(BaseModel):
    """RSS and timing for one benchmark stage."""

    model_config = ConfigDict(frozen=True)

    variant: BackendVariant
    n_k: int
    stage: str
    elapsed_seconds: float
    rss_before_mb: float
    rss_after_mb: float
    max_rss_after_mb: float
    rss_delta_mb: float


class TaigeArrayByteEstimate(BaseModel):
    """Shape-based byte estimates for major Taige HF arrays."""

    model_config = ConfigDict(frozen=True)

    n_k: int
    n_blocks: int
    n_q: int
    n_g: int
    dim: int
    n_active: int
    lambda_blocks_mb: float
    compact_lambda_blocks_mb: float
    target_minus_q_mb: float
    v_over_a_mb: float
    dense_tve_mb: float
    packed_tve_mb: float
    one_projector_field_mb: float
    one_q_full_lambda_slab_mb: float
    one_q_compact_lambda_slab_mb: float


class TaigeCorrectnessRecord(BaseModel):
    """Numerical comparison of one variant against the baseline backend."""

    model_config = ConfigDict(frozen=True)

    variant: BackendVariant
    n_k: int
    compared_to_baseline: bool
    max_abs_fock_error: float | None = None
    max_abs_hf_error: float | None = None
    total_energy_abs_error: float | None = None
    one_body_abs_error: float | None = None
    hartree_abs_error: float | None = None
    fock_abs_error: float | None = None
    direct_gap_abs_error: float | None = None
    indirect_gap_abs_error: float | None = None
    tolerance: float
    passed: bool


class TaigeVariantSummary(BaseModel):
    """Scalar summary row for one variant and mesh."""

    model_config = ConfigDict(frozen=True)

    variant: BackendVariant
    n_k: int
    skipped: bool = False
    skip_reason: str | None = None
    total_elapsed_seconds: float | None = None
    final_rss_mb: float | None = None
    peak_rss_mb: float | None = None
    fock_repeats: int
    fock_repeats_elapsed_seconds: float | None = None
    fock_apply_seconds_per_call: float | None = None
    hf_smoke_elapsed_seconds: float | None = None
    hf_smoke_vp_plus_energy: float | None = None
    hf_smoke_ivc_energy: float | None = None
    keeps_full_lambda_blocks: bool
    keeps_dense_tve: bool
    uses_packed_tve: bool = False
    uses_matrix_free_fock: bool = False
    dtype: str


class TaigeMemoryBenchmarkWorkerResult(BaseModel):
    """Complete output from one fresh benchmark subprocess."""

    model_config = ConfigDict(frozen=True)

    input: TaigeMemoryBenchmarkInput
    variant_spec: TaigeBackendVariantSpec
    summary: TaigeVariantSummary
    stages: tuple[TaigeStageMeasurement, ...]
    correctness: TaigeCorrectnessRecord | None
    estimates: TaigeArrayByteEstimate


class TaigeMemoryBenchmarkRunSummary(BaseModel):
    """Merged local benchmark result."""

    model_config = ConfigDict(frozen=True)

    output_dir: str
    results: tuple[TaigeMemoryBenchmarkWorkerResult, ...]


@dataclass(frozen=True)
class CompactTaigeDensityVertices:
    """Valley-block compact Taige density vertices.

    ``lambda_blocks`` has shape ``(n_q, n_g, n_blocks, 2, n_active, n_active)``.
    """

    q_shifts: tuple[tuple[int, int], ...]
    target_minus_q: np.ndarray
    q_is_zero: np.ndarray
    lambda_blocks: np.ndarray
    v_over_a: np.ndarray
    g_channels: tuple[tuple[int, int], ...]
    channel_in_disk: np.ndarray | None = None
    q_vectors_nm_inv: np.ndarray | None = None
    q_norm_nm_inv: np.ndarray | None = None
    v_q: np.ndarray | None = None

    @property
    def n_active(self) -> int:
        return int(self.lambda_blocks.shape[-1])

    @property
    def dim(self) -> int:
        return 2 * self.n_active


@dataclass(frozen=True)
class PackedHermitianExchange:
    """Packed upper-triangle Hermitian exchange superoperator."""

    ap_upper: np.ndarray
    n: int

    @classmethod
    def from_dense(cls, matrix: np.ndarray) -> "PackedHermitianExchange":
        arr = np.asarray(matrix)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("packed exchange requires a square dense matrix")
        packed = np.concatenate([arr[: j + 1, j] for j in range(arr.shape[0])])
        return cls(ap_upper=np.asarray(packed, dtype=arr.dtype).copy(), n=int(arr.shape[0]))

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        x = np.asarray(vector)
        if x.shape != (self.n,):
            raise ValueError(f"packed matvec expected shape {(self.n,)}, got {x.shape}")
        if self.ap_upper.dtype == np.complex64 or x.dtype == np.complex64:
            return blas.chpmv(self.n, np.complex64(1.0), self.ap_upper, x, lower=0)
        return blas.zhpmv(self.n, 1.0 + 0.0j, self.ap_upper, x, lower=0)


class BenchmarkHFBackend:
    """Small backend-like object for memory benchmark variants."""

    def __init__(
        self,
        *,
        h0: np.ndarray,
        interaction: ContinuumInteractionParams,
        target_minus_q: np.ndarray,
        lambda_blocks: np.ndarray,
        v_over_a: np.ndarray,
        hartree_channels: Sequence[tuple[int, int, float]],
        tVE: np.ndarray | None = None,
        packed_tVE: PackedHermitianExchange | None = None,
        matrix_free: bool = False,
    ) -> None:
        self.h0 = hermitize(np.asarray(h0))
        self.n_blocks, self.dim, _ = self.h0.shape
        self.n_total = self.n_blocks * self.dim
        self.interaction = interaction
        self.target_minus_q = np.asarray(target_minus_q, dtype=int)
        self.lambda_blocks = np.asarray(lambda_blocks)
        self.v_over_a = np.asarray(v_over_a)
        self.n_q = int(self.lambda_blocks.shape[0])
        self.n_g = int(self.lambda_blocks.shape[1]) if self.lambda_blocks.ndim >= 2 else 0
        self.hartree_channels = tuple((int(iq), int(ig), float(v)) for iq, ig, v in hartree_channels)
        self.tVE = None if tVE is None else np.asarray(tVE)
        self.packed_tVE = packed_tVE
        self.matrix_free = bool(matrix_free)
        self.p_ref = np.zeros_like(self.h0)

    def as_block_density(self, P: np.ndarray) -> np.ndarray:
        arr = np.asarray(P)
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

    def hartree_hamiltonian(self, Q: np.ndarray) -> np.ndarray:
        density = self.as_block_density(Q)
        out = np.zeros_like(density)
        for iq, ig, v in self.hartree_channels:
            lam = self.lambda_blocks[iq, ig]
            rho = np.einsum("kab,kba->", lam, density, optimize=True)
            out += 0.5 * v * (np.conj(rho) * lam + rho * np.swapaxes(lam.conj(), -1, -2))
        return hermitize(out)

    def _matrix_free_fock(self, density: np.ndarray) -> np.ndarray:
        out = np.zeros_like(density)
        scale = float(self.interaction.exchange_scale)
        for iq in range(self.n_q):
            targets = self.target_minus_q[iq]
            for ig in range(self.n_g):
                v = scale * float(self.v_over_a[iq, ig])
                if v == 0.0:
                    continue
                lam = self.lambda_blocks[iq, ig]
                for ik in range(self.n_blocks):
                    jk = int(targets[ik])
                    out[ik] -= 0.5 * v * lam[ik] @ density[jk] @ lam[ik].conj().T
                    out[jk] -= 0.5 * v * lam[ik].conj().T @ density[ik] @ lam[ik]
        return hermitize(out)

    def fock_hamiltonian(self, Q: np.ndarray) -> np.ndarray:
        density = self.as_block_density(Q)
        if self.matrix_free:
            return self._matrix_free_fock(density)
        flat = np.reshape(density, (-1,))
        if self.tVE is not None:
            out = -(self.tVE @ flat).reshape(density.shape)
        elif self.packed_tVE is not None:
            out = -self.packed_tVE.matvec(flat).reshape(density.shape)
        else:
            raise RuntimeError("backend has no Fock representation")
        return hermitize(out)

    def hf_hamiltonian(self, P: np.ndarray) -> np.ndarray:
        density = self.as_block_density(P)
        Q = density - self.p_ref
        return hermitize(self.h0 + self.hartree_hamiltonian(Q) + self.fock_hamiltonian(Q))

    def interaction_components(self, Q: np.ndarray) -> tuple[float, float]:
        density = self.as_block_density(Q)
        hartree = 0.0
        for iq, ig, v in self.hartree_channels:
            lam = self.lambda_blocks[iq, ig]
            rho = np.einsum("kab,kba->", lam, density, optimize=True)
            hartree += 0.5 * v * float(np.real(rho * np.conj(rho)))
        fock = 0.5 * block_trace_product(self.fock_hamiltonian(density), density)
        return hartree, fock

    def interaction_energy(self, Q: np.ndarray) -> float:
        hartree, fock = self.interaction_components(Q)
        return float(hartree + fock)

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

    def update_density_per_k(
        self,
        H: np.ndarray,
        n_occ_per_k: int,
        constraint=None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        blocks = hermitize(np.asarray(H))
        if constraint is not None and hasattr(constraint, "update_density"):
            return constraint.update_density(blocks, n_occ_per_k)
        if constraint is not None:
            blocks = constraint.project_operator(blocks)
        return _fixed_per_k_aufbau(blocks, n_occ_per_k)

    def update_density(
        self,
        H: np.ndarray,
        n_particles: float,
        constraint=None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        blocks = hermitize(np.asarray(H))
        if constraint is not None and hasattr(constraint, "update_density_global"):
            return constraint.update_density_global(blocks, n_particles)
        if constraint is not None:
            blocks = constraint.project_operator(blocks)
        return _fixed_per_k_aufbau(blocks, int(round(n_particles / max(self.n_blocks, 1))))


def variant_spec(name: BackendVariant) -> TaigeBackendVariantSpec:
    specs: dict[BackendVariant, TaigeBackendVariantSpec] = {
        "baseline": TaigeBackendVariantSpec(
            name="baseline",
            description="Production dense backend with full lambda_blocks and dense tVE.",
            keeps_full_lambda_blocks=True,
            keeps_dense_tve=True,
        ),
        "hartree_only": TaigeBackendVariantSpec(
            name="hartree_only",
            description="Dense tVE plus only Hartree-channel vertices retained after build.",
            keeps_full_lambda_blocks=False,
            keeps_dense_tve=True,
        ),
        "fused": TaigeBackendVariantSpec(
            name="fused",
            description="Q-slab vertex build directly into dense tVE, retaining only Hartree vertices.",
            keeps_full_lambda_blocks=False,
            keeps_dense_tve=True,
        ),
        "compact": TaigeBackendVariantSpec(
            name="compact",
            description="Valley-block compact vertices expanded only for exchange assembly.",
            keeps_full_lambda_blocks=False,
            keeps_dense_tve=True,
        ),
        "fused_compact": TaigeBackendVariantSpec(
            name="fused_compact",
            description="Q-slab fused build using compact valley-block slab representation.",
            keeps_full_lambda_blocks=False,
            keeps_dense_tve=True,
        ),
        "packed": TaigeBackendVariantSpec(
            name="packed",
            description="Hartree-only vertices plus packed Hermitian tVE storage and BLAS matvec.",
            keeps_full_lambda_blocks=False,
            keeps_dense_tve=False,
            uses_packed_tve=True,
        ),
        "matrix_free": TaigeBackendVariantSpec(
            name="matrix_free",
            description="Full dense vertices retained, no tVE, direct matrix-free Fock application.",
            keeps_full_lambda_blocks=True,
            keeps_dense_tve=False,
            uses_matrix_free_fock=True,
        ),
        "complex64": TaigeBackendVariantSpec(
            name="complex64",
            description="Complex64 dense tVE and Hartree-only vertices; diagnostic only.",
            keeps_full_lambda_blocks=False,
            keeps_dense_tve=True,
            dtype="complex64",
        ),
    }
    return specs[name]


def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return _max_rss_bytes()


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value
    return value * 1024


def _bytes_to_mb(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


def _measure_stage(
    *,
    variant: BackendVariant,
    n_k: int,
    stage: str,
    rows: list[TaigeStageMeasurement],
    fn: Callable[[], Any],
) -> Any:
    gc.collect()
    rss_before = _rss_bytes()
    start = time.perf_counter()
    out = fn()
    elapsed = time.perf_counter() - start
    gc.collect()
    rss_after = _rss_bytes()
    rows.append(
        TaigeStageMeasurement(
            variant=variant,
            n_k=n_k,
            stage=stage,
            elapsed_seconds=float(elapsed),
            rss_before_mb=_bytes_to_mb(rss_before),
            rss_after_mb=_bytes_to_mb(rss_after),
            max_rss_after_mb=_bytes_to_mb(_max_rss_bytes()),
            rss_delta_mb=_bytes_to_mb(rss_after - rss_before),
        )
    )
    return out


def estimate_taige_array_bytes(params: TaigeMemoryBenchmarkInput) -> TaigeArrayByteEstimate:
    grid = MomentumGrid(params.n_k)
    interaction = params.interaction_params()
    n_q = len(q_transfers(grid, interaction))
    n_g = len(reciprocal_box(params.local_field_cutoff))
    n_blocks = grid.size
    n_active = int(params.n_active_bands_per_valley)
    dim = 2 * n_active
    complex128_bytes = np.dtype(np.complex128).itemsize
    int_bytes = np.dtype(np.int64).itemsize
    float_bytes = np.dtype(float).itemsize
    lambda_bytes = n_q * n_g * n_blocks * dim * dim * complex128_bytes
    compact_bytes = n_q * n_g * n_blocks * 2 * n_active * n_active * complex128_bytes
    block_dim = dim * dim
    tve_bytes = (n_blocks * block_dim) ** 2 * complex128_bytes
    packed_elems = (n_blocks * block_dim) * (n_blocks * block_dim + 1) // 2
    projector_bytes = n_blocks * dim * dim * complex128_bytes
    return TaigeArrayByteEstimate(
        n_k=int(params.n_k),
        n_blocks=int(n_blocks),
        n_q=int(n_q),
        n_g=int(n_g),
        dim=int(dim),
        n_active=int(n_active),
        lambda_blocks_mb=_bytes_to_mb(lambda_bytes),
        compact_lambda_blocks_mb=_bytes_to_mb(compact_bytes),
        target_minus_q_mb=_bytes_to_mb(n_q * n_blocks * int_bytes),
        v_over_a_mb=_bytes_to_mb(n_q * n_g * float_bytes),
        dense_tve_mb=_bytes_to_mb(tve_bytes),
        packed_tve_mb=_bytes_to_mb(packed_elems * complex128_bytes),
        one_projector_field_mb=_bytes_to_mb(projector_bytes),
        one_q_full_lambda_slab_mb=_bytes_to_mb(n_g * n_blocks * dim * dim * complex128_bytes),
        one_q_compact_lambda_slab_mb=_bytes_to_mb(
            n_g * n_blocks * 2 * n_active * n_active * complex128_bytes
        ),
    )


def compact_lambdas_from_dense(lambdas: np.ndarray, n_active: int) -> np.ndarray:
    arr = np.asarray(lambdas)
    if arr.shape[-1] != 2 * int(n_active) or arr.shape[-2] != 2 * int(n_active):
        raise ValueError("dense lambda shape is incompatible with n_active")
    compact = np.empty(arr.shape[:-2] + (2, int(n_active), int(n_active)), dtype=arr.dtype)
    for iv in range(2):
        start = iv * int(n_active)
        stop = start + int(n_active)
        compact[..., iv, :, :] = arr[..., start:stop, start:stop]
    return compact


def dense_lambdas_from_compact(compact: np.ndarray) -> np.ndarray:
    arr = np.asarray(compact)
    if arr.ndim < 4 or arr.shape[-3] != 2 or arr.shape[-1] != arr.shape[-2]:
        raise ValueError("compact lambdas must end in (2, n_active, n_active)")
    n_active = int(arr.shape[-1])
    dense = np.zeros(arr.shape[:-3] + (2 * n_active, 2 * n_active), dtype=arr.dtype)
    for iv in range(2):
        start = iv * n_active
        stop = start + n_active
        dense[..., start:stop, start:stop] = arr[..., iv, :, :]
    return dense


def compact_vertices_from_dense(
    vertices: DensityVertices,
    *,
    n_active: int,
) -> CompactTaigeDensityVertices:
    return CompactTaigeDensityVertices(
        q_shifts=vertices.q_shifts,
        target_minus_q=np.asarray(vertices.target_minus_q, dtype=int).copy(),
        q_is_zero=np.asarray(vertices.q_is_zero, dtype=bool).copy(),
        lambda_blocks=compact_lambdas_from_dense(vertices.lambda_blocks, n_active).copy(),
        v_over_a=np.asarray(vertices.v_over_a).copy(),
        g_channels=vertices.g_channels,
        channel_in_disk=None
        if vertices.channel_in_disk is None
        else np.asarray(vertices.channel_in_disk, dtype=bool).copy(),
        q_vectors_nm_inv=None
        if vertices.q_vectors_nm_inv is None
        else np.asarray(vertices.q_vectors_nm_inv).copy(),
        q_norm_nm_inv=None
        if vertices.q_norm_nm_inv is None
        else np.asarray(vertices.q_norm_nm_inv).copy(),
        v_q=None if vertices.v_q is None else np.asarray(vertices.v_q).copy(),
    )


def dense_vertices_from_compact(vertices: CompactTaigeDensityVertices) -> DensityVertices:
    return DensityVertices(
        q_shifts=vertices.q_shifts,
        target_minus_q=np.asarray(vertices.target_minus_q, dtype=int).copy(),
        q_is_zero=np.asarray(vertices.q_is_zero, dtype=bool).copy(),
        lambda_blocks=dense_lambdas_from_compact(vertices.lambda_blocks).copy(),
        v_over_a=np.asarray(vertices.v_over_a).copy(),
        g_channels=vertices.g_channels,
        channel_in_disk=None
        if vertices.channel_in_disk is None
        else np.asarray(vertices.channel_in_disk, dtype=bool).copy(),
        q_vectors_nm_inv=None
        if vertices.q_vectors_nm_inv is None
        else np.asarray(vertices.q_vectors_nm_inv).copy(),
        q_norm_nm_inv=None
        if vertices.q_norm_nm_inv is None
        else np.asarray(vertices.q_norm_nm_inv).copy(),
        v_q=None if vertices.v_q is None else np.asarray(vertices.v_q).copy(),
    )


def _scatter_exchange_contribution(
    *,
    tVE: np.ndarray,
    target_minus_q_iq: np.ndarray,
    block_rows: np.ndarray,
    block_dim: int,
    forward: np.ndarray,
    reverse: np.ndarray,
) -> None:
    local = np.arange(block_dim)
    target_rows = np.asarray(target_minus_q_iq, dtype=int)[:, None] * block_dim + local[None, :]
    n_blocks = int(target_rows.shape[0])
    tVE[block_rows[:, :, None], target_rows[:, None, :]] += forward.reshape(
        n_blocks,
        block_dim,
        block_dim,
    )
    tVE[target_rows[:, :, None], block_rows[:, None, :]] += reverse.reshape(
        n_blocks,
        block_dim,
        block_dim,
    )


def _hartree_channel_keys(
    *,
    q_is_zero: np.ndarray,
    v_over_a: np.ndarray,
    q_norm_nm_inv: np.ndarray | None,
    interaction: ContinuumInteractionParams,
    q_shifts: Sequence[tuple[int, int]],
    g_channels: Sequence[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    keys: dict[tuple[int, int], float] = {}
    scale = float(interaction.hartree_scale)
    if scale == 0.0:
        return keys
    for iq in range(len(q_is_zero)):
        if not bool(q_is_zero[iq]):
            continue
        for ig in range(v_over_a.shape[1]):
            if interaction.q0_hartree == "omit_uniform":
                if q_norm_nm_inv is not None:
                    is_uniform = bool(float(q_norm_nm_inv[int(iq), int(ig)]) < 1e-12)
                else:
                    q = q_shifts[int(iq)]
                    g = g_channels[int(ig)]
                    is_uniform = int(q[0]) + int(g[0]) == 0 and int(q[1]) + int(g[1]) == 0
                if is_uniform:
                    continue
            v = scale * float(v_over_a[iq, ig])
            if v != 0.0:
                keys[(int(iq), int(ig))] = v
    return keys


def _hartree_only_arrays_from_dense_backend(
    backend: ContinuumHFBackend,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, int, float], ...]]:
    channels = tuple(backend.hartree_channels)
    if not channels:
        empty_lambdas = np.zeros((0, 0, backend.n_blocks, backend.dim, backend.dim), dtype=backend.h0.dtype)
        empty_targets = np.zeros((0, backend.n_blocks), dtype=int)
        empty_v = np.zeros((0, 0), dtype=float)
        return empty_targets, empty_lambdas, empty_v, ()
    lambdas = np.empty(
        (len(channels), 1, backend.n_blocks, backend.dim, backend.dim),
        dtype=backend.lambda_blocks.dtype,
    )
    target_minus_q = np.empty((len(channels), backend.n_blocks), dtype=int)
    v_over_a = np.empty((len(channels), 1), dtype=float)
    remapped: list[tuple[int, int, float]] = []
    for new_iq, (old_iq, old_ig, v) in enumerate(channels):
        lambdas[new_iq, 0] = backend.lambda_blocks[int(old_iq), int(old_ig)]
        target_minus_q[new_iq] = backend.target_minus_q[int(old_iq)]
        v_over_a[new_iq, 0] = backend.v_over_a[int(old_iq), int(old_ig)]
        remapped.append((new_iq, 0, float(v)))
    return target_minus_q, lambdas, v_over_a, tuple(remapped)


def _hartree_only_arrays_from_vertices(
    vertices: DensityVertices,
    interaction: ContinuumInteractionParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, int, float], ...]]:
    q_norm = None if vertices.q_norm_nm_inv is None else np.asarray(vertices.q_norm_nm_inv)
    keys = _hartree_channel_keys(
        q_is_zero=np.asarray(vertices.q_is_zero, dtype=bool),
        v_over_a=np.asarray(vertices.v_over_a, dtype=float),
        q_norm_nm_inv=q_norm,
        interaction=interaction,
        q_shifts=vertices.q_shifts,
        g_channels=vertices.g_channels,
    )
    n_blocks = int(vertices.lambda_blocks.shape[2])
    dim = int(vertices.lambda_blocks.shape[-1])
    if not keys:
        return _empty_hartree_retention(
            n_blocks=n_blocks,
            dim=dim,
            dtype=np.asarray(vertices.lambda_blocks).dtype,
        )
    retained_targets: list[np.ndarray] = []
    retained_lambdas: list[np.ndarray] = []
    retained_v: list[float] = []
    for iq, ig in sorted(keys):
        retained_targets.append(np.asarray(vertices.target_minus_q[iq], dtype=int).copy())
        retained_lambdas.append(np.asarray(vertices.lambda_blocks[iq, ig]).copy())
        retained_v.append(keys[(iq, ig)])
    return _finalize_retained_hartree(
        retained_targets,
        retained_lambdas,
        retained_v,
        n_blocks=n_blocks,
        dim=dim,
        dtype=np.asarray(vertices.lambda_blocks).dtype,
    )


def hartree_only_backend_from_dense(
    backend: ContinuumHFBackend,
    *,
    packed: bool = False,
    dtype: np.dtype | type | None = None,
) -> BenchmarkHFBackend:
    target, lambdas, v_over_a, channels = _hartree_only_arrays_from_dense_backend(backend)
    out_dtype = np.dtype(dtype or backend.tVE.dtype)
    tVE = np.asarray(backend.tVE, dtype=out_dtype)
    h0 = np.asarray(backend.h0, dtype=out_dtype)
    lambdas = np.asarray(lambdas, dtype=out_dtype)
    if packed:
        packed_tVE = PackedHermitianExchange.from_dense(tVE)
        return BenchmarkHFBackend(
            h0=h0,
            interaction=backend.interaction,
            target_minus_q=target,
            lambda_blocks=lambdas,
            v_over_a=v_over_a,
            hartree_channels=channels,
            packed_tVE=packed_tVE,
        )
    return BenchmarkHFBackend(
        h0=h0,
        interaction=backend.interaction,
        target_minus_q=target,
        lambda_blocks=lambdas,
        v_over_a=v_over_a,
        hartree_channels=channels,
        tVE=tVE,
    )


def matrix_free_backend_from_dense(backend: ContinuumHFBackend) -> BenchmarkHFBackend:
    return BenchmarkHFBackend(
        h0=backend.h0.copy(),
        interaction=backend.interaction,
        target_minus_q=backend.target_minus_q.copy(),
        lambda_blocks=backend.lambda_blocks.copy(),
        v_over_a=backend.v_over_a.copy(),
        hartree_channels=tuple(backend.hartree_channels),
        matrix_free=True,
    )


def matrix_free_backend_from_vertices(
    active: ContinuumActiveSpace,
    vertices: DensityVertices,
    interaction: ContinuumInteractionParams,
) -> BenchmarkHFBackend:
    keys = _hartree_channel_keys(
        q_is_zero=np.asarray(vertices.q_is_zero, dtype=bool),
        v_over_a=np.asarray(vertices.v_over_a, dtype=float),
        q_norm_nm_inv=None if vertices.q_norm_nm_inv is None else np.asarray(vertices.q_norm_nm_inv),
        interaction=interaction,
        q_shifts=vertices.q_shifts,
        g_channels=vertices.g_channels,
    )
    channels = tuple((iq, ig, v) for (iq, ig), v in sorted(keys.items()))
    return BenchmarkHFBackend(
        h0=active.h0.copy(),
        interaction=interaction,
        target_minus_q=np.asarray(vertices.target_minus_q, dtype=int).copy(),
        lambda_blocks=np.asarray(vertices.lambda_blocks).copy(),
        v_over_a=np.asarray(vertices.v_over_a, dtype=float).copy(),
        hartree_channels=channels,
        matrix_free=True,
    )


def _taige_vertex_metadata(
    active: ContinuumActiveSpace,
    interaction: ContinuumInteractionParams,
) -> dict[str, Any]:
    if active.bands is None or active.geometry is None:
        raise ValueError("Taige fused benchmark requires bandstructure and geometry")
    grid = active.grid
    q_list = q_transfers(grid, interaction)
    g_channels = reciprocal_box(interaction.local_field_cutoff)
    channel_in_disk = _channel_mask(
        active.geometry,
        grid,
        q_list,
        g_channels,
        interaction.local_field_cutoff,
    )
    q_vectors_nm_inv = active.geometry.mesh_q_vectors_nm_inv(grid, q_list, g_channels)
    if interaction.coulomb_kind == "dual_gate":
        v_q, v_over_a, q_zero_channels = _physical_v_over_a(
            active.geometry,
            grid,
            q_vectors_nm_inv,
            channel_in_disk,
            interaction,
        )
    else:
        v_q, v_over_a, q_zero_channels = _dimensionless_v_over_a(q_list, g_channels, interaction)
        v_over_a = np.where(channel_in_disk, v_over_a, 0.0)
        v_q = np.where(channel_in_disk, v_q, 0.0)
    q_is_zero = np.asarray([q == (0, 0) for q in q_list], dtype=bool)
    q_is_zero = np.logical_or(q_is_zero, np.any(q_zero_channels, axis=-1))
    q_norm_nm_inv = np.linalg.norm(q_vectors_nm_inv, axis=-1)
    return {
        "q_list": q_list,
        "g_channels": g_channels,
        "channel_in_disk": channel_in_disk,
        "q_vectors_nm_inv": q_vectors_nm_inv,
        "q_norm_nm_inv": q_norm_nm_inv,
        "v_q": v_q,
        "v_over_a": v_over_a,
        "q_is_zero": q_is_zero,
    }


def _empty_hartree_retention(
    *,
    n_blocks: int,
    dim: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, int, float], ...]]:
    return (
        np.zeros((0, n_blocks), dtype=int),
        np.zeros((0, 0, n_blocks, dim, dim), dtype=dtype),
        np.zeros((0, 0), dtype=float),
        (),
    )


def _finalize_retained_hartree(
    retained_targets: list[np.ndarray],
    retained_lambdas: list[np.ndarray],
    retained_v: list[float],
    *,
    n_blocks: int,
    dim: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[tuple[int, int, float], ...]]:
    if not retained_lambdas:
        return _empty_hartree_retention(n_blocks=n_blocks, dim=dim, dtype=dtype)
    target = np.stack(retained_targets, axis=0).astype(int, copy=False)
    lambdas = np.stack(retained_lambdas, axis=0).astype(dtype, copy=False)[:, None, :, :, :]
    v_over_a = np.asarray(retained_v, dtype=float)[:, None]
    channels = tuple((iq, 0, float(v)) for iq, v in enumerate(retained_v))
    return target, lambdas, v_over_a, channels


def _dense_exchange_from_dense_vertices(
    vertices: DensityVertices,
    interaction: ContinuumInteractionParams,
    *,
    dim: int,
    dtype: np.dtype | type = np.complex128,
) -> np.ndarray:
    block_dim = int(dim) * int(dim)
    n_blocks = int(vertices.target_minus_q.shape[1])
    size = n_blocks * block_dim
    tVE = np.zeros((size, size), dtype=np.dtype(dtype))
    block_rows = np.arange(n_blocks)[:, None] * block_dim + np.arange(block_dim)[None, :]
    scale = float(interaction.exchange_scale)
    if scale == 0.0:
        return tVE
    lambdas = np.asarray(vertices.lambda_blocks, dtype=np.dtype(dtype))
    v_over_a = np.asarray(vertices.v_over_a, dtype=float)
    for iq in range(lambdas.shape[0]):
        for local_iq, forward, reverse in _exchange_tve_q_slab(
            q_start=iq,
            q_stop=iq + 1,
            lambda_blocks=lambdas,
            v_over_a=v_over_a,
            exchange_scale=scale,
        ):
            _scatter_exchange_contribution(
                tVE=tVE,
                target_minus_q_iq=vertices.target_minus_q[int(local_iq)],
                block_rows=block_rows,
                block_dim=block_dim,
                forward=forward,
                reverse=reverse,
            )
    return _hermitize_dense_in_place(tVE)


def _dense_exchange_from_compact_vertices(
    vertices: CompactTaigeDensityVertices,
    interaction: ContinuumInteractionParams,
    *,
    dtype: np.dtype | type = np.complex128,
) -> np.ndarray:
    block_dim = vertices.dim * vertices.dim
    n_blocks = int(vertices.target_minus_q.shape[1])
    size = n_blocks * block_dim
    out_dtype = np.dtype(dtype)
    tVE = np.zeros((size, size), dtype=out_dtype)
    block_rows = np.arange(n_blocks)[:, None] * block_dim + np.arange(block_dim)[None, :]
    scale = float(interaction.exchange_scale)
    if scale == 0.0:
        return tVE
    for iq in range(vertices.lambda_blocks.shape[0]):
        dense_one_q = dense_lambdas_from_compact(vertices.lambda_blocks[iq : iq + 1]).astype(
            out_dtype,
            copy=False,
        )
        for _local_iq, forward, reverse in _exchange_tve_q_slab(
            q_start=0,
            q_stop=1,
            lambda_blocks=dense_one_q,
            v_over_a=np.asarray(vertices.v_over_a[iq : iq + 1], dtype=float),
            exchange_scale=scale,
        ):
            _scatter_exchange_contribution(
                tVE=tVE,
                target_minus_q_iq=vertices.target_minus_q[iq],
                block_rows=block_rows,
                block_dim=block_dim,
                forward=forward,
                reverse=reverse,
            )
    return _hermitize_dense_in_place(tVE)


def _build_fused_taige_backend(
    active: ContinuumActiveSpace,
    interaction: ContinuumInteractionParams,
    *,
    compact_slabs: bool = False,
    dtype: np.dtype | type = np.complex128,
) -> BenchmarkHFBackend:
    metadata = _taige_vertex_metadata(active, interaction)
    q_list: tuple[tuple[int, int], ...] = metadata["q_list"]
    g_channels: tuple[tuple[int, int], ...] = metadata["g_channels"]
    v_over_a = np.asarray(metadata["v_over_a"], dtype=float)
    q_is_zero = np.asarray(metadata["q_is_zero"], dtype=bool)
    q_norm_nm_inv = np.asarray(metadata["q_norm_nm_inv"], dtype=float)
    channel_in_disk = np.asarray(metadata["channel_in_disk"], dtype=bool)
    hartree_keys = _hartree_channel_keys(
        q_is_zero=q_is_zero,
        v_over_a=v_over_a,
        q_norm_nm_inv=q_norm_nm_inv,
        interaction=interaction,
        q_shifts=q_list,
        g_channels=g_channels,
    )
    n_blocks = active.grid.size
    dim = active.dim
    block_dim = dim * dim
    out_dtype = np.dtype(dtype)
    tVE = np.zeros((n_blocks * block_dim, n_blocks * block_dim), dtype=out_dtype)
    block_rows = np.arange(n_blocks)[:, None] * block_dim + np.arange(block_dim)[None, :]
    shell_index = {g: i for i, g in enumerate(active.shell)}
    electron_vectors = np.asarray(active.bands.electron_vectors)
    retained_targets: list[np.ndarray] = []
    retained_lambdas: list[np.ndarray] = []
    retained_v: list[float] = []
    scale = float(interaction.exchange_scale)
    for q_start, q in enumerate(q_list):
        _slab_start, target_slab, _q_zero_slab, lambda_slab = _taige_density_vertex_q_slab(
            q_start=q_start,
            q_stop=q_start + 1,
            q_list=q_list,
            g_channels=g_channels,
            channel_in_disk=channel_in_disk,
            n_k=active.grid.n_k,
            n_active=active.n_active,
            dim=active.dim,
            shell=active.shell,
            shell_index=shell_index,
            electron_vectors=electron_vectors,
            source_index=active.source_index,
        )
        if compact_slabs:
            compact = compact_lambdas_from_dense(lambda_slab, active.n_active)
            lambda_slab_for_exchange = dense_lambdas_from_compact(compact).astype(
                out_dtype,
                copy=False,
            )
        else:
            lambda_slab_for_exchange = np.asarray(lambda_slab, dtype=out_dtype)
        if scale != 0.0:
            for _local_iq, forward, reverse in _exchange_tve_q_slab(
                q_start=0,
                q_stop=1,
                lambda_blocks=lambda_slab_for_exchange,
                v_over_a=v_over_a[q_start : q_start + 1],
                exchange_scale=scale,
            ):
                _scatter_exchange_contribution(
                    tVE=tVE,
                    target_minus_q_iq=target_slab[0],
                    block_rows=block_rows,
                    block_dim=block_dim,
                    forward=forward,
                    reverse=reverse,
                )
        for ig in range(len(g_channels)):
            if (q_start, ig) in hartree_keys:
                retained_targets.append(target_slab[0].copy())
                retained_lambdas.append(np.asarray(lambda_slab[0, ig], dtype=out_dtype).copy())
                retained_v.append(hartree_keys[(q_start, ig)])
        _unused_q = q
    tVE = _hermitize_dense_in_place(tVE)
    target, lambdas, retained_v_over_a, channels = _finalize_retained_hartree(
        retained_targets,
        retained_lambdas,
        retained_v,
        n_blocks=n_blocks,
        dim=dim,
        dtype=out_dtype,
    )
    return BenchmarkHFBackend(
        h0=np.asarray(active.h0, dtype=out_dtype),
        interaction=interaction,
        target_minus_q=target,
        lambda_blocks=lambdas,
        v_over_a=retained_v_over_a,
        hartree_channels=channels,
        tVE=tVE,
    )


def _build_compact_taige_backend(
    active: ContinuumActiveSpace,
    vertices: DensityVertices,
    interaction: ContinuumInteractionParams,
    *,
    dtype: np.dtype | type = np.complex128,
) -> BenchmarkHFBackend:
    compact = compact_vertices_from_dense(vertices, n_active=active.n_active)
    tVE = _dense_exchange_from_compact_vertices(compact, interaction, dtype=dtype)
    target, lambdas, v_over_a, channels = _hartree_only_arrays_from_vertices(vertices, interaction)
    out_dtype = np.dtype(dtype)
    return BenchmarkHFBackend(
        h0=np.asarray(active.h0, dtype=out_dtype),
        interaction=interaction,
        target_minus_q=target,
        lambda_blocks=np.asarray(lambdas, dtype=out_dtype),
        v_over_a=v_over_a,
        hartree_channels=channels,
        tVE=np.asarray(tVE, dtype=out_dtype),
    )


def _random_density_for_backend(backend: Any, *, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=backend.h0.shape) + 1j * rng.normal(size=backend.h0.shape)
    return hermitize(raw)


def _direct_indirect_gaps(backend: Any, H: np.ndarray) -> tuple[float, float]:
    _P, _evals, direct, indirect = backend.update_density_per_k(H, 1)
    return float(direct), float(indirect)


def _variant_estimated_peak_mb(
    estimates: TaigeArrayByteEstimate,
    variant: BackendVariant,
) -> float:
    full = estimates.lambda_blocks_mb
    compact = estimates.compact_lambda_blocks_mb
    dense_tve = estimates.dense_tve_mb
    packed = estimates.packed_tve_mb
    one_q = estimates.one_q_full_lambda_slab_mb
    one_q_compact = estimates.one_q_compact_lambda_slab_mb
    if variant == "baseline":
        return full + dense_tve
    if variant == "hartree_only":
        return full + dense_tve
    if variant == "fused":
        return dense_tve + one_q
    if variant == "compact":
        return full + compact + dense_tve
    if variant == "fused_compact":
        return dense_tve + one_q + one_q_compact
    if variant == "packed":
        return full + dense_tve + packed
    if variant == "matrix_free":
        return full
    if variant == "complex64":
        return 0.5 * (full + dense_tve)
    raise ValueError(f"unknown variant {variant!r}")


def _build_variant_backend(
    params: TaigeMemoryBenchmarkInput,
    variant: BackendVariant,
    stages: list[TaigeStageMeasurement],
) -> tuple[ContinuumActiveSpace, Any]:
    interaction = params.interaction_params()
    active = _measure_stage(
        variant=variant,
        n_k=params.n_k,
        stage="active_space",
        rows=stages,
        fn=lambda: build_active_space(params.grid_params(), params.model_params()),
    )
    if variant == "fused":
        backend = _measure_stage(
            variant=variant,
            n_k=params.n_k,
            stage="fused_vertex_exchange",
            rows=stages,
            fn=lambda: _build_fused_taige_backend(active, interaction, compact_slabs=False),
        )
        return active, backend
    if variant == "fused_compact":
        backend = _measure_stage(
            variant=variant,
            n_k=params.n_k,
            stage="fused_compact_vertex_exchange",
            rows=stages,
            fn=lambda: _build_fused_taige_backend(active, interaction, compact_slabs=True),
        )
        return active, backend

    vertices = _measure_stage(
        variant=variant,
        n_k=params.n_k,
        stage="density_vertices",
        rows=stages,
        fn=lambda: build_taige_density_vertices(active, interaction),
    )
    if variant == "baseline":
        backend = _measure_stage(
            variant=variant,
            n_k=params.n_k,
            stage="dense_exchange_backend",
            rows=stages,
            fn=lambda: ContinuumHFBackend(active.h0, vertices, interaction),
        )
        return active, backend
    if variant == "hartree_only":
        dense_backend = _measure_stage(
            variant=variant,
            n_k=params.n_k,
            stage="dense_exchange_backend",
            rows=stages,
            fn=lambda: ContinuumHFBackend(active.h0, vertices, interaction),
        )
        backend = _measure_stage(
            variant=variant,
            n_k=params.n_k,
            stage="retain_hartree_only",
            rows=stages,
            fn=lambda: hartree_only_backend_from_dense(dense_backend),
        )
        del dense_backend, vertices
        gc.collect()
        return active, backend
    if variant == "packed":
        dense_backend = _measure_stage(
            variant=variant,
            n_k=params.n_k,
            stage="dense_exchange_backend",
            rows=stages,
            fn=lambda: ContinuumHFBackend(active.h0, vertices, interaction),
        )
        backend = _measure_stage(
            variant=variant,
            n_k=params.n_k,
            stage="pack_exchange",
            rows=stages,
            fn=lambda: hartree_only_backend_from_dense(dense_backend, packed=True),
        )
        del dense_backend, vertices
        gc.collect()
        return active, backend
    if variant == "matrix_free":
        backend = _measure_stage(
            variant=variant,
            n_k=params.n_k,
            stage="matrix_free_backend",
            rows=stages,
            fn=lambda: matrix_free_backend_from_vertices(active, vertices, interaction),
        )
        del vertices
        gc.collect()
        return active, backend
    if variant == "compact":
        backend = _measure_stage(
            variant=variant,
            n_k=params.n_k,
            stage="compact_exchange_backend",
            rows=stages,
            fn=lambda: _build_compact_taige_backend(active, vertices, interaction),
        )
        del vertices
        gc.collect()
        return active, backend
    if variant == "complex64":
        vertices64 = DensityVertices(
            q_shifts=vertices.q_shifts,
            target_minus_q=np.asarray(vertices.target_minus_q, dtype=int).copy(),
            q_is_zero=np.asarray(vertices.q_is_zero, dtype=bool).copy(),
            lambda_blocks=np.asarray(vertices.lambda_blocks, dtype=np.complex64).copy(),
            v_over_a=np.asarray(vertices.v_over_a, dtype=float).copy(),
            g_channels=vertices.g_channels,
            channel_in_disk=None
            if vertices.channel_in_disk is None
            else np.asarray(vertices.channel_in_disk, dtype=bool).copy(),
            q_vectors_nm_inv=None
            if vertices.q_vectors_nm_inv is None
            else np.asarray(vertices.q_vectors_nm_inv, dtype=float).copy(),
            q_norm_nm_inv=None
            if vertices.q_norm_nm_inv is None
            else np.asarray(vertices.q_norm_nm_inv, dtype=float).copy(),
            v_q=None if vertices.v_q is None else np.asarray(vertices.v_q, dtype=float).copy(),
        )
        target, lambdas, v_over_a, channels = _hartree_only_arrays_from_vertices(
            vertices64,
            interaction,
        )

        def _complex64_backend() -> BenchmarkHFBackend:
            tVE = _dense_exchange_from_dense_vertices(
                vertices64,
                interaction,
                dim=active.dim,
                dtype=np.complex64,
            )
            return BenchmarkHFBackend(
                h0=np.asarray(active.h0, dtype=np.complex64),
                interaction=interaction,
                target_minus_q=target,
                lambda_blocks=np.asarray(lambdas, dtype=np.complex64),
                v_over_a=v_over_a,
                hartree_channels=channels,
                tVE=tVE,
            )

        backend = _measure_stage(
            variant=variant,
            n_k=params.n_k,
            stage="complex64_exchange_backend",
            rows=stages,
            fn=_complex64_backend,
        )
        del vertices, vertices64
        gc.collect()
        return active, backend
    raise ValueError(f"unknown variant {variant!r}")


def _reference_payload(backend: Any, Q: np.ndarray) -> dict[str, Any]:
    fock = backend.fock_hamiltonian(Q)
    hf = backend.hf_hamiltonian(Q)
    energy = backend.energy(Q)
    direct, indirect = _direct_indirect_gaps(backend, hf)
    return {
        "fock": fock,
        "hf": hf,
        "energy_total": float(energy.total),
        "energy_one_body": float(energy.one_body),
        "energy_hartree": float(energy.hartree),
        "energy_fock": float(energy.fock),
        "direct_gap": float(direct),
        "indirect_gap": float(indirect),
    }


def _save_reference_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _load_reference_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _correctness_against_reference(
    *,
    variant: BackendVariant,
    n_k: int,
    payload: dict[str, Any],
    reference: dict[str, Any] | None,
) -> TaigeCorrectnessRecord | None:
    if reference is None:
        return None
    tolerance = 5e-4 if variant == "complex64" else 1e-8
    fock_error = float(np.max(np.abs(payload["fock"] - reference["fock"])))
    hf_error = float(np.max(np.abs(payload["hf"] - reference["hf"])))
    total_error = abs(float(payload["energy_total"]) - float(reference["energy_total"]))
    one_body_error = abs(float(payload["energy_one_body"]) - float(reference["energy_one_body"]))
    hartree_error = abs(float(payload["energy_hartree"]) - float(reference["energy_hartree"]))
    fock_energy_error = abs(float(payload["energy_fock"]) - float(reference["energy_fock"]))
    direct_error = abs(float(payload["direct_gap"]) - float(reference["direct_gap"]))
    indirect_error = abs(float(payload["indirect_gap"]) - float(reference["indirect_gap"]))
    passed = max(
        fock_error,
        hf_error,
        total_error,
        one_body_error,
        hartree_error,
        fock_energy_error,
        direct_error,
        indirect_error,
    ) <= tolerance
    return TaigeCorrectnessRecord(
        variant=variant,
        n_k=n_k,
        compared_to_baseline=True,
        max_abs_fock_error=fock_error,
        max_abs_hf_error=hf_error,
        total_energy_abs_error=total_error,
        one_body_abs_error=one_body_error,
        hartree_abs_error=hartree_error,
        fock_abs_error=fock_energy_error,
        direct_gap_abs_error=direct_error,
        indirect_gap_abs_error=indirect_error,
        tolerance=float(tolerance),
        passed=bool(passed),
    )


def _run_hf_smoke(active: ContinuumActiveSpace, backend: Any, params: TaigeMemoryBenchmarkInput) -> dict[str, float]:
    bundle = ContinuumBundle(
        grid=active.grid,
        active=active,
        vertices=DensityVertices(
            q_shifts=(),
            target_minus_q=np.zeros((0, active.grid.size), dtype=int),
            q_is_zero=np.zeros(0, dtype=bool),
            lambda_blocks=np.zeros((0, 0, active.grid.size, active.dim, active.dim), dtype=complex),
            v_over_a=np.zeros((0, 0), dtype=float),
            g_channels=(),
        ),
        backend=backend,
        params=params.model_params(),
        interaction=params.interaction_params(),
        bands=active.bands,
        geometry=active.geometry,
    )
    refs = build_symmetric_hf_references(
        bundle,
        ContinuumHFParams(
            max_iter=params.hf_max_iter,
            min_iter=1,
            mixing_method="linear",
            mixing=0.4,
            n_occ_per_k=1,
            seed_random_weight=0.0,
        ),
    )
    return {
        "vp_plus_energy": float(refs.vp_plus.energy),
        "ivc_energy": float(refs.ivc.energy),
    }


def run_taige_memory_benchmark_worker(
    *,
    params: TaigeMemoryBenchmarkInput,
    variant: BackendVariant,
    reference_input: Path | None = None,
    reference_output: Path | None = None,
) -> TaigeMemoryBenchmarkWorkerResult:
    spec = variant_spec(variant)
    estimates = estimate_taige_array_bytes(params)
    estimated_peak = _variant_estimated_peak_mb(estimates, variant)
    if params.max_rss_gb is not None and estimated_peak > float(params.max_rss_gb) * 1024.0:
        summary = TaigeVariantSummary(
            variant=variant,
            n_k=params.n_k,
            skipped=True,
            skip_reason=(
                f"estimated variant peak {estimated_peak:.1f} MB exceeds "
                f"max_rss_gb={params.max_rss_gb}"
            ),
            fock_repeats=params.fock_repeats,
            keeps_full_lambda_blocks=spec.keeps_full_lambda_blocks,
            keeps_dense_tve=spec.keeps_dense_tve,
            uses_packed_tve=spec.uses_packed_tve,
            uses_matrix_free_fock=spec.uses_matrix_free_fock,
            dtype=spec.dtype,
        )
        correctness = TaigeCorrectnessRecord(
            variant=variant,
            n_k=params.n_k,
            compared_to_baseline=reference_input is not None,
            tolerance=5e-4 if variant == "complex64" else 1e-8,
            passed=False,
        )
        return TaigeMemoryBenchmarkWorkerResult(
            input=params,
            variant_spec=spec,
            summary=summary,
            stages=(),
            correctness=correctness,
            estimates=estimates,
        )

    stages: list[TaigeStageMeasurement] = []
    start = time.perf_counter()
    active, backend = _build_variant_backend(params, variant, stages)
    Q = _random_density_for_backend(backend)
    payload = _measure_stage(
        variant=variant,
        n_k=params.n_k,
        stage="single_fock_hf_energy",
        rows=stages,
        fn=lambda: _reference_payload(backend, Q),
    )
    if reference_output is not None:
        _save_reference_payload(reference_output, payload)
    reference = None if reference_input is None else _load_reference_payload(reference_input)
    correctness = _correctness_against_reference(
        variant=variant,
        n_k=params.n_k,
        payload=payload,
        reference=reference,
    )

    def _repeat_fock() -> None:
        for _idx in range(params.fock_repeats):
            backend.fock_hamiltonian(Q)

    repeat_start = time.perf_counter()
    _measure_stage(
        variant=variant,
        n_k=params.n_k,
        stage="fock_repeats",
        rows=stages,
        fn=_repeat_fock,
    )
    repeat_elapsed = time.perf_counter() - repeat_start
    hf_smoke_elapsed: float | None = None
    hf_smoke: dict[str, float] = {}
    if params.run_hf_smoke:
        smoke_start = time.perf_counter()
        hf_smoke = _measure_stage(
            variant=variant,
            n_k=params.n_k,
            stage="hf_smoke",
            rows=stages,
            fn=lambda: _run_hf_smoke(active, backend, params),
        )
        hf_smoke_elapsed = time.perf_counter() - smoke_start
    total_elapsed = time.perf_counter() - start
    summary = TaigeVariantSummary(
        variant=variant,
        n_k=params.n_k,
        skipped=False,
        total_elapsed_seconds=float(total_elapsed),
        final_rss_mb=_bytes_to_mb(_rss_bytes()),
        peak_rss_mb=_bytes_to_mb(_max_rss_bytes()),
        fock_repeats=params.fock_repeats,
        fock_repeats_elapsed_seconds=float(repeat_elapsed),
        fock_apply_seconds_per_call=float(repeat_elapsed / params.fock_repeats),
        hf_smoke_elapsed_seconds=hf_smoke_elapsed,
        hf_smoke_vp_plus_energy=hf_smoke.get("vp_plus_energy"),
        hf_smoke_ivc_energy=hf_smoke.get("ivc_energy"),
        keeps_full_lambda_blocks=spec.keeps_full_lambda_blocks,
        keeps_dense_tve=spec.keeps_dense_tve,
        uses_packed_tve=spec.uses_packed_tve,
        uses_matrix_free_fock=spec.uses_matrix_free_fock,
        dtype=spec.dtype,
    )
    return TaigeMemoryBenchmarkWorkerResult(
        input=params,
        variant_spec=spec,
        summary=summary,
        stages=tuple(stages),
        correctness=correctness,
        estimates=estimates,
    )


def parse_backend_variants(value: str | Sequence[str]) -> tuple[BackendVariant, ...]:
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",")]
    else:
        raw = [str(part).strip() for part in value]
    variants: list[BackendVariant] = []
    for part in raw:
        if not part:
            continue
        if part == "all":
            variants.extend(_ALL_VARIANTS)
            continue
        if part not in _ALL_VARIANTS:
            raise ValueError(f"unknown backend variant {part!r}; expected one of {_ALL_VARIANTS}")
        variants.append(part)  # type: ignore[arg-type]
    deduped: list[BackendVariant] = []
    for variant in variants:
        if variant not in deduped:
            deduped.append(variant)
    return tuple(deduped or ("baseline",))


def parse_n_k_list(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",")]
        return tuple(int(part) for part in raw if part)
    return tuple(int(part) for part in value)


def _run_worker_subprocess(
    *,
    script_path: Path,
    params: TaigeMemoryBenchmarkInput,
    variant: BackendVariant,
    reference_input: Path | None,
    reference_output: Path | None,
) -> TaigeMemoryBenchmarkWorkerResult:
    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        "--params-json",
        params.model_dump_json(),
        "--variant",
        variant,
    ]
    if reference_input is not None:
        cmd.extend(["--reference-input", str(reference_input)])
    if reference_output is not None:
        cmd.extend(["--reference-output", str(reference_output)])
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return TaigeMemoryBenchmarkWorkerResult.model_validate_json(proc.stdout)


def _run_worker_in_process(
    *,
    script_path: Path,
    params: TaigeMemoryBenchmarkInput,
    variant: BackendVariant,
    reference_input: Path | None,
    reference_output: Path | None,
) -> TaigeMemoryBenchmarkWorkerResult:
    _unused = script_path
    return run_taige_memory_benchmark_worker(
        params=params,
        variant=variant,
        reference_input=reference_input,
        reference_output=reference_output,
    )


def _model_dump(row: BaseModel) -> dict[str, Any]:
    return row.model_dump(mode="json")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def write_taige_memory_benchmark_outputs(
    output_dir: Path,
    results: Sequence[TaigeMemoryBenchmarkWorkerResult],
) -> TaigeMemoryBenchmarkRunSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = TaigeMemoryBenchmarkRunSummary(
        output_dir=str(output_dir),
        results=tuple(results),
    )
    (output_dir / "benchmark_results.json").write_text(
        summary.model_dump_json(indent=2),
    )
    _write_csv(
        output_dir / "variant_summary.csv",
        (
            {
                **_model_dump(result.summary),
                "estimated_lambda_blocks_mb": result.estimates.lambda_blocks_mb,
                "estimated_compact_lambda_blocks_mb": result.estimates.compact_lambda_blocks_mb,
                "estimated_dense_tve_mb": result.estimates.dense_tve_mb,
                "estimated_packed_tve_mb": result.estimates.packed_tve_mb,
                "estimated_one_q_full_lambda_slab_mb": result.estimates.one_q_full_lambda_slab_mb,
                "estimated_one_q_compact_lambda_slab_mb": result.estimates.one_q_compact_lambda_slab_mb,
            }
            for result in results
        ),
    )
    _write_csv(
        output_dir / "stage_measurements.csv",
        (_model_dump(stage) for result in results for stage in result.stages),
    )
    _write_csv(
        output_dir / "correctness.csv",
        (
            _model_dump(result.correctness)
            for result in results
            if result.correctness is not None
        ),
    )
    _write_csv(
        output_dir / "array_estimates.csv",
        (_model_dump(result.estimates) for result in results),
    )
    _write_markdown_report(output_dir / "benchmark_report.md", results)
    return summary


def _write_markdown_report(path: Path, results: Sequence[TaigeMemoryBenchmarkWorkerResult]) -> None:
    lines = [
        "# Taige Local Memory Backend Benchmark",
        "",
        "This report is generated by the local benchmark script. RSS values are process-level measurements from fresh worker runs.",
        "",
        "| n_k | variant | peak RSS MB | final RSS MB | total s | Fock s/call | correctness | notes |",
        "|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for result in results:
        summary = result.summary
        correctness = "n/a" if result.correctness is None else ("pass" if result.correctness.passed else "fail")
        notes = summary.skip_reason or ""
        lines.append(
            "| {n_k} | {variant} | {peak} | {final} | {total} | {fock} | {correctness} | {notes} |".format(
                n_k=summary.n_k,
                variant=summary.variant,
                peak="" if summary.peak_rss_mb is None else f"{summary.peak_rss_mb:.1f}",
                final="" if summary.final_rss_mb is None else f"{summary.final_rss_mb:.1f}",
                total="" if summary.total_elapsed_seconds is None else f"{summary.total_elapsed_seconds:.3g}",
                fock=""
                if summary.fock_apply_seconds_per_call is None
                else f"{summary.fock_apply_seconds_per_call:.3g}",
                correctness=correctness,
                notes=notes,
            )
        )
    lines.append("")
    lines.append("Important interpretation: variants that build from the dense baseline can reduce retained RSS but not peak RSS. The fused variants are the ones designed to reduce peak vertex memory.")
    path.write_text("\n".join(lines) + "\n")


def run_taige_memory_benchmark_suite(
    *,
    output_dir: Path,
    script_path: Path,
    base_params: TaigeMemoryBenchmarkInput,
    n_k_list: Sequence[int],
    variants: Sequence[BackendVariant],
    use_subprocess: bool = True,
) -> TaigeMemoryBenchmarkRunSummary:
    selected = tuple(variants)
    if any(variant != "baseline" for variant in selected) and "baseline" not in selected:
        selected = ("baseline", *selected)
    ordered = tuple(dict.fromkeys(("baseline", *selected)).keys()) if "baseline" in selected else selected
    results: list[TaigeMemoryBenchmarkWorkerResult] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = _run_worker_subprocess if use_subprocess else _run_worker_in_process
    for n_k in n_k_list:
        params = base_params.model_copy(update={"n_k": int(n_k)})
        reference_path = output_dir / f"baseline_reference_nk{int(n_k)}.npz"
        for variant in ordered:
            reference_input = None if variant == "baseline" else reference_path
            reference_output = reference_path if variant == "baseline" else None
            result = runner(
                script_path=script_path,
                params=params,
                variant=variant,
                reference_input=(
                    reference_input
                    if reference_input is not None and reference_input.exists()
                    else None
                ),
                reference_output=reference_output,
            )
            results.append(result)
    return write_taige_memory_benchmark_outputs(output_dir, results)


__all__ = [
    "BackendVariant",
    "BenchmarkHFBackend",
    "CompactTaigeDensityVertices",
    "PackedHermitianExchange",
    "TaigeArrayByteEstimate",
    "TaigeBackendVariantSpec",
    "TaigeCorrectnessRecord",
    "TaigeMemoryBenchmarkInput",
    "TaigeMemoryBenchmarkRunSummary",
    "TaigeMemoryBenchmarkWorkerResult",
    "TaigeStageMeasurement",
    "TaigeVariantSummary",
    "compact_lambdas_from_dense",
    "compact_vertices_from_dense",
    "dense_lambdas_from_compact",
    "dense_vertices_from_compact",
    "estimate_taige_array_bytes",
    "hartree_only_backend_from_dense",
    "matrix_free_backend_from_dense",
    "matrix_free_backend_from_vertices",
    "parse_backend_variants",
    "parse_n_k_list",
    "run_taige_memory_benchmark_suite",
    "run_taige_memory_benchmark_worker",
    "variant_spec",
    "write_taige_memory_benchmark_outputs",
]
