"""Native continuum/HF data models and small linear-algebra helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from chiral_dw.config import (
    ContinuumFiniteQParams,
    ContinuumGridParams,
    ContinuumHFParams,
    ContinuumInteractionParams,
    ContinuumModelParams,
)

VALLEY_K = "K"
VALLEY_KPRIME = "Kprime"
VALLEY_ORDER = (VALLEY_K, VALLEY_KPRIME)


def hermitize(blocks: np.ndarray) -> np.ndarray:
    """Return the Hermitian part on the final two axes."""

    arr = np.asarray(blocks, dtype=complex)
    return 0.5 * (arr + arr.conj().swapaxes(-1, -2))


def block_trace_product(A: np.ndarray, B: np.ndarray) -> float:
    """Return Re sum_k Tr[A_k B_k]."""

    value = np.einsum("kab,kba->", np.asarray(A, complex), np.asarray(B, complex))
    return float(np.real_if_close(value, tol=1000).real)


def projector_idempotency_errors(P: np.ndarray) -> tuple[float, float]:
    """Return Frobenius and max-absolute idempotency errors."""

    arr = np.asarray(P, dtype=complex)
    err = arr @ arr - arr
    return float(np.linalg.norm(err)), float(np.max(np.abs(err)))


def finite_q_shift_metadata(
    finite_q: ContinuumFiniteQParams | None,
    grid: "MomentumGrid" | None = None,
) -> dict:
    """Return JSON-friendly metadata for the finite-Q active-frame convention."""

    controls = finite_q or ContinuumFiniteQParams()
    enabled = bool(controls.enabled)
    q_coord = (int(controls.q_coord[0]), int(controls.q_coord[1]))
    metadata = {
        "enabled": enabled,
        "q_coord": [q_coord[0], q_coord[1]],
        "shift_convention": "K: k-Q/2, Kprime: k+Q/2",
        "momentum_frame": (
            "finite-Q symmetric active frame"
            if enabled
            else "translation-symmetric Q=0 active frame"
        ),
    }
    if grid is None:
        metadata.update(
            {
                "grid": None,
                "q_fractional": [0.0, 0.0],
                "half_shift_coord": [0, 0],
                "half_shift_fractional": [0.0, 0.0],
                "half_shift_centered_fractional": [0.0, 0.0],
                "valley_shifts_fractional": {
                    VALLEY_K: [0.0, 0.0],
                    VALLEY_KPRIME: [0.0, 0.0],
                },
            }
        )
        return metadata

    if enabled:
        half = grid.assert_half_q_on_mesh(q_coord, controls.half_shift_coord)
        half_frac = np.array([half[0] / grid.n1, half[1] / grid.n2], dtype=float)
        q_frac = [q_coord[0] / grid.n1, q_coord[1] / grid.n2]
    else:
        half = (0, 0)
        half_frac = np.zeros(2, dtype=float)
        q_frac = [0.0, 0.0]
    metadata.update(
        {
            "grid": {"n1": int(grid.n1), "n2": int(grid.n2)},
            "q_fractional": [float(q_frac[0]), float(q_frac[1])],
            "half_shift_coord": [int(half[0]), int(half[1])],
            "half_shift_fractional": [float(half_frac[0]), float(half_frac[1])],
            "half_shift_centered_fractional": [
                float(x) for x in (np.mod(half_frac + 0.5, 1.0) - 0.5)
            ],
            "valley_shifts_fractional": {
                VALLEY_K: [float(-half_frac[0]), float(-half_frac[1])],
                VALLEY_KPRIME: [float(half_frac[0]), float(half_frac[1])],
            },
        }
    )
    return metadata


@dataclass(frozen=True)
class MomentumGrid:
    """Square fractional moire momentum grid."""

    n_k: int

    def __post_init__(self) -> None:
        if int(self.n_k) < 1:
            raise ValueError("n_k must be positive")
        object.__setattr__(self, "n_k", int(self.n_k))

    @property
    def size(self) -> int:
        return self.n_k * self.n_k

    @property
    def n1(self) -> int:
        return self.n_k

    @property
    def n2(self) -> int:
        return self.n_k

    def coord_of(self, index: int) -> tuple[int, int]:
        idx = int(index)
        return idx // self.n_k, idx % self.n_k

    def index_of(self, coord: tuple[int, int]) -> int:
        i, j = int(coord[0]) % self.n_k, int(coord[1]) % self.n_k
        return i * self.n_k + j

    def fold_grid_coord(self, coord: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        i, j = int(coord[0]), int(coord[1])
        fi = i % self.n_k
        fj = j % self.n_k
        return (fi, fj), ((i - fi) // self.n_k, (j - fj) // self.n_k)

    def shift_plus_q(self, coord: tuple[int, int], q_coord: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return self.fold_grid_coord((int(coord[0]) + int(q_coord[0]), int(coord[1]) + int(q_coord[1])))

    def shift_minus_q(self, coord: tuple[int, int], q_coord: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return self.fold_grid_coord((int(coord[0]) - int(q_coord[0]), int(coord[1]) - int(q_coord[1])))

    def assert_half_q_on_mesh(
        self,
        q_coord: tuple[int, int],
        half_shift_coord: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        """Return a folded Q/2 coordinate and validate it lies on the mesh."""

        q = (int(q_coord[0]), int(q_coord[1]))
        if half_shift_coord is not None:
            h = (int(half_shift_coord[0]), int(half_shift_coord[1]))
            folded, _shift = self.fold_grid_coord(h)
            if (2 * folded[0] - q[0]) % self.n1 or (2 * folded[1] - q[1]) % self.n2:
                raise ValueError(
                    "finite-Q half_shift_coord must satisfy 2*half_shift_coord = "
                    f"q_coord modulo the mesh; got half_shift_coord={h}, q_coord={q}"
                )
            return folded
        if q[0] % 2 or q[1] % 2:
            raise ValueError(f"finite-Q symmetric basis requires Q/2 on mesh; got {q}")
        return q[0] // 2, q[1] // 2

    def finite_q_physical_coord(
        self,
        coord: tuple[int, int],
        q_coord: tuple[int, int],
        valley: str,
        half_shift_coord: tuple[int, int] | None = None,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Map active-frame k to physical k in the symmetric finite-Q frame."""

        half = self.assert_half_q_on_mesh(q_coord, half_shift_coord)
        k = (int(coord[0]), int(coord[1]))
        if valley == VALLEY_K:
            return self.fold_grid_coord((k[0] - half[0], k[1] - half[1]))
        if valley == VALLEY_KPRIME:
            return self.fold_grid_coord((k[0] + half[0], k[1] + half[1]))
        raise ValueError(f"unknown valley {valley!r}")

    def fractional_coords(self) -> np.ndarray:
        coords = np.zeros((self.size, 2), dtype=float)
        for ik in range(self.size):
            i, j = self.coord_of(ik)
            coords[ik] = (i / self.n_k, j / self.n_k)
        return coords

    def centered_coords(self) -> np.ndarray:
        frac = self.fractional_coords()
        return ((frac + 0.5) % 1.0) - 0.5


@dataclass(frozen=True)
class ContinuumActiveSpace:
    """Native two-valley active space used by the continuum HF backend."""

    grid: MomentumGrid
    n_active: int
    h0: np.ndarray
    hole_energies: np.ndarray
    band_vectors: np.ndarray
    model: ContinuumModelParams
    shell: tuple[tuple[int, int], ...] = ()
    n_plane_waves: int = 0
    electron_energies: np.ndarray | None = None
    electron_vectors: np.ndarray | None = None
    source_index: np.ndarray | None = None
    source_shift: np.ndarray | None = None
    finite_q_enabled: bool = False
    q_coord: tuple[int, int] | None = None
    half_shift_coord: tuple[int, int] | None = None
    geometry: object | None = None
    bands: object | None = None

    @property
    def n_k(self) -> int:
        return self.grid.size

    @property
    def dim(self) -> int:
        return 2 * self.n_active

    @property
    def valley_order(self) -> tuple[str, str]:
        return VALLEY_ORDER

    def valley_index(self, valley: str) -> int:
        return self.valley_order.index(str(valley))


@dataclass(frozen=True)
class PhysicalDensityChannels:
    """Flat table of nonzero physical interaction channels.

    Every row represents one accepted physical ``q+G`` transfer and owns the
    corresponding momentum permutation, interaction weight, compact
    valley-diagonal form factor, and Hartree eligibility flag.
    """

    physical_transfer_nm_inv: np.ndarray
    momentum_permutation: np.ndarray
    weight: np.ndarray
    compact_form_factor: np.ndarray
    hartree: np.ndarray
    mesh_transfer: np.ndarray

    def __post_init__(self) -> None:
        transfer = np.asarray(self.physical_transfer_nm_inv, dtype=float)
        permutation = np.asarray(self.momentum_permutation, dtype=int)
        weight = np.asarray(self.weight, dtype=float)
        form_factor = np.asarray(self.compact_form_factor, dtype=complex)
        hartree = np.asarray(self.hartree, dtype=bool)
        mesh_transfer = np.asarray(self.mesh_transfer, dtype=int)
        if transfer.ndim != 2 or transfer.shape[1] != 2:
            raise ValueError("physical transfers must have shape (n_channels, 2)")
        n_channels = int(transfer.shape[0])
        if permutation.ndim != 2 or permutation.shape[0] != n_channels:
            raise ValueError(
                "momentum permutations must have shape (n_channels, n_blocks)"
            )
        if weight.shape != (n_channels,):
            raise ValueError("physical-channel weights must have shape (n_channels,)")
        if hartree.shape != (n_channels,):
            raise ValueError("Hartree flags must have shape (n_channels,)")
        if mesh_transfer.shape != (n_channels, 2):
            raise ValueError("mesh transfers must have shape (n_channels, 2)")
        if form_factor.ndim != 5 or form_factor.shape[:2] != (
            n_channels,
            permutation.shape[1],
        ):
            raise ValueError(
                "compact form factors must have shape "
                "(n_channels, n_blocks, 2, n_active, n_active)"
            )
        if form_factor.shape[2] != 2 or form_factor.shape[-1] != form_factor.shape[-2]:
            raise ValueError(
                "compact form factors must end in (2, n_active, n_active)"
            )
        object.__setattr__(self, "physical_transfer_nm_inv", transfer)
        object.__setattr__(self, "momentum_permutation", permutation)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "compact_form_factor", form_factor)
        object.__setattr__(self, "hartree", hartree)
        object.__setattr__(self, "mesh_transfer", mesh_transfer)

    @property
    def n_channels(self) -> int:
        return int(self.weight.size)

    @property
    def n_blocks(self) -> int:
        return int(self.momentum_permutation.shape[1])

    @property
    def n_active(self) -> int:
        return int(self.compact_form_factor.shape[-1])

    @property
    def storage_bytes(self) -> int:
        return int(
            self.physical_transfer_nm_inv.nbytes
            + self.momentum_permutation.nbytes
            + self.weight.nbytes
            + self.compact_form_factor.nbytes
            + self.hartree.nbytes
            + self.mesh_transfer.nbytes
        )


@dataclass(frozen=True)
class DensityVertices:
    """Projected density vertices for the native block HF backend."""

    q_shifts: tuple[tuple[int, int], ...]
    target_minus_q: np.ndarray
    q_is_zero: np.ndarray
    lambda_blocks: np.ndarray
    v_over_a: np.ndarray
    g_channels: tuple[tuple[int, int], ...] = ((0, 0),)
    vertex_layout: Literal["dense", "valley_compact"] = "dense"
    lambda_compact: np.ndarray | None = None
    channel_in_disk: np.ndarray | None = None
    q_vectors_nm_inv: np.ndarray | None = None
    q_norm_nm_inv: np.ndarray | None = None
    v_q: np.ndarray | None = None
    physical_channels: PhysicalDensityChannels | None = None
    prebuilt_exchange_sectors: np.ndarray | None = None


def compact_lambdas_from_dense(lambdas: np.ndarray, n_active: int) -> np.ndarray:
    """Return valley-diagonal blocks as ``(..., 2, n_active, n_active)``."""

    arr = np.asarray(lambdas)
    n = int(n_active)
    if arr.shape[-2:] != (2 * n, 2 * n):
        raise ValueError("dense lambda shape is incompatible with n_active")
    compact = np.empty(arr.shape[:-2] + (2, n, n), dtype=arr.dtype)
    for iv in range(2):
        start = iv * n
        stop = start + n
        compact[..., iv, :, :] = arr[..., start:stop, start:stop]
    return compact


def dense_lambdas_from_compact(lambda_compact: np.ndarray) -> np.ndarray:
    """Expand valley-compact lambdas into dense active-basis matrices."""

    compact = np.asarray(lambda_compact)
    if compact.ndim < 4 or compact.shape[-3] != 2 or compact.shape[-1] != compact.shape[-2]:
        raise ValueError("compact lambdas must end in (2, n_active, n_active)")
    n = int(compact.shape[-1])
    dense = np.zeros(compact.shape[:-3] + (2 * n, 2 * n), dtype=compact.dtype)
    for iv in range(2):
        start = iv * n
        stop = start + n
        dense[..., start:stop, start:stop] = compact[..., iv, :, :]
    return dense


def density_vertices_dense_lambdas(vertices: DensityVertices) -> np.ndarray:
    """Return dense lambda blocks for either supported vertex layout."""

    if vertices.vertex_layout == "dense":
        return np.asarray(vertices.lambda_blocks)
    if vertices.vertex_layout == "valley_compact":
        if vertices.lambda_compact is None:
            raise ValueError("valley_compact DensityVertices require lambda_compact")
        return dense_lambdas_from_compact(vertices.lambda_compact)
    raise ValueError(f"unknown density vertex layout {vertices.vertex_layout!r}")


@dataclass(frozen=True)
class ContinuumBundle:
    """All numerical arrays needed for one native continuum HF problem."""

    grid: MomentumGrid
    active: ContinuumActiveSpace
    vertices: DensityVertices
    backend: object
    params: ContinuumModelParams
    interaction: ContinuumInteractionParams
    finite_q: ContinuumFiniteQParams = field(default_factory=ContinuumFiniteQParams)
    bands: object | None = None
    geometry: object | None = None
    form_factors: object | None = None


class ContinuumHFDiagnostics(BaseModel):
    """Scalar diagnostics for one native HF density."""

    model_config = ConfigDict(frozen=True)

    energy: float
    delta_energy: float
    delta_P: float
    idempotency_error_fro: float
    idempotency_error_max: float
    constraint_error: float
    aufbau_residual_norm: float
    commutator_norm: float
    trace_error: float
    direct_gap_min: float
    indirect_gap: float
    iteration: int
    constraint_name: str | None = None
    lambda_value: float | None = None
    fallback_reason: str | None = None
    occupation_mode: Literal["fixed_per_k", "global"] = "fixed_per_k"
    density_kind: Literal["mixed", "final_idempotent"] = "mixed"
    self_consistency_warning: bool = False


class ReferenceHamiltonianDiagnostics(BaseModel):
    """Channel diagnostics for one raw HF reference Hamiltonian."""

    model_config = ConfigDict(frozen=True)

    scalar_norm: float
    traceless_norm: float
    valley_diagonal_norm: float
    intervalley_norm: float
    hermiticity_error: float


@dataclass(frozen=True)
class ContinuumHFResult:
    """Final projector, HF Hamiltonian, and diagnostics for one HF reference."""

    P: np.ndarray
    H_hf: np.ndarray
    energy: float
    converged: bool
    n_iter: int
    diagnostics: ContinuumHFDiagnostics
    history: tuple[ContinuumHFDiagnostics, ...] = field(default_factory=tuple)
    snapshots: tuple["ContinuumHFIterationSnapshot", ...] = field(default_factory=tuple)
    seed: str = ""
    constraint_name: str | None = None


@dataclass(frozen=True)
class ContinuumHFIterationSnapshot:
    """Stored projector state from one HF iteration."""

    iteration: int
    P: np.ndarray
    energy: float
    diagnostics: ContinuumHFDiagnostics


@dataclass(frozen=True)
class SymmetricHFReferences:
    """The three symmetry-related HF references used by the convex path."""

    vp_plus: ContinuumHFResult
    vp_minus: ContinuumHFResult
    ivc: ContinuumHFResult
    n_occ_per_k: int = 1

    @property
    def n_particles(self) -> float:
        return float(self.n_occ_per_k * self.n_blocks)

    @property
    def H_vp_plus(self) -> np.ndarray:
        return self.vp_plus.H_hf

    @property
    def H_vp_minus(self) -> np.ndarray:
        return self.vp_minus.H_hf

    @property
    def H_ivc(self) -> np.ndarray:
        return self.ivc.H_hf

    @property
    def dim(self) -> int:
        return int(self.H_vp_plus.shape[-1])

    @property
    def n_blocks(self) -> int:
        return int(self.H_vp_plus.shape[0])


@dataclass(frozen=True)
class ConvexPathDiagnostics:
    """Diagnostics for one theta point in the symmetric convex path."""

    theta: float
    phi: float
    w_vp_plus: float
    w_vp_minus: float
    w_ivc: float
    direct_gap_min: float
    indirect_gap: float
    projector_idempotency_error_fro: float
    projector_idempotency_error_max: float
