"""Native continuum/HF data models and small linear-algebra helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from chiral_dw.config import (
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
class DensityVertices:
    """Projected density vertices for the native block HF backend."""

    q_shifts: tuple[tuple[int, int], ...]
    target_minus_q: np.ndarray
    q_is_zero: np.ndarray
    lambda_blocks: np.ndarray
    v_over_a: np.ndarray
    g_channels: tuple[tuple[int, int], ...] = ((0, 0),)
    channel_in_disk: np.ndarray | None = None
    q_vectors_nm_inv: np.ndarray | None = None
    q_norm_nm_inv: np.ndarray | None = None
    v_q: np.ndarray | None = None


@dataclass(frozen=True)
class ContinuumBundle:
    """All numerical arrays needed for one native continuum HF problem."""

    grid: MomentumGrid
    active: ContinuumActiveSpace
    vertices: DensityVertices
    backend: object
    params: ContinuumModelParams
    interaction: ContinuumInteractionParams
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
