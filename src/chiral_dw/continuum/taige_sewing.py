"""Reciprocal-boundary sewing for the Taige plane-wave basis.

For ``k_raw = k_fold + s1*b1 + s2*b2``, the same physical plane wave obeys
``|k_raw, G> = |k_fold, G+s>``.  Consequently the folded-to-raw transport is
``(T_s v)_G = v_{G+s}``.  At finite plane-wave cutoff this map is a partial
isometry; the unmatched coefficients are set to zero and are never silently
renormalized.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

GridCoord = tuple[int, int]


def reciprocal_shift_gather(
    shell: tuple[GridCoord, ...], shift: GridCoord
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw ``G`` and folded ``G+s`` indices for a reciprocal shift."""

    shell_tuple = tuple((int(g1), int(g2)) for g1, g2 in shell)
    shell_index = {g: i for i, g in enumerate(shell_tuple)}
    s1, s2 = int(shift[0]), int(shift[1])
    source: list[int] = []
    target: list[int] = []
    for i, (g1, g2) in enumerate(shell_tuple):
        j = shell_index.get((g1 + s1, g2 + s2))
        if j is not None:
            source.append(i)
            target.append(j)
    return np.asarray(source, dtype=int), np.asarray(target, dtype=int)


class SewingFrameDiagnostics(BaseModel):
    """Finite-shell diagnostics for one transported frame."""

    model_config = ConfigDict(frozen=True)

    shift: GridCoord
    matched_plane_waves: int = Field(ge=0)
    total_plane_waves: int = Field(ge=1)
    geometric_coverage: float = Field(ge=0.0, le=1.0)
    min_retained_state_weight: float = Field(ge=0.0)
    max_retained_state_weight: float = Field(ge=0.0)
    max_gram_loss: float = Field(ge=0.0)


@dataclass(frozen=True)
class TaigeReciprocalTransport:
    """Cached gather implementing a reciprocal shift in ``(internal, G)`` order."""

    shell: tuple[GridCoord, ...]
    shift: GridCoord
    _source: np.ndarray = field(init=False, repr=False, compare=False)
    _target: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        shell = tuple((int(g1), int(g2)) for g1, g2 in self.shell)
        if not shell:
            raise ValueError("plane-wave shell must not be empty")
        if len(set(shell)) != len(shell):
            raise ValueError("plane-wave shell contains duplicate coordinates")
        shift = (int(self.shift[0]), int(self.shift[1]))
        source, target = reciprocal_shift_gather(shell, shift)
        object.__setattr__(self, "shell", shell)
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_target", target)

    @property
    def source_indices(self) -> np.ndarray:
        """Raw-basis plane-wave indices receiving matched coefficients."""

        return self._source.copy()

    @property
    def target_indices(self) -> np.ndarray:
        """Folded-basis indices corresponding to ``source_indices``."""

        return self._target.copy()

    @property
    def matched_plane_waves(self) -> int:
        return int(self._source.size)

    @property
    def geometric_coverage(self) -> float:
        return float(self.matched_plane_waves / len(self.shell))

    def folded_to_raw_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Transport vectors with leading microscopic dimension ``n_internal*n_G``."""

        arr, was_vector = self._as_frame(vectors)
        n_internal = arr.shape[0] // len(self.shell)
        blocks = arr.reshape(n_internal, len(self.shell), arr.shape[1])
        out = np.zeros_like(blocks)
        out[:, self._source, :] = blocks[:, self._target, :]
        result = out.reshape(arr.shape)
        return result[:, 0] if was_vector else result

    def raw_to_folded_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Apply the adjoint partial transport ``T_s^dagger``."""

        arr, was_vector = self._as_frame(vectors)
        n_internal = arr.shape[0] // len(self.shell)
        blocks = arr.reshape(n_internal, len(self.shell), arr.shape[1])
        out = np.zeros_like(blocks)
        out[:, self._target, :] = blocks[:, self._source, :]
        result = out.reshape(arr.shape)
        return result[:, 0] if was_vector else result

    def folded_to_raw_operator(self, operator: np.ndarray) -> np.ndarray:
        """Return ``T_s O T_s^dagger`` without constructing a dense ``T_s``."""

        op = self._as_operator(operator)
        flat_source = self._flat_indices(self._source, op.shape[0])
        flat_target = self._flat_indices(self._target, op.shape[0])
        out = np.zeros_like(op)
        out[np.ix_(flat_source, flat_source)] = op[np.ix_(flat_target, flat_target)]
        return out

    def raw_to_folded_operator(self, operator: np.ndarray) -> np.ndarray:
        """Return ``T_s^dagger O T_s`` without constructing a dense ``T_s``."""

        op = self._as_operator(operator)
        flat_source = self._flat_indices(self._source, op.shape[0])
        flat_target = self._flat_indices(self._target, op.shape[0])
        out = np.zeros_like(op)
        out[np.ix_(flat_target, flat_target)] = op[np.ix_(flat_source, flat_source)]
        return out

    def sewn_overlap(self, left_raw: np.ndarray, right_folded: np.ndarray) -> np.ndarray:
        """Return ``left_raw^dagger T_s right_folded`` for vectors or frames."""

        left, left_was_vector = self._as_frame(left_raw)
        right, right_was_vector = self._as_frame(right_folded)
        if left.shape[0] != right.shape[0]:
            raise ValueError("sewn overlap frames have incompatible microscopic dimensions")
        overlap = left.conj().T @ self.folded_to_raw_vectors(right)
        if left_was_vector and right_was_vector:
            return np.asarray(overlap[0, 0])
        return overlap

    def frame_diagnostics(self, frame_folded: np.ndarray) -> SewingFrameDiagnostics:
        """Measure norm retained by the finite-shell transport for a frame."""

        frame, _was_vector = self._as_frame(frame_folded)
        transported = self.folded_to_raw_vectors(frame)
        gram = transported.conj().T @ transported
        weights = np.real(np.diag(gram))
        loss = gram - np.eye(gram.shape[0], dtype=complex)
        return SewingFrameDiagnostics(
            shift=self.shift,
            matched_plane_waves=self.matched_plane_waves,
            total_plane_waves=len(self.shell),
            geometric_coverage=self.geometric_coverage,
            min_retained_state_weight=float(np.min(weights)),
            max_retained_state_weight=float(np.max(weights)),
            max_gram_loss=float(np.max(np.abs(loss))),
        )

    def _as_frame(self, vectors: np.ndarray) -> tuple[np.ndarray, bool]:
        arr = np.asarray(vectors, dtype=complex)
        was_vector = arr.ndim == 1
        if was_vector:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError("vectors must have shape (microscopic_dim,) or (microscopic_dim, n)")
        if arr.shape[0] % len(self.shell):
            raise ValueError("microscopic dimension must be a multiple of the shell size")
        return arr, was_vector

    def _as_operator(self, operator: np.ndarray) -> np.ndarray:
        op = np.asarray(operator, dtype=complex)
        if op.ndim != 2 or op.shape[0] != op.shape[1]:
            raise ValueError("operator must be a square matrix")
        if op.shape[0] % len(self.shell):
            raise ValueError("operator dimension must be a multiple of the shell size")
        return op

    def _flat_indices(self, plane_wave_indices: np.ndarray, dimension: int) -> np.ndarray:
        n_internal = dimension // len(self.shell)
        return np.concatenate(
            [block * len(self.shell) + plane_wave_indices for block in range(n_internal)]
        )


def taige_reciprocal_transport(
    shell: tuple[GridCoord, ...], shift: GridCoord
) -> TaigeReciprocalTransport:
    """Construct the canonical reciprocal transport for one boundary shift."""

    return TaigeReciprocalTransport(shell=shell, shift=shift)
