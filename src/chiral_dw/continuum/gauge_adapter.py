"""Reusable momentum, flavor, gauge, and matrix-index active-space adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActiveSpaceGaugeAdapter:
    """Transport block matrices between two finite active-space conventions.

    ``target_index_of_source`` maps each source momentum block to its target
    block. ``source_index_in_target_order`` states which source flavor occupies
    each target flavor slot before the unitary basis rotation. The optional
    transpose handles libraries that store the two single-particle matrix
    indices in opposite order.
    """

    target_index_of_source: np.ndarray
    source_index_in_target_order: np.ndarray
    unitary_target_from_reordered_source: np.ndarray
    transpose_source_matrices: bool = False

    def __post_init__(self) -> None:
        momentum_map = np.asarray(self.target_index_of_source, dtype=int)
        flavor_map = np.asarray(self.source_index_in_target_order, dtype=int)
        unitaries = np.asarray(
            self.unitary_target_from_reordered_source,
            dtype=complex,
        )
        if momentum_map.ndim != 1 or momentum_map.size < 1:
            raise ValueError("target_index_of_source must be a nonempty 1D array")
        if not np.array_equal(np.sort(momentum_map), np.arange(momentum_map.size)):
            raise ValueError("target_index_of_source must be a momentum permutation")
        if flavor_map.ndim != 1 or flavor_map.size < 1:
            raise ValueError("source_index_in_target_order must be a nonempty 1D array")
        if not np.array_equal(np.sort(flavor_map), np.arange(flavor_map.size)):
            raise ValueError("source_index_in_target_order must be a flavor permutation")
        expected = (momentum_map.size, flavor_map.size, flavor_map.size)
        if unitaries.shape != expected:
            raise ValueError(
                "unitary_target_from_reordered_source must have shape "
                f"{expected}, got {unitaries.shape}"
            )
        identity = np.eye(flavor_map.size, dtype=complex)
        gram = np.einsum(
            "kba,kbc->kac",
            unitaries.conj(),
            unitaries,
            optimize=True,
        )
        if not np.allclose(gram, identity[None, :, :], atol=1.0e-10, rtol=0.0):
            raise ValueError("active-space gauge matrices must be unitary")
        object.__setattr__(self, "target_index_of_source", momentum_map.copy())
        object.__setattr__(
            self,
            "source_index_in_target_order",
            flavor_map.copy(),
        )
        object.__setattr__(
            self,
            "unitary_target_from_reordered_source",
            unitaries.copy(),
        )
        object.__setattr__(
            self,
            "transpose_source_matrices",
            bool(self.transpose_source_matrices),
        )

    @classmethod
    def from_diagonal_phases(
        cls,
        *,
        target_index_of_source: np.ndarray,
        source_index_in_target_order: np.ndarray,
        phases_target_order_by_source: np.ndarray,
        transpose_source_matrices: bool = False,
    ) -> "ActiveSpaceGaugeAdapter":
        """Construct an adapter for a nondegenerate band gauge."""

        phases = np.asarray(phases_target_order_by_source, dtype=complex)
        if phases.ndim != 2:
            raise ValueError("phases_target_order_by_source must be a 2D array")
        magnitudes = np.abs(phases)
        if not np.all(np.isfinite(phases)) or not np.allclose(
            magnitudes,
            1.0,
            atol=1.0e-10,
            rtol=0.0,
        ):
            raise ValueError("diagonal gauge phases must be finite and unit modulus")
        unitaries = np.zeros(
            (phases.shape[0], phases.shape[1], phases.shape[1]),
            dtype=complex,
        )
        diagonal = np.arange(phases.shape[1])
        unitaries[:, diagonal, diagonal] = phases
        return cls(
            target_index_of_source=target_index_of_source,
            source_index_in_target_order=source_index_in_target_order,
            unitary_target_from_reordered_source=unitaries,
            transpose_source_matrices=transpose_source_matrices,
        )

    @property
    def n_momenta(self) -> int:
        return int(self.target_index_of_source.size)

    @property
    def dimension(self) -> int:
        return int(self.source_index_in_target_order.size)

    def to_target(self, source_matrices: np.ndarray) -> np.ndarray:
        """Transport source matrices into the target active-space convention."""

        source = np.asarray(source_matrices)
        expected = (self.n_momenta, self.dimension, self.dimension)
        if source.shape != expected:
            raise ValueError(f"source matrices must have shape {expected}, got {source.shape}")
        target = np.empty(expected, dtype=np.result_type(source.dtype, complex))
        order = self.source_index_in_target_order
        for source_index, target_index in enumerate(self.target_index_of_source):
            block = source[source_index]
            if self.transpose_source_matrices:
                block = block.T
            reordered = block[np.ix_(order, order)]
            unitary = self.unitary_target_from_reordered_source[source_index]
            target[target_index] = unitary @ reordered @ unitary.conj().T
        return target

    def to_source(self, target_matrices: np.ndarray) -> np.ndarray:
        """Transport target matrices back into the source convention."""

        target = np.asarray(target_matrices)
        expected = (self.n_momenta, self.dimension, self.dimension)
        if target.shape != expected:
            raise ValueError(f"target matrices must have shape {expected}, got {target.shape}")
        source = np.empty(expected, dtype=np.result_type(target.dtype, complex))
        inverse_order = np.argsort(self.source_index_in_target_order)
        for source_index, target_index in enumerate(self.target_index_of_source):
            unitary = self.unitary_target_from_reordered_source[source_index]
            reordered = unitary.conj().T @ target[target_index] @ unitary
            block = reordered[np.ix_(inverse_order, inverse_order)]
            source[source_index] = (
                block.T if self.transpose_source_matrices else block
            )
        return source

    def npz_payload(self) -> dict[str, np.ndarray]:
        """Return a serialization payload independent of either backend."""

        return {
            "target_index_of_source": self.target_index_of_source,
            "source_index_in_target_order": self.source_index_in_target_order,
            "unitary_target_from_reordered_source": (
                self.unitary_target_from_reordered_source
            ),
            "transpose_source_matrices": np.asarray(
                self.transpose_source_matrices,
                dtype=bool,
            ),
        }


__all__ = ["ActiveSpaceGaugeAdapter"]
