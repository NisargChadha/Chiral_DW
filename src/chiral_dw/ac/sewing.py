"""Reciprocal-boundary transport for the finite-LL magnetic-Bloch basis.

For an up-valley cell-periodic Landau-level basis state,

``u_{k+G}(r) = eta_G exp(-i G.r + i l_B^2 G x k / 2) u_k(r)``.

The conjugate valley has the opposite momentum-dependent phase.  The common
``exp(-i G.r)`` operator is represented by shifting the reciprocal argument of
the LL density form factor; the remaining phases act in the two-valley active
coefficient space and are centralized here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chiral_dw.ac.nonideal import NonIdealACLLModel, cross2
from chiral_dw.continuum.models import hermitize

ReciprocalShift = tuple[int, int]


def reciprocal_parity(shift: ReciprocalShift) -> float:
    """Return the magnetic parity ``eta_G`` for ``G=m*b1+n*b2``."""

    m, n = int(shift[0]), int(shift[1])
    return 1.0 if m % 2 == 0 and n % 2 == 0 else -1.0


@dataclass(frozen=True)
class ACReciprocalTransport:
    """Exact reciprocal transport in the conjugate finite-LL active frame."""

    model: NonIdealACLLModel

    def fold_fractional(
        self,
        fractional: np.ndarray,
        *,
        atol: float = 1e-12,
    ) -> tuple[np.ndarray, ReciprocalShift]:
        """Fold fractional momentum into ``[0,1)^2`` and return its shift."""

        raw = np.asarray(fractional, dtype=float)
        if raw.shape != (2,):
            raise ValueError("fractional momentum must have shape (2,)")
        nearest = np.rint(raw)
        snapped = np.where(np.abs(raw - nearest) <= float(atol), nearest, raw)
        shift_array = np.floor(snapped).astype(int)
        folded = snapped - shift_array
        folded = np.where(np.abs(folded) <= float(atol), 0.0, folded)
        return folded.astype(float), (int(shift_array[0]), int(shift_array[1]))

    def reciprocal_vector(self, shift: ReciprocalShift) -> np.ndarray:
        """Return the Cartesian reciprocal vector for an integer shift."""

        m, n = int(shift[0]), int(shift[1])
        b1, b2 = self.model.fields.G_shell[0], self.model.fields.G_shell[1]
        return m * b1 + n * b2

    def valley_phase(
        self,
        k_folded: np.ndarray,
        shift: ReciprocalShift,
        *,
        valley: int,
    ) -> complex:
        """Return the scalar coefficient-space sewing phase for one valley.

        ``valley=+1`` is the up/C=+1 sector and ``valley=-1`` is its
        time-reversed conjugate.
        """

        sign = int(valley)
        if sign not in {-1, 1}:
            raise ValueError("valley must be +1 or -1")
        G = self.reciprocal_vector(shift)
        phase = 0.5 * self.model.l2 * cross2(G, np.asarray(k_folded, dtype=float))
        return complex(reciprocal_parity(shift) * np.exp(1j * sign * phase))

    def active_sewing_matrix(
        self,
        k_folded: np.ndarray,
        shift: ReciprocalShift,
    ) -> np.ndarray:
        """Return ``diag(s_+, s_-)`` in the up/down active coefficient frame."""

        return np.diag(
            [
                self.valley_phase(k_folded, shift, valley=1),
                self.valley_phase(k_folded, shift, valley=-1),
            ]
        ).astype(complex)

    def folded_to_raw_vector(
        self,
        vector: np.ndarray,
        k_folded: np.ndarray,
        shift: ReciprocalShift,
    ) -> np.ndarray:
        """Transport folded coefficients to the basis at ``k_folded+G``."""

        arr = np.asarray(vector, dtype=complex)
        if arr.shape != (2,):
            raise ValueError("active vector must have shape (2,)")
        sewing = self.active_sewing_matrix(k_folded, shift)
        return sewing.conj().T @ arr

    def folded_to_raw_projector(
        self,
        projector: np.ndarray,
        k_folded: np.ndarray,
        shift: ReciprocalShift,
    ) -> np.ndarray:
        """Transport a folded active projector to ``k_folded+G``."""

        arr = np.asarray(projector, dtype=complex)
        if arr.shape != (2, 2):
            raise ValueError("active projector must have shape (2,2)")
        sewing = self.active_sewing_matrix(k_folded, shift)
        return hermitize(sewing.conj().T @ arr @ sewing)

    def cocycle_residual(
        self,
        k_folded: np.ndarray,
        first: ReciprocalShift,
        second: ReciprocalShift,
    ) -> float:
        """Return the residual for two-step versus combined reciprocal sewing."""

        k = np.asarray(k_folded, dtype=float)
        first_vector = self.reciprocal_vector(first)
        combined = (int(first[0]) + int(second[0]), int(first[1]) + int(second[1]))
        sequential = self.active_sewing_matrix(k, first) @ self.active_sewing_matrix(
            k + first_vector,
            second,
        )
        direct = self.active_sewing_matrix(k, combined)
        return float(np.max(np.abs(sequential - direct)))


__all__ = [
    "ACReciprocalTransport",
    "ReciprocalShift",
    "reciprocal_parity",
]
