"""Momentum-space projector observables for continuum HF notebooks."""

from __future__ import annotations

import numpy as np

from chiral_dw.continuum.models import ContinuumActiveSpace


def valley_occupations(P: np.ndarray, active: ContinuumActiveSpace) -> tuple[np.ndarray, np.ndarray]:
    """Return per-k K and Kprime occupations."""

    arr = np.asarray(P, dtype=complex)
    n = active.n_active
    occ_k = np.real(np.trace(arr[:, :n, :n], axis1=-2, axis2=-1))
    occ_kp = np.real(np.trace(arr[:, n:, n:], axis1=-2, axis2=-1))
    return occ_k, occ_kp


def valley_polarization(P: np.ndarray, active: ContinuumActiveSpace) -> np.ndarray:
    """Return per-k valley polarization n_K - n_Kprime."""

    occ_k, occ_kp = valley_occupations(P, active)
    return occ_k - occ_kp


def ivc_order_parameter(P: np.ndarray, active: ContinuumActiveSpace) -> np.ndarray:
    """Return per-k trace of the K,Kprime coherence block."""

    arr = np.asarray(P, dtype=complex)
    n = active.n_active
    return np.trace(arr[:, :n, n:], axis1=-2, axis2=-1)


def projector_maps(P: np.ndarray, active: ContinuumActiveSpace) -> dict[str, np.ndarray]:
    """Return square-grid maps used to visualize continuum HF projectors."""

    n_k = active.grid.n_k
    occ_k, occ_kp = valley_occupations(P, active)
    ivc = ivc_order_parameter(P, active)
    return {
        "K": occ_k.reshape(n_k, n_k),
        "Kprime": occ_kp.reshape(n_k, n_k),
        "VP": (occ_k - occ_kp).reshape(n_k, n_k),
        "IVC_abs": np.abs(ivc).reshape(n_k, n_k),
        "IVC_phase": np.angle(ivc).reshape(n_k, n_k),
    }


__all__ = [
    "ivc_order_parameter",
    "projector_maps",
    "valley_occupations",
    "valley_polarization",
]
