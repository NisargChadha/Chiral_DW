"""Momentum-space projector observables for continuum HF notebooks."""

from __future__ import annotations

import numpy as np

from chiral_dw.continuum.models import ContinuumActiveSpace


def active_basis_frames(active: ContinuumActiveSpace) -> np.ndarray:
    """Return direct-sum Bloch basis frames for embedded response calculations."""

    vectors = np.asarray(active.band_vectors, dtype=complex)
    if vectors.ndim != 4 or vectors.shape[0] != active.n_k or vectors.shape[1] != 2:
        raise ValueError("active.band_vectors must have shape (n_k,2,full_valley_dim,n_active)")
    valley_dim = vectors.shape[2]
    frames = np.zeros((active.n_k, 2 * valley_dim, active.dim), dtype=complex)
    n = active.n_active
    frames[:, :valley_dim, :n] = vectors[:, 0]
    frames[:, valley_dim:, n:] = vectors[:, 1]
    return frames


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


def valley_projector_matrix(P: np.ndarray, active: ContinuumActiveSpace) -> np.ndarray:
    """Return the traced 2x2 valley projector matrix at every momentum."""

    arr = np.asarray(P, dtype=complex)
    n = active.n_active
    matrix = np.empty((arr.shape[0], 2, 2), dtype=complex)
    matrix[:, 0, 0] = np.trace(arr[:, :n, :n], axis1=-2, axis2=-1)
    matrix[:, 0, 1] = np.trace(arr[:, :n, n:], axis1=-2, axis2=-1)
    matrix[:, 1, 0] = np.trace(arr[:, n:, :n], axis1=-2, axis2=-1)
    matrix[:, 1, 1] = np.trace(arr[:, n:, n:], axis1=-2, axis2=-1)
    return matrix


def projector_maps(P: np.ndarray, active: ContinuumActiveSpace) -> dict[str, np.ndarray]:
    """Return square-grid maps used to visualize continuum HF projectors."""

    n_k = active.grid.n_k
    occ_k, occ_kp = valley_occupations(P, active)
    ivc = ivc_order_parameter(P, active)
    valley_matrix = valley_projector_matrix(P, active)
    return {
        "K": occ_k.reshape(n_k, n_k),
        "Kprime": occ_kp.reshape(n_k, n_k),
        "VP": (occ_k - occ_kp).reshape(n_k, n_k),
        "IVC_abs": np.abs(ivc).reshape(n_k, n_k),
        "IVC_phase": np.angle(ivc).reshape(n_k, n_k),
        "P_KK": np.real(valley_matrix[:, 0, 0]).reshape(n_k, n_k),
        "P_KKprime_abs": np.abs(valley_matrix[:, 0, 1]).reshape(n_k, n_k),
        "P_KprimeK_abs": np.abs(valley_matrix[:, 1, 0]).reshape(n_k, n_k),
        "P_KprimeKprime": np.real(valley_matrix[:, 1, 1]).reshape(n_k, n_k),
    }


__all__ = [
    "active_basis_frames",
    "ivc_order_parameter",
    "projector_maps",
    "valley_projector_matrix",
    "valley_occupations",
    "valley_polarization",
]
