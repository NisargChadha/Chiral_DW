"""Momentum-space projector observables for continuum HF notebooks."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from chiral_dw.continuum.models import ContinuumActiveSpace


class ContinuumOrderDiagnostics(BaseModel):
    """Scalar VP/IVC order diagnostics for an active-space projector."""

    model_config = ConfigDict(frozen=True)

    Nz_block: float
    Nz_abs: float
    C_IVC_block: float
    IVC_amplitude_block: float
    C_IVC_scalar: float | None = None
    IVC_amplitude_scalar: float | None = None


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


def order_diagnostics(
    P: np.ndarray,
    active: ContinuumActiveSpace,
    *,
    n_occ_per_k: int = 1,
) -> ContinuumOrderDiagnostics:
    """Return normalized VP/IVC order magnitudes for one projector path endpoint."""

    arr = np.asarray(P, dtype=complex)
    expected = (active.n_k, active.dim, active.dim)
    if arr.shape != expected:
        raise ValueError(f"projector must have shape {expected}, got {arr.shape}")
    n = int(active.n_active)
    occ = float(n_occ_per_k)
    if occ <= 0.0:
        raise ValueError("n_occ_per_k must be positive")

    p_kk = arr[:, :n, :n]
    p_kkp = arr[:, :n, n:]
    p_kpkp = arr[:, n:, n:]
    nz = (
        np.mean(
            np.trace(p_kk, axis1=-2, axis2=-1)
            - np.trace(p_kpkp, axis1=-2, axis2=-1)
        ).real
        / occ
    )
    c_block = float(np.mean(np.sum(np.abs(p_kkp) ** 2, axis=(-2, -1))) / occ)
    c_scalar = None
    amp_scalar = None
    if n == 1:
        c_scalar = float(np.mean(np.abs(arr[:, 0, 1]) ** 2))
        amp_scalar = float(2.0 * np.sqrt(max(c_scalar, 0.0)))
    return ContinuumOrderDiagnostics(
        Nz_block=float(nz),
        Nz_abs=float(abs(nz)),
        C_IVC_block=c_block,
        IVC_amplitude_block=float(np.sqrt(max(c_block, 0.0))),
        C_IVC_scalar=c_scalar,
        IVC_amplitude_scalar=amp_scalar,
    )


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
    "ContinuumOrderDiagnostics",
    "active_basis_frames",
    "ivc_order_parameter",
    "order_diagnostics",
    "projector_maps",
    "valley_projector_matrix",
    "valley_occupations",
    "valley_polarization",
]
