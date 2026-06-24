"""Projector response, U(1) reconstruction, K(theta), and cG."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TAU_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


@dataclass(frozen=True)
class KThetaResult:
    """Dimensionless response kernel sampled on theta nodes."""

    theta: np.ndarray
    K: np.ndarray
    cG: float


def hermitian_part(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=complex)
    return 0.5 * (arr + arr.conj().swapaxes(-1, -2))


def u1_rotation(phi: float) -> np.ndarray:
    """Return U_phi=exp(-i phi tau_z/2)."""
    angle = float(phi)
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)]).astype(complex)


def rotate_projector_phi(P: np.ndarray, phi: float) -> np.ndarray:
    """Rotate flavor projector(s) by U_phi."""
    U = u1_rotation(phi)
    rotated = U @ np.asarray(P, dtype=complex) @ U.conj().T
    return hermitian_part(rotated)


def phi_derivative_projector(P: np.ndarray) -> np.ndarray:
    """Analytic derivative d_phi P at phi=0 for exp(-i phi tau_z/2)."""
    arr = np.asarray(P, dtype=complex)
    comm = TAU_Z @ arr - arr @ TAU_Z
    return -0.5j * comm


def projector_grid_from_theta(P_theta: np.ndarray, phi_nodes: np.ndarray) -> np.ndarray:
    """Return P[theta, phi, k1, k2, 2, 2] from P[theta, k1, k2, 2, 2]."""
    base = np.asarray(P_theta, dtype=complex)
    if base.ndim != 5 or base.shape[-2:] != (2, 2) or base.shape[1] != base.shape[2]:
        raise ValueError("P_theta must have shape (n_theta,n_k,n_k,2,2)")
    phi = np.asarray(phi_nodes, dtype=float)
    out = np.zeros((base.shape[0], len(phi), base.shape[1], base.shape[2], 2, 2), dtype=complex)
    for ip, value in enumerate(phi):
        out[:, ip] = rotate_projector_phi(base, float(value))
    return out


def projector_errors(P: np.ndarray) -> dict[str, float]:
    """Return max Hermiticity, idempotency, and trace-one errors."""
    arr = np.asarray(P, dtype=complex)
    herm = np.max(np.abs(arr - arr.conj().swapaxes(-1, -2)))
    idem = np.max(np.abs(arr @ arr - arr))
    trace = np.max(np.abs(np.trace(arr, axis1=-2, axis2=-1) - 1.0))
    return {
        "hermiticity": float(herm),
        "idempotency": float(idem),
        "trace": float(trace),
    }


def _periodic_derivative(arr: np.ndarray, axis: int, step: float) -> np.ndarray:
    return (np.roll(arr, -1, axis=axis) - np.roll(arr, 1, axis=axis)) / (2.0 * float(step))


def _theta_derivative(P_theta: np.ndarray, theta: np.ndarray) -> np.ndarray:
    if len(theta) < 3:
        raise ValueError("at least three theta nodes are required")
    edge_order = 2 if len(theta) >= 3 else 1
    return np.gradient(P_theta, theta, axis=0, edge_order=edge_order)


def projector_berry_curvature(P: np.ndarray, d_a: np.ndarray, d_b: np.ndarray) -> np.ndarray:
    """Return F_ab=-i Tr P[d_aP,d_bP] at every non-matrix grid point."""
    comm = d_a @ d_b - d_b @ d_a
    value = -1j * np.trace(P @ comm, axis1=-2, axis2=-1)
    return np.real_if_close(value, tol=1000).real


def k_theta_from_projectors(P_theta: np.ndarray, theta: np.ndarray) -> KThetaResult:
    """Compute dimensionless K(theta) and cG from P[theta,k1,k2,2,2]."""
    P = hermitian_part(np.asarray(P_theta, dtype=complex))
    theta_arr = np.asarray(theta, dtype=float)
    if P.ndim != 5 or P.shape[-2:] != (2, 2):
        raise ValueError("P_theta must have shape (n_theta,n_k,n_k,2,2)")
    if P.shape[0] != len(theta_arr):
        raise ValueError("theta length must match P_theta leading dimension")
    n_k = P.shape[1]
    if P.shape[2] != n_k:
        raise ValueError("momentum grid must be square")

    du = dv = 1.0 / float(n_k)
    d_u = _periodic_derivative(P, axis=1, step=du)
    d_v = _periodic_derivative(P, axis=2, step=dv)
    d_th = _theta_derivative(P, theta_arr)
    d_ph = phi_derivative_projector(P)

    Fuv = projector_berry_curvature(P, d_u, d_v)
    Fthph = projector_berry_curvature(P, d_th, d_ph)
    Fu_th = projector_berry_curvature(P, d_u, d_th)
    Fv_ph = projector_berry_curvature(P, d_v, d_ph)
    Fu_ph = projector_berry_curvature(P, d_u, d_ph)
    Fv_th = projector_berry_curvature(P, d_v, d_th)

    density = (Fuv * Fthph - Fu_th * Fv_ph + Fu_ph * Fv_th) / (4.0 * np.pi**2)
    K = np.mean(density, axis=(1, 2))
    return KThetaResult(theta=theta_arr, K=np.asarray(K, dtype=float), cG=compute_cG(theta_arr, K))


def compute_cG(theta: np.ndarray, K_theta: np.ndarray) -> float:
    """Return dimensionless cG = integral K(theta) log tan(theta/2) dtheta."""
    th = np.asarray(theta, dtype=float)
    K = np.asarray(K_theta, dtype=float)
    if th.shape != K.shape:
        raise ValueError("theta and K_theta must have matching shapes")
    eps = 1e-12
    clipped = np.clip(th, eps, np.pi - eps)
    weight = np.log(np.tan(0.5 * clipped))
    return float(np.trapezoid(K * weight, th))
