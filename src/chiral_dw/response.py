"""Projector response, U(1) reconstruction, K(theta), and cG."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KThetaResult:
    """Dimensionless response kernel sampled on theta nodes."""

    theta: np.ndarray
    K: np.ndarray
    cG: float


def hermitian_part(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=complex)
    return 0.5 * (arr + arr.conj().swapaxes(-1, -2))


def flavor_tau_z(dim: int) -> np.ndarray:
    """Return tau_z=diag(+I_n,-I_n) for an even active flavor dimension."""

    d = int(dim)
    if d <= 0 or d % 2:
        raise ValueError("dim must be a positive even integer")
    n = d // 2
    charges = np.concatenate([np.ones(n, dtype=float), -np.ones(n, dtype=float)])
    return np.diag(charges.astype(complex))


def u1_rotation(phi: float, dim: int = 2) -> np.ndarray:
    """Return U_phi=exp(-i phi tau_z/2) in valley-block flavor space."""
    angle = float(phi)
    tau = np.diag(flavor_tau_z(dim)).real
    phases = np.exp(-0.5j * angle * tau)
    return np.diag(phases.astype(complex))


def rotate_projector_phi(P: np.ndarray, phi: float) -> np.ndarray:
    """Rotate flavor projector(s) by U_phi."""
    arr = np.asarray(P, dtype=complex)
    if arr.shape[-1] != arr.shape[-2]:
        raise ValueError("projector blocks must be square in the final two axes")
    U = u1_rotation(phi, dim=arr.shape[-1])
    rotated = np.einsum("ab,...bc,dc->...ad", U, arr, U.conj(), optimize=True)
    return hermitian_part(rotated)


def phi_derivative_projector(P: np.ndarray) -> np.ndarray:
    """Analytic derivative d_phi P at phi=0 for exp(-i phi tau_z/2)."""
    arr = np.asarray(P, dtype=complex)
    if arr.shape[-1] != arr.shape[-2]:
        raise ValueError("projector blocks must be square in the final two axes")
    tau = flavor_tau_z(arr.shape[-1])
    comm = (
        np.einsum("ab,...bc->...ac", tau, arr, optimize=True)
        - np.einsum("...ab,bc->...ac", arr, tau, optimize=True)
    )
    return -0.5j * comm


def projector_grid_from_theta(P_theta: np.ndarray, phi_nodes: np.ndarray) -> np.ndarray:
    """Return P[theta,phi,k1,k2,dim,dim] from P[theta,k1,k2,dim,dim]."""
    base = np.asarray(P_theta, dtype=complex)
    if base.ndim != 5 or base.shape[-1] != base.shape[-2] or base.shape[1] != base.shape[2]:
        raise ValueError("P_theta must have shape (n_theta,n_k,n_k,dim,dim)")
    dim = base.shape[-1]
    flavor_tau_z(dim)
    phi = np.asarray(phi_nodes, dtype=float)
    out = np.zeros((base.shape[0], len(phi), base.shape[1], base.shape[2], dim, dim), dtype=complex)
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
    """Compute dimensionless K(theta) and cG from P[theta,k1,k2,dim,dim]."""
    P = hermitian_part(np.asarray(P_theta, dtype=complex))
    theta_arr = np.asarray(theta, dtype=float)
    if P.ndim != 5 or P.shape[-1] != P.shape[-2]:
        raise ValueError("P_theta must have shape (n_theta,n_k,n_k,dim,dim)")
    flavor_tau_z(P.shape[-1])
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


def _embedded_curvature(
    basis_center: np.ndarray,
    projector_center: np.ndarray,
    terms_a: tuple[tuple[np.ndarray, np.ndarray, float], ...],
    terms_b: tuple[tuple[np.ndarray, np.ndarray, float], ...],
) -> float:
    """Return -i Tr P[d_a P,d_b P] using low-rank embedded active-basis terms."""

    wc = np.asarray(basis_center, dtype=complex)
    pc = np.asarray(projector_center, dtype=complex)
    total = 0.0j
    for wi, ai, ci in terms_a:
        wi_arr = np.asarray(wi, dtype=complex)
        ai_arr = np.asarray(ai, dtype=complex)
        o_ci = wc.conj().T @ wi_arr
        o_ic = o_ci.conj().T
        for wj, bj, cj in terms_b:
            wj_arr = np.asarray(wj, dtype=complex)
            bj_arr = np.asarray(bj, dtype=complex)
            o_cj = wc.conj().T @ wj_arr
            o_jc = o_cj.conj().T
            o_ij = wi_arr.conj().T @ wj_arr
            o_ji = o_ij.conj().T
            forward = np.trace(pc @ o_ci @ ai_arr @ o_ij @ bj_arr @ o_jc)
            reverse = np.trace(pc @ o_cj @ bj_arr @ o_ji @ ai_arr @ o_ic)
            total += float(ci) * float(cj) * (forward - reverse)
    return float(np.real_if_close(-1j * total, tol=1000).real)


def k_theta_from_projectors_with_basis(
    P_theta: np.ndarray,
    theta: np.ndarray,
    basis_frames: np.ndarray,
) -> KThetaResult:
    """Compute K(theta) after embedding active projectors into Bloch basis frames.

    ``P_theta`` has shape ``(n_theta, n_k, n_k, dim, dim)``. ``basis_frames``
    has shape ``(n_k, n_k, full_dim, dim)`` and stores orthonormal columns
    spanning the active basis at each momentum. This keeps the phi derivative
    in active valley space while letting momentum derivatives see the
    k-dependent Chern-band geometry.
    """

    P = hermitian_part(np.asarray(P_theta, dtype=complex))
    theta_arr = np.asarray(theta, dtype=float)
    W = np.asarray(basis_frames, dtype=complex)
    if P.ndim != 5 or P.shape[-1] != P.shape[-2]:
        raise ValueError("P_theta must have shape (n_theta,n_k,n_k,dim,dim)")
    flavor_tau_z(P.shape[-1])
    if P.shape[0] != len(theta_arr):
        raise ValueError("theta length must match P_theta leading dimension")
    if W.shape[:2] != P.shape[1:3] or W.shape[-1] != P.shape[-1]:
        raise ValueError("basis_frames must have shape (n_k,n_k,full_dim,dim)")

    n_theta, n_k, _, _dim, _ = P.shape
    du = dv = 1.0 / float(n_k)
    d_th = _theta_derivative(P, theta_arr)
    d_ph = phi_derivative_projector(P)
    density = np.zeros((n_theta, n_k, n_k), dtype=float)

    for it in range(n_theta):
        for i in range(n_k):
            ip = (i + 1) % n_k
            im = (i - 1) % n_k
            for j in range(n_k):
                jp = (j + 1) % n_k
                jm = (j - 1) % n_k
                wc = W[i, j]
                pc = P[it, i, j]
                du_terms = (
                    (W[ip, j], P[it, ip, j], 0.5 / du),
                    (W[im, j], P[it, im, j], -0.5 / du),
                )
                dv_terms = (
                    (W[i, jp], P[it, i, jp], 0.5 / dv),
                    (W[i, jm], P[it, i, jm], -0.5 / dv),
                )
                dtheta_terms = ((wc, d_th[it, i, j], 1.0),)
                dphi_terms = ((wc, d_ph[it, i, j], 1.0),)

                Fuv = _embedded_curvature(wc, pc, du_terms, dv_terms)
                Fthph = _embedded_curvature(wc, pc, dtheta_terms, dphi_terms)
                Fu_th = _embedded_curvature(wc, pc, du_terms, dtheta_terms)
                Fv_ph = _embedded_curvature(wc, pc, dv_terms, dphi_terms)
                Fu_ph = _embedded_curvature(wc, pc, du_terms, dphi_terms)
                Fv_th = _embedded_curvature(wc, pc, dv_terms, dtheta_terms)
                density[it, i, j] = (
                    Fuv * Fthph - Fu_th * Fv_ph + Fu_ph * Fv_th
                ) / (4.0 * np.pi**2)

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
