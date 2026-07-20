"""Gauge-covariant projector orbital magnetization on a periodic mesh.

The implementation evaluates the projector traces without materializing dense
microscopic projectors or their derivatives.  ``P`` is the occupied-hole
projector and ``Q`` is the explicitly retained empty-hole projector; ``Q`` is
not replaced by ``I-P`` when remote bands are truncated.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from chiral_dw.continuum.models import MomentumGrid

HBAR2_OVER_2ME_MEV_NM2 = 38.0998212
FrameTransport = Callable[[np.ndarray, tuple[int, int]], np.ndarray]


class OrbitalMagnetizationSummary(BaseModel):
    """Scalar projector-formula result for one chemical potential."""

    model_config = ConfigDict(frozen=True)

    chemical_potential_hole_mev: float
    orbital_magnetization_mu_b_per_cell: float
    self_rotation_mu_b_per_cell: float
    im_w_minus_n_mean_mev_nm2: float
    im_w_plus_n_mean_mev_nm2: float
    streda_slope_mu_b_per_mev: float
    streda_chern_from_retained_pq: float
    moire_cell_area_nm2: float = Field(gt=0.0)
    max_occupied_orthonormality_error: float = Field(ge=0.0)
    max_empty_orthonormality_error: float = Field(ge=0.0)
    max_occupied_empty_overlap: float = Field(ge=0.0)


@dataclass(frozen=True)
class OrbitalMagnetizationEvaluation:
    """Summary plus k-resolved complex ``W_xy`` and ``N_xy`` traces."""

    summary: OrbitalMagnetizationSummary
    w_xy_mev_nm2: np.ndarray
    n_xy_mev_nm2: np.ndarray


def evaluate_projector_orbital_magnetization(
    *,
    grid: MomentumGrid,
    occupied_frames: np.ndarray,
    empty_frames: np.ndarray,
    hamiltonian_on_occupied: np.ndarray,
    hamiltonian_on_empty: np.ndarray,
    reciprocal_basis_nm_inv: np.ndarray,
    chemical_potential_hole_mev: float,
    transport: FrameTransport | None = None,
) -> OrbitalMagnetizationEvaluation:
    """Evaluate the positive-charge hole formula using sewn central differences.

    The returned moment is also the physical doped-electron magnetization
    relative to the filled-valence reference when ``mu_hole = -mu_electron``.
    Hamiltonian actions must come from the full common-basis Hamiltonian, not a
    retained-band spectral reconstruction.
    """

    occupied = _validate_frames("occupied_frames", occupied_frames, grid)
    empty = _validate_frames("empty_frames", empty_frames, grid)
    if occupied.shape[1] != empty.shape[1]:
        raise ValueError("occupied and empty frames must share a microscopic basis")
    h_occ = _validate_action("hamiltonian_on_occupied", hamiltonian_on_occupied, occupied)
    h_emp = _validate_action("hamiltonian_on_empty", hamiltonian_on_empty, empty)
    basis = np.asarray(reciprocal_basis_nm_inv, dtype=float)
    if basis.shape != (2, 2) or not np.all(np.isfinite(basis)):
        raise ValueError("reciprocal_basis_nm_inv must be a finite 2x2 matrix")
    determinant = float(np.linalg.det(basis))
    if abs(determinant) < 1e-14:
        raise ValueError("reciprocal basis must be invertible")
    inverse_basis = np.linalg.inv(basis)
    transport_frame = _identity_transport if transport is None else transport

    occ_orth, emp_orth, cross = _frame_errors(occupied, empty)
    if max(occ_orth, emp_orth, cross) > 5e-6:
        raise ValueError(
            "occupied/empty frames must be orthonormal and mutually orthogonal; "
            f"errors are {occ_orth:.3e}, {emp_orth:.3e}, {cross:.3e}"
        )

    mu = float(chemical_potential_hole_mev)
    w_xy = np.zeros(grid.size, dtype=complex)
    n_xy = np.zeros(grid.size, dtype=complex)
    chemical_slope_integrand = np.zeros(grid.size, dtype=complex)
    for ik in range(grid.size):
        dpx = _projector_derivative_terms(
            grid, occupied, ik, inverse_basis[:, 0], transport_frame
        )
        dpy = _projector_derivative_terms(
            grid, occupied, ik, inverse_basis[:, 1], transport_frame
        )
        dqx = _projector_derivative_terms(
            grid, empty, ik, inverse_basis[:, 0], transport_frame
        )
        dqy = _projector_derivative_terms(
            grid, empty, ik, inverse_basis[:, 1], transport_frame
        )
        p = occupied[ik]
        q = empty[ik]
        hmu_p = h_occ[ik] - mu * p
        hmu_q = h_emp[ik] - mu * q

        w_xy[ik] = np.trace(p.conj().T @ _apply_terms(dpx, _apply_terms(dqy, hmu_p)))
        n_xy[ik] = np.trace(q.conj().T @ _apply_terms(dpx, _apply_terms(dqy, hmu_q)))

        # Differentiate the same finite-difference trace analytically with
        # respect to mu.  In the continuum this is twice the retained-PQ Berry
        # curvature; evaluating the discrete trace keeps the reported slope
        # exactly consistent with the reported magnetization.
        chemical_slope_integrand[ik] = -np.trace(
            p.conj().T @ _apply_terms(dpx, _apply_terms(dqy, p))
        ) + np.trace(q.conj().T @ _apply_terms(dpx, _apply_terms(dqy, q)))

    im_difference = float(np.mean(np.imag(w_xy - n_xy)))
    im_sum = float(np.mean(np.imag(w_xy + n_xy)))
    moment = im_difference / HBAR2_OVER_2ME_MEV_NM2
    # Generalized charge q gives m_SR=-(q/hbar) Im(W+N).  Here q=+e
    # because the primary computation is in the direct-hole convention.
    self_rotation = -im_sum / HBAR2_OVER_2ME_MEV_NM2
    reciprocal_area = abs(determinant)
    moire_area = (2.0 * np.pi) ** 2 / reciprocal_area
    streda_slope = float(
        np.mean(np.imag(chemical_slope_integrand)) / HBAR2_OVER_2ME_MEV_NM2
    )
    # Match the overlap orientation used by ``taige.chern_number_on_grid``:
    # dM/dmu_h = +C*A_M/(2*pi*E0).  Since mu_e=-mu_h, the physical electron
    # chemical-potential slope carries the opposite sign.
    streda_chern = float(
        streda_slope * (2.0 * np.pi * HBAR2_OVER_2ME_MEV_NM2) / moire_area
    )
    summary = OrbitalMagnetizationSummary(
        chemical_potential_hole_mev=mu,
        orbital_magnetization_mu_b_per_cell=moment,
        self_rotation_mu_b_per_cell=self_rotation,
        im_w_minus_n_mean_mev_nm2=im_difference,
        im_w_plus_n_mean_mev_nm2=im_sum,
        streda_slope_mu_b_per_mev=streda_slope,
        streda_chern_from_retained_pq=streda_chern,
        moire_cell_area_nm2=moire_area,
        max_occupied_orthonormality_error=occ_orth,
        max_empty_orthonormality_error=emp_orth,
        max_occupied_empty_overlap=cross,
    )
    return OrbitalMagnetizationEvaluation(summary=summary, w_xy_mev_nm2=w_xy, n_xy_mev_nm2=n_xy)


def _projector_derivative_terms(
    grid: MomentumGrid,
    frames: np.ndarray,
    index: int,
    fractional_coefficients: np.ndarray,
    transport: FrameTransport,
) -> tuple[tuple[float, np.ndarray], ...]:
    coord = grid.coord_of(index)
    terms: list[tuple[float, np.ndarray]] = []
    for axis, coefficient in enumerate(np.asarray(fractional_coefficients, dtype=float)):
        if abs(coefficient) < 1e-15:
            continue
        delta = (1, 0) if axis == 0 else (0, 1)
        minus_delta = (-delta[0], -delta[1])
        plus_coord, plus_shift = grid.shift_plus_q(coord, delta)
        minus_coord, minus_shift = grid.shift_plus_q(coord, minus_delta)
        central_weight = float(coefficient * grid.n_k / 2.0)
        plus = transport(frames[grid.index_of(plus_coord)], plus_shift)
        minus = transport(frames[grid.index_of(minus_coord)], minus_shift)
        terms.append((central_weight, np.asarray(plus, dtype=complex)))
        terms.append((-central_weight, np.asarray(minus, dtype=complex)))
    return tuple(terms)


def _apply_terms(terms: tuple[tuple[float, np.ndarray], ...], vectors: np.ndarray) -> np.ndarray:
    source = np.asarray(vectors, dtype=complex)
    out = np.zeros_like(source)
    for coefficient, frame in terms:
        out += coefficient * frame @ (frame.conj().T @ source)
    return out


def _identity_transport(frame: np.ndarray, _shift: tuple[int, int]) -> np.ndarray:
    return np.asarray(frame, dtype=complex)


def _validate_frames(name: str, value: np.ndarray, grid: MomentumGrid) -> np.ndarray:
    arr = np.asarray(value, dtype=complex)
    if arr.ndim != 3 or arr.shape[0] != grid.size or arr.shape[1] < 1 or arr.shape[2] < 1:
        raise ValueError(f"{name} must have shape (grid.size, microscopic_dim, rank)")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _validate_action(name: str, value: np.ndarray, frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=complex)
    if arr.shape != frame.shape:
        raise ValueError(f"{name} must have the same shape as its frame")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _frame_errors(occupied: np.ndarray, empty: np.ndarray) -> tuple[float, float, float]:
    occ_identity = np.eye(occupied.shape[2], dtype=complex)
    emp_identity = np.eye(empty.shape[2], dtype=complex)
    occ_gram = occupied.conj().swapaxes(1, 2) @ occupied
    emp_gram = empty.conj().swapaxes(1, 2) @ empty
    cross_gram = occupied.conj().swapaxes(1, 2) @ empty
    return (
        float(np.max(np.abs(occ_gram - occ_identity))),
        float(np.max(np.abs(emp_gram - emp_identity))),
        float(np.max(np.abs(cross_gram))),
    )
