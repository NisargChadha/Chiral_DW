from __future__ import annotations

import numpy as np
import pytest

from chiral_dw.continuum.models import MomentumGrid
from chiral_dw.continuum.orbital_magnetization import (
    HBAR2_OVER_2ME_MEV_NM2,
    evaluate_projector_orbital_magnetization,
)


def _qwz_data(n_k: int = 24, mass: float = -1.0) -> tuple:
    grid = MomentumGrid(n_k)
    occupied = np.zeros((grid.size, 2, 1), dtype=complex)
    empty = np.zeros_like(occupied)
    h_occ = np.zeros_like(occupied)
    h_emp = np.zeros_like(empty)
    hamiltonians = np.zeros((grid.size, 2, 2), dtype=complex)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.diag([1.0, -1.0]).astype(complex)
    for ik in range(grid.size):
        i, j = grid.coord_of(ik)
        kx = 2.0 * np.pi * i / n_k
        ky = 2.0 * np.pi * j / n_k
        h = (
            np.sin(kx) * sigma_x
            + np.sin(ky) * sigma_y
            + (mass + np.cos(kx) + np.cos(ky)) * sigma_z
        )
        energies, vectors = np.linalg.eigh(h)
        occupied[ik] = vectors[:, :1]
        empty[ik] = vectors[:, 1:]
        h_occ[ik] = h @ occupied[ik]
        h_emp[ik] = h @ empty[ik]
        hamiltonians[ik] = h
    reciprocal_basis = 2.0 * np.pi * np.eye(2)
    return grid, occupied, empty, h_occ, h_emp, hamiltonians, reciprocal_basis


def _evaluate(data: tuple, mu: float = 0.0):
    grid, occupied, empty, h_occ, h_emp, _hamiltonians, basis = data
    return evaluate_projector_orbital_magnetization(
        grid=grid,
        occupied_frames=occupied,
        empty_frames=empty,
        hamiltonian_on_occupied=h_occ,
        hamiltonian_on_empty=h_emp,
        reciprocal_basis_nm_inv=basis,
        chemical_potential_hole_mev=mu,
    )


def _raw_fukui_chern(grid: MomentumGrid, frame: np.ndarray) -> float:
    total = 0.0
    for i in range(grid.n_k):
        for j in range(grid.n_k):
            u00 = frame[grid.index_of((i, j)), :, 0]
            u10 = frame[grid.index_of((i + 1, j)), :, 0]
            u11 = frame[grid.index_of((i + 1, j + 1)), :, 0]
            u01 = frame[grid.index_of((i, j + 1)), :, 0]
            overlaps = (
                np.vdot(u00, u10),
                np.vdot(u10, u11),
                np.vdot(u11, u01),
                np.vdot(u01, u00),
            )
            total += np.angle(np.prod([z / abs(z) for z in overlaps]))
    return float(total / (2.0 * np.pi))


def _dense_projector_trace(data: tuple, mu: float) -> tuple[np.ndarray, np.ndarray]:
    grid, occupied, empty, _h_occ, _h_emp, hamiltonians, basis = data
    inverse_basis = np.linalg.inv(basis)
    w_xy = np.zeros(grid.size, dtype=complex)
    n_xy = np.zeros(grid.size, dtype=complex)
    projectors = occupied @ occupied.conj().swapaxes(1, 2)
    complements = empty @ empty.conj().swapaxes(1, 2)
    for ik in range(grid.size):
        coord = grid.coord_of(ik)
        derivatives: list[tuple[np.ndarray, np.ndarray]] = []
        for cartesian_axis in range(2):
            dp = np.zeros((2, 2), dtype=complex)
            dq = np.zeros((2, 2), dtype=complex)
            for fractional_axis in range(2):
                delta = (1, 0) if fractional_axis == 0 else (0, 1)
                plus, _ = grid.shift_plus_q(coord, delta)
                minus, _ = grid.shift_plus_q(coord, (-delta[0], -delta[1]))
                coefficient = inverse_basis[fractional_axis, cartesian_axis] * grid.n_k / 2
                dp += coefficient * (
                    projectors[grid.index_of(plus)] - projectors[grid.index_of(minus)]
                )
                dq += coefficient * (
                    complements[grid.index_of(plus)] - complements[grid.index_of(minus)]
                )
            derivatives.append((dp, dq))
        p = projectors[ik]
        q = complements[ik]
        hmu = hamiltonians[ik] - mu * np.eye(2)
        w_xy[ik] = np.trace(p @ derivatives[0][0] @ derivatives[1][1] @ hmu)
        n_xy[ik] = np.trace(q @ derivatives[0][0] @ derivatives[1][1] @ hmu)
    return w_xy, n_xy


def test_low_rank_projector_trace_matches_explicit_dense_trace() -> None:
    data = _qwz_data(n_k=10)
    result = _evaluate(data, mu=0.37)
    dense_w, dense_n = _dense_projector_trace(data, mu=0.37)
    assert np.allclose(result.w_xy_mev_nm2, dense_w, atol=2e-14)
    assert np.allclose(result.n_xy_mev_nm2, dense_n, atol=2e-14)


def test_particle_hole_symmetric_qwz_has_zero_midgap_magnetization() -> None:
    result = _evaluate(_qwz_data(n_k=18), mu=0.0)
    assert abs(result.summary.orbital_magnetization_mu_b_per_cell) < 2e-16


def test_streda_slope_matches_existing_fukui_chern_orientation() -> None:
    data = _qwz_data(n_k=36)
    minus = _evaluate(data, mu=-0.1).summary
    plus = _evaluate(data, mu=0.1).summary
    slope = (
        plus.orbital_magnetization_mu_b_per_cell
        - minus.orbital_magnetization_mu_b_per_cell
    ) / 0.2
    grid, occupied, *_rest = data
    chern = _raw_fukui_chern(grid, occupied)
    expected = chern / (2.0 * np.pi * HBAR2_OVER_2ME_MEV_NM2)
    assert slope == pytest.approx(expected, rel=0.02)
    assert plus.streda_chern_from_retained_pq == pytest.approx(chern, rel=0.02)


def test_energy_origin_shift_and_independent_frame_phases_do_not_change_result() -> None:
    data = _qwz_data(n_k=12)
    baseline = _evaluate(data, mu=0.23).summary
    grid, occupied, empty, h_occ, h_emp, hamiltonians, basis = data
    shift = 4.7
    shifted = evaluate_projector_orbital_magnetization(
        grid=grid,
        occupied_frames=occupied,
        empty_frames=empty,
        hamiltonian_on_occupied=h_occ + shift * occupied,
        hamiltonian_on_empty=h_emp + shift * empty,
        reciprocal_basis_nm_inv=basis,
        chemical_potential_hole_mev=0.23 + shift,
    ).summary
    assert shifted.orbital_magnetization_mu_b_per_cell == pytest.approx(
        baseline.orbital_magnetization_mu_b_per_cell, abs=2e-15
    )

    rng = np.random.default_rng(5)
    occ_phase = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(grid.size, 1, 1)))
    emp_phase = np.exp(1j * rng.uniform(0, 2 * np.pi, size=(grid.size, 1, 1)))
    gauged = evaluate_projector_orbital_magnetization(
        grid=grid,
        occupied_frames=occupied * occ_phase,
        empty_frames=empty * emp_phase,
        hamiltonian_on_occupied=h_occ * occ_phase,
        hamiltonian_on_empty=h_emp * emp_phase,
        reciprocal_basis_nm_inv=basis,
        chemical_potential_hole_mev=0.23,
    ).summary
    assert gauged.orbital_magnetization_mu_b_per_cell == pytest.approx(
        baseline.orbital_magnetization_mu_b_per_cell, abs=2e-15
    )


def test_retained_empty_projector_is_not_replaced_by_full_complement() -> None:
    data = _qwz_data(n_k=10)
    grid, occupied_2d, empty_2d, h_occ_2d, h_emp_2d, _h, basis = data
    occupied = np.pad(occupied_2d, ((0, 0), (0, 1), (0, 0)))
    empty = np.pad(empty_2d, ((0, 0), (0, 1), (0, 0)))
    h_occ = np.pad(h_occ_2d, ((0, 0), (0, 1), (0, 0)))
    h_emp = np.pad(h_emp_2d, ((0, 0), (0, 1), (0, 0)))
    result = evaluate_projector_orbital_magnetization(
        grid=grid,
        occupied_frames=occupied,
        empty_frames=empty,
        hamiltonian_on_occupied=h_occ,
        hamiltonian_on_empty=h_emp,
        reciprocal_basis_nm_inv=basis,
        chemical_potential_hole_mev=0.2,
    )
    assert np.isfinite(result.summary.orbital_magnetization_mu_b_per_cell)
    assert occupied.shape[2] + empty.shape[2] < occupied.shape[1]


def test_invalid_nonorthogonal_frames_are_rejected() -> None:
    data = list(_qwz_data(n_k=4))
    data[2] = data[1].copy()
    with pytest.raises(ValueError, match="mutually orthogonal"):
        _evaluate(tuple(data))
