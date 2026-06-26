import numpy as np
import pytest

from chiral_dw.config import DomainWallParams, UnitsParams
from chiral_dw.domain_wall import charge_density_radial, dtheta_dr, theta_profile
from chiral_dw.response import (
    compute_cG,
    flavor_tau_z,
    phi_derivative_projector,
    k_theta_from_projectors,
    k_theta_from_projectors_with_basis,
    projector_errors,
    projector_grid_from_theta,
    rotate_projector_phi,
    u1_rotation,
)


def _constant_projector_grid(n_theta: int = 5, n_k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, np.pi, n_theta)
    P0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=complex)
    P = np.broadcast_to(P0, (n_theta, n_k, n_k, 2, 2)).copy()
    return theta, P


def test_u1_rotation_preserves_projector_and_is_periodic():
    P = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
    rotated = rotate_projector_phi(P, 0.7)
    periodic = rotate_projector_phi(P, 2.0 * np.pi)

    assert projector_errors(rotated)["hermiticity"] < 1e-12
    assert projector_errors(rotated)["idempotency"] < 1e-12
    assert np.allclose(periodic, P, atol=1e-12)


def test_flavor_tau_z_and_u1_rotation_general_dim():
    phi = 0.37

    assert np.allclose(flavor_tau_z(4), np.diag([1.0, 1.0, -1.0, -1.0]))
    assert np.allclose(
        u1_rotation(phi, dim=4),
        np.diag(
            [
                np.exp(-0.5j * phi),
                np.exp(-0.5j * phi),
                np.exp(0.5j * phi),
                np.exp(0.5j * phi),
            ]
        ),
    )
    with pytest.raises(ValueError, match="positive even"):
        flavor_tau_z(3)


def test_phi_derivative_matches_finite_difference_for_general_dim():
    eps = 1e-6
    for dim in (2, 4, 6):
        rng = np.random.default_rng(dim)
        z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        q, _ = np.linalg.qr(z)
        occ = q[:, : max(1, dim // 3)]
        P = occ @ occ.conj().T

        finite_difference = (rotate_projector_phi(P, eps) - rotate_projector_phi(P, -eps)) / (2.0 * eps)
        assert np.allclose(phi_derivative_projector(P), finite_difference, atol=1e-8)


def test_projector_grid_from_theta_preserves_projectors():
    theta, P = _constant_projector_grid()
    grid = projector_grid_from_theta(P, np.array([0.0, 0.3, 2.0 * np.pi]))

    assert grid.shape == (len(theta), 3, 4, 4, 2, 2)
    assert projector_errors(grid)["idempotency"] < 1e-12
    assert np.allclose(grid[:, 0], grid[:, 2], atol=1e-12)


def test_projector_grid_from_theta_accepts_dim4():
    n_theta = 5
    n_k = 3
    dim = 4
    theta = np.linspace(0.0, np.pi, n_theta)
    P = np.zeros((n_theta, n_k, n_k, dim, dim), dtype=complex)
    for it, th in enumerate(theta):
        spinor = np.array([np.cos(0.5 * th), 0.0, np.sin(0.5 * th), 0.0], dtype=complex)
        P[it, :, :] = spinor[:, None] * spinor.conj()[None, :]

    grid = projector_grid_from_theta(P, np.array([0.0, 0.4]))

    assert grid.shape == (n_theta, 2, n_k, n_k, dim, dim)
    assert projector_errors(grid)["hermiticity"] < 1e-12
    assert projector_errors(grid)["idempotency"] < 1e-12


def test_trivial_projector_has_zero_response():
    theta, P = _constant_projector_grid(n_theta=7, n_k=5)
    result = k_theta_from_projectors(P, theta)

    assert np.allclose(result.K, 0.0, atol=1e-12)
    assert abs(result.cG) < 1e-12


def test_embedded_response_matches_active_response_for_constant_basis():
    n_theta = 7
    n_k = 4
    theta = np.linspace(0.1, np.pi - 0.1, n_theta)
    P = np.zeros((n_theta, n_k, n_k, 2, 2), dtype=complex)
    for it, th in enumerate(theta):
        for i in range(n_k):
            for j in range(n_k):
                phase = 2.0 * np.pi * (i + 2 * j) / n_k
                spinor = np.array(
                    [np.cos(0.5 * th), np.exp(1j * phase) * np.sin(0.5 * th)],
                    dtype=complex,
                )
                P[it, i, j] = spinor[:, None] * spinor.conj()[None, :]

    basis = np.broadcast_to(np.eye(2, dtype=complex), (n_k, n_k, 2, 2)).copy()
    active = k_theta_from_projectors(P, theta)
    embedded = k_theta_from_projectors_with_basis(P, theta, basis)

    assert np.allclose(embedded.K, active.K, atol=1e-12)
    assert embedded.cG == pytest.approx(active.cG, abs=1e-12)


def test_embedded_response_matches_active_response_for_constant_basis_dim4():
    n_theta = 7
    n_k = 4
    theta = np.linspace(0.1, np.pi - 0.1, n_theta)
    P = np.zeros((n_theta, n_k, n_k, 4, 4), dtype=complex)
    for it, th in enumerate(theta):
        for i in range(n_k):
            for j in range(n_k):
                phase = 2.0 * np.pi * (i + 2 * j) / n_k
                spinor = np.array(
                    [np.cos(0.5 * th), 0.0, np.exp(1j * phase) * np.sin(0.5 * th), 0.0],
                    dtype=complex,
                )
                P[it, i, j] = spinor[:, None] * spinor.conj()[None, :]

    basis = np.broadcast_to(np.eye(4, dtype=complex), (n_k, n_k, 4, 4)).copy()
    active = k_theta_from_projectors(P, theta)
    embedded = k_theta_from_projectors_with_basis(P, theta, basis)

    assert np.allclose(embedded.K, active.K, atol=1e-12)
    assert embedded.cG == pytest.approx(active.cG, abs=1e-12)


def test_odd_k_theta_input_has_finite_cg_and_midpoint_zero():
    theta = np.linspace(0.0, np.pi, 101)
    K = np.cos(theta)
    cG = compute_cG(theta, K)

    assert np.isfinite(cG)
    assert np.isclose(np.interp(0.5 * np.pi, theta, K), 0.0, atol=1e-12)
    assert np.max(np.abs(K + K[::-1])) < 1e-12


def test_domain_wall_charge_profile_scales_with_winding_and_units():
    theta = np.linspace(0.0, np.pi, 101)
    K = np.cos(theta)
    r = np.linspace(1.0, 20.0, 200)
    params = DomainWallParams(radius=10.0, width=2.0, winding=1)
    profile = charge_density_radial(r, theta, K, params)
    opposite = charge_density_radial(r, theta, K, params.model_copy(update={"winding": -1}))

    assert np.allclose(theta_profile(r, params), profile.theta)
    assert np.allclose(dtheta_dr(r, params), 1.0 / (params.width * np.cosh((r - params.radius) / params.width)))
    assert np.allclose(opposite.rho_dimless, -profile.rho_dimless)
    assert np.allclose(profile.rho_physical(UnitsParams(a_m=2.0)), 0.25 * profile.rho_dimless)
    assert np.max(np.abs(profile.rho_dimless)) > 0.0
