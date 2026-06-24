import numpy as np

from chiral_dw.config import DomainWallParams, UnitsParams
from chiral_dw.domain_wall import charge_density_radial, dtheta_dr, theta_profile
from chiral_dw.response import (
    compute_cG,
    k_theta_from_projectors,
    projector_errors,
    projector_grid_from_theta,
    rotate_projector_phi,
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


def test_projector_grid_from_theta_preserves_projectors():
    theta, P = _constant_projector_grid()
    grid = projector_grid_from_theta(P, np.array([0.0, 0.3, 2.0 * np.pi]))

    assert grid.shape == (len(theta), 3, 4, 4, 2, 2)
    assert projector_errors(grid)["idempotency"] < 1e-12
    assert np.allclose(grid[:, 0], grid[:, 2], atol=1e-12)


def test_trivial_projector_has_zero_response():
    theta, P = _constant_projector_grid(n_theta=7, n_k=5)
    result = k_theta_from_projectors(P, theta)

    assert np.allclose(result.K, 0.0, atol=1e-12)
    assert abs(result.cG) < 1e-12


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
