from types import SimpleNamespace

import numpy as np
import pytest

from chiral_dw.config import (
    ContinuumGridParams,
    ContinuumHFParams,
    ContinuumInteractionParams,
    ContinuumModelParams,
)
from chiral_dw.continuum import (
    TPrimeConstraint,
    ValleyU1Constraint,
    build_continuum_bundle,
    build_symmetric_hf_references,
    convex_weights,
    mesh_inversion_map,
    reference_diagnostics,
    rotate_valley_u1,
    solve_reference_hf,
    symmetric_convex_hamiltonian,
    symmetric_convex_path,
    symmetric_convex_projector,
)
from chiral_dw.continuum.models import SymmetricHFReferences, hermitize
from chiral_dw.continuum.seeds import valley_polarized_seed
from chiral_dw.response import compute_cG, k_theta_from_projectors


def _small_bundle():
    return build_continuum_bundle(
        model=ContinuumModelParams(displacement_mev=0.0),
        grid=ContinuumGridParams(n_k=3),
        interaction=ContinuumInteractionParams(v0=0.2, q_shell=0, gate_distance=1.0),
    )


def test_valley_u1_constraint_projects_intervalley_blocks_and_preserves_vp_seed():
    bundle = _small_bundle()
    active = bundle.active
    constraint = ValleyU1Constraint(active)
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(active.n_k, active.dim, active.dim)) + 1j * rng.normal(
        size=(active.n_k, active.dim, active.dim)
    )
    blocks = hermitize(raw)
    projected = constraint.project_density(blocks)
    n = active.n_active

    assert np.allclose(projected[:, :n, n:], 0.0)
    assert np.allclose(projected[:, n:, :n], 0.0)
    assert np.allclose(projected, projected.conj().swapaxes(-1, -2))
    assert np.allclose(np.trace(projected, axis1=-2, axis2=-1), np.trace(blocks, axis1=-2, axis2=-1))
    assert np.allclose(constraint.project_density(projected), projected)

    vp = valley_polarized_seed(active, "K")
    assert np.allclose(constraint.project_density(vp), vp)


def test_tprime_constraint_is_involutive_and_final_aufbau_is_idempotent():
    bundle = _small_bundle()
    active = bundle.active
    constraint = TPrimeConstraint(active)
    partner = mesh_inversion_map(active.grid)

    assert np.all(partner[partner] == np.arange(active.n_k))

    rng = np.random.default_rng(3)
    raw = rng.normal(size=(active.n_k, active.dim, active.dim)) + 1j * rng.normal(
        size=(active.n_k, active.dim, active.dim)
    )
    blocks = hermitize(raw)
    assert np.allclose(constraint.transform(constraint.transform(blocks)), blocks)

    projected = constraint.project_operator(blocks)
    assert constraint.symmetry_error(projected) < 1e-12

    P, _evals, _direct, _indirect = constraint.update_density(projected, 1)
    assert np.linalg.norm(P @ P - P) < 1e-12
    assert constraint.symmetry_error(P) < 1e-12


def test_hf_solver_reports_idempotent_final_projectors_for_reference_states():
    bundle = _small_bundle()
    params = ContinuumHFParams(max_iter=6, min_iter=1, mixing=0.6, tolerance=1e-9)

    vp_plus = solve_reference_hf(
        bundle,
        "vp_plus",
        params,
        constraint=ValleyU1Constraint(bundle.active),
    )
    vp_minus = solve_reference_hf(
        bundle,
        "vp_minus",
        params,
        constraint=ValleyU1Constraint(bundle.active),
    )
    ivc = solve_reference_hf(
        bundle,
        "ivc",
        params,
        constraint=TPrimeConstraint(bundle.active),
    )

    for result in (vp_plus, vp_minus, ivc):
        assert result.diagnostics.density_kind == "final_idempotent"
        assert result.diagnostics.idempotency_error_fro < 1e-8
        assert result.diagnostics.idempotency_error_max < 1e-8
        assert result.diagnostics.trace_error < 1e-8
        assert np.isfinite(result.diagnostics.aufbau_residual_norm)

    assert np.real(np.trace(vp_plus.P[:, 0:1, 0:1], axis1=-2, axis2=-1).sum()) > 0.9 * bundle.active.n_k
    assert np.real(np.trace(vp_minus.P[:, 1:2, 1:2], axis1=-2, axis2=-1).sum()) > 0.9 * bundle.active.n_k
    assert ivc.diagnostics.constraint_name == "tprime"


def _toy_refs_with_scalar_terms(n_blocks: int = 4) -> SymmetricHFReferences:
    h0 = np.zeros((n_blocks, 2, 2), dtype=complex)
    scalar = np.broadcast_to(3.0 * np.eye(2, dtype=complex), h0.shape).copy()
    H_plus = scalar - np.broadcast_to(np.diag([1.0, -1.0]).astype(complex), h0.shape)
    H_minus = scalar + np.broadcast_to(np.diag([1.0, -1.0]).astype(complex), h0.shape)
    H_ivc = scalar - np.broadcast_to(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex), h0.shape)

    def result(H):
        return SimpleNamespace(H_hf=H, P=np.zeros_like(H), diagnostics=None)

    return SymmetricHFReferences(
        vp_plus=result(H_plus),
        vp_minus=result(H_minus),
        ivc=result(H_ivc),
        n_occ_per_k=1,
    )


def test_symmetric_convex_path_preserves_scalar_terms_and_rotates_ivc():
    refs = _toy_refs_with_scalar_terms()
    weights = np.asarray([convex_weights(x) for x in np.linspace(0.0, np.pi, 7)])

    assert np.all(weights >= -1e-15)
    assert np.allclose(np.sum(weights, axis=1), 1.0)
    assert convex_weights(0.0) == pytest.approx((1.0, 0.0, 0.0))
    assert convex_weights(0.5 * np.pi) == pytest.approx((0.0, 0.0, 1.0))
    assert convex_weights(np.pi) == pytest.approx((0.0, 1.0, 0.0))

    H0 = symmetric_convex_hamiltonian(refs, 0.0)
    Hpi = symmetric_convex_hamiltonian(refs, np.pi)
    Hmid = symmetric_convex_hamiltonian(refs, 0.5 * np.pi)
    assert np.allclose(np.trace(H0, axis1=-2, axis2=-1), 6.0)
    assert np.allclose(np.trace(Hpi, axis1=-2, axis2=-1), 6.0)
    assert np.allclose(np.trace(Hmid, axis1=-2, axis2=-1), 6.0)
    assert np.allclose(H0 - 3.0 * np.eye(2), -(Hpi - 3.0 * np.eye(2)))

    phi = np.pi / 4.0
    P_phi, H_phi, diag = symmetric_convex_projector(refs, 0.5 * np.pi, phi)
    P_zero, _H_zero, _diag_zero = symmetric_convex_projector(refs, 0.5 * np.pi, 0.0)
    assert np.allclose(P_phi, rotate_valley_u1(P_zero, phi), atol=1e-12)
    assert np.allclose(H_phi, rotate_valley_u1(Hmid, phi), atol=1e-12)
    assert diag.projector_idempotency_error_fro < 1e-12


def test_tiny_end_to_end_continuum_response_smoke():
    bundle = _small_bundle()
    params = ContinuumHFParams(max_iter=5, min_iter=1, mixing=0.7)
    refs = build_symmetric_hf_references(bundle, params)
    theta = np.linspace(0.0, np.pi, 5)
    projectors, diagnostics = symmetric_convex_path(refs, theta)
    response_projectors = projectors.reshape(5, 3, 3, 2, 2)
    response = k_theta_from_projectors(response_projectors, theta)

    assert projectors.shape == (5, bundle.active.n_k, 2, 2)
    assert len(diagnostics) == 5
    assert np.all(np.isfinite(response.K))
    assert np.isfinite(compute_cG(response.theta, response.K))
    assert refs.vp_plus.diagnostics.idempotency_error_fro < 1e-8
    assert refs.vp_minus.diagnostics.idempotency_error_fro < 1e-8
    assert refs.ivc.diagnostics.idempotency_error_fro < 1e-8

    ref_diag = reference_diagnostics(refs)
    assert ref_diag["ivc"].intervalley_norm >= 0.0
