import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from chiral_dw.ac.kahler import IdealACKahlerModel
from chiral_dw.ac.nonideal import NonIdealACLLModel
from chiral_dw.ac.projected import (
    build_ac_active_space,
    build_ac_density_vertices,
    build_ac_projected_bundle,
)
from chiral_dw.ac.response import (
    ACBandOverlapProvider,
    ac_projector_chern,
    ac_reference_cherns_are_valid,
    k_theta_from_ac_projectors,
)
from chiral_dw.config import (
    ACProjectedHFParams,
    ContinuumGridParams,
    ContinuumHFParams,
    ContinuumInteractionParams,
    FirstShellACParams,
    ResponseParams,
)
from chiral_dw.continuum import (
    ContinuumHFBackend,
    MomentumGrid,
    TPrimeConstraint,
    ValleyU1Constraint,
    active_basis_frames,
    build_symmetric_hf_references,
    symmetric_convex_path,
)
from chiral_dw.response import k_theta_from_projectors_with_basis


def _small_params() -> ACProjectedHFParams:
    return ACProjectedHFParams(
        grid=ContinuumGridParams(n_k=3),
        ac=FirstShellACParams(b1=0.12, u1=0.04, n_ll=3),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_dual_gate",
            v0=0.03,
            q_shell=1,
            local_field_cutoff=1,
            gate_distance=1.5,
        ),
        hf=ContinuumHFParams(max_iter=4, min_iter=1, mixing=0.6, tolerance=1e-9),
        band_diagnostics_n_k=3,
    )


def _ideal_lll_params(n_k: int = 5, n_theta: int = 20) -> ACProjectedHFParams:
    return ACProjectedHFParams(
        grid=ContinuumGridParams(n_k=n_k),
        ac=FirstShellACParams(b1=0.0, u1=0.0, n_ll=1),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_dual_gate",
            v0=0.2,
            q_shell=1,
            local_field_cutoff=1,
            gate_distance=2.0,
        ),
        hf=ContinuumHFParams(
            max_iter=220,
            min_iter=2,
            mixing_method="oda",
            mixing=0.45,
            tolerance=1e-8,
            final_residual_tolerance=1e-7,
        ),
        band_diagnostics_n_k=max(5, n_k),
        response=ResponseParams(n_theta=n_theta),
    )


def _uniform_vp_ivc_path(theta_edges: np.ndarray, n_k: int) -> np.ndarray:
    projectors = np.zeros((len(theta_edges), n_k, n_k, 2, 2), dtype=complex)
    for it, theta in enumerate(theta_edges):
        spinor = np.array([np.cos(0.5 * theta), np.sin(0.5 * theta)], dtype=complex)
        projectors[it, :, :] = spinor[:, None] * spinor.conj()[None, :]
    return projectors


def test_kahler_chi_solves_periodic_poisson_equation():
    ll_model = NonIdealACLLModel(FirstShellACParams(b1=0.15, u1=0.0, n_ll=3))
    G, b_coeff, _A = ll_model.vector_potential_coefficients()
    kahler = IdealACKahlerModel(ll_model.fields)
    sol = kahler.solve_chi_from_fourier(G, b_coeff, n_grid=32)

    G2 = np.sum(sol.g_vectors**2, axis=-1)
    lap_coeff = -G2 * sol.chi_coeffs
    nonzero = G2 > 1e-14

    assert abs(float(np.mean(sol.chi))) < 1e-12
    assert np.allclose(lap_coeff[nonzero], sol.b_coeffs[nonzero], atol=1e-11)
    assert abs(sol.chi_coeffs[0, 0]) < 1e-12


def test_kahler_uniform_limit_has_unit_dimensionless_berry_curvature():
    ll_model = NonIdealACLLModel(FirstShellACParams(b1=0.0, u1=0.0, n_ll=2))
    kahler = IdealACKahlerModel(ll_model.fields)
    sol = kahler.solve_chi_from_fourier(np.empty((0, 2)), np.empty(0), n_grid=16)
    G, coeffs, _phi = kahler.exp2chi_fourier_coeffs(sol)
    b1, b2 = ll_model.fields.G_shell[0], ll_model.fields.G_shell[1]
    pts = np.linspace(0.0, 1.0, 5, endpoint=False)
    uu, vv = np.meshgrid(pts, pts, indexing="ij")
    k = uu[..., None] * b1 + vv[..., None] * b2
    omega = kahler.dimensionless_berry_curvature(k, G, coeffs)

    assert np.allclose(omega, 1.0, atol=1e-12)


def test_ac_projected_bundle_shapes_tprime_and_density_identities():
    bundle = build_ac_projected_bundle(_small_params())
    active = bundle.active
    vertices = bundle.vertices

    assert active.h0.shape == (9, 2, 2)
    assert active.band_vectors.shape == (9, 2, 3, 1)
    assert vertices.lambda_blocks.shape[:3] == (9, 9, 9)
    assert active_basis_frames(active).shape == (9, 6, 2)
    assert TPrimeConstraint(active).symmetry_error(active.h0) < 1e-12
    assert ValleyU1Constraint(active).symmetry_error(vertices.lambda_blocks[0, 0]) < 1e-12

    q0 = vertices.q_shifts.index((0, 0))
    g0 = vertices.g_channels.index((0, 0))
    q0_lambda = vertices.lambda_blocks[q0, g0]
    assert np.allclose(q0_lambda[:, 0, 0], 1.0, atol=1e-12)
    assert np.allclose(q0_lambda[:, 1, 1], 1.0, atol=1e-12)
    assert np.allclose(q0_lambda[:, 0, 1], 0.0, atol=1e-12)
    assert np.allclose(q0_lambda[:, 1, 0], 0.0, atol=1e-12)

    q_plus = vertices.q_shifts.index((1, 0))
    q_minus = vertices.q_shifts.index((-1, 0))
    for ik in range(active.n_k):
        jk = int(vertices.target_minus_q[q_plus, ik])
        lhs = vertices.lambda_blocks[q_plus, g0, ik]
        rhs = vertices.lambda_blocks[q_minus, g0, jk].conj().T
        assert np.allclose(lhs, rhs, atol=1e-10)


def test_explicit_active_space_and_vertex_path_matches_bundle_builder():
    params = _small_params()
    model = NonIdealACLLModel(params.ac)
    grid = MomentumGrid(params.grid.n_k)
    active, _bands = build_ac_active_space(
        model,
        grid,
        active_band=params.active_band,
        diagnostics_n_k=params.band_diagnostics_n_k,
    )
    vertices = build_ac_density_vertices(
        model,
        active,
        params.interaction,
        moire_length_nm=params.moire_length_nm,
        energy_unit_mev=params.energy_unit_mev,
    )
    backend = ContinuumHFBackend(active.h0, vertices, params.interaction)
    bundle = build_ac_projected_bundle(params)

    assert np.allclose(active.h0, bundle.active.h0)
    assert np.allclose(vertices.lambda_blocks, bundle.vertices.lambda_blocks)
    assert np.allclose(backend.h0, bundle.backend.h0)


def test_dimensionless_dual_gate_weights_include_correct_q0_limit():
    params = _small_params()
    bundle = build_ac_projected_bundle(params)
    vertices = bundle.vertices
    q0 = vertices.q_shifts.index((0, 0))
    g0 = vertices.g_channels.index((0, 0))
    q1 = vertices.q_shifts.index((1, 0))

    v0 = params.interaction.v0
    d = params.interaction.gate_distance
    assert vertices.v_q is not None
    assert np.isclose(vertices.v_q[q0, g0], 2.0 * np.pi * v0 * d)

    q_norm = np.linalg.norm(
        bundle.form_factors.fields.G_shell[0] / params.grid.n_k
    )
    expected = 2.0 * np.pi * v0 * np.tanh(q_norm * d) / q_norm
    assert np.isclose(vertices.v_q[q1, g0], expected)

    without_q0 = params.model_copy(
        update={
            "interaction": params.interaction.model_copy(update={"include_q0": False})
        }
    )
    no_q0_vertices = build_ac_projected_bundle(without_q0).vertices
    assert no_q0_vertices.v_q is not None
    assert no_q0_vertices.v_q[q0, g0] == 0.0


def test_ac_overlap_provider_has_opposite_cherns_and_tprime_overlaps():
    model = NonIdealACLLModel(FirstShellACParams(b1=0.0, u1=0.0, n_ll=1))
    provider = ACBandOverlapProvider(model)
    up_chern, down_chern = provider.band_cherns(n_k=5)
    k = 0.13 * model.fields.G_shell[0] + 0.21 * model.fields.G_shell[1]
    p = -0.17 * model.fields.G_shell[0] + 0.08 * model.fields.G_shell[1]
    boundary = provider.up_overlap(k, k + model.fields.G_shell[0])

    assert np.isclose(up_chern, 1.0, atol=5e-3)
    assert np.isclose(down_chern, -1.0, atol=5e-3)
    assert np.allclose(provider.down_overlap(k, p), np.conj(provider.up_overlap(-k, -p)))
    assert abs(boundary) > 1e-12


def test_active_frame_overlap_provider_removes_only_eigensolver_phase(monkeypatch):
    params = _small_params()
    model = NonIdealACLLModel(params.ac)
    grid = MomentumGrid(params.grid.n_k)
    active, _bands = build_ac_active_space(
        model,
        grid,
        active_band=params.active_band,
        diagnostics_n_k=params.band_diagnostics_n_k,
    )
    b1 = model.fields.G_shell[0]
    wrapped = b1 / grid.n_k
    shifted = wrapped + b1
    baseline = ACBandOverlapProvider(model, active=active).up_overlap(wrapped, shifted)
    original_solve = model.solve

    def phase_flipped_solve(k, active_band=0):
        solution = original_solve(k, active_band=active_band)
        if not np.allclose(k, shifted, atol=1e-12, rtol=0.0):
            return solution
        eigenvectors = solution.eigenvectors.copy()
        eigenvectors[:, int(active_band)] *= -1j
        return replace(solution, eigenvectors=eigenvectors)

    monkeypatch.setattr(model, "solve", phase_flipped_solve)
    anchored = ACBandOverlapProvider(model, active=active)
    unanchored = ACBandOverlapProvider(model)

    expected_wrapped = active.band_vectors[grid.index_of((1, 0)), 0, :, 0]
    assert np.allclose(anchored.up_coefficients(wrapped), expected_wrapped)
    assert np.allclose(anchored.up_overlap(wrapped, shifted), baseline, atol=1e-12)
    assert np.allclose(unanchored.up_overlap(wrapped, shifted), -1j * baseline, atol=1e-12)


def test_reference_chern_validation_requires_symmetry_values():
    assert ac_reference_cherns_are_valid(
        {"vp_plus": 1.0, "vp_minus": -1.0, "ivc": 0.0}
    )
    assert not ac_reference_cherns_are_valid(
        {"vp_plus": 1.0, "vp_minus": -1.0, "ivc": 0.5}
    )
    assert not ac_reference_cherns_are_valid({"vp_plus": 1.0, "vp_minus": -1.0})


def test_ideal_lll_hf_reference_cherns_use_ac_overlaps():
    params = _ideal_lll_params(n_k=5, n_theta=12)
    bundle = build_ac_projected_bundle(params)
    provider = ACBandOverlapProvider(bundle.form_factors, active=bundle.active)
    refs = build_symmetric_hf_references(bundle, params.hf)

    assert refs.vp_plus.converged
    assert refs.vp_minus.converged
    assert refs.ivc.converged
    assert np.isclose(ac_projector_chern(provider, bundle.grid, refs.vp_plus.P), 1.0, atol=5e-3)
    assert np.isclose(ac_projector_chern(provider, bundle.grid, refs.vp_minus.P), -1.0, atol=5e-3)
    assert abs(ac_projector_chern(provider, bundle.grid, refs.ivc.P)) < 5e-3


def test_ideal_lll_ac_response_is_nonzero_and_coefficient_response_is_zero():
    params = _ideal_lll_params(n_k=5, n_theta=20)
    bundle = build_ac_projected_bundle(params)
    provider = ACBandOverlapProvider(bundle.form_factors, active=bundle.active)
    theta_edges = np.linspace(0.0, np.pi, params.response.n_theta + 1)
    phi_nodes = np.arange(2, dtype=float) * params.response.phi_step
    projectors = _uniform_vp_ivc_path(theta_edges, params.grid.n_k)

    response = k_theta_from_ac_projectors(provider, projectors, theta_edges, phi_nodes)
    frames = active_basis_frames(bundle.active).reshape(
        params.grid.n_k,
        params.grid.n_k,
        -1,
        bundle.active.dim,
    )
    coefficient_response = k_theta_from_projectors_with_basis(projectors, theta_edges, frames)

    assert abs(response.cG + 1.0 / (4.0 * np.pi)) < 1e-2
    assert abs(coefficient_response.cG) < 1e-12
    assert np.allclose(response.K + response.K[::-1], 0.0, atol=1e-10)


def test_ac_projected_bundle_runs_symmetric_hf_and_embedded_response():
    params = _small_params()
    bundle = build_ac_projected_bundle(params)
    refs = build_symmetric_hf_references(bundle, params.hf)

    assert refs.vp_plus.constraint_name == "valley_u1"
    assert refs.vp_minus.constraint_name == "valley_u1"
    assert refs.ivc.constraint_name == "tprime"

    theta = np.linspace(1e-4, np.pi - 1e-4, 5)
    projectors, _diag = symmetric_convex_path(refs, theta)
    n_k = bundle.grid.n_k
    frames = active_basis_frames(bundle.active).reshape(n_k, n_k, -1, bundle.active.dim)
    response = k_theta_from_projectors_with_basis(
        projectors.reshape(len(theta), n_k, n_k, bundle.active.dim, bundle.active.dim),
        theta,
        frames,
    )

    assert np.all(np.isfinite(response.K))
    assert np.isfinite(response.cG)


def test_ac_projected_bundle_runs_overlap_response_smoke():
    params = _ideal_lll_params(n_k=3, n_theta=8)
    bundle = build_ac_projected_bundle(params)
    refs = build_symmetric_hf_references(bundle, params.hf)
    provider = ACBandOverlapProvider(bundle.form_factors, active=bundle.active)
    theta_edges = np.linspace(0.0, np.pi, params.response.n_theta + 1)
    projectors, _diag = symmetric_convex_path(refs, theta_edges)
    projector_grid = projectors.reshape(
        len(theta_edges),
        params.grid.n_k,
        params.grid.n_k,
        bundle.active.dim,
        bundle.active.dim,
    )
    phi_nodes = np.arange(2, dtype=float) * params.response.phi_step
    response = k_theta_from_ac_projectors(provider, projector_grid, theta_edges, phi_nodes)

    assert refs.vp_plus.converged
    assert refs.vp_minus.converged
    assert refs.ivc.converged
    assert np.all(np.isfinite(response.K))
    assert np.isfinite(response.cG)
    assert abs(response.cG) > 1e-3


def test_ac_projected_notebook_is_paired_when_present():
    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks" / "ac_adiabatic_projected_hf.ipynb"
    script = root / "notebooks" / "ac_adiabatic_projected_hf.py"
    if not notebook.exists():
        return
    data = json.loads(notebook.read_text())
    assert data["metadata"]["jupytext"]["formats"] == "ipynb,py:percent"
    assert script.exists()
    text = script.read_text()
    assert "build_ac_active_space" in text
    assert "build_ac_density_vertices" in text
    assert "ContinuumHFBackend" in text
    assert "allow_nonconverged_references" in text
    assert "TPrimeConstraint" in text
    assert "ValleyU1Constraint" in text
    assert "ACBandOverlapProvider" in text
    assert "k_theta_from_ac_projectors" in text
    assert "ac_projector_chern" in text
    assert "b2 = 0.0" in text
    assert "u2 = 0.0" in text
    assert "FirstShellACParams(b1=b1, u1=u1, b2=b2, u2=u2, n_ll=n_ll)" in text
    assert "scan_ac_projected_hf_b2_u2.py" in text
    assert "jobs/scan_ac_projected_hf_b2_u2_array.sh" in text
