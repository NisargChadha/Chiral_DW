from pathlib import Path

import numpy as np

from chiral_dw.config import (
    IdealConjugateLLLChargeBenchmarkParams,
    MomentumGridParams,
    RealSpaceGridParams,
)
from chiral_dw.ideal_conjugate_lll import (
    IdealConjugate4DCurvatureEvaluator,
    IdealConjugateLLLBasis,
    IdealConjugateProjectorSolution,
    IdealConjugateTrialProjector,
    explicit_texture_solution_from_spinors,
    k_theta_from_ideal_conjugate_projectors,
    plaquette_average,
    projectors_from_spinors,
    run_explicit_chiral_domain_wall_texture_response,
    run_ideal_conjugate_lll_charge_benchmark,
    spinors_from_unit_vector_texture,
    uniform_conjugate_projector_path,
)


def _small_params(
    *,
    n_k: int = 3,
    n_r: int = 11,
    m0: float = 1.0,
    winding: int = 1,
    output_dir: str = "results/test_ideal_conjugate_lll",
) -> IdealConjugateLLLChargeBenchmarkParams:
    return IdealConjugateLLLChargeBenchmarkParams(
        grid=MomentumGridParams(n_k=n_k),
        real_space=RealSpaceGridParams(n_r=n_r),
        radius_lB=4.0,
        width_lB=1.0,
        patch_length_lB=18.0,
        winding=winding,
        m0=m0,
        output_dir=output_dir,
    )


def test_ideal_conjugate_basis_has_opposite_cherns_and_flat_bands():
    params = _small_params(n_k=3, n_r=5)
    basis = IdealConjugateLLLBasis(params)
    up_chern, down_chern = basis.band_cherns()
    diagnostics = basis.model.band_diagnostics(n_k=4, active_band=0)
    up_bandwidth, down_bandwidth = basis.band_bandwidths()

    assert np.isclose(up_chern, 1.0, atol=5e-3)
    assert np.isclose(down_chern, -1.0, atol=5e-3)
    assert np.isclose(diagnostics["chern"], 1.0, atol=5e-3)
    assert up_bandwidth < 1e-12
    assert down_bandwidth < 1e-12
    assert diagnostics["bandwidth"] < 1e-12


def test_ideal_conjugate_projectors_are_rank_one_and_align_with_texture():
    basis = IdealConjugateLLLBasis(_small_params(n_k=3, n_r=7))
    solution = IdealConjugateTrialProjector(basis).solve(extended_k=True)

    p = solution.band_projectors
    assert np.allclose(p, p.conj().swapaxes(-1, -2), atol=1e-12)
    assert np.allclose(p @ p, p, atol=1e-12)
    assert np.allclose(np.trace(p, axis1=-2, axis2=-1), 1.0, atol=1e-12)
    assert np.allclose(solution.spin_expectation, solution.wall_field, atol=1e-12)
    assert np.isclose(np.min(solution.gaps), 2.0 * basis.params.m0, atol=1e-12)

    full = IdealConjugateTrialProjector(basis).full_projector(
        solution.k_points[1, 1],
        solution.band_projectors[1, 1, 2, 2],
    )
    assert np.allclose(full, full.conj().T, atol=1e-12)
    assert np.allclose(full @ full, full, atol=1e-12)
    assert np.isclose(np.trace(full), 1.0, atol=1e-12)


def test_ideal_conjugate_curvature_rejects_nonextended_solution():
    basis = IdealConjugateLLLBasis(_small_params(n_k=3, n_r=5))
    solution = IdealConjugateTrialProjector(basis).solve(extended_k=False)

    try:
        IdealConjugate4DCurvatureEvaluator(basis, solution)
    except ValueError as exc:
        assert "extended_k" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected nonextended solution to fail")


def test_ideal_conjugate_charge_is_independent_of_m0():
    params_a = _small_params(m0=0.4)
    params_b = _small_params(m0=2.3)
    result_a = run_ideal_conjugate_lll_charge_benchmark(params_a)
    result_b = run_ideal_conjugate_lll_charge_benchmark(params_b)

    assert np.allclose(result_a.rho_top, result_b.rho_top, atol=1e-12)
    assert np.allclose(result_a.rho_analytic, result_b.rho_analytic, atol=1e-12)
    assert np.allclose(
        result_a.solution.band_projectors,
        result_b.solution.band_projectors,
        atol=1e-12,
    )
    assert np.isclose(result_a.summary.local_gap_min, 2.0 * params_a.m0, atol=1e-12)
    assert np.isclose(result_b.summary.local_gap_min, 2.0 * params_b.m0, atol=1e-12)


def test_ideal_conjugate_charge_matches_discrete_analytic_plaquettes():
    result = run_ideal_conjugate_lll_charge_benchmark(_small_params(n_r=15))
    expected_shape = (3, 3, 14, 14)
    rho = result.rho_top.reshape(-1)
    target = result.rho_analytic.reshape(-1)
    correlation = np.corrcoef(rho, target)[0, 1]
    wrong_sign_rms = float(np.sqrt(np.mean((rho + target) ** 2)))

    assert all(component.shape == expected_shape for component in result.curvature_components.values())
    assert result.rho_top.shape == (14, 14)
    assert result.rho_analytic.shape == result.rho_top.shape
    assert result.q_sk.shape == result.rho_top.shape
    assert result.summary.charge_error_max < 2e-3
    assert result.summary.charge_error_rms < 8e-4
    assert correlation > 0.998
    assert wrong_sign_rms > 10.0 * result.summary.charge_error_rms
    assert np.allclose(result.rho_top, result.rho_analytic, atol=2e-3)
    assert np.isclose(np.sum(result.q_sk), 1.0, atol=5e-3)
    assert np.isclose(
        result.summary.integrated_charge,
        result.summary.integrated_analytic_charge,
        atol=2e-2,
    )
    assert result.summary.valid_analytic_charge


def test_ideal_conjugate_charge_winding_zero_and_sign_flip():
    zero = run_ideal_conjugate_lll_charge_benchmark(_small_params(winding=0))
    plus = run_ideal_conjugate_lll_charge_benchmark(_small_params(winding=1))
    minus = run_ideal_conjugate_lll_charge_benchmark(_small_params(winding=-1))

    assert np.max(np.abs(zero.rho_top)) < 1e-12
    assert np.allclose(plus.rho_top, -minus.rho_top, atol=1e-12)
    assert np.allclose(plus.rho_analytic, -minus.rho_analytic, atol=1e-12)
    assert np.sign(plus.summary.dipole_moment) == -np.sign(minus.summary.dipole_moment)


def test_ideal_conjugate_charge_is_gauge_invariant_under_local_spinor_phase():
    params = _small_params(n_r=9)
    basis = IdealConjugateLLLBasis(params)
    solution = IdealConjugateTrialProjector(basis).solve(extended_k=True)
    evaluator = IdealConjugate4DCurvatureEvaluator(basis, solution)
    rho = evaluator.charge_per_realspace_plaquette_centered(
        evaluator.curvature_components_centered()
    )
    k_phase = 0.37 * solution.k_fractional[..., 0] + 0.19 * solution.k_fractional[..., 1]
    xy = solution.xy
    real_phase = 0.23 * xy[..., 0] + 0.11 * xy[..., 1]
    phase = np.exp(1j * (k_phase[:, :, None, None] + real_phase[None, None, :, :]))
    phased = IdealConjugateProjectorSolution(
        k_points=solution.k_points,
        k_fractional=solution.k_fractional,
        xy=solution.xy,
        wall_field=solution.wall_field,
        spinors=solution.spinors * phase[..., None],
        band_projectors=solution.band_projectors,
        eigenvalues=solution.eigenvalues,
        gaps=solution.gaps,
    )
    phased_eval = IdealConjugate4DCurvatureEvaluator(basis, phased)
    rho_phased = phased_eval.charge_per_realspace_plaquette_centered(
        phased_eval.curvature_components_centered()
    )

    assert np.allclose(rho, rho_phased, atol=1e-12)


def test_ideal_conjugate_wall_charge_is_localized_and_dipolar():
    params = _small_params(n_r=15)
    result = run_ideal_conjugate_lll_charge_benchmark(params)
    basis = IdealConjugateLLLBasis(params)
    centers = plaquette_average(result.solution.xy)
    radius = np.linalg.norm(centers, axis=-1)
    wall_mask = np.abs(radius - basis.radius) < 1.5 * basis.width
    far_mask = (radius < basis.radius - 2.0 * basis.width) | (
        radius > basis.radius + 2.0 * basis.width
    )
    absolute_charge = np.sum(np.abs(result.rho_top))

    assert np.max(result.rho_top) > 0.0
    assert np.min(result.rho_top) < 0.0
    assert np.mean(np.abs(result.rho_top[wall_mask])) > 10.0 * np.mean(
        np.abs(result.rho_top[far_mask])
    )
    assert abs(result.summary.integrated_charge) < 0.05 * absolute_charge


def test_ideal_conjugate_charge_writes_artifacts(tmp_path: Path):
    params = _small_params(
        n_k=3,
        n_r=9,
        output_dir=str(tmp_path),
    )
    result = run_ideal_conjugate_lll_charge_benchmark(
        params,
        write_outputs=True,
        write_plots=False,
    )

    assert result.manifest is not None
    assert result.manifest.passed
    assert (tmp_path / "ideal_conjugate_lll_charge.npz").exists()
    assert (tmp_path / "ideal_conjugate_lll_summary.json").exists()
    assert (tmp_path / "ideal_conjugate_lll_profiles.csv").exists()
    arrays = np.load(tmp_path / "ideal_conjugate_lll_charge.npz")
    for key in ("rho_top", "rho_analytic", "q_sk", "n_z_center", "charge_error", "Fkxky", "Fxy"):
        assert key in arrays


def test_explicit_spinor_texture_builds_rank_one_projectors():
    basis = IdealConjugateLLLBasis(_small_params(n_k=3, n_r=7))
    xy = basis.real_space_grid()
    field = np.zeros((*xy.shape[:2], 3), dtype=float)
    field[..., 0] = 0.3
    field[..., 1] = 0.4
    field[..., 2] = np.sqrt(1.0 - 0.3**2 - 0.4**2)
    spinors = spinors_from_unit_vector_texture(field)
    solution = explicit_texture_solution_from_spinors(basis, spinors, wall_field=field)

    p = solution.band_projectors
    assert p.shape == (4, 4, 7, 7, 2, 2)
    assert np.allclose(p, p.conj().swapaxes(-1, -2), atol=1e-12)
    assert np.allclose(p @ p, p, atol=1e-12)
    assert np.allclose(np.trace(p, axis1=-2, axis2=-1), 1.0, atol=1e-12)
    assert np.allclose(solution.spin_expectation, field, atol=1e-12)


def test_explicit_uniform_conjugate_projectors_give_expected_cg():
    params = _small_params(n_k=5, n_r=7)
    basis = IdealConjugateLLLBasis(params)
    theta_edges = np.linspace(0.0, np.pi, 21)
    projectors = uniform_conjugate_projector_path(theta_edges, params.grid.n_k)
    response = k_theta_from_ideal_conjugate_projectors(
        basis,
        projectors,
        theta_edges,
        np.array([0.0, 0.2]),
    )

    assert abs(response.cG + 1.0 / (4.0 * np.pi)) < 1e-2
    assert np.allclose(response.K + response.K[::-1], 0.0, atol=1e-10)


def test_explicit_chiral_texture_response_matches_trial_charge_path():
    params = _small_params(n_k=3, n_r=11)
    explicit = run_explicit_chiral_domain_wall_texture_response(
        params,
        theta_edges=np.linspace(0.0, np.pi, 17),
        phi_nodes=np.array([0.0, 0.2]),
    )
    trial = run_ideal_conjugate_lll_charge_benchmark(params)

    assert np.isfinite(explicit.response.cG)
    assert abs(explicit.response.cG) > 1e-3
    assert np.allclose(explicit.rho_top, trial.rho_top, atol=1e-12)
    assert np.allclose(explicit.rho_analytic, trial.rho_analytic, atol=1e-12)
    assert np.allclose(explicit.solution.band_projectors, trial.solution.band_projectors, atol=1e-12)
    assert np.all(np.isnan(explicit.solution.gaps))
    assert explicit.charge_profile.rho_dimless.shape == (500,)


def test_explicit_texture_charge_is_gauge_invariant_under_supplied_spinor_phase():
    params = _small_params(n_k=3, n_r=9)
    basis = IdealConjugateLLLBasis(params)
    xy = basis.real_space_grid()
    field = np.dstack(
        [
            np.zeros(xy.shape[:2]),
            np.zeros(xy.shape[:2]),
            np.ones(xy.shape[:2]),
        ]
    )
    spinors = spinors_from_unit_vector_texture(field)
    phase = np.exp(1j * (0.31 * xy[..., 0] + 0.17 * xy[..., 1]))
    solution = explicit_texture_solution_from_spinors(
        basis,
        spinors * phase[..., None],
        wall_field=field,
    )

    assert np.allclose(
        solution.band_projectors,
        projectors_from_spinors(spinors)[None, None],
        atol=1e-12,
    )
