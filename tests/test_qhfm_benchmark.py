import numpy as np

from chiral_dw.config import (
    FirstShellACParams,
    MomentumGridParams,
    QHFMChargeBenchmarkParams,
    RealSpaceGridParams,
)
from chiral_dw.qhfm_benchmark import (
    GeneralProjector4DCurvature,
    SameChernQHFMTrial,
    periodic_skyrmion_lattice_field,
    run_qhfm_charge_benchmark,
)


def _small_params(n_k: int = 4, n_r: int = 4) -> QHFMChargeBenchmarkParams:
    return QHFMChargeBenchmarkParams(
        grid=MomentumGridParams(n_k=n_k),
        real_space=RealSpaceGridParams(n_r=n_r),
        ac=FirstShellACParams(b1=0.0, u1=0.0, n_ll=3),
        n_form_factors=1,
    )


def test_same_chern_qhfm_projectors_are_rank_one():
    trial = SameChernQHFMTrial(_small_params(n_k=3, n_r=2))
    coeffs = np.zeros((3, 1))
    coeffs[2, 0] = -1.0
    texture = -0.2 * periodic_skyrmion_lattice_field(trial.fractional_r_grid(), mass=0.5)
    solution = trial.solve(coeffs, texture_field=texture)

    p = solution.flavor_projectors
    assert np.allclose(p, p.conj().swapaxes(-1, -2), atol=1e-12)
    assert np.allclose(p @ p, p, atol=1e-12)
    assert np.allclose(np.trace(p, axis1=-2, axis2=-1), 1.0, atol=1e-12)

    full = trial.full_projector(solution.k_points[1, 1], solution.flavor_projectors[1, 1, 0, 0])
    assert np.allclose(full, full.conj().T, atol=1e-12)
    assert np.allclose(full @ full, full, atol=1e-12)
    assert np.isclose(np.trace(full), 1.0, atol=1e-12)


def test_qhfm_projector_is_invariant_under_occupied_state_phase():
    trial = SameChernQHFMTrial(_small_params(n_k=3, n_r=1))
    coeffs = np.zeros((3, 1))
    coeffs[2, 0] = -1.0
    solution = trial.solve(coeffs)
    phases = np.exp(1j * np.linspace(0.0, 1.7, solution.spinors.size // 2)).reshape(
        solution.spinors.shape[:-1]
    )
    phased = solution.spinors * phases[..., None]
    p_phased = phased[..., :, None] * phased[..., None, :].conj()

    assert np.allclose(p_phased, solution.flavor_projectors, atol=1e-12)


def test_same_chern_qhfm_ac_band_has_unit_chern():
    trial = SameChernQHFMTrial(_small_params(n_k=3, n_r=1))
    diagnostics = trial.model.band_diagnostics(n_k=6)

    assert np.isclose(diagnostics["chern"], 1.0, atol=5e-3)


def test_qhfm_factorized_charge_matches_skyrmion_density():
    result = run_qhfm_charge_benchmark(_small_params(n_k=4, n_r=4))

    assert result.summary.mixed_curvature_max < 1e-10
    assert np.isclose(result.summary.orbital_chern, 1.0, atol=5e-3)
    assert result.summary.charge_error_max < 5e-3
    assert np.allclose(result.rho_top, -result.q_sk, atol=5e-3)
    assert np.isclose(
        result.summary.integrated_charge,
        -result.summary.integrated_skyrmion_charge,
        atol=5e-3,
    )
    assert result.summary.valid_charge_normalization


def test_qhfm_curvature_rejects_nonextended_solution():
    trial = SameChernQHFMTrial(_small_params(n_k=3, n_r=2))
    coeffs = np.zeros((3, 1))
    solution = trial.solve(coeffs)

    try:
        GeneralProjector4DCurvature(trial, solution)
    except ValueError as exc:
        assert "extended_k" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected nonextended solution to fail")
