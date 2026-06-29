import pytest
from pydantic import ValidationError

from chiral_dw.artifacts import RunArtifact, RunManifest
from chiral_dw.config import (
    ACConventionParams,
    ACResponseWorkflowParams,
    ConjugateACBiasSweepParams,
    ContinuumGridParams,
    ContinuumHFParams,
    ContinuumInteractionParams,
    ContinuumModelParams,
    ContinuumWorkflowParams,
    FirstShellACParams,
    FourierACParams,
    FourierCoefficient,
    IdealConjugateLLLChargeBenchmarkParams,
    PhysicalCoulombACPreset,
    QHFMChargeBenchmarkParams,
    RealSpaceGridParams,
    ResponseParams,
    TMoTe2ACParams,
    UnitsParams,
)


def test_units_keep_dimensionless_density_convention():
    units = UnitsParams(a_m=2.0)

    assert units.charge_density_convention == "dimensionless_per_a_m_squared"
    assert units.physical_density_scale == 0.25


def test_ac_conventions_record_sign_choices_from_reference_notes():
    conventions = ACConventionParams()

    assert conventions.average_field_formula == "B0=-2*pi/A_M"
    assert conventions.dimensionless_field == "minus_B_A_M_over_2pi"
    assert conventions.phi_rotation_sign == "exp_minus_i_phi_sz_over_2"


def test_models_are_frozen():
    params = FirstShellACParams(b1=0.2)

    with pytest.raises(ValidationError):
        params.b1 = 0.3


def test_fourier_params_require_matching_lengths():
    coeff = FourierCoefficient(real=1.0, imag=-0.5)
    ok = FourierACParams(
        g_vectors=((1.0, 0.0),),
        u_coefficients=(coeff,),
        b_coefficients=(FourierCoefficient(),),
    )

    assert ok.u_coefficients[0].as_complex() == 1.0 - 0.5j
    with pytest.raises(ValidationError):
        FourierACParams(
            g_vectors=((1.0, 0.0), (0.0, 1.0)),
            u_coefficients=(coeff,),
            b_coefficients=(FourierCoefficient(),),
        )


def test_tmd_hf_canonical_ac_params_record_folded_signs():
    params = TMoTe2ACParams()

    assert params.phi_deg == 107.7
    assert params.continuum_w_mev == -23.8
    assert params.folded_ac_w_mev == 23.8


def test_native_continuum_params_record_self_contained_hf_defaults():
    params = ContinuumWorkflowParams(
        grid=ContinuumGridParams(n_k=3),
        model=ContinuumModelParams(n_active_bands_per_valley=1),
        interaction=ContinuumInteractionParams(v0=0.2),
        hf=ContinuumHFParams(n_occ_per_k=1),
    )

    assert params.grid.n_total == 9
    assert params.model.active_model == "qiwuzhang"
    assert params.interaction.q0_hartree == "omit_uniform"
    assert params.hf.n_occ_per_k == 1


def test_response_params_validate_theta_window():
    with pytest.raises(ValidationError):
        ResponseParams(theta_min=1.0, theta_max=1.0)


def test_workflow_params_nest_frozen_models():
    params = ACResponseWorkflowParams(ac=FirstShellACParams(b1=0.1, u1=0.05))

    assert params.ac.b1 == 0.1
    assert params.conventions.average_field_sign == "B0_negative"
    assert params.source.field_policy == "raw_hermitian"


def test_qhfm_benchmark_params_record_same_chern_defaults():
    params = QHFMChargeBenchmarkParams()

    assert params.grid.n_k == 7
    assert params.real_space.n_r == 9
    assert params.ac.b1 == 0.2
    assert params.skyrmion.expected_charge_relation == "rho_top=-q_sk"


def test_ideal_conjugate_lll_params_enforce_flat_opposite_chern_limit():
    params = IdealConjugateLLLChargeBenchmarkParams()

    assert params.ac.n_ll == 1
    assert params.ac.b1 == 0.0
    assert params.radius_lB == 10.0
    assert params.magnetic_length_convention == "magnetic_length"
    with pytest.raises(ValidationError):
        IdealConjugateLLLChargeBenchmarkParams(ac=FirstShellACParams(u1=0.1, n_ll=1))
    with pytest.raises(ValidationError):
        IdealConjugateLLLChargeBenchmarkParams(ac=FirstShellACParams(b2=0.1, n_ll=1))
    with pytest.raises(ValidationError):
        IdealConjugateLLLChargeBenchmarkParams(ac=FirstShellACParams(u2=0.1, n_ll=1))
    with pytest.raises(ValidationError):
        IdealConjugateLLLChargeBenchmarkParams(ac=FirstShellACParams(n_ll=2))
    with pytest.raises(ValidationError):
        IdealConjugateLLLChargeBenchmarkParams(real_space=RealSpaceGridParams(n_r=2))


def test_conjugate_ac_bias_sweep_params_record_old_physical_coulomb_preset():
    preset = PhysicalCoulombACPreset()
    interaction = preset.interaction_params(interaction_shell=1)
    params = ConjugateACBiasSweepParams(
        sweep_parameter="b1_c3",
        n_ll=3,
        interaction=interaction,
        use_physical_coulomb=True,
    )

    assert preset.units == "omega_c"
    assert interaction.v0 == 0.267
    assert interaction.gate_distance == 5.80
    assert interaction.interaction_shell == 1
    assert params.sweep_parameter == "b1_c3"
    with pytest.raises(ValidationError):
        ConjugateACBiasSweepParams(active_band=1)


def test_run_manifest_reports_missing_required_artifacts():
    artifacts = (
        RunArtifact(name="summary", path="summary.json", kind="json", description="summary"),
        RunArtifact(
            name="optional_plot",
            path="plot.png",
            kind="plot",
            description="plot",
            required=False,
        ),
    )
    manifest = RunManifest.from_artifacts(
        run_id="demo",
        result_dir="results/demo",
        artifacts=artifacts,
    )

    assert manifest.missing_required == ("summary",)
    assert not manifest.passed
