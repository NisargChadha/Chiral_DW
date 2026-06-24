import pytest
from pydantic import ValidationError

from chiral_dw.artifacts import RunArtifact, RunManifest
from chiral_dw.config import (
    ACConventionParams,
    ACResponseWorkflowParams,
    FirstShellACParams,
    FourierACParams,
    FourierCoefficient,
    QHFMChargeBenchmarkParams,
    ResponseParams,
    TMoTe2ACParams,
    TMDHFReferenceParams,
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


def test_tmd_hf_reference_params_record_raw_source_convention():
    params = TMDHFReferenceParams()

    assert params.tmd_hf_path_hint == "/Users/nisargchadha/Documents/TMD_HF"
    assert params.source_convention == "Delta=H_HF(P)-H0"
    assert params.n_occ_per_block == 1


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
