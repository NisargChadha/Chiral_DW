"""Frozen Pydantic parameters and convention records."""

from __future__ import annotations

from math import pi
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chiral_dw.artifacts import RunArtifact, RunManifest

Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]


class MomentumGridParams(BaseModel):
    """Uniform primitive-cell momentum mesh."""

    model_config = ConfigDict(frozen=True)

    n_k: int = Field(default=7, ge=1)

    @property
    def n_total(self) -> int:
        return self.n_k * self.n_k


class RealSpaceGridParams(BaseModel):
    """Uniform primitive-cell real-space mesh."""

    model_config = ConfigDict(frozen=True)

    n_r: int = Field(default=9, ge=1)

    @property
    def n_total(self) -> int:
        return self.n_r * self.n_r


class UnitsParams(BaseModel):
    """Units used for dimensionless moire-cell calculations."""

    model_config = ConfigDict(frozen=True)

    a_m: float = Field(default=1.0, gt=0.0)
    length_unit: Literal["moire_period"] = "moire_period"
    charge_density_convention: Literal["dimensionless_per_a_m_squared"] = (
        "dimensionless_per_a_m_squared"
    )

    @property
    def physical_density_scale(self) -> float:
        """Multiplier converting dimensionless density to charge per length^2."""
        return 1.0 / (self.a_m * self.a_m)


class ACConventionParams(BaseModel):
    """Sign and unit conventions for the nonideal finite-LL AC backend."""

    model_config = ConfigDict(frozen=True)

    average_field_sign: Literal["B0_negative"] = "B0_negative"
    average_field_formula: Literal["B0=-2*pi/A_M"] = "B0=-2*pi/A_M"
    magnetic_length_formula: Literal["l2=A_M/(2*pi)"] = "l2=A_M/(2*pi)"
    dimensionless_field: Literal["minus_B_A_M_over_2pi"] = "minus_B_A_M_over_2pi"
    phi_rotation_sign: Literal["exp_minus_i_phi_sz_over_2"] = (
        "exp_minus_i_phi_sz_over_2"
    )
    projector_basis: Literal["two_flavor_active_band"] = "two_flavor_active_band"


class FourierCoefficient(BaseModel):
    """JSON-friendly complex Fourier coefficient."""

    model_config = ConfigDict(frozen=True)

    real: float = 0.0
    imag: float = 0.0

    @classmethod
    def from_complex(cls, value: complex) -> "FourierCoefficient":
        z = complex(value)
        return cls(real=float(z.real), imag=float(z.imag))

    def as_complex(self) -> complex:
        return complex(self.real, self.imag)


class FirstShellACParams(BaseModel):
    """Low-harmonic nonideal AC parameters in units of omega_c."""

    model_config = ConfigDict(frozen=True)

    b1: float = 0.0
    u1: float = 0.0
    b2: float = 0.0
    u2: float = 0.0
    b1_c3: float = 0.0
    u1_c3: float = 0.0
    n_ll: int = Field(default=5, ge=1)
    material: str = "MoTe2_TMD_HF_canonical_first_shell"
    a_m: float = Field(default=1.0, gt=0.0)


class FourierACParams(BaseModel):
    """Arbitrary Fourier coefficients for the finite-LL AC backend."""

    model_config = ConfigDict(frozen=True)

    g_vectors: tuple[Vector2, ...]
    u_coefficients: tuple[FourierCoefficient, ...]
    b_coefficients: tuple[FourierCoefficient, ...]
    n_ll: int = Field(default=5, ge=1)
    material: str = "FourierAC"
    a_m: float = Field(default=1.0, gt=0.0)

    @model_validator(mode="after")
    def _matching_lengths(self) -> "FourierACParams":
        n_g = len(self.g_vectors)
        if len(self.u_coefficients) != n_g or len(self.b_coefficients) != n_g:
            raise ValueError("g_vectors, u_coefficients, and b_coefficients must match")
        return self


class TMoTe2ACParams(BaseModel):
    """TMD_HF-canonical tMoTe2 parameters folded into the AC convention."""

    model_config = ConfigDict(frozen=True)

    theta_deg: float = Field(default=3.5, gt=0.0)
    a0_angstrom: float = Field(default=3.52, gt=0.0)
    m_eff: float = Field(default=0.6, gt=0.0)
    V_mev: float = 20.8
    phi_deg: float = 107.7
    continuum_w_mev: float = -23.8
    folded_ac_w_mev: float = 23.8
    uD_mev: float = 0.0
    a_m: float = Field(default=1.0, gt=0.0)
    grid_size: int = Field(default=96, ge=8)
    g_shell_cutoff: int = Field(default=3, ge=1)
    coefficient_cutoff: float = Field(default=1e-10, ge=0.0)
    n_ll: int = Field(default=5, ge=1)
    material: str = "MoTe2_TMD_HF_canonical_folded_AC"


class ResponseParams(BaseModel):
    """Numerical controls for K(theta), cG, and phi reconstruction."""

    model_config = ConfigDict(frozen=True)

    n_theta: int = Field(default=41, ge=3)
    n_phi: int = Field(default=5, ge=1)
    phi_step: float = Field(default=0.2, gt=0.0)
    theta_min: float = Field(default=0.0, ge=0.0)
    theta_max: float = Field(default=pi, gt=0.0)
    endpoint_eps: float = Field(default=1e-5, ge=0.0)
    derivative_method: Literal["finite_difference"] = "finite_difference"
    phi_rotation_sign: Literal["exp_minus_i_phi_sz_over_2"] = "exp_minus_i_phi_sz_over_2"

    @model_validator(mode="after")
    def _theta_window_is_valid(self) -> "ResponseParams":
        if self.theta_max <= self.theta_min:
            raise ValueError("theta_max must exceed theta_min")
        if self.n_theta < 3:
            raise ValueError("n_theta must be at least 3")
        return self


class DomainWallParams(BaseModel):
    """Circular chiral domain-wall texture parameters in moire units."""

    model_config = ConfigDict(frozen=True)

    radius: float = Field(default=20.0, gt=0.0)
    width: float = Field(default=3.0, gt=0.0)
    winding: int = 1
    profile: Literal["tanh", "logistic"] = "tanh"


class SourceInterpolationParams(BaseModel):
    """Controls for simple VP/IVC source-field interpolation."""

    model_config = ConfigDict(frozen=True)

    source_scale: float = 1.0
    occupy: Literal["lowest", "highest"] = "lowest"
    n_occ_per_block: int = Field(default=1, ge=1)
    field_policy: Literal["raw_hermitian"] = "raw_hermitian"
    include_scalar_diagnostics: bool = True


class ContinuumGridParams(BaseModel):
    """Square momentum grid for native continuum/HF workflows."""

    model_config = ConfigDict(frozen=True)

    n_k: int = Field(default=5, ge=1)

    @property
    def n_total(self) -> int:
        return self.n_k * self.n_k


class ContinuumFiniteQParams(BaseModel):
    """Finite-Q active-frame controls for native continuum/HF workflows."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    q_coord: tuple[int, int] = (0, 0)
    half_shift_coord: tuple[int, int] | None = None

    @field_validator("q_coord")
    @classmethod
    def _q_coord_is_integral(cls, value: tuple[int, int]) -> tuple[int, int]:
        if len(value) != 2:
            raise ValueError("q_coord must have length 2")
        return int(value[0]), int(value[1])

    @field_validator("half_shift_coord")
    @classmethod
    def _half_shift_coord_is_integral(
        cls, value: tuple[int, int] | None
    ) -> tuple[int, int] | None:
        if value is None:
            return None
        if len(value) != 2:
            raise ValueError("half_shift_coord must have length 2")
        return int(value[0]), int(value[1])


class ContinuumModelParams(BaseModel):
    """Native continuum active-band parameters."""

    model_config = ConfigDict(frozen=True)

    theta_deg: float = Field(default=3.5, gt=0.0)
    a0_angstrom: float = Field(default=3.47, gt=0.0)
    m_eff: float = Field(default=0.62, gt=0.0)
    moire_potential_mev: float = 11.2
    phi_deg: float = 91.0
    tunneling_mev: float = -13.3
    displacement_mev: float = 0.0
    plane_wave_shell: int = Field(default=1, ge=0)
    n_bands: int = Field(default=2, ge=1)
    n_active_bands_per_valley: int = Field(default=1, ge=1)
    active_model: Literal["qiwuzhang", "taige", "ac_finite_ll"] = "qiwuzhang"


class ContinuumInteractionParams(BaseModel):
    """Screened Coulomb controls for the native continuum HF backend."""

    model_config = ConfigDict(frozen=True)

    v0: float = Field(default=1.0, ge=0.0)
    gate_distance: float = Field(default=2.0, gt=0.0)
    q0_hartree: Literal["omit_uniform"] = "omit_uniform"
    q_shell: int = Field(default=1, ge=0)
    q_mesh: Literal["shell", "full"] = "shell"
    local_field_cutoff: int = Field(default=0, ge=0)
    coulomb_kind: Literal["dimensionless_screened", "dual_gate"] = "dimensionless_screened"
    epsilon: float = Field(default=16.7, gt=0.0)
    gate_distance_nm: float = Field(default=30.0, gt=0.0)
    include_q0: bool = True
    smear_length_nm: float = Field(default=0.347, ge=0.0)
    exchange_scale: float = Field(default=1.0, ge=0.0)
    hartree_scale: float = Field(default=1.0, ge=0.0)
    reference_density: Literal["zero"] = "zero"
    vertex_workers: int = Field(default=1, ge=0)
    exchange_workers: int = Field(default=1, ge=0)
    density_vertex_retention: Literal["full", "hartree_only"] = "full"
    density_vertex_layout: Literal["auto", "dense", "valley_compact"] = "auto"
    exchange_representation: Literal["auto", "dense", "valley_sector"] = "auto"
    form_factor_backend: Literal["auto", "scalar", "cached_gather", "vectorized"] = "auto"


class ContinuumHFParams(BaseModel):
    """Zero-temperature fixed-per-k native continuum HF controls."""

    model_config = ConfigDict(frozen=True)

    n_occ_per_k: int = Field(default=1, ge=1)
    max_iter: int = Field(default=80, ge=1)
    min_iter: int = Field(default=2, ge=0)
    mixing_method: Literal["linear", "oda"] = "linear"
    mixing: float = Field(default=0.45, ge=0.0, le=1.0)
    oda_lambda_min: float = Field(default=1e-4, ge=0.0)
    tolerance: float = Field(default=1e-8, gt=0.0)
    energy_tolerance: float = Field(default=1e-10, gt=0.0)
    idempotency_tolerance: float = Field(default=1e-8, gt=0.0)
    final_residual_tolerance: float = Field(default=1e-7, gt=0.0)
    ivc_angle: float = 0.5 * pi
    ivc_phase: float = 0.0
    random_seed: int = 1
    seed_ordered_weight: float = Field(default=1.0, ge=0.0)
    seed_random_weight: float = Field(default=0.0, ge=0.0)
    store_projector_snapshots: bool = False
    snapshot_interval: int = Field(default=10, ge=1)
    first_iteration_snapshot: bool = True

    @model_validator(mode="after")
    def _seed_weights_sum_to_one(self) -> "ContinuumHFParams":
        total = self.seed_ordered_weight + self.seed_random_weight
        if abs(total - 1.0) > 1e-12:
            raise ValueError("seed_ordered_weight and seed_random_weight must sum to one")
        return self


class ContinuumWorkflowParams(BaseModel):
    """Top-level native continuum symmetric-HF response controls."""

    model_config = ConfigDict(frozen=True)

    grid: ContinuumGridParams = Field(default_factory=ContinuumGridParams)
    model: ContinuumModelParams = Field(default_factory=ContinuumModelParams)
    interaction: ContinuumInteractionParams = Field(default_factory=ContinuumInteractionParams)
    hf: ContinuumHFParams = Field(default_factory=ContinuumHFParams)
    response: ResponseParams = Field(default_factory=ResponseParams)
    domain_wall: DomainWallParams = Field(default_factory=DomainWallParams)
    output_dir: str = "results/continuum_symmetric_hf"


class ACProjectedHFParams(BaseModel):
    """Top-level finite-LL AC projected symmetric-HF response controls."""

    model_config = ConfigDict(frozen=True)

    grid: ContinuumGridParams = Field(default_factory=lambda: ContinuumGridParams(n_k=5))
    ac: FirstShellACParams | FourierACParams | TMoTe2ACParams = Field(
        default_factory=lambda: FirstShellACParams(b1=0.2, u1=0.05, n_ll=5)
    )
    interaction: ContinuumInteractionParams = Field(
        default_factory=lambda: ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.2,
            q_shell=1,
            local_field_cutoff=1,
            gate_distance=2.0,
        )
    )
    hf: ContinuumHFParams = Field(default_factory=ContinuumHFParams)
    response: ResponseParams = Field(default_factory=lambda: ResponseParams(n_theta=21))
    active_band: int = Field(default=0, ge=0)
    band_diagnostics_n_k: int = Field(default=9, ge=2)
    moire_length_nm: float = Field(default=1.0, gt=0.0)
    energy_unit_mev: float = Field(default=1.0, gt=0.0)
    output_dir: str = "results/ac_projected_hf"


class GatedInteractionParams(BaseModel):
    """Dimensionless screened Coulomb interaction used by AC workflows."""

    model_config = ConfigDict(frozen=True)

    v0: float = Field(default=1.0, ge=0.0)
    gate_distance: float = Field(default=2.0, gt=0.0)
    interaction_shell: int = Field(default=2, ge=0)
    q0_policy: Literal["omit_uniform_hartree"] = "omit_uniform_hartree"


class PhysicalCoulombACPreset(BaseModel):
    """Old-reference physical-Coulomb match in AC omega_c units."""

    model_config = ConfigDict(frozen=True)

    units: Literal["omega_c"] = "omega_c"
    interaction_matching: Literal["physical_coulomb"] = "physical_coulomb"
    v0: float = Field(default=0.267, ge=0.0)
    gate_distance: float = Field(default=5.80, gt=0.0)
    source: str = "Variational_Calculation_tMoTe2 physical Coulomb default"

    def interaction_params(self, interaction_shell: int = 2) -> GatedInteractionParams:
        return GatedInteractionParams(
            v0=self.v0,
            gate_distance=self.gate_distance,
            interaction_shell=interaction_shell,
        )


class SkyrmionTextureParams(BaseModel):
    """Periodic real-space skyrmion-lattice texture controls."""

    model_config = ConfigDict(frozen=True)

    mass: float = 0.5
    texture: Literal["periodic_qwz"] = "periodic_qwz"
    expected_charge_relation: Literal["rho_top=-q_sk"] = "rho_top=-q_sk"


class QHFMChargeBenchmarkParams(BaseModel):
    """Same-Chern QHFM real-space charge normalization benchmark parameters."""

    model_config = ConfigDict(frozen=True)

    grid: MomentumGridParams = Field(default_factory=lambda: MomentumGridParams(n_k=7))
    real_space: RealSpaceGridParams = Field(default_factory=RealSpaceGridParams)
    ac: FirstShellACParams = Field(
        default_factory=lambda: FirstShellACParams(b1=0.2, u1=0.1, n_ll=5)
    )
    skyrmion: SkyrmionTextureParams = Field(default_factory=SkyrmionTextureParams)
    active_band: int = Field(default=0, ge=0)
    n_form_factors: int = Field(default=1, ge=1, le=7)
    output_dir: str = "results/qhfm_charge_benchmark"
    write_curvature_npz: bool = True


class QHFMChargeSummary(BaseModel):
    """Scalar summary of the same-Chern QHFM charge benchmark."""

    model_config = ConfigDict(frozen=True)

    orbital_chern: float
    mixed_curvature_max: float
    charge_error_max: float
    integrated_charge: float
    integrated_skyrmion_charge: float
    slope: float
    intercept: float
    correlation: float
    valid_charge_normalization: bool


class IdealConjugateLLLChargeBenchmarkParams(BaseModel):
    """Flat opposite-Chern LLL real-space charge benchmark parameters."""

    model_config = ConfigDict(frozen=True)

    grid: MomentumGridParams = Field(default_factory=lambda: MomentumGridParams(n_k=7))
    real_space: RealSpaceGridParams = Field(default_factory=lambda: RealSpaceGridParams(n_r=41))
    ac: FirstShellACParams = Field(default_factory=lambda: FirstShellACParams(n_ll=1))
    active_band: int = Field(default=0, ge=0)
    radius_lB: float = Field(default=10.0, gt=0.0)
    width_lB: float = Field(default=3.5, gt=0.0)
    patch_length_lB: float = Field(default=56.0, gt=0.0)
    winding: int = 1
    helicity: float = 0.0
    magnetic_length_convention: Literal["magnetic_length"] = "magnetic_length"
    m0: float = Field(default=1.0, gt=0.0)
    output_dir: str = "results/ideal_conjugate_lll_charge"
    write_curvature_npz: bool = True
    charge_tolerance: float = Field(default=5e-3, gt=0.0)

    @model_validator(mode="after")
    def _ideal_lll_limit_is_valid(self) -> "IdealConjugateLLLChargeBenchmarkParams":
        if self.active_band != 0:
            raise ValueError("the ideal conjugate LLL benchmark supports active_band=0")
        if self.real_space.n_r < 3:
            raise ValueError("real_space.n_r must be at least 3 for open-patch plaquettes")
        if self.ac.n_ll != 1:
            raise ValueError("the ideal conjugate LLL benchmark requires ac.n_ll=1")
        harmonics = (
            self.ac.b1,
            self.ac.u1,
            self.ac.b2,
            self.ac.u2,
            self.ac.b1_c3,
            self.ac.u1_c3,
        )
        if any(abs(float(value)) > 1e-15 for value in harmonics):
            raise ValueError("ideal conjugate LLL benchmark requires b1=u1=b2=u2=b1_c3=u1_c3=0")
        if self.width_lB >= self.radius_lB:
            raise ValueError("width_lB must be smaller than radius_lB")
        if self.radius_lB + 3.0 * self.width_lB >= 0.5 * self.patch_length_lB:
            raise ValueError("wall plus three widths must fit inside half the open patch")
        return self


class IdealConjugateLLLChargeSummary(BaseModel):
    """Scalar summary for the flat opposite-Chern LLL benchmark."""

    model_config = ConfigDict(frozen=True)

    up_chern: float
    down_chern: float
    up_bandwidth: float
    down_bandwidth: float
    local_gap_min: float
    spin_alignment_error: float
    projector_hermiticity_error: float
    projector_idempotency_error: float
    charge_error_max: float
    charge_error_rms: float
    integrated_charge: float
    integrated_analytic_charge: float
    integrated_skyrmion_charge: float
    dipole_moment: float
    m0: float
    valid_analytic_charge: bool


class M0SourceScanParams(BaseModel):
    """Controls for one-parameter source-field projector scans."""

    model_config = ConfigDict(frozen=True)

    n_vec: Vector3 = (0.0, 0.0, 1.0)
    active_band: int = Field(default=0, ge=0)
    occupy: Literal["lowest", "highest"] = "lowest"
    m0_min: float = Field(default=0.0, ge=0.0)
    m0_max: float = Field(default=1.0, gt=0.0)
    n_m0: int = Field(default=21, ge=2)

    @field_validator("n_vec")
    @classmethod
    def _n_vec_has_nonzero_norm(cls, value: Vector3) -> Vector3:
        norm2 = sum(float(x) * float(x) for x in value)
        if norm2 <= 0.0:
            raise ValueError("n_vec must have nonzero norm")
        return float(value[0]), float(value[1]), float(value[2])


class ACResponseWorkflowParams(BaseModel):
    """Top-level nonideal AC cG workflow parameters."""

    model_config = ConfigDict(frozen=True)

    grid: MomentumGridParams = Field(default_factory=MomentumGridParams)
    units: UnitsParams = Field(default_factory=UnitsParams)
    conventions: ACConventionParams = Field(default_factory=ACConventionParams)
    ac: FirstShellACParams = Field(default_factory=FirstShellACParams)
    response: ResponseParams = Field(default_factory=ResponseParams)
    domain_wall: DomainWallParams = Field(default_factory=DomainWallParams)
    source: SourceInterpolationParams = Field(default_factory=SourceInterpolationParams)
    interaction: GatedInteractionParams = Field(default_factory=GatedInteractionParams)
    output_dir: str = "results/ac_cg"


class ConjugateACBiasSweepParams(BaseModel):
    """Old-compatible conjugate-AC C3-bias sweep parameters."""

    model_config = ConfigDict(frozen=True)

    sweep_parameter: Literal["u1_c3", "b1_c3"] = "u1_c3"
    output_dir: str = "results/conjugate_ac_bias_sweep"
    b1: float = 0.0
    u1: float = 0.0
    b1_c3_fixed: float = 0.0
    u1_c3_fixed: float = 0.0
    bias_min: float = 0.0
    bias_max: float = 0.2
    n_bias: int = Field(default=11, ge=2)
    n_ll: int = Field(default=1, ge=1)
    active_band: int = Field(default=0, ge=0)
    grid: MomentumGridParams = Field(default_factory=lambda: MomentumGridParams(n_k=7))
    response: ResponseParams = Field(default_factory=ResponseParams)
    source: SourceInterpolationParams = Field(default_factory=SourceInterpolationParams)
    interaction: GatedInteractionParams = Field(default_factory=GatedInteractionParams)
    domain_wall: DomainWallParams = Field(default_factory=DomainWallParams)
    dispersion_points: int = Field(default=80, ge=1)
    n_phi_check: int = Field(default=3, ge=1)
    cg_phi_step: float = Field(default=0.2, gt=0.0)
    max_iter: int = Field(default=120, ge=1)
    energy_tol: float = Field(default=1e-9, gt=0.0)
    projector_tol: float = Field(default=1e-7, gt=0.0)
    min_gap_tol: float = Field(default=1e-10, ge=0.0)
    use_physical_coulomb: bool = False
    write_plots: bool = True

    @model_validator(mode="after")
    def _active_band_supported(self) -> "ConjugateACBiasSweepParams":
        if self.active_band != 0:
            raise ValueError("the current source-field AC workflow supports active_band=0")
        return self


class ConjugateACBiasSweepSummary(BaseModel):
    """Scalar summary for one conjugate-AC bias sweep."""

    model_config = ConfigDict(frozen=True)

    sweep_parameter: Literal["u1_c3", "b1_c3"]
    n_bias: int
    cG_min: float
    cG_max: float
    max_dispersion_split: float
    gap_min: float
    projection: str


class ChargeResponseSummary(BaseModel):
    """Small scalar summary for response outputs."""

    model_config = ConfigDict(frozen=True)

    cG: float
    cG_dimension: Literal["dimensionless"] = "dimensionless"
    kappa_min: float
    kappa_max: float
    gap_min: float | None = None
    valid_local_gap: bool = True


__all__ = [
    "ACConventionParams",
    "ACProjectedHFParams",
    "ACResponseWorkflowParams",
    "ChargeResponseSummary",
    "ConjugateACBiasSweepParams",
    "ConjugateACBiasSweepSummary",
    "ContinuumGridParams",
    "ContinuumHFParams",
    "ContinuumInteractionParams",
    "ContinuumModelParams",
    "ContinuumWorkflowParams",
    "DomainWallParams",
    "FirstShellACParams",
    "FourierACParams",
    "FourierCoefficient",
    "GatedInteractionParams",
    "IdealConjugateLLLChargeBenchmarkParams",
    "IdealConjugateLLLChargeSummary",
    "M0SourceScanParams",
    "MomentumGridParams",
    "PhysicalCoulombACPreset",
    "QHFMChargeBenchmarkParams",
    "QHFMChargeSummary",
    "RealSpaceGridParams",
    "ResponseParams",
    "RunArtifact",
    "RunManifest",
    "SkyrmionTextureParams",
    "SourceInterpolationParams",
    "TMoTe2ACParams",
    "UnitsParams",
]
