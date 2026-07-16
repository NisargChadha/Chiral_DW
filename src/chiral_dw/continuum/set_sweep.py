"""Taige scanning-SET point workflow near hole filling one."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from chiral_dw.config import (
    ContinuumFiniteQParams,
    ContinuumGridParams,
    ContinuumHFParams,
    ContinuumInteractionParams,
    ContinuumModelParams,
)
from chiral_dw.continuum.builder import build_continuum_bundle
from chiral_dw.continuum.hf import retarget_global_density, solve_global_hf
from chiral_dw.continuum.hf_bands import hf_band_chern_table, hf_mesh_band_data
from chiral_dw.continuum.models import ContinuumBundle, ContinuumHFResult
from chiral_dw.continuum.references import solve_reference_hf
from chiral_dw.continuum.set_signal import (
    HFBandValiditySummary,
    SETChemicalPotentialRow,
    SETFillingEnergyRow,
    SETGapSummary,
    SETInverseCompressibilityRow,
    chemical_potential_rows,
    gaussian_dos,
    hf_band_validity_summary,
    inverse_compressibility_rows,
    set_gap_summary,
)
from chiral_dw.continuum.symmetry import ValleyU1Constraint
from chiral_dw.continuum.taige import MoireGeometry


class TaigeSETWorkflowParams(BaseModel):
    """Frozen controls for one homogeneous Taige scanning-SET calculation."""

    model_config = ConfigDict(frozen=True)

    model: ContinuumModelParams
    grid: ContinuumGridParams = Field(default_factory=lambda: ContinuumGridParams(n_k=18))
    interaction: ContinuumInteractionParams
    hf: ContinuumHFParams
    particle_offsets: tuple[int, ...] = tuple(range(-12, 13))
    temperature_kbt_mev: float = Field(default=0.0, ge=0.0)
    gaussian_broadening_mev: float = Field(default=0.1, gt=0.0)
    dos_energy_points: int = Field(default=801, ge=101)
    direct_gap_tolerance_mev: float = Field(default=1e-6, ge=0.0)
    indirect_gap_tolerance_mev: float = 0.0

    @model_validator(mode="after")
    def _validate_particle_offsets(self) -> "TaigeSETWorkflowParams":
        offsets = tuple(sorted(set(int(value) for value in self.particle_offsets)))
        if offsets != self.particle_offsets:
            raise ValueError("particle_offsets must be sorted and unique")
        if not {-1, 0, 1}.issubset(offsets):
            raise ValueError("particle_offsets must contain -1, 0, and 1")
        if offsets != tuple(range(offsets[0], offsets[-1] + 1)):
            raise ValueError("particle_offsets must be consecutive")
        if self.temperature_kbt_mev != 0.0:
            raise ValueError("the current SET workflow implements zero-temperature HF only")
        return self


class FixedVPReferenceSummary(BaseModel):
    """One fixed-per-k VP reference used for topology diagnostics."""

    model_config = ConfigDict(frozen=True)

    reference: str
    energy_total_mev: float
    energy_per_cell_mev: float
    converged: bool
    n_iter: int
    direct_gap_mev: float
    indirect_gap_mev: float
    aufbau_residual_norm: float
    self_consistency_warning: bool


class TaigeSETPointSummary(BaseModel):
    """Scalar summary for one displacement-field SET point."""

    model_config = ConfigDict(frozen=True)

    material: str
    theta_deg: float
    u_D_mev: float
    n_k: int
    n_cells: int
    n_fillings: int
    selected_fixed_vp_reference: str
    fixed_vp_plus: FixedVPReferenceSummary
    fixed_vp_minus: FixedVPReferenceSummary
    hf_band_chern: float
    band_validity: HFBandValiditySummary
    set_gap: SETGapSummary
    all_global_fillings_converged: bool
    n_unconverged_global_fillings: int
    global_filling_one_energy_per_cell_mev: float
    global_filling_one_direct_gap_mev: float
    global_filling_one_indirect_gap_mev: float
    gaussian_broadening_mev: float
    temperature_kbt_mev: float
    uniform_capacitance_convention: str = (
        "native HF omits uniform Hartree; raw SET adds it in postprocessing"
    )

    def as_csv_row(self) -> dict[str, Any]:
        """Return a flat row suitable for displacement-sweep CSV output."""

        return {
            "material": self.material,
            "theta_deg": self.theta_deg,
            "u_D_meV": self.u_D_mev,
            "n_k": self.n_k,
            "n_cells": self.n_cells,
            "n_fillings": self.n_fillings,
            "selected_fixed_vp_reference": self.selected_fixed_vp_reference,
            "hf_band_chern": self.hf_band_chern,
            "hf_chern_interpretable": self.band_validity.chern_physically_interpretable,
            "fixed_per_k_valid_insulator": self.band_validity.valid_fixed_per_k_insulator,
            "fixed_per_k_invalid_reason": self.band_validity.invalid_reason,
            "fixed_direct_gap_meV": self.band_validity.direct_gap_mev,
            "fixed_indirect_gap_meV": self.band_validity.indirect_gap_mev,
            "charge_gap_raw_meV": self.set_gap.charge_gap_raw_mev,
            "charge_gap_intrinsic_meV": self.set_gap.charge_gap_intrinsic_mev,
            "mu_minus_hole_raw_meV": self.set_gap.mu_minus_hole_raw_mev,
            "mu_plus_hole_raw_meV": self.set_gap.mu_plus_hole_raw_mev,
            "mu_minus_hole_intrinsic_meV": self.set_gap.mu_minus_hole_intrinsic_mev,
            "mu_plus_hole_intrinsic_meV": self.set_gap.mu_plus_hole_intrinsic_mev,
            "all_global_fillings_converged": self.all_global_fillings_converged,
            "n_unconverged_global_fillings": self.n_unconverged_global_fillings,
            "global_filling_one_energy_per_cell_meV": (
                self.global_filling_one_energy_per_cell_mev
            ),
            "global_filling_one_direct_gap_meV": self.global_filling_one_direct_gap_mev,
            "global_filling_one_indirect_gap_meV": self.global_filling_one_indirect_gap_mev,
            "gaussian_broadening_meV": self.gaussian_broadening_mev,
            "temperature_kBT_meV": self.temperature_kbt_mev,
        }


@dataclass(frozen=True)
class TaigeSETPointResult:
    """In-memory SET point outputs and restartable HF arrays."""

    params: TaigeSETWorkflowParams
    summary: TaigeSETPointSummary
    bundle: ContinuumBundle
    fixed_vp_plus: ContinuumHFResult
    fixed_vp_minus: ContinuumHFResult
    filling_results: dict[int, ContinuumHFResult]
    filling_energy_rows: tuple[SETFillingEnergyRow, ...]
    chemical_potential_rows: tuple[SETChemicalPotentialRow, ...]
    inverse_compressibility_rows: tuple[SETInverseCompressibilityRow, ...]
    hf_chern_rows: tuple[dict[str, Any], ...]
    dos_rows: tuple[dict[str, float], ...]


def _fixed_reference_summary(
    name: str,
    result: ContinuumHFResult,
    *,
    n_cells: int,
) -> FixedVPReferenceSummary:
    diag = result.diagnostics
    return FixedVPReferenceSummary(
        reference=name,
        energy_total_mev=float(result.energy),
        energy_per_cell_mev=float(result.energy / n_cells),
        converged=bool(result.converged),
        n_iter=int(result.n_iter),
        direct_gap_mev=float(diag.direct_gap_min),
        indirect_gap_mev=float(diag.indirect_gap),
        aufbau_residual_norm=float(diag.aufbau_residual_norm),
        self_consistency_warning=bool(diag.self_consistency_warning),
    )


def _select_fixed_reference(
    vp_plus: ContinuumHFResult,
    vp_minus: ContinuumHFResult,
) -> tuple[str, ContinuumHFResult]:
    candidates = [("VP+", vp_plus), ("VP-", vp_minus)]
    clean = [item for item in candidates if item[1].converged]
    pool = clean or candidates
    return min(pool, key=lambda item: float(item[1].energy))


def _solve_global_filling_sequence(
    bundle: ContinuumBundle,
    selected: ContinuumHFResult,
    params: TaigeSETWorkflowParams,
) -> dict[int, ContinuumHFResult]:
    constraint = ValleyU1Constraint(bundle.active)
    n0 = int(bundle.grid.size)
    target_numbers = tuple(n0 + offset for offset in params.particle_offsets)
    if target_numbers[0] < 0 or target_numbers[-1] > bundle.backend.n_total:
        raise ValueError("particle offset window lies outside the active Hilbert space")

    center_seed = retarget_global_density(
        bundle.backend,
        selected.P,
        n0,
        constraint=constraint,
    )
    center = solve_global_hf(
        bundle.backend,
        center_seed,
        n0,
        params.hf,
        constraint=constraint,
        seed="fixed_vp_to_global_N0",
    )
    results: dict[int, ContinuumHFResult] = {n0: center}

    previous = center
    for target in range(n0 + 1, target_numbers[-1] + 1):
        seed = retarget_global_density(
            bundle.backend,
            previous.P,
            target,
            constraint=constraint,
        )
        previous = solve_global_hf(
            bundle.backend,
            seed,
            target,
            params.hf,
            constraint=constraint,
            seed=f"warm_up_N{target}",
        )
        results[target] = previous

    previous = center
    for target in range(n0 - 1, target_numbers[0] - 1, -1):
        seed = retarget_global_density(
            bundle.backend,
            previous.P,
            target,
            constraint=constraint,
        )
        previous = solve_global_hf(
            bundle.backend,
            seed,
            target,
            params.hf,
            constraint=constraint,
            seed=f"warm_down_N{target}",
        )
        results[target] = previous
    return dict(sorted(results.items()))


def _filling_energy_rows(
    bundle: ContinuumBundle,
    results: dict[int, ContinuumHFResult],
) -> tuple[SETFillingEnergyRow, ...]:
    n_cells = int(bundle.grid.size)
    rows: list[SETFillingEnergyRow] = []
    for n_particles, result in sorted(results.items()):
        components = bundle.backend.energy(result.P)
        uniform = bundle.backend.uniform_hartree_energy(result.P - bundle.backend.p_ref)
        intrinsic = float(components.total)
        raw = float(intrinsic + uniform)
        diag = result.diagnostics
        rows.append(
            SETFillingEnergyRow(
                n_particles=int(n_particles),
                filling_holes=n_particles / float(n_cells),
                energy_total_mev=raw,
                energy_per_cell_mev=raw / n_cells,
                uniform_hartree_energy_mev=float(uniform),
                intrinsic_energy_total_mev=intrinsic,
                intrinsic_energy_per_cell_mev=intrinsic / n_cells,
                one_body_energy_mev=float(components.one_body),
                hartree_energy_mev=float(components.hartree),
                fock_energy_mev=float(components.fock),
                converged=bool(result.converged),
                n_iter=int(result.n_iter),
                trace_error=float(diag.trace_error),
                aufbau_residual_norm=float(diag.aufbau_residual_norm),
                commutator_norm=float(diag.commutator_norm),
            )
        )
    return tuple(rows)


def _dos_rows(
    bundle: ContinuumBundle,
    fixed_result: ContinuumHFResult,
    validity: HFBandValiditySummary,
    params: TaigeSETWorkflowParams,
) -> tuple[dict[str, float], ...]:
    mesh = hf_mesh_band_data(bundle.active, fixed_result.H_hf)
    n_occ = int(params.hf.n_occ_per_k)
    valence_max = float(np.max(mesh.energies[:, n_occ - 1]))
    conduction_min = float(np.min(mesh.energies[:, n_occ]))
    energy_zero = 0.5 * (valence_max + conduction_min)
    sigma = float(params.gaussian_broadening_mev)
    energy = np.linspace(
        float(np.min(mesh.energies) - energy_zero - 5.0 * sigma),
        float(np.max(mesh.energies) - energy_zero + 5.0 * sigma),
        int(params.dos_energy_points),
    )
    dos = gaussian_dos(mesh.energies - energy_zero, energy, sigma_mev=sigma)
    return tuple(
        {
            "energy_relative_midgap_meV": float(value),
            "dos_per_cell_per_meV": float(weight),
            "energy_zero_meV": float(energy_zero),
            "fixed_direct_gap_meV": float(validity.direct_gap_mev),
            "fixed_indirect_gap_meV": float(validity.indirect_gap_mev),
        }
        for value, weight in zip(energy, dos)
    )


def run_taige_set_point(params: TaigeSETWorkflowParams) -> TaigeSETPointResult:
    """Run one displacement-field SET point through fixed and global HF paths."""

    bundle = build_continuum_bundle(
        model=params.model,
        grid=params.grid,
        finite_q=ContinuumFiniteQParams(enabled=False),
        interaction=params.interaction,
    )
    constraint = ValleyU1Constraint(bundle.active)
    vp_plus = solve_reference_hf(bundle, "vp_plus", params.hf, constraint=constraint)
    vp_minus = solve_reference_hf(bundle, "vp_minus", params.hf, constraint=constraint)
    selected_name, selected = _select_fixed_reference(vp_plus, vp_minus)
    validity = hf_band_validity_summary(
        vp_plus.H_hf,
        n_occ_per_k=params.hf.n_occ_per_k,
        direct_gap_tolerance_mev=params.direct_gap_tolerance_mev,
        indirect_gap_tolerance_mev=params.indirect_gap_tolerance_mev,
    )
    vp_plus_chern = hf_band_chern_table(
        bundle.active,
        vp_plus.H_hf,
        reference="VP+",
    )
    vp_minus_chern = hf_band_chern_table(
        bundle.active,
        vp_minus.H_hf,
        reference="VP-",
    )
    chern_rows = tuple(
        row.model_dump(mode="json") for row in (*vp_plus_chern, *vp_minus_chern)
    )
    occupied_chern = float(vp_plus_chern[params.hf.n_occ_per_k - 1].chern)

    filling_results = _solve_global_filling_sequence(bundle, selected, params)
    energy_rows = _filling_energy_rows(bundle, filling_results)
    n_cells = int(bundle.grid.size)
    geometry = bundle.geometry if isinstance(bundle.geometry, MoireGeometry) else MoireGeometry(params.model)
    mu_rows = chemical_potential_rows(energy_rows, n_cells=n_cells)
    kappa_rows = inverse_compressibility_rows(
        energy_rows,
        n_cells=n_cells,
        moire_cell_area_nm2=float(geometry.moire_cell_area_nm2),
    )
    gap = set_gap_summary(energy_rows, n_particles_filling_one=n_cells)
    center = filling_results[n_cells]
    n_unconverged = sum(not result.converged for result in filling_results.values())
    summary = TaigeSETPointSummary(
        material=params.model.material,
        theta_deg=float(params.model.theta_deg),
        u_D_mev=float(params.model.displacement_mev),
        n_k=int(params.grid.n_k),
        n_cells=n_cells,
        n_fillings=len(filling_results),
        selected_fixed_vp_reference=selected_name,
        fixed_vp_plus=_fixed_reference_summary("VP+", vp_plus, n_cells=n_cells),
        fixed_vp_minus=_fixed_reference_summary("VP-", vp_minus, n_cells=n_cells),
        hf_band_chern=occupied_chern,
        band_validity=validity,
        set_gap=gap,
        all_global_fillings_converged=n_unconverged == 0,
        n_unconverged_global_fillings=n_unconverged,
        global_filling_one_energy_per_cell_mev=float(center.energy / n_cells),
        global_filling_one_direct_gap_mev=float(center.diagnostics.direct_gap_min),
        global_filling_one_indirect_gap_mev=float(center.diagnostics.indirect_gap),
        gaussian_broadening_mev=float(params.gaussian_broadening_mev),
        temperature_kbt_mev=float(params.temperature_kbt_mev),
    )
    return TaigeSETPointResult(
        params=params,
        summary=summary,
        bundle=bundle,
        fixed_vp_plus=vp_plus,
        fixed_vp_minus=vp_minus,
        filling_results=filling_results,
        filling_energy_rows=energy_rows,
        chemical_potential_rows=mu_rows,
        inverse_compressibility_rows=kappa_rows,
        hf_chern_rows=chern_rows,
        dos_rows=_dos_rows(bundle, vp_plus, validity, params),
    )


__all__ = [
    "FixedVPReferenceSummary",
    "TaigeSETPointResult",
    "TaigeSETPointSummary",
    "TaigeSETWorkflowParams",
    "run_taige_set_point",
]
