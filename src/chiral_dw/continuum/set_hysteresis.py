"""Branch-resolved HF continuation for scanning-SET thermodynamics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from chiral_dw.config import ContinuumFiniteQParams
from chiral_dw.continuum.builder import build_continuum_bundle
from chiral_dw.continuum.hf import retarget_global_density, solve_global_hf
from chiral_dw.continuum.hf_bands import hf_band_chern_table
from chiral_dw.continuum.models import ContinuumBundle, ContinuumHFResult
from chiral_dw.continuum.set_signal import (
    HFBandValiditySummary,
    SETFillingEnergyRow,
    SETGapSummary,
    hf_band_validity_summary,
    set_gap_summary,
)
from chiral_dw.continuum.set_sweep import TaigeSETWorkflowParams
from chiral_dw.continuum.symmetry import ValleyU1Constraint


SETHysteresisDirection = Literal["up", "down"]


class SETBranchTopologySummary(BaseModel):
    """Neutral-filling topology diagnostic on one continuation branch."""

    model_config = ConfigDict(frozen=True)

    direction: SETHysteresisDirection
    hf_band_chern: float
    band_validity: HFBandValiditySummary
    one_state_per_k_max_error: float = Field(ge=0.0)
    chern_physically_interpretable: bool


class SETBranchPointSummary(BaseModel):
    """One displacement point on one SET hysteresis branch."""

    model_config = ConfigDict(frozen=True)

    direction: SETHysteresisDirection
    u_D_mev: float
    n_cells: int = Field(ge=1)
    filling_energy_rows: tuple[SETFillingEnergyRow, ...]
    neutral_topology: SETBranchTopologySummary
    all_fillings_converged: bool
    n_unconverged_fillings: int = Field(ge=0)


class SETHysteresisEnvelope(BaseModel):
    """Lower-energy envelope selected independently at every particle number."""

    model_config = ConfigDict(frozen=True)

    n_particles_filling_one: int = Field(ge=1)
    selected_energy_rows: tuple[SETFillingEnergyRow, ...]
    selected_direction_by_particles: dict[int, SETHysteresisDirection]
    down_minus_up_intrinsic_energy_mev: dict[int, float]
    set_gap: SETGapSummary


@dataclass(frozen=True)
class SETBranchPointResult:
    """In-memory branch point with restartable HF states."""

    params: TaigeSETWorkflowParams
    bundle: ContinuumBundle
    summary: SETBranchPointSummary
    filling_results: dict[int, ContinuumHFResult]


def set_filling_energy_row(
    bundle: ContinuumBundle,
    n_particles: int,
    result: ContinuumHFResult,
) -> SETFillingEnergyRow:
    """Convert one branch-resolved HF state into a SET energy row."""

    n_cells = int(bundle.grid.size)
    components = bundle.backend.energy(result.P)
    uniform = bundle.backend.uniform_hartree_energy(result.P - bundle.backend.p_ref)
    intrinsic = float(components.total)
    raw = float(intrinsic + uniform)
    diag = result.diagnostics
    return SETFillingEnergyRow(
        n_particles=int(n_particles),
        filling_holes=float(n_particles) / float(n_cells),
        energy_total_mev=raw,
        energy_per_cell_mev=raw / float(n_cells),
        uniform_hartree_energy_mev=float(uniform),
        intrinsic_energy_total_mev=intrinsic,
        intrinsic_energy_per_cell_mev=intrinsic / float(n_cells),
        one_body_energy_mev=float(components.one_body),
        hartree_energy_mev=float(components.hartree),
        fock_energy_mev=float(components.fock),
        converged=bool(result.converged),
        n_iter=int(result.n_iter),
        trace_error=float(diag.trace_error),
        aufbau_residual_norm=float(diag.aufbau_residual_norm),
        commutator_norm=float(diag.commutator_norm),
    )


def run_taige_set_hysteresis_branch_point(
    params: TaigeSETWorkflowParams,
    initial_projectors: Mapping[int, np.ndarray],
    *,
    direction: SETHysteresisDirection,
) -> SETBranchPointResult:
    """Continue every requested particle number into one displacement point."""

    bundle = build_continuum_bundle(
        model=params.model,
        grid=params.grid,
        finite_q=ContinuumFiniteQParams(enabled=False),
        interaction=params.interaction,
    )
    constraint = ValleyU1Constraint(bundle.active)
    n_cells = int(bundle.grid.size)
    targets = tuple(n_cells + int(offset) for offset in params.particle_offsets)
    if set(initial_projectors) != set(targets):
        raise ValueError(
            "initial_projectors keys must exactly match the requested particle numbers"
        )

    results: dict[int, ContinuumHFResult] = {}
    for target in targets:
        seed = bundle.backend.as_block_density(initial_projectors[target])
        trace = float(np.real(np.trace(seed, axis1=-2, axis2=-1).sum()))
        if abs(trace - float(target)) > params.hf.tolerance:
            seed = retarget_global_density(
                bundle.backend,
                seed,
                target,
                constraint=constraint,
            )
        results[target] = solve_global_hf(
            bundle.backend,
            seed,
            target,
            params.hf,
            constraint=constraint,
            seed=f"set_hysteresis_{direction}_N{target}",
        )

    rows = tuple(
        set_filling_energy_row(bundle, target, results[target]) for target in targets
    )
    center = results[n_cells]
    per_k_trace = np.trace(center.P, axis1=-2, axis2=-1).real
    one_per_k_error = float(np.max(np.abs(per_k_trace - params.hf.n_occ_per_k)))
    validity = hf_band_validity_summary(
        center.H_hf,
        n_occ_per_k=params.hf.n_occ_per_k,
        direct_gap_tolerance_mev=params.direct_gap_tolerance_mev,
        indirect_gap_tolerance_mev=params.indirect_gap_tolerance_mev,
    )
    chern_rows = hf_band_chern_table(
        bundle.active,
        center.H_hf,
        reference=f"SET {direction}",
    )
    occupied_chern = float(chern_rows[params.hf.n_occ_per_k - 1].chern)
    topology = SETBranchTopologySummary(
        direction=direction,
        hf_band_chern=occupied_chern,
        band_validity=validity,
        one_state_per_k_max_error=one_per_k_error,
        chern_physically_interpretable=bool(
            validity.chern_physically_interpretable and one_per_k_error <= 1e-6
        ),
    )
    n_unconverged = sum(not result.converged for result in results.values())
    summary = SETBranchPointSummary(
        direction=direction,
        u_D_mev=float(params.model.displacement_mev),
        n_cells=n_cells,
        filling_energy_rows=rows,
        neutral_topology=topology,
        all_fillings_converged=n_unconverged == 0,
        n_unconverged_fillings=n_unconverged,
    )
    return SETBranchPointResult(
        params=params,
        bundle=bundle,
        summary=summary,
        filling_results=results,
    )


def select_set_hysteresis_envelope(
    up_rows: Sequence[SETFillingEnergyRow],
    down_rows: Sequence[SETFillingEnergyRow],
    *,
    n_particles_filling_one: int,
) -> SETHysteresisEnvelope:
    """Select the clean lower-energy branch separately for every filling."""

    by_direction = {
        "up": {row.n_particles: row for row in up_rows},
        "down": {row.n_particles: row for row in down_rows},
    }
    particle_numbers = sorted(by_direction["up"])
    if particle_numbers != sorted(by_direction["down"]):
        raise ValueError("up/down branches must contain the same particle numbers")
    if not particle_numbers:
        raise ValueError("hysteresis envelope needs at least one energy row")

    selected: list[SETFillingEnergyRow] = []
    selected_directions: dict[int, SETHysteresisDirection] = {}
    energy_differences: dict[int, float] = {}
    for n_particles in particle_numbers:
        up = by_direction["up"][n_particles]
        down = by_direction["down"][n_particles]
        energy_differences[n_particles] = float(
            down.intrinsic_energy_total_mev - up.intrinsic_energy_total_mev
        )
        candidates: list[tuple[SETHysteresisDirection, SETFillingEnergyRow]] = [
            ("up", up),
            ("down", down),
        ]
        clean = [item for item in candidates if item[1].converged]
        if not clean:
            raise ValueError(f"both hysteresis branches are unconverged at N={n_particles}")
        direction, row = min(
            clean,
            key=lambda item: (
                float(item[1].intrinsic_energy_total_mev),
                0 if item[0] == "up" else 1,
            ),
        )
        selected.append(row)
        selected_directions[n_particles] = direction

    selected_rows = tuple(sorted(selected, key=lambda row: row.n_particles))
    gap = set_gap_summary(
        selected_rows,
        n_particles_filling_one=int(n_particles_filling_one),
    )
    return SETHysteresisEnvelope(
        n_particles_filling_one=int(n_particles_filling_one),
        selected_energy_rows=selected_rows,
        selected_direction_by_particles=selected_directions,
        down_minus_up_intrinsic_energy_mev=energy_differences,
        set_gap=gap,
    )


__all__ = [
    "SETBranchPointResult",
    "SETBranchPointSummary",
    "SETBranchTopologySummary",
    "SETHysteresisDirection",
    "SETHysteresisEnvelope",
    "run_taige_set_hysteresis_branch_point",
    "select_set_hysteresis_envelope",
    "set_filling_energy_row",
]
