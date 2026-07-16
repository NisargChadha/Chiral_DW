"""Homogeneous scanning-SET thermodynamics from continuum HF energies."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class SETFillingEnergyRow(BaseModel):
    """One relaxed HF energy at a fixed total hole number."""

    model_config = ConfigDict(frozen=True)

    n_particles: int = Field(ge=0)
    filling_holes: float
    energy_total_mev: float
    energy_per_cell_mev: float
    uniform_hartree_energy_mev: float = 0.0
    intrinsic_energy_total_mev: float
    intrinsic_energy_per_cell_mev: float
    one_body_energy_mev: float
    hartree_energy_mev: float
    fock_energy_mev: float
    converged: bool
    n_iter: int = Field(ge=0)
    trace_error: float = Field(ge=0.0)
    aufbau_residual_norm: float = Field(ge=0.0)
    commutator_norm: float = Field(ge=0.0)


class SETChemicalPotentialRow(BaseModel):
    """Addition chemical potential between neighboring particle numbers."""

    model_config = ConfigDict(frozen=True)

    n_lower: int
    n_upper: int
    filling_midpoint_holes: float
    mu_hole_raw_mev: float
    mu_hole_intrinsic_mev: float
    mu_electron_raw_mev: float
    mu_electron_intrinsic_mev: float


class SETInverseCompressibilityRow(BaseModel):
    """Centered inverse-compressibility estimate at one particle number."""

    model_config = ConfigDict(frozen=True)

    n_particles: int
    filling_holes: float
    dmu_dnu_raw_mev: float
    dmu_dnu_intrinsic_mev: float
    dmu_dn_raw_mev_nm2: float
    dmu_dn_intrinsic_mev_nm2: float


class SETGapSummary(BaseModel):
    """Addition/removal chemical potentials and charge gap at filling one."""

    model_config = ConfigDict(frozen=True)

    n_particles_filling_one: int
    mu_minus_hole_raw_mev: float
    mu_plus_hole_raw_mev: float
    charge_gap_raw_mev: float
    midgap_hole_raw_mev: float
    mu_minus_hole_intrinsic_mev: float
    mu_plus_hole_intrinsic_mev: float
    charge_gap_intrinsic_mev: float
    midgap_hole_intrinsic_mev: float


class HFBandValiditySummary(BaseModel):
    """Fixed-per-k HF-band gap checks used to interpret a band Chern number."""

    model_config = ConfigDict(frozen=True)

    n_occ_per_k: int = Field(ge=1)
    direct_gap_mev: float
    indirect_gap_mev: float
    direct_gap_k_index: int = Field(ge=0)
    occupied_band_max_k_index: int = Field(ge=0)
    empty_band_min_k_index: int = Field(ge=0)
    valid_fixed_per_k_insulator: bool
    chern_resolved_by_direct_gap: bool
    chern_physically_interpretable: bool
    invalid_reason: str | None = None


def hf_band_validity_summary(
    H_blocks: np.ndarray,
    *,
    n_occ_per_k: int = 1,
    direct_gap_tolerance_mev: float = 1e-6,
    indirect_gap_tolerance_mev: float = 0.0,
) -> HFBandValiditySummary:
    """Return direct/indirect gaps and fixed-per-k validity flags."""

    H = np.asarray(H_blocks, dtype=complex)
    if H.ndim != 3 or H.shape[-1] != H.shape[-2]:
        raise ValueError("H_blocks must have shape (n_k, dim, dim)")
    n_occ = int(n_occ_per_k)
    if n_occ < 1 or n_occ >= H.shape[-1]:
        raise ValueError("n_occ_per_k must leave at least one empty band")
    evals = np.linalg.eigvalsh(0.5 * (H + H.conj().swapaxes(-1, -2)))
    direct_by_k = evals[:, n_occ] - evals[:, n_occ - 1]
    occupied = evals[:, n_occ - 1]
    empty = evals[:, n_occ]
    direct_idx = int(np.argmin(direct_by_k))
    occupied_idx = int(np.argmax(occupied))
    empty_idx = int(np.argmin(empty))
    direct = float(direct_by_k[direct_idx])
    indirect = float(empty[empty_idx] - occupied[occupied_idx])
    valid_insulator = bool(indirect > float(indirect_gap_tolerance_mev))
    chern_resolved = bool(direct > float(direct_gap_tolerance_mev))
    if not valid_insulator:
        reason = "nonpositive_indirect_gap"
    elif not chern_resolved:
        reason = "direct_gap_unresolved"
    else:
        reason = None
    return HFBandValiditySummary(
        n_occ_per_k=n_occ,
        direct_gap_mev=direct,
        indirect_gap_mev=indirect,
        direct_gap_k_index=direct_idx,
        occupied_band_max_k_index=occupied_idx,
        empty_band_min_k_index=empty_idx,
        valid_fixed_per_k_insulator=valid_insulator,
        chern_resolved_by_direct_gap=chern_resolved,
        chern_physically_interpretable=bool(valid_insulator and chern_resolved),
        invalid_reason=reason,
    )


def chemical_potential_rows(
    energies: list[SETFillingEnergyRow] | tuple[SETFillingEnergyRow, ...],
    *,
    n_cells: int,
) -> tuple[SETChemicalPotentialRow, ...]:
    """Return neighboring-particle chemical potentials in hole/electron conventions."""

    rows = sorted(energies, key=lambda row: row.n_particles)
    out: list[SETChemicalPotentialRow] = []
    for lower, upper in zip(rows[:-1], rows[1:]):
        delta_n = int(upper.n_particles - lower.n_particles)
        if delta_n != 1:
            raise ValueError("SET chemical potentials require consecutive integer particle numbers")
        mu_raw = float(upper.energy_total_mev - lower.energy_total_mev)
        mu_intrinsic = float(
            upper.intrinsic_energy_total_mev - lower.intrinsic_energy_total_mev
        )
        out.append(
            SETChemicalPotentialRow(
                n_lower=lower.n_particles,
                n_upper=upper.n_particles,
                filling_midpoint_holes=(lower.n_particles + 0.5) / float(n_cells),
                mu_hole_raw_mev=mu_raw,
                mu_hole_intrinsic_mev=mu_intrinsic,
                mu_electron_raw_mev=-mu_raw,
                mu_electron_intrinsic_mev=-mu_intrinsic,
            )
        )
    return tuple(out)


def inverse_compressibility_rows(
    energies: list[SETFillingEnergyRow] | tuple[SETFillingEnergyRow, ...],
    *,
    n_cells: int,
    moire_cell_area_nm2: float,
) -> tuple[SETInverseCompressibilityRow, ...]:
    """Return centered dmu/dnu and dmu/dn from consecutive integer energies."""

    rows = sorted(energies, key=lambda row: row.n_particles)
    out: list[SETInverseCompressibilityRow] = []
    for lower, center, upper in zip(rows[:-2], rows[1:-1], rows[2:]):
        if not (
            center.n_particles == lower.n_particles + 1
            and upper.n_particles == center.n_particles + 1
        ):
            raise ValueError("inverse compressibility requires consecutive particle numbers")
        second_raw = (
            upper.energy_total_mev
            - 2.0 * center.energy_total_mev
            + lower.energy_total_mev
        )
        second_intrinsic = (
            upper.intrinsic_energy_total_mev
            - 2.0 * center.intrinsic_energy_total_mev
            + lower.intrinsic_energy_total_mev
        )
        dmu_dnu_raw = float(n_cells * second_raw)
        dmu_dnu_intrinsic = float(n_cells * second_intrinsic)
        out.append(
            SETInverseCompressibilityRow(
                n_particles=center.n_particles,
                filling_holes=center.n_particles / float(n_cells),
                dmu_dnu_raw_mev=dmu_dnu_raw,
                dmu_dnu_intrinsic_mev=dmu_dnu_intrinsic,
                dmu_dn_raw_mev_nm2=float(moire_cell_area_nm2 * dmu_dnu_raw),
                dmu_dn_intrinsic_mev_nm2=float(
                    moire_cell_area_nm2 * dmu_dnu_intrinsic
                ),
            )
        )
    return tuple(out)


def set_gap_summary(
    energies: list[SETFillingEnergyRow] | tuple[SETFillingEnergyRow, ...],
    *,
    n_particles_filling_one: int,
) -> SETGapSummary:
    """Return the relaxed thermodynamic charge gap at filling one."""

    by_n = {row.n_particles: row for row in energies}
    n0 = int(n_particles_filling_one)
    try:
        lower, center, upper = by_n[n0 - 1], by_n[n0], by_n[n0 + 1]
    except KeyError as exc:
        raise ValueError("SET gap needs N0-1, N0, and N0+1 energies") from exc
    mu_minus_raw = float(center.energy_total_mev - lower.energy_total_mev)
    mu_plus_raw = float(upper.energy_total_mev - center.energy_total_mev)
    mu_minus_intrinsic = float(
        center.intrinsic_energy_total_mev - lower.intrinsic_energy_total_mev
    )
    mu_plus_intrinsic = float(
        upper.intrinsic_energy_total_mev - center.intrinsic_energy_total_mev
    )
    return SETGapSummary(
        n_particles_filling_one=n0,
        mu_minus_hole_raw_mev=mu_minus_raw,
        mu_plus_hole_raw_mev=mu_plus_raw,
        charge_gap_raw_mev=mu_plus_raw - mu_minus_raw,
        midgap_hole_raw_mev=0.5 * (mu_plus_raw + mu_minus_raw),
        mu_minus_hole_intrinsic_mev=mu_minus_intrinsic,
        mu_plus_hole_intrinsic_mev=mu_plus_intrinsic,
        charge_gap_intrinsic_mev=mu_plus_intrinsic - mu_minus_intrinsic,
        midgap_hole_intrinsic_mev=0.5 * (mu_plus_intrinsic + mu_minus_intrinsic),
    )


def gaussian_dos(
    eigenvalues_mev: np.ndarray,
    energy_grid_mev: np.ndarray,
    *,
    sigma_mev: float,
) -> np.ndarray:
    """Return a normalized Gaussian-broadened DOS per momentum block."""

    eigenvalues = np.asarray(eigenvalues_mev, dtype=float)
    energy = np.asarray(energy_grid_mev, dtype=float)
    sigma = float(sigma_mev)
    if eigenvalues.ndim != 2:
        raise ValueError("eigenvalues_mev must have shape (n_k, n_bands)")
    if energy.ndim != 1:
        raise ValueError("energy_grid_mev must be one-dimensional")
    if sigma <= 0.0:
        raise ValueError("sigma_mev must be positive")
    x = (energy[:, None, None] - eigenvalues[None, :, :]) / sigma
    kernel = np.exp(-0.5 * x * x) / (np.sqrt(2.0 * np.pi) * sigma)
    return np.sum(kernel, axis=(1, 2)) / float(eigenvalues.shape[0])


__all__ = [
    "HFBandValiditySummary",
    "SETChemicalPotentialRow",
    "SETFillingEnergyRow",
    "SETGapSummary",
    "SETInverseCompressibilityRow",
    "chemical_potential_rows",
    "gaussian_dos",
    "hf_band_validity_summary",
    "inverse_compressibility_rows",
    "set_gap_summary",
]
