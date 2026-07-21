"""Frozen-remote and enlarged-HF orbital-magnetization workflow helpers."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import tempfile
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from chiral_dw.config import (
    ContinuumFiniteQParams,
    ContinuumModelParams,
    TaigeOrbitalMagnetizationWorkflowParams,
)
from chiral_dw.continuum.models import (
    ContinuumActiveSpace,
    ContinuumBundle,
    ContinuumHFDiagnostics,
    ContinuumHFResult,
    MomentumGrid,
)
from chiral_dw.continuum.observables import active_basis_frames, valley_polarization
from chiral_dw.continuum.orbital_magnetization import (
    HBAR2_OVER_2ME_MEV_NM2,
    evaluate_projector_orbital_magnetization,
)
from chiral_dw.continuum.taige import (
    MoireGeometry,
    TaigeBandStructure,
    TaigeContinuumModel,
    VALLEY_ORDER,
    active_space_from_taige_bands,
    build_taige_density_vertices,
    chern_number_on_grid,
    compute_taige_bandstructure,
)
from chiral_dw.continuum.taige_sewing import TaigeReciprocalTransport

TAIGE_BAND_CACHE_SCHEMA = "taige_orbital_band_cache_v1"


class TaigeBandCacheManifest(BaseModel):
    """Versioned cache metadata for a maximum-cutoff continuum eigensystem."""

    model_config = ConfigDict(frozen=True)

    schema_name: str = TAIGE_BAND_CACHE_SCHEMA
    cache_hash: str
    path: str
    signature: dict[str, Any]
    n_k: int = Field(ge=1)
    n_bands: int = Field(ge=1)
    microscopic_valley_dim: int = Field(ge=1)


class FrozenEmbeddingDiagnostics(BaseModel):
    """Common-basis residuals for one frozen-band cutoff."""

    model_config = ConfigDict(frozen=True)

    n_active_bands_per_valley: int = Field(ge=1)
    n_remote_bands_per_valley: int = Field(ge=0)
    occupied_rank: int = Field(ge=1)
    empty_rank: int = Field(ge=1)
    max_occupied_hamiltonian_residual_mev: float = Field(ge=0.0)
    max_empty_hamiltonian_residual_mev: float = Field(ge=0.0)
    max_active_remote_overlap: float = Field(ge=0.0)
    max_active_remote_self_energy_mev: float = Field(ge=0.0)
    max_projector_occupation_error: float = Field(ge=0.0)


class HoleGapSummary(BaseModel):
    """Hole gap and equivalent physical-electron chemical-potential edges."""

    model_config = ConfigDict(frozen=True)

    hole_occupied_max_mev: float
    hole_empty_min_mev: float
    hole_indirect_gap_mev: float
    hole_min_direct_gap_mev: float
    electron_vbm_mev: float
    electron_cbm_mev: float
    electron_midgap_mev: float
    hole_mu_at_electron_vbm_mev: float
    hole_mu_at_electron_cbm_mev: float
    hole_mu_midgap_mev: float


class SewingCutoffDiagnostics(BaseModel):
    """Worst retained state weights across all wrapped central-difference links."""

    model_config = ConfigDict(frozen=True)

    n_boundary_frames: int = Field(ge=0)
    min_occupied_state_weight: float = Field(ge=0.0)
    min_empty_state_weight: float = Field(ge=0.0)
    max_occupied_gram_loss: float = Field(ge=0.0)
    max_empty_gram_loss: float = Field(ge=0.0)


class StageBenchmark(BaseModel):
    """Wall-time and process-memory record for one workflow stage."""

    model_config = ConfigDict(frozen=True)

    stage: str
    n_active_bands_per_valley: int | None = Field(default=None, ge=1)
    n_remote_bands_per_valley: int | None = Field(default=None, ge=0)
    repeats: int = Field(default=1, ge=1)
    elapsed_seconds_min: float = Field(ge=0.0)
    elapsed_seconds_median: float = Field(ge=0.0)
    process_peak_rss_mb: float = Field(ge=0.0)


class OrbitalMagnetizationRow(BaseModel):
    """One source/cutoff/chemical-potential convergence row."""

    model_config = ConfigDict(frozen=True)

    source_kind: str
    n_hf_bands_per_valley: int = Field(ge=1)
    n_remote_bands_per_valley: int = Field(ge=0)
    n_total_bands_per_valley: int = Field(ge=1)
    chemical_potential_point: str
    chemical_potential_electron_mev: float
    chemical_potential_hole_mev: float
    orbital_magnetization_mu_b_per_cell: float
    self_rotation_mu_b_per_cell: float
    streda_slope_hole_mu_b_per_mev: float
    streda_slope_electron_mu_b_per_mev: float
    occupied_hole_chern_fukui: float
    streda_chern_from_retained_pq: float
    streda_electron_slope_error_mu_b_per_mev: float
    hole_indirect_gap_mev: float
    hole_min_direct_gap_mev: float
    valley_polarization: float
    hf_energy_per_cell_mev: float
    hf_converged: bool
    hf_iterations: int = Field(ge=0)
    active_remote_mixing_lambda: float = Field(ge=0.0)
    occupied_projector_overlap_with_hf2_mean: float = Field(ge=0.0, le=1.000001)
    occupied_projector_overlap_with_hf2_min: float = Field(ge=0.0, le=1.000001)
    min_occupied_sewing_weight: float = Field(ge=0.0)
    min_empty_sewing_weight: float = Field(ge=0.0)
    max_occupied_hamiltonian_residual_mev: float = Field(ge=0.0)
    max_empty_hamiltonian_residual_mev: float = Field(ge=0.0)


class MatchedCutoffComparison(BaseModel):
    """Frozen-versus-self-consistent comparison at the same total cutoff."""

    model_config = ConfigDict(frozen=True)

    n_total_bands_per_valley: int = Field(ge=1)
    frozen_remote_bands_per_valley: int = Field(ge=0)
    common_electron_mu_mev: float
    common_mu_inside_both_gaps: bool
    frozen_magnetization_mu_b_per_cell: float
    hf_magnetization_mu_b_per_cell: float
    signed_delta_mu_b_per_cell: float
    absolute_delta_mu_b_per_cell: float = Field(ge=0.0)
    relative_delta: float = Field(ge=0.0)


class OrbitalMagnetizationWorkflowSummary(BaseModel):
    """Artifact-level summary of a completed convergence workflow."""

    model_config = ConfigDict(frozen=True)

    result_dir: str
    band_cache_hash: str
    frozen_rows: int = Field(ge=0)
    enlarged_hf_rows: int = Field(ge=0)
    matched_rows: int = Field(ge=0)
    benchmark_rows: int = Field(ge=0)
    largest_remote_cutoff_completed: int = Field(ge=0)
    largest_hf_cutoff_completed: int = Field(ge=1)
    all_hf_converged: bool
    manifest_passed: bool
    frozen_midgap_last_step_absolute_mu_b: float | None = Field(default=None, ge=0.0)
    frozen_midgap_last_step_relative: float | None = Field(default=None, ge=0.0)
    frozen_absolute_tolerance_passed: bool | None = None
    frozen_relative_tolerance_passed: bool | None = None
    largest_matched_hf_relaxation_absolute_mu_b: float = Field(ge=0.0)
    largest_matched_hf_relaxation_relative: float = Field(ge=0.0)
    observable_scope: str = "valence_continuum"
    excluded_moments: str = "true conduction-band, microscopic atomic-orbital, and spin moments"


@dataclass(frozen=True)
class FrozenHoleSubspaces:
    """Retained P/Q frames and exact full-Hamiltonian actions."""

    occupied_frames: np.ndarray
    empty_frames: np.ndarray
    hamiltonian_on_occupied: np.ndarray
    hamiltonian_on_empty: np.ndarray
    occupied_projected_hamiltonian: np.ndarray
    empty_projected_hamiltonian: np.ndarray
    diagnostics: FrozenEmbeddingDiagnostics
    gap: HoleGapSummary


def taige_band_cache_signature(model: ContinuumModelParams, grid: MomentumGrid) -> dict[str, Any]:
    """Return the signature for a reusable maximum-band eigensystem."""

    model_payload = model.model_dump(mode="json")
    model_payload.pop("n_active_bands_per_valley", None)
    return {
        "schema": TAIGE_BAND_CACHE_SCHEMA,
        "model": model_payload,
        "grid": {"n_k": int(grid.n_k)},
        "basis": "electron plane-wave/layer; Kprime T-prime generated",
        "band_order": "electron energies descending from valence-band maximum",
        "cutoff_convention": (
            "one maximum-band T-prime frame is Loewdin orthonormalized once, then nested slices are used"
        ),
    }


def save_taige_band_cache(
    path: str | Path,
    bands: TaigeBandStructure,
) -> TaigeBandCacheManifest:
    """Atomically save the maximum-cutoff electron eigensystem."""

    output = Path(path).expanduser().resolve()
    signature = taige_band_cache_signature(bands.model, bands.grid)
    cache_hash = _stable_hash(signature)
    metadata = {
        "schema": TAIGE_BAND_CACHE_SCHEMA,
        "cache_hash": cache_hash,
        "signature": signature,
        "model": bands.model.model_dump(mode="json"),
        "shell": [list(g) for g in bands.shell],
        "n_shell": int(bands.n_shell),
        "n_bands": int(bands.n_bands),
        "n_plane_waves": int(bands.n_plane_waves),
    }
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "electron_energies": np.asarray(bands.electron_energies, dtype=float),
        "electron_vectors": np.asarray(bands.electron_vectors, dtype=complex),
        "tprime_partner_index": np.asarray(bands.tprime_partner_index, dtype=int),
        "tprime_sewing_quality": np.asarray(bands.tprime_sewing_quality, dtype=float),
    }
    _write_npz_atomically(output, arrays)
    return TaigeBandCacheManifest(
        cache_hash=cache_hash,
        path=str(output),
        signature=signature,
        n_k=bands.grid.n_k,
        n_bands=bands.n_bands,
        microscopic_valley_dim=2 * bands.n_plane_waves,
    )


def load_taige_band_cache(
    path: str | Path,
    *,
    expected_model: ContinuumModelParams | None = None,
    expected_grid: MomentumGrid | None = None,
) -> tuple[TaigeBandStructure, TaigeBandCacheManifest]:
    """Load and validate a cached continuum eigensystem."""

    cache_path = Path(path).expanduser().resolve()
    try:
        data_context = np.load(cache_path, allow_pickle=False)
    except Exception as exc:  # pragma: no cover - exact numpy exception is format-dependent
        raise ValueError(f"could not load Taige band cache {cache_path}: {exc}") from exc
    with data_context as data:
        try:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
        except Exception as exc:
            raise ValueError("Taige band cache has invalid metadata") from exc
        if metadata.get("schema") != TAIGE_BAND_CACHE_SCHEMA:
            raise ValueError("Taige band cache schema mismatch")
        model = ContinuumModelParams.model_validate(metadata["model"])
        grid = MomentumGrid(int(metadata["signature"]["grid"]["n_k"]))
        signature = taige_band_cache_signature(model, grid)
        cache_hash = _stable_hash(signature)
        if metadata.get("cache_hash") != cache_hash or metadata.get("signature") != signature:
            raise ValueError("Taige band cache signature/hash mismatch")
        if expected_model is not None:
            expected_signature = taige_band_cache_signature(expected_model, grid)
            if expected_signature != signature:
                raise ValueError("Taige band cache does not match the expected model")
        if expected_grid is not None and expected_grid != grid:
            raise ValueError("Taige band cache does not match the expected grid")
        electron_energies = np.asarray(data["electron_energies"], dtype=float)
        electron_vectors = np.asarray(data["electron_vectors"], dtype=complex)
        partner = np.asarray(data["tprime_partner_index"], dtype=int)
        quality = np.asarray(data["tprime_sewing_quality"], dtype=float)

    n_bands = int(metadata["n_bands"])
    n_plane_waves = int(metadata["n_plane_waves"])
    expected_energy_shape = (grid.size, 2, n_bands)
    expected_vector_shape = (grid.size, 2, 2 * n_plane_waves, n_bands)
    if electron_energies.shape != expected_energy_shape:
        raise ValueError("Taige band cache energy shape mismatch")
    if electron_vectors.shape != expected_vector_shape:
        raise ValueError("Taige band cache vector shape mismatch")
    shell = tuple((int(g[0]), int(g[1])) for g in metadata["shell"])
    bands = TaigeBandStructure(
        model=model,
        grid=grid,
        n_shell=int(metadata["n_shell"]),
        n_bands=n_bands,
        shell=shell,
        n_plane_waves=n_plane_waves,
        electron_energies=electron_energies,
        electron_vectors=electron_vectors,
        hole_energies=-electron_energies,
        hole_vectors=np.conj(electron_vectors),
        geometry=MoireGeometry(model),
        tprime_partner_index=partner,
        tprime_sewing_quality=quality,
    )
    manifest = TaigeBandCacheManifest(
        cache_hash=cache_hash,
        path=str(cache_path),
        signature=signature,
        n_k=grid.n_k,
        n_bands=n_bands,
        microscopic_valley_dim=2 * n_plane_waves,
    )
    return bands, manifest


def build_frozen_hole_subspaces(
    *,
    active: ContinuumActiveSpace,
    bands: TaigeBandStructure,
    hf: ContinuumHFResult,
    n_remote_bands_per_valley: int,
    n_occ_holes_per_k: int = 1,
) -> FrozenHoleSubspaces:
    """Embed active HF plus bare remote bands in the common hole basis."""

    if active.finite_q_enabled:
        raise NotImplementedError("orbital magnetization currently requires the Q=0 active frame")
    n_active = int(active.n_active)
    n_remote = int(n_remote_bands_per_valley)
    n_occ = int(n_occ_holes_per_k)
    if n_remote < 0 or n_active + n_remote > bands.n_bands:
        raise ValueError("remote cutoff exceeds the cached continuum bands")
    if hf.P.shape != (active.n_k, active.dim, active.dim):
        raise ValueError("HF projector shape is incompatible with the active space")
    if not 1 <= n_occ < active.dim:
        raise ValueError("occupied-hole rank must lie inside the active dimension")

    active_frames = active_basis_frames(active)
    valley_dim = bands.hole_vectors.shape[2]
    microscopic_dim = 2 * valley_dim
    empty_rank = active.dim - n_occ + 2 * n_remote
    occupied = np.zeros((active.n_k, microscopic_dim, n_occ), dtype=complex)
    empty = np.zeros((active.n_k, microscopic_dim, empty_rank), dtype=complex)
    sigma = np.asarray(hf.H_hf - active.h0, dtype=complex)
    occupation_error = 0.0

    for ik in range(active.n_k):
        energies, vectors = np.linalg.eigh(hf.H_hf[ik])
        weights = np.real(np.diag(vectors.conj().T @ hf.P[ik] @ vectors))
        occupied_indices = np.argsort(weights)[-n_occ:]
        empty_indices = np.asarray(
            [index for index in range(active.dim) if index not in set(occupied_indices)],
            dtype=int,
        )
        occupied[ik] = active_frames[ik] @ vectors[:, occupied_indices]
        empty_active = active_frames[ik] @ vectors[:, empty_indices]
        empty[ik, :, : empty_active.shape[1]] = empty_active
        occupation_error = max(
            occupation_error,
            float(np.max(np.abs(weights[occupied_indices] - 1.0))),
            float(np.max(np.abs(weights[empty_indices]))),
        )
        if n_remote:
            offset = active.dim - n_occ
            empty[ik, :valley_dim, offset : offset + n_remote] = bands.hole_vectors[
                ik, 0, :, n_active : n_active + n_remote
            ]
            empty[ik, valley_dim:, offset + n_remote :] = bands.hole_vectors[
                ik, 1, :, n_active : n_active + n_remote
            ]

    h_occ = apply_frozen_hole_hamiltonian(
        active=active, bands=bands, sigma=sigma, frames=occupied
    )
    h_emp = apply_frozen_hole_hamiltonian(active=active, bands=bands, sigma=sigma, frames=empty)
    projected_occ = occupied.conj().swapaxes(1, 2) @ h_occ
    projected_emp = empty.conj().swapaxes(1, 2) @ h_emp
    occ_residual = h_occ - occupied @ projected_occ
    emp_residual = h_emp - empty @ projected_emp
    cross = active_frames.conj().swapaxes(1, 2) @ empty[:, :, active.dim - n_occ :]
    sigma_remote = np.einsum("kij,kjr->kir", sigma, cross, optimize=True)
    diagnostics = FrozenEmbeddingDiagnostics(
        n_active_bands_per_valley=n_active,
        n_remote_bands_per_valley=n_remote,
        occupied_rank=n_occ,
        empty_rank=empty_rank,
        max_occupied_hamiltonian_residual_mev=float(np.max(np.linalg.norm(occ_residual, axis=1))),
        max_empty_hamiltonian_residual_mev=float(np.max(np.linalg.norm(emp_residual, axis=1))),
        max_active_remote_overlap=float(np.max(np.abs(cross), initial=0.0)),
        max_active_remote_self_energy_mev=float(np.max(np.abs(sigma_remote), initial=0.0)),
        max_projector_occupation_error=occupation_error,
    )
    return FrozenHoleSubspaces(
        occupied_frames=occupied,
        empty_frames=empty,
        hamiltonian_on_occupied=h_occ,
        hamiltonian_on_empty=h_emp,
        occupied_projected_hamiltonian=projected_occ,
        empty_projected_hamiltonian=projected_emp,
        diagnostics=diagnostics,
        gap=hole_gap_summary(projected_occ, projected_emp),
    )


def apply_frozen_hole_hamiltonian(
    *,
    active: ContinuumActiveSpace,
    bands: TaigeBandStructure,
    sigma: np.ndarray,
    frames: np.ndarray,
) -> np.ndarray:
    """Apply ``H0_h^PW + F_A Sigma_A F_A^dag`` without storing dense mesh blocks."""

    vectors = np.asarray(frames, dtype=complex)
    if vectors.ndim != 3 or vectors.shape[0] != active.n_k:
        raise ValueError("frames have incompatible mesh shape")
    active_frames = active_basis_frames(active)
    valley_dim = bands.hole_vectors.shape[2]
    if vectors.shape[1] != 2 * valley_dim:
        raise ValueError("frames use an incompatible microscopic dimension")
    continuum = TaigeContinuumModel(bands.model)
    result = np.zeros_like(vectors)
    for ik in range(active.n_k):
        coord = active.grid.coord_of(ik)
        k_frac = np.asarray((coord[0] / active.grid.n1, coord[1] / active.grid.n2))
        for iv, valley in enumerate(VALLEY_ORDER):
            sl = slice(iv * valley_dim, (iv + 1) * valley_dim)
            h0_hole = -continuum.hamiltonian(k_frac, valley).conj()
            result[ik, sl] = h0_hole @ vectors[ik, sl]
        frame = active_frames[ik]
        result[ik] += frame @ sigma[ik] @ (frame.conj().T @ vectors[ik])
    return result


def hole_gap_summary(
    occupied_projected_hamiltonian: np.ndarray,
    empty_projected_hamiltonian: np.ndarray,
) -> HoleGapSummary:
    """Return indirect/direct hole gaps and their electron-edge mapping."""

    occupied_h = np.asarray(occupied_projected_hamiltonian, dtype=complex)
    empty_h = np.asarray(empty_projected_hamiltonian, dtype=complex)
    occ_energies = np.linalg.eigvalsh(occupied_h)
    emp_energies = np.linalg.eigvalsh(empty_h)
    occupied_max = float(np.max(occ_energies))
    empty_min = float(np.min(emp_energies))
    direct = float(np.min(np.min(emp_energies, axis=1) - np.max(occ_energies, axis=1)))
    electron_vbm = -empty_min
    electron_cbm = -occupied_max
    electron_midgap = 0.5 * (electron_vbm + electron_cbm)
    return HoleGapSummary(
        hole_occupied_max_mev=occupied_max,
        hole_empty_min_mev=empty_min,
        hole_indirect_gap_mev=empty_min - occupied_max,
        hole_min_direct_gap_mev=direct,
        electron_vbm_mev=electron_vbm,
        electron_cbm_mev=electron_cbm,
        electron_midgap_mev=electron_midgap,
        hole_mu_at_electron_vbm_mev=empty_min,
        hole_mu_at_electron_cbm_mev=occupied_max,
        hole_mu_midgap_mev=-electron_midgap,
    )


def taige_transport_factory(shell: tuple[tuple[int, int], ...]):
    """Return a cached common-basis folded-to-raw frame transport."""

    cache: dict[tuple[int, int], TaigeReciprocalTransport] = {}

    def transport(frame: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
        key = (int(shift[0]), int(shift[1]))
        if key not in cache:
            cache[key] = TaigeReciprocalTransport(shell, key)
        return cache[key].folded_to_raw_vectors(frame)

    return transport


def sewing_cutoff_diagnostics(
    *,
    grid: MomentumGrid,
    shell: tuple[tuple[int, int], ...],
    occupied_frames: np.ndarray,
    empty_frames: np.ndarray,
) -> SewingCutoffDiagnostics:
    """Aggregate retained-weight diagnostics over every wrapped central link."""

    occupied_weights: list[float] = []
    empty_weights: list[float] = []
    occupied_losses: list[float] = []
    empty_losses: list[float] = []
    for ik in range(grid.size):
        coord = grid.coord_of(ik)
        for delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            folded, shift = grid.shift_plus_q(coord, delta)
            if shift == (0, 0):
                continue
            transport = TaigeReciprocalTransport(shell, shift)
            neighbor = grid.index_of(folded)
            occ = transport.frame_diagnostics(occupied_frames[neighbor])
            emp = transport.frame_diagnostics(empty_frames[neighbor])
            occupied_weights.append(occ.min_retained_state_weight)
            empty_weights.append(emp.min_retained_state_weight)
            occupied_losses.append(occ.max_gram_loss)
            empty_losses.append(emp.max_gram_loss)
    return SewingCutoffDiagnostics(
        n_boundary_frames=len(occupied_weights),
        min_occupied_state_weight=min(occupied_weights, default=1.0),
        min_empty_state_weight=min(empty_weights, default=1.0),
        max_occupied_gram_loss=max(occupied_losses, default=0.0),
        max_empty_gram_loss=max(empty_losses, default=0.0),
    )


def run_taige_orbital_magnetization_workflow(
    params: TaigeOrbitalMagnetizationWorkflowParams | None = None,
) -> OrbitalMagnetizationWorkflowSummary:
    """Run frozen r=0..6 and self-consistent N=2,3,4 VP convergence."""

    controls = params or TaigeOrbitalMagnetizationWorkflowParams()
    result_dir = Path(controls.output_dir).expanduser().resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    previous_benchmarks = _load_benchmark_csv(result_dir / "benchmarks.csv")
    _write_json_atomic(result_dir / "parameters.json", controls.model_dump(mode="json"))
    benchmarks: list[StageBenchmark] = []
    grid = MomentumGrid(controls.grid.n_k)

    band_path = result_dir / "continuum_bands_max.npz"
    expected_signature = taige_band_cache_signature(controls.model, grid)
    if controls.reuse_completed_stages and band_path.exists():
        (bands, band_manifest), benchmark = _benchmark_call(
            "continuum_band_cache_load",
            lambda: load_taige_band_cache(
                band_path, expected_model=controls.model, expected_grid=grid
            ),
        )
        if band_manifest.signature != expected_signature:
            raise ValueError("loaded band cache has an unexpected signature")
        benchmarks.append(benchmark)
    else:
        bands, benchmark = _benchmark_call(
            "continuum_diagonalization",
            lambda: compute_taige_bandstructure(controls.model, grid),
        )
        benchmarks.append(benchmark)
        band_manifest = save_taige_band_cache(band_path, bands)

    hf_data: dict[int, tuple[ContinuumActiveSpace, ContinuumHFResult]] = {}
    all_hf_cutoffs = tuple(controls.orbital.enlarged_hf_bands_per_valley)
    n_base = controls.orbital.n_active_bands_per_valley
    base_active, base_hf, stage_benchmarks = _solve_or_load_vp_hf(
        controls=controls,
        bands=bands,
        band_cache_hash=band_manifest.cache_hash,
        n_active=n_base,
        result_dir=result_dir,
    )
    hf_data[n_base] = (base_active, base_hf)
    benchmarks.extend(stage_benchmarks)
    base_subspace = build_frozen_hole_subspaces(
        active=base_active,
        bands=bands,
        hf=base_hf,
        n_remote_bands_per_valley=0,
        n_occ_holes_per_k=controls.orbital.n_occ_holes_per_k,
    )
    reference_occupied = base_subspace.occupied_frames
    reciprocal_basis = bands.geometry.kM_inv_nm * np.column_stack(
        (bands.geometry.b1, bands.geometry.b2)
    )
    transport = taige_transport_factory(bands.shell)
    frozen_rows: list[OrbitalMagnetizationRow] = []
    hf_rows: list[OrbitalMagnetizationRow] = []
    frozen_subspaces: dict[int, FrozenHoleSubspaces] = {}
    hf_subspaces: dict[int, FrozenHoleSubspaces] = {}
    k_resolved: dict[str, np.ndarray] = {}

    for n_remote in controls.orbital.remote_cutoffs_per_valley:
        subspace, embedding_benchmark = _benchmark_call(
            "frozen_embedding",
            lambda n_remote=n_remote: build_frozen_hole_subspaces(
                active=base_active,
                bands=bands,
                hf=base_hf,
                n_remote_bands_per_valley=n_remote,
                n_occ_holes_per_k=controls.orbital.n_occ_holes_per_k,
            ),
            n_active=n_base,
            n_remote=n_remote,
        )
        benchmarks.append(embedding_benchmark)
        frozen_subspaces[n_remote] = subspace
        rows, arrays, observable_benchmark = _evaluate_subspace_rows(
            source_kind="frozen_remote",
            n_hf=n_base,
            n_remote=n_remote,
            active=base_active,
            hf=base_hf,
            subspace=subspace,
            bands=bands,
            reciprocal_basis=reciprocal_basis,
            transport=transport,
            reference_occupied=reference_occupied,
            repeats=controls.orbital.benchmark_repeats,
        )
        frozen_rows.extend(rows)
        k_resolved.update(arrays)
        benchmarks.append(observable_benchmark)

    # Only after the full frozen sequence succeeds do we allocate and solve the
    # larger HF backends.  This preserves the single-point smoke/scale order.
    for n_active in all_hf_cutoffs:
        if n_active not in hf_data:
            active, hf, stage_benchmarks = _solve_or_load_vp_hf(
                controls=controls,
                bands=bands,
                band_cache_hash=band_manifest.cache_hash,
                n_active=n_active,
                result_dir=result_dir,
            )
            hf_data[n_active] = (active, hf)
            benchmarks.extend(stage_benchmarks)
        active, hf = hf_data[n_active]
        subspace = build_frozen_hole_subspaces(
            active=active,
            bands=bands,
            hf=hf,
            n_remote_bands_per_valley=0,
            n_occ_holes_per_k=controls.orbital.n_occ_holes_per_k,
        )
        hf_subspaces[n_active] = subspace
        rows, arrays, observable_benchmark = _evaluate_subspace_rows(
            source_kind="self_consistent_hf",
            n_hf=n_active,
            n_remote=0,
            active=active,
            hf=hf,
            subspace=subspace,
            bands=bands,
            reciprocal_basis=reciprocal_basis,
            transport=transport,
            reference_occupied=reference_occupied,
            repeats=controls.orbital.benchmark_repeats,
        )
        hf_rows.extend(rows)
        k_resolved.update(arrays)
        benchmarks.append(observable_benchmark)

    matched = _matched_cutoff_rows(
        base_active=base_active,
        base_hf=base_hf,
        hf_data=hf_data,
        frozen_subspaces=frozen_subspaces,
        hf_subspaces=hf_subspaces,
        bands=bands,
        reciprocal_basis=reciprocal_basis,
        transport=transport,
    )

    _write_model_csv(result_dir / "remote_convergence.csv", frozen_rows)
    _write_model_csv(result_dir / "hf_active_space_convergence.csv", hf_rows)
    _write_model_csv(result_dir / "matched_cutoff_comparison.csv", matched)
    benchmarks = _merge_benchmarks(previous_benchmarks, benchmarks)
    _write_model_csv(result_dir / "benchmarks.csv", benchmarks)
    _write_json_atomic(
        result_dir / "benchmarks.json", [row.model_dump(mode="json") for row in benchmarks]
    )
    if controls.orbital.store_k_resolved_terms:
        _write_npz_atomically(result_dir / "k_resolved_terms.npz", k_resolved)

    all_converged = all(hf.converged for _active, hf in hf_data.values())
    convergence_marker = result_dir / "hf_convergence.ok"
    if all_converged:
        convergence_marker.write_text("all requested VP HF states converged\n")
    elif convergence_marker.exists():
        convergence_marker.unlink()
    artifacts = _workflow_artifacts(result_dir, all_hf_cutoffs)
    from chiral_dw.artifacts import RunManifest

    manifest = RunManifest.from_artifacts(
        run_id=result_dir.name,
        result_dir=str(result_dir),
        artifacts=artifacts,
    )
    midgap_frozen = [
        row for row in frozen_rows if row.chemical_potential_point == "midgap"
    ]
    midgap_frozen.sort(key=lambda row: row.n_remote_bands_per_valley)
    if len(midgap_frozen) >= 2:
        last_step = abs(
            midgap_frozen[-1].orbital_magnetization_mu_b_per_cell
            - midgap_frozen[-2].orbital_magnetization_mu_b_per_cell
        )
        last_scale = max(
            abs(midgap_frozen[-1].orbital_magnetization_mu_b_per_cell), 1e-12
        )
        last_relative = last_step / last_scale
    else:
        last_step = None
        last_relative = None
    largest_match = max(matched, key=lambda row: row.n_total_bands_per_valley)
    summary = OrbitalMagnetizationWorkflowSummary(
        result_dir=str(result_dir),
        band_cache_hash=band_manifest.cache_hash,
        frozen_rows=len(frozen_rows),
        enlarged_hf_rows=len(hf_rows),
        matched_rows=len(matched),
        benchmark_rows=len(benchmarks),
        largest_remote_cutoff_completed=max(controls.orbital.remote_cutoffs_per_valley),
        largest_hf_cutoff_completed=max(all_hf_cutoffs),
        all_hf_converged=all_converged,
        manifest_passed=manifest.passed,
        frozen_midgap_last_step_absolute_mu_b=last_step,
        frozen_midgap_last_step_relative=last_relative,
        frozen_absolute_tolerance_passed=(
            None
            if last_step is None
            else last_step <= controls.orbital.convergence_abs_mu_b
        ),
        frozen_relative_tolerance_passed=(
            None
            if last_relative is None
            else last_relative <= controls.orbital.convergence_rel
        ),
        largest_matched_hf_relaxation_absolute_mu_b=(
            largest_match.absolute_delta_mu_b_per_cell
        ),
        largest_matched_hf_relaxation_relative=largest_match.relative_delta,
    )
    _write_json_atomic(result_dir / "summary.json", summary.model_dump(mode="json"))
    # Refresh after summary exists.
    manifest = RunManifest.from_artifacts(
        run_id=result_dir.name,
        result_dir=str(result_dir),
        artifacts=_workflow_artifacts(result_dir, all_hf_cutoffs),
    )
    _write_json_atomic(result_dir / "run_manifest.json", manifest.model_dump(mode="json"))
    if summary.manifest_passed != manifest.passed:
        summary = summary.model_copy(update={"manifest_passed": manifest.passed})
        _write_json_atomic(result_dir / "summary.json", summary.model_dump(mode="json"))
    return summary


def _solve_or_load_vp_hf(
    *,
    controls: TaigeOrbitalMagnetizationWorkflowParams,
    bands: TaigeBandStructure,
    band_cache_hash: str,
    n_active: int,
    result_dir: Path,
) -> tuple[ContinuumActiveSpace, ContinuumHFResult, list[StageBenchmark]]:
    model = controls.model.model_copy(update={"n_active_bands_per_valley": int(n_active)})
    active = active_space_from_taige_bands(bands.grid, model, bands)
    signature = _hf_state_signature(controls, band_cache_hash, n_active)
    state_path = result_dir / f"hf_active_{n_active}.npz"
    benchmarks: list[StageBenchmark] = []
    if controls.reuse_completed_stages and state_path.exists():
        hf, benchmark = _benchmark_call(
            "hf_state_cache_load",
            lambda: _load_hf_state(state_path, signature),
            n_active=n_active,
        )
        benchmarks.append(benchmark)
        return active, hf, benchmarks

    vertices, benchmark = _benchmark_call(
        "density_vertices",
        lambda: build_taige_density_vertices(active, controls.interaction),
        n_active=n_active,
    )
    benchmarks.append(benchmark)
    from chiral_dw.continuum.hf import ContinuumHFBackend

    backend, benchmark = _benchmark_call(
        "exchange_backend",
        lambda: ContinuumHFBackend(active.h0, vertices, controls.interaction),
        n_active=n_active,
    )
    benchmarks.append(benchmark)
    bundle = ContinuumBundle(
        grid=bands.grid,
        active=active,
        vertices=backend.vertices,
        backend=backend,
        params=model,
        interaction=controls.interaction,
        finite_q=ContinuumFiniteQParams(),
        bands=bands,
        geometry=bands.geometry,
    )
    from chiral_dw.continuum.references import solve_reference_hf
    from chiral_dw.continuum.symmetry import ValleyU1Constraint

    hf, benchmark = _benchmark_call(
        "vp_hf_solve",
        lambda: solve_reference_hf(
            bundle,
            "vp_plus",
            controls.hf,
            constraint=ValleyU1Constraint(active, pinned_valley="K"),
        ),
        n_active=n_active,
    )
    benchmarks.append(benchmark)
    _save_hf_state(state_path, hf, signature)
    return active, hf, benchmarks


def _evaluate_subspace_rows(
    *,
    source_kind: str,
    n_hf: int,
    n_remote: int,
    active: ContinuumActiveSpace,
    hf: ContinuumHFResult,
    subspace: FrozenHoleSubspaces,
    bands: TaigeBandStructure,
    reciprocal_basis: np.ndarray,
    transport,
    reference_occupied: np.ndarray,
    repeats: int,
) -> tuple[list[OrbitalMagnetizationRow], dict[str, np.ndarray], StageBenchmark]:
    gap = subspace.gap
    mu_points = {
        "vbm": (gap.electron_vbm_mev, gap.hole_mu_at_electron_vbm_mev),
        "midgap": (gap.electron_midgap_mev, gap.hole_mu_midgap_mev),
        "cbm": (gap.electron_cbm_mev, gap.hole_mu_at_electron_cbm_mev),
    }
    sewing = sewing_cutoff_diagnostics(
        grid=bands.grid,
        shell=bands.shell,
        occupied_frames=subspace.occupied_frames,
        empty_frames=subspace.empty_frames,
    )
    chern = chern_number_on_grid(
        bands.grid, subspace.occupied_frames, 0, shell=bands.shell
    )
    overlap_values = np.abs(
        np.einsum(
            "kdi,kdj->kij",
            reference_occupied.conj(),
            subspace.occupied_frames,
            optimize=True,
        )
    ) ** 2
    overlap_mean = float(np.mean(overlap_values))
    overlap_min = float(np.min(overlap_values))
    mixing = active_remote_mixing_lambda(active, hf, n_base=2)
    polarization = float(np.mean(valley_polarization(hf.P, active)))

    def evaluate(mu_hole: float):
        return evaluate_projector_orbital_magnetization(
            grid=bands.grid,
            occupied_frames=subspace.occupied_frames,
            empty_frames=subspace.empty_frames,
            hamiltonian_on_occupied=subspace.hamiltonian_on_occupied,
            hamiltonian_on_empty=subspace.hamiltonian_on_empty,
            reciprocal_basis_nm_inv=reciprocal_basis,
            chemical_potential_hole_mev=mu_hole,
            transport=transport,
        )

    mid_evaluation, benchmark = _benchmark_call(
        f"observable_{source_kind}",
        lambda: evaluate(mu_points["midgap"][1]),
        repeats=repeats,
        n_active=n_hf,
        n_remote=n_remote,
    )
    evaluations = {
        "vbm": evaluate(mu_points["vbm"][1]),
        "midgap": mid_evaluation,
        "cbm": evaluate(mu_points["cbm"][1]),
    }
    expected_electron_slope = (
        -chern
        * bands.geometry.moire_cell_area_nm2
        / (2.0 * np.pi * HBAR2_OVER_2ME_MEV_NM2)
    )
    rows: list[OrbitalMagnetizationRow] = []
    arrays: dict[str, np.ndarray] = {}
    for point, evaluation in evaluations.items():
        summary = evaluation.summary
        electron_mu, hole_mu = mu_points[point]
        electron_slope = -summary.streda_slope_mu_b_per_mev
        rows.append(
            OrbitalMagnetizationRow(
                source_kind=source_kind,
                n_hf_bands_per_valley=n_hf,
                n_remote_bands_per_valley=n_remote,
                n_total_bands_per_valley=n_hf + n_remote,
                chemical_potential_point=point,
                chemical_potential_electron_mev=electron_mu,
                chemical_potential_hole_mev=hole_mu,
                orbital_magnetization_mu_b_per_cell=summary.orbital_magnetization_mu_b_per_cell,
                self_rotation_mu_b_per_cell=summary.self_rotation_mu_b_per_cell,
                streda_slope_hole_mu_b_per_mev=summary.streda_slope_mu_b_per_mev,
                streda_slope_electron_mu_b_per_mev=electron_slope,
                occupied_hole_chern_fukui=chern,
                streda_chern_from_retained_pq=summary.streda_chern_from_retained_pq,
                streda_electron_slope_error_mu_b_per_mev=electron_slope
                - expected_electron_slope,
                hole_indirect_gap_mev=gap.hole_indirect_gap_mev,
                hole_min_direct_gap_mev=gap.hole_min_direct_gap_mev,
                valley_polarization=polarization,
                hf_energy_per_cell_mev=float(hf.energy / active.n_k),
                hf_converged=hf.converged,
                hf_iterations=hf.n_iter,
                active_remote_mixing_lambda=mixing,
                occupied_projector_overlap_with_hf2_mean=overlap_mean,
                occupied_projector_overlap_with_hf2_min=overlap_min,
                min_occupied_sewing_weight=sewing.min_occupied_state_weight,
                min_empty_sewing_weight=sewing.min_empty_state_weight,
                max_occupied_hamiltonian_residual_mev=(
                    subspace.diagnostics.max_occupied_hamiltonian_residual_mev
                ),
                max_empty_hamiltonian_residual_mev=(
                    subspace.diagnostics.max_empty_hamiltonian_residual_mev
                ),
            )
        )
        key = f"{source_kind}_hf{n_hf}_r{n_remote}_{point}"
        arrays[f"{key}_w_xy_mev_nm2"] = evaluation.w_xy_mev_nm2
        arrays[f"{key}_n_xy_mev_nm2"] = evaluation.n_xy_mev_nm2
    return rows, arrays, benchmark


def _matched_cutoff_rows(
    *,
    base_active: ContinuumActiveSpace,
    base_hf: ContinuumHFResult,
    hf_data: dict[int, tuple[ContinuumActiveSpace, ContinuumHFResult]],
    frozen_subspaces: dict[int, FrozenHoleSubspaces],
    hf_subspaces: dict[int, FrozenHoleSubspaces],
    bands: TaigeBandStructure,
    reciprocal_basis: np.ndarray,
    transport,
) -> list[MatchedCutoffComparison]:
    comparisons: list[MatchedCutoffComparison] = []
    for n_total, (_active, _hf) in sorted(hf_data.items()):
        n_remote = n_total - base_active.n_active
        if n_remote not in frozen_subspaces:
            continue
        frozen = frozen_subspaces[n_remote]
        enlarged = hf_subspaces[n_total]
        electron_mu = enlarged.gap.electron_midgap_mev
        hole_mu = -electron_mu

        def moment(subspace: FrozenHoleSubspaces) -> float:
            return evaluate_projector_orbital_magnetization(
                grid=bands.grid,
                occupied_frames=subspace.occupied_frames,
                empty_frames=subspace.empty_frames,
                hamiltonian_on_occupied=subspace.hamiltonian_on_occupied,
                hamiltonian_on_empty=subspace.hamiltonian_on_empty,
                reciprocal_basis_nm_inv=reciprocal_basis,
                chemical_potential_hole_mev=hole_mu,
                transport=transport,
            ).summary.orbital_magnetization_mu_b_per_cell

        frozen_m = moment(frozen)
        hf_m = moment(enlarged)
        delta = hf_m - frozen_m
        scale = max(abs(frozen_m), abs(hf_m), 1e-12)
        in_frozen = (
            frozen.gap.electron_vbm_mev <= electron_mu <= frozen.gap.electron_cbm_mev
        )
        in_hf = (
            enlarged.gap.electron_vbm_mev <= electron_mu <= enlarged.gap.electron_cbm_mev
        )
        comparisons.append(
            MatchedCutoffComparison(
                n_total_bands_per_valley=n_total,
                frozen_remote_bands_per_valley=n_remote,
                common_electron_mu_mev=electron_mu,
                common_mu_inside_both_gaps=bool(in_frozen and in_hf),
                frozen_magnetization_mu_b_per_cell=frozen_m,
                hf_magnetization_mu_b_per_cell=hf_m,
                signed_delta_mu_b_per_cell=delta,
                absolute_delta_mu_b_per_cell=abs(delta),
                relative_delta=abs(delta) / scale,
            )
        )
    return comparisons


def active_remote_mixing_lambda(
    active: ContinuumActiveSpace,
    hf: ContinuumHFResult,
    *,
    n_base: int,
) -> float:
    """Return max ||Sigma_RA||/Delta_RA in the bare active-band basis."""

    n = active.n_active
    if n <= n_base:
        return 0.0
    active_indices = np.asarray(list(range(n_base)) + list(range(n, n + n_base)))
    remote_indices = np.asarray(list(range(n_base, n)) + list(range(n + n_base, 2 * n)))
    sigma = np.asarray(hf.H_hf - active.h0, dtype=complex)
    numerator = max(
        float(np.linalg.norm(block, ord=2))
        for block in sigma[:, remote_indices][:, :, active_indices]
    )
    active_energies = active.hole_energies[:, :, :n_base]
    remote_energies = active.hole_energies[:, :, n_base:n]
    separations = np.abs(remote_energies[..., :, None] - active_energies[..., None, :])
    denominator = max(float(np.min(separations)), 1e-12)
    return numerator / denominator


def _benchmark_call(
    stage: str,
    operation,
    *,
    repeats: int = 1,
    n_active: int | None = None,
    n_remote: int | None = None,
):
    elapsed: list[float] = []
    result = None
    for _ in range(int(repeats)):
        start = time.perf_counter()
        result = operation()
        elapsed.append(time.perf_counter() - start)
    benchmark = StageBenchmark(
        stage=stage,
        n_active_bands_per_valley=n_active,
        n_remote_bands_per_valley=n_remote,
        repeats=repeats,
        elapsed_seconds_min=min(elapsed),
        elapsed_seconds_median=statistics.median(elapsed),
        process_peak_rss_mb=_process_peak_rss_mb(),
    )
    return result, benchmark


def _process_peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0**2)
    return value / 1024.0


def _hf_state_signature(
    controls: TaigeOrbitalMagnetizationWorkflowParams,
    band_cache_hash: str,
    n_active: int,
) -> dict[str, Any]:
    return {
        "schema": "taige_orbital_hf_state_v1",
        "band_cache_hash": band_cache_hash,
        "n_active_bands_per_valley": int(n_active),
        "interaction": controls.interaction.model_dump(mode="json"),
        "hf": controls.hf.model_dump(mode="json"),
        "constraint": "valley_u1_K",
    }


def _save_hf_state(path: Path, hf: ContinuumHFResult, signature: dict[str, Any]) -> None:
    metadata = {
        "signature": signature,
        "energy": float(hf.energy),
        "converged": bool(hf.converged),
        "n_iter": int(hf.n_iter),
        "diagnostics": hf.diagnostics.model_dump(mode="json"),
        "seed": hf.seed,
        "constraint_name": hf.constraint_name,
    }
    _write_npz_atomically(
        path,
        {
            "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
            "P": np.asarray(hf.P, dtype=complex),
            "H_hf": np.asarray(hf.H_hf, dtype=complex),
        },
    )


def _load_hf_state(path: Path, signature: dict[str, Any]) -> ContinuumHFResult:
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            p = np.asarray(data["P"], dtype=complex)
            h_hf = np.asarray(data["H_hf"], dtype=complex)
    except Exception as exc:
        raise ValueError(f"could not load HF state {path}: {exc}") from exc
    if metadata.get("signature") != signature:
        raise ValueError(f"HF state signature mismatch for {path}")
    return ContinuumHFResult(
        P=p,
        H_hf=h_hf,
        energy=float(metadata["energy"]),
        converged=bool(metadata["converged"]),
        n_iter=int(metadata["n_iter"]),
        diagnostics=ContinuumHFDiagnostics.model_validate(metadata["diagnostics"]),
        seed=str(metadata.get("seed") or ""),
        constraint_name=metadata.get("constraint_name"),
    )


def _write_model_csv(path: Path, rows: list[BaseModel]) -> None:
    payloads = [row.model_dump(mode="json") for row in rows]
    if not payloads:
        raise ValueError(f"refusing to write empty required table {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payloads[0]))
        writer.writeheader()
        writer.writerows(payloads)
    os.replace(temporary, path)


def _load_benchmark_csv(path: Path) -> list[StageBenchmark]:
    if not path.exists():
        return []
    try:
        with path.open(newline="") as handle:
            rows = []
            for raw in csv.DictReader(handle):
                for key in (
                    "n_active_bands_per_valley",
                    "n_remote_bands_per_valley",
                ):
                    if raw.get(key) == "":
                        raw[key] = None
                rows.append(StageBenchmark.model_validate(raw))
            return rows
    except Exception as exc:
        raise ValueError(f"could not parse existing benchmark table {path}") from exc


def _merge_benchmarks(
    previous: list[StageBenchmark], current: list[StageBenchmark]
) -> list[StageBenchmark]:
    """Keep cold and warm stage records across restart runs without duplicates."""

    merged: dict[tuple[str, int | None, int | None], StageBenchmark] = {}
    for row in [*previous, *current]:
        key = (
            row.stage,
            row.n_active_bands_per_valley,
            row.n_remote_bands_per_valley,
        )
        merged[key] = row
    return sorted(
        merged.values(),
        key=lambda row: (
            row.n_active_bands_per_valley is not None,
            row.n_active_bands_per_valley or 0,
            row.n_remote_bands_per_valley is not None,
            row.n_remote_bands_per_valley or 0,
            row.stage,
        ),
    )


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _workflow_artifacts(result_dir: Path, hf_cutoffs: tuple[int, ...]):
    from chiral_dw.artifacts import RunArtifact

    definitions = [
        ("parameters", "parameters.json", "json", True),
        ("continuum_bands", "continuum_bands_max.npz", "array", True),
        ("remote_convergence", "remote_convergence.csv", "table", True),
        ("hf_convergence", "hf_active_space_convergence.csv", "table", True),
        ("matched_comparison", "matched_cutoff_comparison.csv", "table", True),
        ("benchmarks_csv", "benchmarks.csv", "table", True),
        ("benchmarks_json", "benchmarks.json", "json", True),
        ("k_resolved_terms", "k_resolved_terms.npz", "array", False),
        ("summary", "summary.json", "json", True),
        ("hf_convergence_marker", "hf_convergence.ok", "text", True),
    ]
    definitions.extend(
        (f"hf_active_{n}", f"hf_active_{n}.npz", "array", True) for n in hf_cutoffs
    )
    artifacts = []
    for name, relative, kind, required in definitions:
        path = result_dir / relative
        artifacts.append(
            RunArtifact(
                name=name,
                path=str(path),
                kind=kind,
                description=f"Taige orbital-magnetization workflow artifact: {name}",
                required=required,
                exists=path.exists(),
                size_bytes=path.stat().st_size if path.exists() else None,
            )
        )
    return artifacts


def _stable_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _write_npz_atomically(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
