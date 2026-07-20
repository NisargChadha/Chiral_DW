"""Frozen-remote and enlarged-HF orbital-magnetization workflow helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from chiral_dw.config import ContinuumModelParams
from chiral_dw.continuum.models import ContinuumActiveSpace, ContinuumHFResult, MomentumGrid
from chiral_dw.continuum.observables import active_basis_frames
from chiral_dw.continuum.taige import (
    MoireGeometry,
    TaigeBandStructure,
    TaigeContinuumModel,
    VALLEY_ORDER,
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
