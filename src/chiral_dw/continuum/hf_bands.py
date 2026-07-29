"""Fixed-density HF band spectra and Chern diagnostics for Taige active spaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict

from chiral_dw.config import ContinuumInteractionParams
from chiral_dw.continuum.models import (
    ContinuumActiveSpace,
    ContinuumBundle,
    VALLEY_K,
    VALLEY_KPRIME,
    hermitize,
)
from chiral_dw.continuum.observables import active_basis_frames
from chiral_dw.continuum.taige import (
    MoireGeometry,
    TaigeContinuumModel,
    chern_number_on_grid,
    coulomb_potential_mev_nm2,
    dual_gate_q0_limit_mev_nm2,
    sew_tprime_electron_vectors,
    taige_momentum_path,
)


class HFPathSpectrumRow(BaseModel):
    """One fixed-density HF band energy along a momentum path."""

    model_config = ConfigDict(frozen=True)

    reference: str
    path_index: int
    k_frac_1: float
    k_frac_2: float
    k_distance: float
    label: str | None = None
    band: int
    energy: float
    valley_K_weight: float
    valley_Kprime_weight: float


class HFBandChernRow(BaseModel):
    """One embedded HF band Chern number on the active momentum mesh."""

    model_config = ConfigDict(frozen=True)

    reference: str
    band: int
    chern: float
    energy_min: float
    energy_max: float


@dataclass(frozen=True)
class FineMomentumFrame:
    """Continuum eigenbasis data for one arbitrary active-frame momentum."""

    k_frac: np.ndarray
    hole_energies: np.ndarray
    electron_vectors: np.ndarray
    physical_k_frac: np.ndarray
    physical_shift: np.ndarray


@dataclass(frozen=True)
class HFPathSpectrum:
    """Fixed-density HF bands and valley weights on a momentum path."""

    rows: tuple[HFPathSpectrumRow, ...]
    energies: np.ndarray
    valley_weights: np.ndarray
    path_frac: np.ndarray
    distances: np.ndarray
    ticks: tuple[int, ...]
    labels: tuple[str, ...]


@dataclass(frozen=True)
class HFMeshBandData:
    """HF eigensystem on the active momentum mesh."""

    energies: np.ndarray
    active_vectors: np.ndarray
    embedded_vectors: np.ndarray


def _require_taige_active(active: ContinuumActiveSpace) -> None:
    if active.model.active_model != "taige":
        raise ValueError("HF path/Chern diagnostics currently require a Taige active space")
    if active.geometry is None or not active.shell:
        raise ValueError("Taige active space is missing geometry or reciprocal shell metadata")
    if active.source_shift is None:
        raise ValueError("Taige active space is missing source_shift metadata")


def _fold_fractional(k_frac: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(k_frac, dtype=float)
    shift = np.floor(raw).astype(int)
    return raw - shift, shift


def _active_to_physical_fractional(
    active: ContinuumActiveSpace,
    k_frac: np.ndarray,
    valley: str,
) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(k_frac, dtype=float)
    if active.finite_q_enabled:
        if active.q_coord is None:
            raise ValueError("finite-Q active space is missing q_coord")
        half = active.grid.assert_half_q_on_mesh(active.q_coord, active.half_shift_coord)
        half_frac = np.array((half[0] / active.grid.n1, half[1] / active.grid.n2), dtype=float)
        if valley == VALLEY_K:
            k = k - half_frac
        elif valley == VALLEY_KPRIME:
            k = k + half_frac
        else:
            raise ValueError(f"unknown valley {valley!r}")
    return _fold_fractional(k)


def taige_active_fine_frame(
    active: ContinuumActiveSpace,
    k_frac: np.ndarray,
    *,
    continuum: TaigeContinuumModel | None = None,
) -> FineMomentumFrame:
    """Solve active Taige hole states at one arbitrary active-frame momentum."""

    _require_taige_active(active)
    model = continuum or TaigeContinuumModel(active.model)
    n_active = int(active.n_active)
    n_valley = len(active.valley_order)
    n_pw2 = 2 * int(active.n_plane_waves)
    hole_energies = np.empty((n_valley, n_active), dtype=float)
    electron_vectors = np.empty((n_valley, n_pw2, n_active), dtype=complex)
    physical_k_frac = np.empty((n_valley, 2), dtype=float)
    physical_shift = np.empty((n_valley, 2), dtype=int)

    k_index = active.valley_index(VALLEY_K)
    kprime_index = active.valley_index(VALLEY_KPRIME)
    active_k = np.asarray(k_frac, dtype=float)

    physical_k, shift = _active_to_physical_fractional(active, active_k, VALLEY_K)
    physical_k_frac[k_index] = physical_k
    physical_shift[k_index] = shift
    evals, evecs = np.linalg.eigh(model.hamiltonian(physical_k, VALLEY_K))
    order = np.argsort(evals)[::-1][:n_active]
    hole_energies[k_index] = -evals[order]
    electron_vectors[k_index] = evecs[:, order]

    kprime_physical, kprime_shift = _active_to_physical_fractional(
        active,
        active_k,
        VALLEY_KPRIME,
    )
    physical_k_frac[kprime_index] = kprime_physical
    physical_shift[kprime_index] = kprime_shift
    partner_physical_k, partner_shift = _fold_fractional(-kprime_physical)
    partner_evals, partner_evecs = np.linalg.eigh(model.hamiltonian(partner_physical_k, VALLEY_K))
    partner_order = np.argsort(partner_evals)[::-1][:n_active]
    hole_energies[kprime_index] = -partner_evals[partner_order]
    generated, _quality = sew_tprime_electron_vectors(
        partner_evecs[:, partner_order],
        active.shell,
        tuple(int(x) for x in partner_shift),
    )
    electron_vectors[kprime_index] = generated

    return FineMomentumFrame(
        k_frac=active_k,
        hole_energies=hole_energies,
        electron_vectors=electron_vectors,
        physical_k_frac=physical_k_frac,
        physical_shift=physical_shift,
    )


def _coarse_fractional_grid(active: ContinuumActiveSpace) -> np.ndarray:
    return active.grid.fractional_coords()


def _center_transfer_array(q_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(q_raw, dtype=float)
    centered = np.mod(raw + 0.5, 1.0) - 0.5
    centered = np.where(np.isclose(centered, -0.5, atol=1e-12), 0.5, centered)
    shift = np.rint(centered - raw).astype(int)
    return raw + shift, shift


def _shift_gather(
    shell: tuple[tuple[int, int], ...],
    shell_index: dict[tuple[int, int], int],
    shift: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    src: list[int] = []
    tgt: list[int] = []
    s1, s2 = int(shift[0]), int(shift[1])
    for i, (g1, g2) in enumerate(shell):
        j = shell_index.get((g1 + s1, g2 + s2))
        if j is not None:
            src.append(i)
            tgt.append(j)
    return np.asarray(src, dtype=int), np.asarray(tgt, dtype=int)


def _electron_overlap(
    active: ContinuumActiveSpace,
    left: np.ndarray,
    right: np.ndarray,
    shift: tuple[int, int],
) -> np.ndarray:
    shell = tuple((int(g1), int(g2)) for g1, g2 in active.shell)
    shell_index = {g: i for i, g in enumerate(shell)}
    src, tgt = _shift_gather(shell, shell_index, shift)
    na_l = left.shape[1]
    na_r = right.shape[1]
    out = np.zeros((na_l, na_r), dtype=complex)
    if src.size == 0:
        return out
    left_blocks = np.asarray(left, dtype=complex).reshape(2, len(shell), na_l)
    right_blocks = np.asarray(right, dtype=complex).reshape(2, len(shell), na_r)
    for layer in range(2):
        out += np.conj(left_blocks[layer, src, :]).T @ right_blocks[layer, tgt, :]
    return out


def _electron_overlap_grid_to_point(
    active: ContinuumActiveSpace,
    left_grid: np.ndarray,
    right: np.ndarray,
    shift: tuple[int, int],
) -> np.ndarray:
    shell = tuple((int(g1), int(g2)) for g1, g2 in active.shell)
    shell_index = {g: i for i, g in enumerate(shell)}
    src, tgt = _shift_gather(shell, shell_index, shift)
    out = np.zeros((left_grid.shape[0], left_grid.shape[2], right.shape[1]), dtype=complex)
    if src.size == 0:
        return out
    left_blocks = np.asarray(left_grid, dtype=complex).reshape(
        left_grid.shape[0],
        2,
        len(shell),
        left_grid.shape[2],
    )
    right_blocks = np.asarray(right, dtype=complex).reshape(2, len(shell), right.shape[1])
    for layer in range(2):
        out += np.einsum(
            "kxa,xb->kab",
            np.conj(left_blocks[:, layer, src, :]),
            right_blocks[layer, tgt, :],
            optimize=True,
        )
    return out


def _electron_overlap_point_to_grid(
    active: ContinuumActiveSpace,
    left: np.ndarray,
    right_grid: np.ndarray,
    shift: tuple[int, int],
) -> np.ndarray:
    shell = tuple((int(g1), int(g2)) for g1, g2 in active.shell)
    shell_index = {g: i for i, g in enumerate(shell)}
    src, tgt = _shift_gather(shell, shell_index, shift)
    out = np.zeros((right_grid.shape[0], left.shape[1], right_grid.shape[2]), dtype=complex)
    if src.size == 0:
        return out
    left_blocks = np.asarray(left, dtype=complex).reshape(2, len(shell), left.shape[1])
    right_blocks = np.asarray(right_grid, dtype=complex).reshape(
        right_grid.shape[0],
        2,
        len(shell),
        right_grid.shape[2],
    )
    for layer in range(2):
        out += np.einsum(
            "xa,kxb->kab",
            np.conj(left_blocks[layer, src, :]),
            right_blocks[:, layer, tgt, :],
            optimize=True,
        )
    return out


def _fine_lambda_ff(
    active: ContinuumActiveSpace,
    frame: FineMomentumFrame,
    g_channel: tuple[int, int],
) -> np.ndarray:
    out = np.zeros((active.dim, active.dim), dtype=complex)
    for iv in range(len(active.valley_order)):
        start = iv * active.n_active
        stop = start + active.n_active
        overlap = _electron_overlap(
            active,
            frame.electron_vectors[iv],
            frame.electron_vectors[iv],
            g_channel,
        )
        out[start:stop, start:stop] = overlap.T
    return out


def _fine_lambda_fc_grid(
    active: ContinuumActiveSpace,
    frame: FineMomentumFrame,
    fold_shift: np.ndarray,
    g_channel: tuple[int, int],
    coarse_electron_vectors: np.ndarray,
) -> np.ndarray:
    out = np.zeros((active.n_k, active.dim, active.dim), dtype=complex)
    source_shift = np.zeros((active.n_k, 2, 2), dtype=int) if active.source_shift is None else active.source_shift
    g = np.asarray(g_channel, dtype=int)
    for iv in range(len(active.valley_order)):
        start = iv * active.n_active
        stop = start + active.n_active
        shifts = (
            fold_shift
            + frame.physical_shift[iv][None, :]
            - source_shift[:, iv, :]
            + g[None, :]
        ).astype(int)
        coarse_electron = coarse_electron_vectors[:, iv]
        for shift, indices in _group_indices_by_shift(shifts):
            overlap = _electron_overlap_grid_to_point(
                active,
                coarse_electron[indices],
                frame.electron_vectors[iv],
                shift,
            )
            out[indices, start:stop, start:stop] = np.swapaxes(overlap, -1, -2)
    return out


def _fine_lambda_cf_grid(
    active: ContinuumActiveSpace,
    frame: FineMomentumFrame,
    fold_shift: np.ndarray,
    g_channel: tuple[int, int],
    coarse_electron_vectors: np.ndarray,
) -> np.ndarray:
    out = np.zeros((active.n_k, active.dim, active.dim), dtype=complex)
    source_shift = np.zeros((active.n_k, 2, 2), dtype=int) if active.source_shift is None else active.source_shift
    g = np.asarray(g_channel, dtype=int)
    for iv in range(len(active.valley_order)):
        start = iv * active.n_active
        stop = start + active.n_active
        shifts = (
            fold_shift
            + source_shift[:, iv, :]
            - frame.physical_shift[iv][None, :]
            + g[None, :]
        ).astype(int)
        coarse_electron = coarse_electron_vectors[:, iv]
        for shift, indices in _group_indices_by_shift(shifts):
            overlap = _electron_overlap_point_to_grid(
                active,
                frame.electron_vectors[iv],
                coarse_electron[indices],
                shift,
            )
            out[indices, start:stop, start:stop] = np.swapaxes(overlap, -1, -2)
    return out


def _group_indices_by_shift(shifts: np.ndarray):
    unique, inverse = np.unique(np.asarray(shifts, dtype=int), axis=0, return_inverse=True)
    for idx, shift in enumerate(unique):
        yield (int(shift[0]), int(shift[1])), np.flatnonzero(inverse == idx)


def _q_dimless_vectors(
    geometry: MoireGeometry,
    q_frac: np.ndarray,
) -> np.ndarray:
    q = np.asarray(q_frac, dtype=float)
    return q[..., 0, None] * geometry.b1 + q[..., 1, None] * geometry.b2


def _channel_allowed(
    active: ContinuumActiveSpace,
    interaction: ContinuumInteractionParams,
    q_frac: np.ndarray,
) -> np.ndarray:
    q = np.asarray(q_frac, dtype=float)
    physical_cutoff = interaction.momentum_transfer_cutoff_km
    if int(interaction.local_field_cutoff) <= 0 and physical_cutoff is None:
        return np.ones(q.shape[:-1], dtype=bool)
    geometry = active.geometry
    if not isinstance(geometry, MoireGeometry):
        geometry = MoireGeometry(active.model)
    cutoff = (
        float(interaction.local_field_cutoff) * np.sqrt(3.0) / 2.0
        if physical_cutoff is None
        else float(physical_cutoff)
    )
    return np.linalg.norm(_q_dimless_vectors(geometry, q), axis=-1) < cutoff


def _fine_v_over_a(
    active: ContinuumActiveSpace,
    interaction: ContinuumInteractionParams,
    q_frac: np.ndarray,
) -> np.ndarray:
    q = np.asarray(q_frac, dtype=float)
    allowed = _channel_allowed(active, interaction, q)
    out = np.zeros(q.shape[:-1], dtype=float)
    if not np.any(allowed):
        return out

    geometry = active.geometry
    if not isinstance(geometry, MoireGeometry):
        geometry = MoireGeometry(active.model)
    q_dimless = _q_dimless_vectors(geometry, q)
    if interaction.coulomb_kind == "dual_gate":
        q_norm = geometry.kM_inv_nm * np.linalg.norm(q_dimless, axis=-1)
        v_q = np.zeros_like(out, dtype=float)
        nonzero = (q_norm >= 1e-12) & allowed
        for idx in np.ndindex(q_norm.shape):
            if not allowed[idx]:
                continue
            if q_norm[idx] < 1e-12:
                if interaction.include_q0:
                    v_q[idx] = dual_gate_q0_limit_mev_nm2(interaction)
            elif nonzero[idx]:
                v_q[idx] = coulomb_potential_mev_nm2(float(q_norm[idx]), interaction)
        area_nm2 = float(active.grid.size * geometry.moire_cell_area_nm2)
        out = v_q / area_nm2
    else:
        q2 = np.sum(np.square(q), axis=-1)
        out = float(interaction.v0) / (1.0 + q2 * float(interaction.gate_distance) ** 2)
    return np.where(allowed, out, 0.0)


def _fine_hartree_hamiltonian(
    bundle: ContinuumBundle,
    frame: FineMomentumFrame,
    q_density: np.ndarray,
) -> np.ndarray:
    active = bundle.active
    vertices = bundle.vertices
    backend = bundle.backend
    interaction = bundle.interaction
    h = np.zeros((active.dim, active.dim), dtype=complex)
    if float(interaction.hartree_scale) == 0.0:
        return h
    try:
        q0_index = vertices.q_shifts.index((0, 0))
    except ValueError:
        return h
    for ig, g_channel in enumerate(vertices.g_channels):
        q_frac = np.asarray(g_channel, dtype=float)
        if interaction.q0_hartree == "omit_uniform" and np.linalg.norm(q_frac) < 1e-12:
            continue
        v = float(interaction.hartree_scale) * float(_fine_v_over_a(active, interaction, q_frac))
        if v == 0.0:
            continue
        coarse_lambda = backend.hartree_lambda_for_channel(q0_index, ig)
        if coarse_lambda is None:
            continue
        density = np.einsum("kab,kba->", coarse_lambda, q_density, optimize=True)
        fine_lambda = _fine_lambda_ff(active, frame, g_channel)
        h += 0.5 * v * (np.conj(density) * fine_lambda + density * fine_lambda.conj().T)
    return h


def _fine_exchange_hamiltonian(
    bundle: ContinuumBundle,
    frame: FineMomentumFrame,
    q_density: np.ndarray,
    coarse_frac: np.ndarray,
    coarse_electron_vectors: np.ndarray,
) -> np.ndarray:
    active = bundle.active
    vertices = bundle.vertices
    interaction = bundle.interaction
    h = np.zeros((active.dim, active.dim), dtype=complex)
    scale = float(interaction.exchange_scale)
    if scale == 0.0:
        return h

    centered, fold_shift = _center_transfer_array(frame.k_frac[None, :] - coarse_frac)
    centered_rev, fold_shift_rev = _center_transfer_array(coarse_frac - frame.k_frac[None, :])
    for g_channel in vertices.g_channels:
        g = np.asarray(g_channel, dtype=float)
        q_forward = centered + g
        v_forward = _fine_v_over_a(active, interaction, q_forward)
        if np.any(v_forward != 0.0):
            lam = _fine_lambda_fc_grid(
                active,
                frame,
                fold_shift,
                g_channel,
                coarse_electron_vectors,
            )
            h += -0.5 * scale * np.einsum(
                "k,kac,kcd,kbd->ab",
                v_forward,
                lam,
                q_density,
                lam.conj(),
                optimize=True,
            )

        q_reverse = centered_rev + g
        v_reverse = _fine_v_over_a(active, interaction, q_reverse)
        if np.any(v_reverse != 0.0):
            lam_rev = _fine_lambda_cf_grid(
                active,
                frame,
                fold_shift_rev,
                g_channel,
                coarse_electron_vectors,
            )
            h += -0.5 * scale * np.einsum(
                "k,kca,kcd,kdb->ab",
                v_reverse,
                lam_rev.conj(),
                q_density,
                lam_rev,
                optimize=True,
            )
    return h


def hf_hamiltonian_at_k(
    bundle: ContinuumBundle,
    P: np.ndarray,
    k_frac: np.ndarray,
    *,
    q_density: np.ndarray | None = None,
    coarse_frac: np.ndarray | None = None,
    coarse_electron_vectors: np.ndarray | None = None,
    continuum: TaigeContinuumModel | None = None,
) -> np.ndarray:
    """Evaluate the fixed-density HF Hamiltonian at one active-frame momentum."""

    active = bundle.active
    _require_taige_active(active)
    backend = bundle.backend
    frame = taige_active_fine_frame(active, k_frac, continuum=continuum)
    Q = backend.as_block_density(P) - backend.p_ref if q_density is None else q_density
    coarse = _coarse_fractional_grid(active) if coarse_frac is None else coarse_frac
    if coarse_electron_vectors is None:
        if active.electron_vectors is not None:
            coarse_electron_vectors = np.asarray(active.electron_vectors, dtype=complex)
        else:
            coarse_electron_vectors = np.conj(active.hole_vectors[:, :, :, : active.n_active])
    h = np.diag(frame.hole_energies.reshape(-1).astype(complex))
    h += _fine_hartree_hamiltonian(bundle, frame, Q)
    h += _fine_exchange_hamiltonian(bundle, frame, Q, coarse, coarse_electron_vectors)
    return hermitize(h)


def evaluate_hf_path(
    bundle: ContinuumBundle,
    P: np.ndarray,
    path_frac: np.ndarray,
    *,
    distances: np.ndarray | None = None,
    labels_by_index: dict[int, str] | None = None,
    reference: str = "",
) -> HFPathSpectrum:
    """Evaluate fixed-density HF bands and valley weights on a momentum path."""

    active = bundle.active
    _require_taige_active(active)
    path = np.asarray(path_frac, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("path_frac must have shape (n_points, 2)")
    if distances is None:
        geometry = active.geometry if isinstance(active.geometry, MoireGeometry) else MoireGeometry(active.model)
        distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff([geometry.k_from_fractional(k) for k in path], axis=0), axis=1))]
    dist = np.asarray(distances, dtype=float)
    if dist.shape != (path.shape[0],):
        raise ValueError("distances must match path length")
    labels = {} if labels_by_index is None else dict(labels_by_index)

    q_density = bundle.backend.as_block_density(P) - bundle.backend.p_ref
    coarse_frac = _coarse_fractional_grid(active)
    coarse_electron = (
        np.asarray(active.electron_vectors, dtype=complex)
        if active.electron_vectors is not None
        else np.conj(active.hole_vectors[:, :, :, : active.n_active])
    )
    continuum = TaigeContinuumModel(active.model)
    energies = np.empty((path.shape[0], active.dim), dtype=float)
    valley_weights = np.empty((path.shape[0], active.dim, len(active.valley_order)), dtype=float)
    rows: list[HFPathSpectrumRow] = []

    for ip, point in enumerate(path):
        h = hf_hamiltonian_at_k(
            bundle,
            P,
            point,
            q_density=q_density,
            coarse_frac=coarse_frac,
            coarse_electron_vectors=coarse_electron,
            continuum=continuum,
        )
        evals, evecs = np.linalg.eigh(h)
        energies[ip] = evals
        for band in range(active.dim):
            vec = evecs[:, band]
            for iv in range(len(active.valley_order)):
                start = iv * active.n_active
                stop = start + active.n_active
                valley_weights[ip, band, iv] = float(np.real(np.vdot(vec[start:stop], vec[start:stop])))
            rows.append(
                HFPathSpectrumRow(
                    reference=str(reference),
                    path_index=int(ip),
                    k_frac_1=float(point[0]),
                    k_frac_2=float(point[1]),
                    k_distance=float(dist[ip]),
                    label=labels.get(ip),
                    band=int(band),
                    energy=float(evals[band]),
                    valley_K_weight=float(valley_weights[ip, band, active.valley_index(VALLEY_K)]),
                    valley_Kprime_weight=float(valley_weights[ip, band, active.valley_index(VALLEY_KPRIME)]),
                )
            )
    return HFPathSpectrum(
        rows=tuple(rows),
        energies=energies,
        valley_weights=valley_weights,
        path_frac=path,
        distances=dist,
        ticks=tuple(sorted(labels)),
        labels=tuple(labels[idx] for idx in sorted(labels)),
    )


def evaluate_hf_high_symmetry_path(
    bundle: ContinuumBundle,
    P: np.ndarray,
    *,
    n_per_segment: int = 24,
    reference: str = "",
) -> HFPathSpectrum:
    """Evaluate fixed-density HF bands on the standard Taige high-symmetry path."""

    path, distances, ticks, labels = taige_momentum_path(
        n_per_segment=int(n_per_segment),
        model=bundle.params,
    )
    labels_by_index = dict(zip(ticks, labels))
    spectrum = evaluate_hf_path(
        bundle,
        P,
        path,
        distances=distances,
        labels_by_index=labels_by_index,
        reference=reference,
    )
    return HFPathSpectrum(
        rows=spectrum.rows,
        energies=spectrum.energies,
        valley_weights=spectrum.valley_weights,
        path_frac=spectrum.path_frac,
        distances=spectrum.distances,
        ticks=tuple(int(x) for x in ticks),
        labels=tuple(str(x) for x in labels),
    )


def hf_mesh_band_data(active: ContinuumActiveSpace, H_blocks: np.ndarray) -> HFMeshBandData:
    """Diagonalize mesh HF blocks and embed active eigenvectors into Bloch frames."""

    _require_taige_active(active)
    H = hermitize(np.asarray(H_blocks, dtype=complex))
    if H.shape != active.h0.shape:
        raise ValueError("H_blocks must have shape (n_k, dim, dim)")
    energies = np.empty((active.n_k, active.dim), dtype=float)
    active_vectors = np.empty((active.n_k, active.dim, active.dim), dtype=complex)
    for ik in range(active.n_k):
        energies[ik], active_vectors[ik] = np.linalg.eigh(H[ik])
    frames = active_basis_frames(active)
    embedded = np.einsum("kfa,kab->kfb", frames, active_vectors, optimize=True)
    return HFMeshBandData(
        energies=energies,
        active_vectors=active_vectors,
        embedded_vectors=embedded,
    )


def hf_band_chern_table(
    active: ContinuumActiveSpace,
    H_blocks: np.ndarray,
    *,
    reference: str = "",
) -> tuple[HFBandChernRow, ...]:
    """Return embedded Fukui Chern numbers for all HF bands on the active mesh."""

    data = hf_mesh_band_data(active, H_blocks)
    rows: list[HFBandChernRow] = []
    for band in range(active.dim):
        rows.append(
            HFBandChernRow(
                reference=str(reference),
                band=int(band),
                chern=chern_number_on_grid(
                    active.grid,
                    data.embedded_vectors,
                    band,
                    shell=active.shell,
                ),
                energy_min=float(np.min(data.energies[:, band])),
                energy_max=float(np.max(data.energies[:, band])),
            )
        )
    return tuple(rows)


__all__ = [
    "FineMomentumFrame",
    "HFBandChernRow",
    "HFMeshBandData",
    "HFPathSpectrum",
    "HFPathSpectrumRow",
    "evaluate_hf_high_symmetry_path",
    "evaluate_hf_path",
    "hf_band_chern_table",
    "hf_hamiltonian_at_k",
    "hf_mesh_band_data",
    "taige_active_fine_frame",
]
