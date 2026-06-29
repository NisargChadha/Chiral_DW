"""Taige-parameter tMoTe2 continuum, topology, and Coulomb helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from chiral_dw.config import ContinuumFiniteQParams, ContinuumInteractionParams, ContinuumModelParams
from chiral_dw.continuum.models import (
    ContinuumActiveSpace,
    DensityVertices,
    MomentumGrid,
    VALLEY_K,
    VALLEY_KPRIME,
    VALLEY_ORDER,
    hermitize,
)

GridCoord = tuple[int, int]

HBAR2_OVER_2ME_MEV_A2 = 3809.98212
E2_MEV_NM = 1439.96454784255
LAYER_BOTTOM = "bottom"
LAYER_TOP = "top"
LAYER_ORDER = (LAYER_BOTTOM, LAYER_TOP)

TAIGE_THETA_DEG = 3.5
TAIGE_A0_ANGSTROM = 3.47
TAIGE_M_EFF = 0.62
TAIGE_V_MEV = 11.2
TAIGE_PHI_DEG = 91.0
TAIGE_W_MEV = -13.3
TAIGE_EPSILON = 16.7
TAIGE_GATE_DISTANCE_NM = 30.0

_INTRALAYER_SHIFTS: tuple[GridCoord, ...] = ((1, 0), (-1, 1), (0, -1))
_C3_TUNNELING_RECIPROCAL_PARTS: tuple[GridCoord, ...] = ((0, 0), (0, 1), (-1, 1))
TAIGE_IVC_MINUS_TARGET_Q_FRACTIONAL = (1.0 / 3.0, 1.0 / 3.0)
TAIGE_IVC_MINUS_TARGET_HALF_SHIFT_FRACTIONAL = (1.0 / 6.0, 2.0 / 3.0)
TaigeFiniteQShiftPolicy = Literal["exact", "nearest_half"]


class ChernNumberRow(BaseModel):
    """One non-interacting band Chern diagnostic."""

    model_config = ConfigDict(frozen=True)

    basis: Literal["electron", "hole"]
    valley: str
    band: int
    chern: float


class TaigeFiniteQShiftChoice(BaseModel):
    """Chosen finite-Q IVC- mesh shift and its target-frame errors."""

    model_config = ConfigDict(frozen=True)

    policy: TaigeFiniteQShiftPolicy
    n_k: int
    q_coord: GridCoord
    half_shift_coord: GridCoord
    exact: bool
    target_q_fractional: tuple[float, float] = TAIGE_IVC_MINUS_TARGET_Q_FRACTIONAL
    target_half_shift_fractional: tuple[float, float] = (
        TAIGE_IVC_MINUS_TARGET_HALF_SHIFT_FRACTIONAL
    )
    q_fractional: tuple[float, float]
    half_shift_fractional: tuple[float, float]
    q_error_fractional: tuple[float, float]
    half_shift_error_fractional: tuple[float, float]
    q_error_grid_units: float
    half_shift_error_grid_units: float
    q_error_fractional_norm: float
    half_shift_error_fractional_norm: float


@dataclass(frozen=True)
class MoireGeometry:
    """Dimensionless and physical moire geometry for the Taige continuum model."""

    model: ContinuumModelParams

    def __post_init__(self) -> None:
        theta_rad = np.deg2rad(self.model.theta_deg)
        a0_angstrom = float(self.model.a0_angstrom)
        a_m_angstrom = a0_angstrom / (2.0 * np.sin(0.5 * theta_rad))
        object.__setattr__(self, "theta_rad", theta_rad)
        object.__setattr__(self, "aM_angstrom", a_m_angstrom)
        object.__setattr__(self, "aM_nm", a_m_angstrom / 10.0)
        object.__setattr__(
            self,
            "kM_inv_angstrom",
            4.0 * np.pi / (np.sqrt(3.0) * a_m_angstrom),
        )
        object.__setattr__(self, "kM_inv_nm", 10.0 * self.kM_inv_angstrom)
        object.__setattr__(self, "b1", np.array([1.0, 0.0]))
        object.__setattr__(self, "b2", np.array([0.5, np.sqrt(3.0) / 2.0]))
        object.__setattr__(self, "kappa_plus", (2.0 * self.b1 - self.b2) / 3.0)
        object.__setattr__(self, "kappa_minus", (self.b1 + self.b2) / 3.0)
        object.__setattr__(
            self,
            "moire_cell_area_nm2",
            (np.sqrt(3.0) / 2.0) * self.aM_nm**2,
        )

    def cartesian(self, coord: GridCoord) -> np.ndarray:
        return int(coord[0]) * self.b1 + int(coord[1]) * self.b2

    def k_from_fractional(self, k_frac: np.ndarray) -> np.ndarray:
        k = np.asarray(k_frac, dtype=float)
        return k[0] * self.b1 + k[1] * self.b2

    def mesh_q_vectors_nm_inv(
        self,
        grid: MomentumGrid,
        q_list: tuple[GridCoord, ...],
        g_channels: tuple[GridCoord, ...],
    ) -> np.ndarray:
        out = np.zeros((len(q_list), len(g_channels), 2), dtype=float)
        for iq, (qi, qj) in enumerate(q_list):
            for ig, (g1, g2) in enumerate(g_channels):
                dimless = (qi / grid.n1 + g1) * self.b1 + (qj / grid.n2 + g2) * self.b2
                out[iq, ig] = self.kM_inv_nm * dimless
        return out


@dataclass(frozen=True)
class TaigeBandStructure:
    """Plane-wave Taige continuum bands in a T-prime-generated valley gauge."""

    model: ContinuumModelParams
    grid: MomentumGrid
    n_shell: int
    n_bands: int
    shell: tuple[GridCoord, ...]
    n_plane_waves: int
    electron_energies: np.ndarray
    electron_vectors: np.ndarray
    hole_energies: np.ndarray
    hole_vectors: np.ndarray
    geometry: MoireGeometry
    tprime_partner_index: np.ndarray
    tprime_sewing_quality: np.ndarray
    gauge: str = "tprime_generated"

    @property
    def valley_order(self) -> tuple[str, str]:
        return VALLEY_ORDER

    def valley_index(self, valley: str) -> int:
        return self.valley_order.index(str(valley))


def taige_model_params(
    *,
    theta_deg: float = TAIGE_THETA_DEG,
    u_D: float = 0.0,
    plane_wave_shell: int = 1,
    n_bands: int = 2,
    n_active_bands_per_valley: int = 1,
) -> ContinuumModelParams:
    """Return Chiral_DW-native Taige model parameters."""

    return ContinuumModelParams(
        theta_deg=float(theta_deg),
        a0_angstrom=TAIGE_A0_ANGSTROM,
        m_eff=TAIGE_M_EFF,
        moire_potential_mev=TAIGE_V_MEV,
        phi_deg=TAIGE_PHI_DEG,
        tunneling_mev=TAIGE_W_MEV,
        displacement_mev=float(u_D),
        plane_wave_shell=int(plane_wave_shell),
        n_bands=int(n_bands),
        n_active_bands_per_valley=int(n_active_bands_per_valley),
        active_model="taige",
    )


def taige_interaction_params(
    *,
    include_q0: bool = True,
    q_mesh: Literal["shell", "full"] = "shell",
    q_shell: int = 1,
    local_field_cutoff: int = 0,
    epsilon: float = TAIGE_EPSILON,
    gate_distance_nm: float = TAIGE_GATE_DISTANCE_NM,
    smear_length_nm: float | None = None,
    interaction_strength_scale: float = 1.0,
    hartree_scale: float = 1.0,
    exchange_scale: float = 1.0,
) -> ContinuumInteractionParams:
    """Return dual-gated smeared Coulomb parameters for Taige MoTe2."""

    smear = TAIGE_A0_ANGSTROM / 10.0 if smear_length_nm is None else float(smear_length_nm)
    return ContinuumInteractionParams(
        v0=float(interaction_strength_scale),
        coulomb_kind="dual_gate",
        epsilon=float(epsilon),
        gate_distance_nm=float(gate_distance_nm),
        include_q0=bool(include_q0),
        smear_length_nm=smear,
        q_mesh=q_mesh,
        q_shell=int(q_shell),
        local_field_cutoff=int(local_field_cutoff),
        hartree_scale=float(hartree_scale),
        exchange_scale=float(exchange_scale),
    )


def taige_ivc_minus_q_coord(n_k: int) -> GridCoord:
    """Return the folded Taige IVC- Q mesh coordinate."""

    n = int(n_k)
    if n % 6:
        raise ValueError("Taige IVC- finite-Q runs require n_k divisible by 6")
    return n // 3, n // 3


def taige_ivc_minus_half_shift_coord(n_k: int) -> GridCoord:
    """Return the unfolded-half Taige IVC- Q/2 mesh representative."""

    n = int(n_k)
    if n % 6:
        raise ValueError("Taige IVC- finite-Q runs require n_k divisible by 6")
    return n // 6, 2 * n // 3


def _centered_fractional_delta(value: float, target: float) -> float:
    return float((float(value) - float(target) + 0.5) % 1.0 - 0.5)


def _triangular_fractional_norm(delta: tuple[float, float]) -> float:
    x, y = float(delta[0]), float(delta[1])
    return float(np.sqrt(max(x * x + y * y + x * y, 0.0)))


def _taige_shift_choice_from_coords(
    *,
    n_k: int,
    policy: TaigeFiniteQShiftPolicy,
    q_coord: GridCoord,
    half_shift_coord: GridCoord,
) -> TaigeFiniteQShiftChoice:
    n = int(n_k)
    q = (int(q_coord[0]) % n, int(q_coord[1]) % n)
    half = (int(half_shift_coord[0]) % n, int(half_shift_coord[1]) % n)
    q_frac = (q[0] / n, q[1] / n)
    half_frac = (half[0] / n, half[1] / n)
    q_error = (
        _centered_fractional_delta(q_frac[0], TAIGE_IVC_MINUS_TARGET_Q_FRACTIONAL[0]),
        _centered_fractional_delta(q_frac[1], TAIGE_IVC_MINUS_TARGET_Q_FRACTIONAL[1]),
    )
    half_error = (
        _centered_fractional_delta(
            half_frac[0], TAIGE_IVC_MINUS_TARGET_HALF_SHIFT_FRACTIONAL[0]
        ),
        _centered_fractional_delta(
            half_frac[1], TAIGE_IVC_MINUS_TARGET_HALF_SHIFT_FRACTIONAL[1]
        ),
    )
    q_norm = _triangular_fractional_norm(q_error)
    half_norm = _triangular_fractional_norm(half_error)
    exact = bool(
        max(abs(q_error[0]), abs(q_error[1]), abs(half_error[0]), abs(half_error[1]))
        < 1e-12
    )
    return TaigeFiniteQShiftChoice(
        policy=policy,
        n_k=n,
        q_coord=q,
        half_shift_coord=half,
        exact=exact,
        q_fractional=q_frac,
        half_shift_fractional=half_frac,
        q_error_fractional=q_error,
        half_shift_error_fractional=half_error,
        q_error_grid_units=float(n * q_norm),
        half_shift_error_grid_units=float(n * half_norm),
        q_error_fractional_norm=q_norm,
        half_shift_error_fractional_norm=half_norm,
    )


def taige_ivc_minus_shift_choice(
    n_k: int,
    *,
    policy: TaigeFiniteQShiftPolicy = "exact",
) -> TaigeFiniteQShiftChoice:
    """Return the Taige IVC- finite-Q mesh shift for a requested policy."""

    n = int(n_k)
    if n < 1:
        raise ValueError("n_k must be positive")
    normalized_policy = str(policy).replace("-", "_")
    if normalized_policy not in {"exact", "nearest_half"}:
        raise ValueError("finite-Q shift policy must be 'exact' or 'nearest_half'")
    policy_key: TaigeFiniteQShiftPolicy = normalized_policy  # type: ignore[assignment]
    if policy_key == "exact":
        return _taige_shift_choice_from_coords(
            n_k=n,
            policy=policy_key,
            q_coord=taige_ivc_minus_q_coord(n),
            half_shift_coord=taige_ivc_minus_half_shift_coord(n),
        )

    best: tuple[float, float, int, int, int, int] | None = None
    for h1 in range(n):
        for h2 in range(n):
            half_frac = (h1 / n, h2 / n)
            half_error = (
                _centered_fractional_delta(
                    half_frac[0], TAIGE_IVC_MINUS_TARGET_HALF_SHIFT_FRACTIONAL[0]
                ),
                _centered_fractional_delta(
                    half_frac[1], TAIGE_IVC_MINUS_TARGET_HALF_SHIFT_FRACTIONAL[1]
                ),
            )
            half_norm = _triangular_fractional_norm(half_error)
            q1 = (2 * h1) % n
            q2 = (2 * h2) % n
            q_frac = (q1 / n, q2 / n)
            q_error = (
                _centered_fractional_delta(
                    q_frac[0], TAIGE_IVC_MINUS_TARGET_Q_FRACTIONAL[0]
                ),
                _centered_fractional_delta(
                    q_frac[1], TAIGE_IVC_MINUS_TARGET_Q_FRACTIONAL[1]
                ),
            )
            q_norm = _triangular_fractional_norm(q_error)
            candidate = (half_norm, q_norm, h1, h2, q1, q2)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError("failed to choose a finite-Q half shift")
    _half_norm, _q_norm, h1, h2, q1, q2 = best
    return _taige_shift_choice_from_coords(
        n_k=n,
        policy=policy_key,
        q_coord=(q1, q2),
        half_shift_coord=(h1, h2),
    )


def reciprocal_shell(n_shell: int) -> tuple[GridCoord, ...]:
    """Hexagonal reciprocal shell in integer b1,b2 coordinates."""

    n = int(n_shell)
    out: list[GridCoord] = []
    for m1 in range(-n, n + 1):
        for m2 in range(-n, n + 1):
            if max(abs(m1), abs(m2), abs(m1 + m2)) <= n:
                out.append((m1, m2))
    out.sort(key=lambda g: (g[0] ** 2 + g[1] ** 2 + g[0] * g[1], g[0], g[1]))
    return tuple(out)


def centered_mesh_transfers(grid: MomentumGrid) -> tuple[GridCoord, ...]:
    """Transfer-momentum representatives centered around zero."""

    def axis(n: int) -> list[int]:
        return [i if i <= n // 2 else i - n for i in range(n)]

    return tuple((i, j) for i in axis(grid.n1) for j in axis(grid.n2))


def reciprocal_box(g_cutoff: int) -> tuple[GridCoord, ...]:
    n = int(g_cutoff)
    coords = [(g1, g2) for g1 in range(-n, n + 1) for g2 in range(-n, n + 1)]
    coords.sort(key=lambda c: (c[0] ** 2 + c[1] ** 2 + c[0] * c[1], c[0], c[1]))
    return tuple(coords)


def valley_to_tau(valley: str) -> int:
    if valley == VALLEY_K:
        return +1
    if valley == VALLEY_KPRIME:
        return -1
    raise ValueError(f"unknown valley {valley!r}")


class TaigeContinuumModel:
    """C3-gauge two-layer continuum Hamiltonian in meV."""

    def __init__(self, model: ContinuumModelParams) -> None:
        self.model = model
        self.geometry = MoireGeometry(model)
        self.shell = reciprocal_shell(model.plane_wave_shell)
        self.shell_index = {g: i for i, g in enumerate(self.shell)}
        self.n_plane_waves = len(self.shell)
        self.dim = 2 * self.n_plane_waves
        self.psi = np.deg2rad(model.phi_deg)
        self._kin_scale = (
            HBAR2_OVER_2ME_MEV_A2
            / float(model.m_eff)
            * self.geometry.kM_inv_angstrom**2
        )
        self._g_cart = np.array([self.geometry.cartesian(g) for g in self.shell])

    def _c3_basis_momentum_offset(self, valley: str, layer: str) -> np.ndarray:
        tau = valley_to_tau(valley)
        if layer == LAYER_BOTTOM:
            return -tau * self.geometry.kappa_plus
        if layer == LAYER_TOP:
            return -tau * self.geometry.kappa_minus
        raise ValueError(f"unknown layer {layer!r}")

    def plane_wave_momenta(self, k_frac: np.ndarray, valley: str, layer: str) -> np.ndarray:
        k_cart = self.geometry.k_from_fractional(k_frac)
        offset = self._c3_basis_momentum_offset(valley, layer)
        return k_cart[None, :] + self._g_cart + offset[None, :]

    def kinetic_energy(self, momenta: np.ndarray) -> np.ndarray:
        return -self._kin_scale * np.einsum("...i,...i->...", momenta, momenta)

    @staticmethod
    def _c3_tunneling_matrix_shifts(tau: int) -> tuple[GridCoord, ...]:
        return tuple((-tau * g1, -tau * g2) for g1, g2 in _C3_TUNNELING_RECIPROCAL_PARTS)

    def hamiltonian(self, k_frac: np.ndarray, valley: str) -> np.ndarray:
        tau = valley_to_tau(valley)
        n = self.n_plane_waves
        h_bb = np.zeros((n, n), dtype=complex)
        h_tt = np.zeros((n, n), dtype=complex)
        h_bt = np.zeros((n, n), dtype=complex)

        kin_b = self.kinetic_energy(self.plane_wave_momenta(k_frac, valley, LAYER_BOTTOM))
        kin_t = self.kinetic_energy(self.plane_wave_momenta(k_frac, valley, LAYER_TOP))
        np.fill_diagonal(h_bb, kin_b - float(self.model.displacement_mev))
        np.fill_diagonal(h_tt, kin_t + float(self.model.displacement_mev))

        v_plus = float(self.model.moire_potential_mev) * np.exp(1j * self.psi)
        v_minus = float(self.model.moire_potential_mev) * np.exp(-1j * self.psi)
        for d1, d2 in _INTRALAYER_SHIFTS:
            for col, (g1, g2) in enumerate(self.shell):
                row = self.shell_index.get((g1 + d1, g2 + d2))
                if row is not None:
                    h_bb[row, col] += v_plus
                    h_tt[row, col] += v_minus
                row = self.shell_index.get((g1 - d1, g2 - d2))
                if row is not None:
                    h_bb[row, col] += v_minus
                    h_tt[row, col] += v_plus

        for d1, d2 in self._c3_tunneling_matrix_shifts(tau):
            for col, (g1, g2) in enumerate(self.shell):
                row = self.shell_index.get((g1 + d1, g2 + d2))
                if row is not None:
                    h_bt[row, col] += float(self.model.tunneling_mev)

        h = np.block([[h_bb, h_bt], [h_bt.conj().T, h_tt]])
        return hermitize(h)


def mesh_inversion_map_with_shifts(grid: MomentumGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return partner indices and reciprocal shifts for k -> -k."""

    partner = np.empty(grid.size, dtype=int)
    shifts = np.empty((grid.size, 2), dtype=int)
    for ik in range(grid.size):
        i, j = grid.coord_of(ik)
        folded, shift = grid.fold_grid_coord((-i, -j))
        partner[ik] = grid.index_of(folded)
        shifts[ik] = shift
    return partner, shifts


def _loewdin_orthonormalize(vectors: np.ndarray, *, eps: float = 1e-12) -> tuple[np.ndarray, float]:
    raw = np.asarray(vectors, dtype=complex)
    gram = hermitize(raw.conj().T @ raw)
    evals, evecs = np.linalg.eigh(gram)
    min_eval = float(np.min(np.real(evals)))
    if min_eval <= eps:
        q, _r = np.linalg.qr(raw)
        return q[:, : raw.shape[1]], min_eval
    inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(np.real(evals))) @ evecs.conj().T
    return raw @ inv_sqrt, min_eval


def sew_tprime_electron_vectors(
    k_vectors: np.ndarray,
    shell: tuple[GridCoord, ...],
    fold_shift: GridCoord,
) -> tuple[np.ndarray, float]:
    """Generate Kprime electron vectors from K vectors by T'=tau_x K."""

    vectors = np.asarray(k_vectors, dtype=complex)
    n_plane_waves = len(shell)
    if vectors.shape[0] != 2 * n_plane_waves:
        raise ValueError("k_vectors size is incompatible with the plane-wave shell")
    shell_index = {g: i for i, g in enumerate(shell)}
    s1, s2 = int(fold_shift[0]), int(fold_shift[1])
    out = np.zeros_like(vectors, dtype=complex)
    for layer in range(2):
        offset = layer * n_plane_waves
        for src, (g1, g2) in enumerate(shell):
            target = shell_index.get((-g1 + s1, -g2 + s2))
            if target is not None:
                out[offset + target] = np.conj(vectors[offset + src])
    return _loewdin_orthonormalize(out)


def compute_taige_bandstructure(model: ContinuumModelParams, grid: MomentumGrid) -> TaigeBandStructure:
    """Diagonalize the Taige continuum model on a momentum mesh."""

    continuum = TaigeContinuumModel(model)
    n_bands = int(model.n_bands)
    if n_bands > continuum.dim:
        raise ValueError("n_bands exceeds the plane-wave Hamiltonian dimension")
    electron_energies = np.empty((grid.size, 2, n_bands), dtype=float)
    electron_vectors = np.empty((grid.size, 2, continuum.dim, n_bands), dtype=complex)
    partner_index, partner_shift = mesh_inversion_map_with_shifts(grid)
    sewing_quality = np.ones(grid.size, dtype=float)
    k_index = VALLEY_ORDER.index(VALLEY_K)
    kp_index = VALLEY_ORDER.index(VALLEY_KPRIME)

    for ik in range(grid.size):
        i, j = grid.coord_of(ik)
        k_frac = np.array((i / grid.n1, j / grid.n2), dtype=float)
        evals, evecs = np.linalg.eigh(continuum.hamiltonian(k_frac, VALLEY_K))
        order = np.argsort(evals)[::-1][:n_bands]
        electron_energies[ik, k_index] = evals[order]
        electron_vectors[ik, k_index] = evecs[:, order]

    for ik in range(grid.size):
        partner = int(partner_index[ik])
        electron_energies[ik, kp_index] = electron_energies[partner, k_index]
        generated, quality = sew_tprime_electron_vectors(
            electron_vectors[partner, k_index],
            continuum.shell,
            tuple(int(x) for x in partner_shift[ik]),
        )
        electron_vectors[ik, kp_index] = generated
        sewing_quality[ik] = quality

    return TaigeBandStructure(
        model=model,
        grid=grid,
        n_shell=model.plane_wave_shell,
        n_bands=n_bands,
        shell=continuum.shell,
        n_plane_waves=continuum.n_plane_waves,
        electron_energies=electron_energies,
        electron_vectors=electron_vectors,
        hole_energies=-electron_energies,
        hole_vectors=np.conj(electron_vectors),
        geometry=continuum.geometry,
        tprime_partner_index=partner_index,
        tprime_sewing_quality=sewing_quality,
    )


def build_taige_active_space(
    grid: MomentumGrid,
    model: ContinuumModelParams,
    finite_q: ContinuumFiniteQParams | None = None,
) -> tuple[ContinuumActiveSpace, TaigeBandStructure]:
    """Build the Taige active hole basis, optionally in a finite-Q frame."""

    bands = compute_taige_bandstructure(model, grid)
    finite_q_params = finite_q or ContinuumFiniteQParams()
    finite_q_enabled = bool(finite_q_params.enabled)
    q_coord = finite_q_params.q_coord if finite_q_enabled else None
    half_shift_coord = finite_q_params.half_shift_coord if finite_q_enabled else None
    if finite_q_enabled:
        grid.assert_half_q_on_mesh(q_coord, half_shift_coord)
    n_active = int(model.n_active_bands_per_valley)
    if n_active > bands.n_bands:
        raise ValueError("n_active_bands_per_valley exceeds computed n_bands")
    dim = 2 * n_active
    hole_energies = np.empty((grid.size, 2, n_active), dtype=float)
    hole_vectors = np.empty((grid.size, 2, 2 * bands.n_plane_waves, n_active), dtype=complex)
    electron_energies = np.empty((grid.size, 2, n_active), dtype=float)
    electron_vectors = np.empty((grid.size, 2, 2 * bands.n_plane_waves, n_active), dtype=complex)
    source_index = np.empty((grid.size, 2), dtype=int)
    source_shift = np.zeros((grid.size, 2, 2), dtype=int)
    for ik in range(grid.size):
        coord = grid.coord_of(ik)
        for iv, valley in enumerate(VALLEY_ORDER):
            if finite_q_enabled:
                folded, shift = grid.finite_q_physical_coord(
                    coord,
                    q_coord,
                    valley,
                    half_shift_coord,
                )
                src = grid.index_of(folded)
                source_shift[ik, iv] = shift
            else:
                src = ik
            source_index[ik, iv] = src
            hole_energies[ik, iv] = bands.hole_energies[src, iv, :n_active]
            hole_vectors[ik, iv] = bands.hole_vectors[src, iv, :, :n_active]
            electron_energies[ik, iv] = bands.electron_energies[src, iv, :n_active]
            electron_vectors[ik, iv] = bands.electron_vectors[src, iv, :, :n_active]
    h0 = np.zeros((grid.size, dim, dim), dtype=complex)
    for ik in range(grid.size):
        h0[ik] = np.diag(hole_energies[ik].reshape(-1).astype(complex))
    active = ContinuumActiveSpace(
        grid=grid,
        n_active=n_active,
        h0=hermitize(h0),
        hole_energies=hole_energies,
        band_vectors=hole_vectors,
        model=model,
        shell=bands.shell,
        n_plane_waves=bands.n_plane_waves,
        electron_energies=electron_energies,
        electron_vectors=electron_vectors,
        source_index=source_index,
        source_shift=source_shift,
        finite_q_enabled=finite_q_enabled,
        q_coord=q_coord,
        half_shift_coord=half_shift_coord,
        geometry=bands.geometry,
        bands=bands,
    )
    return active, bands


def q_transfers(grid: MomentumGrid, interaction: ContinuumInteractionParams) -> tuple[GridCoord, ...]:
    if interaction.q_mesh == "full":
        return centered_mesh_transfers(grid)
    radius = int(interaction.q_shell)
    shifts = [
        (di, dj)
        for di in range(-radius, radius + 1)
        for dj in range(-radius, radius + 1)
        if max(abs(di), abs(dj)) <= radius
    ]
    shifts = sorted(set(shifts), key=lambda x: (abs(x[0]) + abs(x[1]), x[0], x[1]))
    return tuple(shifts)


def _shift_gather(
    shell: tuple[GridCoord, ...],
    shell_index: dict[GridCoord, int],
    shift: GridCoord,
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


def _overlap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    shell: tuple[GridCoord, ...],
    shell_index: dict[GridCoord, int],
    shift: GridCoord,
) -> np.ndarray:
    src, tgt = _shift_gather(shell, shell_index, shift)
    na_l = left.shape[1]
    na_r = right.shape[1]
    out = np.zeros((na_l, na_r), dtype=complex)
    if src.size == 0:
        return out
    n_plane_waves = len(shell)
    left_blocks = left.reshape(2, n_plane_waves, na_l)
    right_blocks = right.reshape(2, n_plane_waves, na_r)
    for layer in range(2):
        out += np.conj(left_blocks[layer][src]).T @ right_blocks[layer][tgt]
    return out


def _channel_mask(
    geometry: MoireGeometry,
    grid: MomentumGrid,
    q_list: tuple[GridCoord, ...],
    g_channels: tuple[GridCoord, ...],
    local_field_cutoff: int,
) -> np.ndarray:
    mask = np.ones((len(q_list), len(g_channels)), dtype=bool)
    if int(local_field_cutoff) <= 0:
        return mask
    cutoff = float(local_field_cutoff) * np.sqrt(3.0) / 2.0
    for iq, (qi, qj) in enumerate(q_list):
        for ig, (g1, g2) in enumerate(g_channels):
            q_cart = (qi / grid.n1 + g1) * geometry.b1 + (qj / grid.n2 + g2) * geometry.b2
            mask[iq, ig] = np.linalg.norm(q_cart) < cutoff
    return mask


def coulomb_potential_mev_nm2(
    q_nm_inv: float,
    interaction: ContinuumInteractionParams,
) -> float:
    """Return V(q) in meV nm^2 for q > 0."""

    q = float(q_nm_inv)
    if q <= 0.0:
        raise ValueError("q=0 is handled separately")
    base = 2.0 * np.pi * E2_MEV_NM / (float(interaction.epsilon) * q)
    value = base * np.tanh(q * float(interaction.gate_distance_nm))
    if interaction.smear_length_nm > 0.0:
        value *= np.exp(-0.5 * (q * interaction.smear_length_nm) ** 2)
    return float(interaction.v0 * value)


def dual_gate_q0_limit_mev_nm2(interaction: ContinuumInteractionParams) -> float:
    return float(
        interaction.v0
        * 2.0
        * np.pi
        * E2_MEV_NM
        * float(interaction.gate_distance_nm)
        / float(interaction.epsilon)
    )


def _physical_v_over_a(
    geometry: MoireGeometry,
    grid: MomentumGrid,
    q_vectors_nm_inv: np.ndarray,
    channel_mask: np.ndarray,
    interaction: ContinuumInteractionParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_norm = np.linalg.norm(q_vectors_nm_inv, axis=-1)
    v_q = np.zeros_like(q_norm, dtype=float)
    q_zero_channels = q_norm < 1e-12
    for idx in np.ndindex(q_norm.shape):
        if not channel_mask[idx]:
            continue
        q = float(q_norm[idx])
        if q < 1e-12:
            if interaction.include_q0:
                v_q[idx] = dual_gate_q0_limit_mev_nm2(interaction)
        else:
            v_q[idx] = coulomb_potential_mev_nm2(q, interaction)
    area_nm2 = float(grid.size * geometry.moire_cell_area_nm2)
    return v_q, v_q / area_nm2, q_zero_channels


def _dimensionless_v_over_a(
    q_list: tuple[GridCoord, ...],
    g_channels: tuple[GridCoord, ...],
    interaction: ContinuumInteractionParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.zeros((len(q_list), len(g_channels)), dtype=float)
    q_zero = np.zeros_like(v, dtype=bool)
    for iq, (di, dj) in enumerate(q_list):
        for ig, (g1, g2) in enumerate(g_channels):
            q2 = float((di + g1) ** 2 + (dj + g2) ** 2)
            q_zero[iq, ig] = q2 == 0.0
            v[iq, ig] = interaction.v0 / (1.0 + q2 * interaction.gate_distance**2)
    return v.copy(), v, q_zero


def build_taige_density_vertices(
    active: ContinuumActiveSpace,
    interaction: ContinuumInteractionParams,
) -> DensityVertices:
    """Build projected hole density vertices and Taige interaction weights."""

    if active.bands is None or active.geometry is None:
        raise ValueError("Taige density vertices require bandstructure and geometry")
    grid = active.grid
    q_list = q_transfers(grid, interaction)
    g_channels = reciprocal_box(interaction.local_field_cutoff)
    n_q = len(q_list)
    n_g = len(g_channels)
    target_minus_q = np.empty((n_q, grid.size), dtype=int)
    q_is_zero = np.zeros(n_q, dtype=bool)
    lambdas = np.zeros((n_q, n_g, grid.size, active.dim, active.dim), dtype=complex)
    shell = active.shell
    shell_index = {g: i for i, g in enumerate(shell)}
    bands = active.bands
    electron_vectors = np.asarray(bands.electron_vectors, dtype=complex)
    geometry = active.geometry
    channel_in_disk = _channel_mask(
        geometry,
        grid,
        q_list,
        g_channels,
        interaction.local_field_cutoff,
    )

    def electron_form_factor(iv: int, ik: int, q_mesh: GridCoord, g_channel: GridCoord) -> np.ndarray:
        forward, rec_shift = grid.shift_plus_q(grid.coord_of(ik), q_mesh)
        ikq = grid.index_of(forward)
        shift = (rec_shift[0] + int(g_channel[0]), rec_shift[1] + int(g_channel[1]))
        left = electron_vectors[ik, iv, :, : active.n_active]
        right = electron_vectors[ikq, iv, :, : active.n_active]
        return _overlap(left, right, shell=shell, shell_index=shell_index, shift=shift)

    def hole_form_factor(iv: int, ik: int, q_mesh: GridCoord, g_channel: GridCoord) -> np.ndarray:
        ik_minus_q = grid.index_of(grid.shift_minus_q(grid.coord_of(ik), q_mesh)[0])
        return electron_form_factor(iv, ik_minus_q, q_mesh, g_channel).T

    for iq, q in enumerate(q_list):
        q_is_zero[iq] = q == (0, 0)
        for ik in range(grid.size):
            folded, _shift = grid.shift_minus_q(grid.coord_of(ik), q)
            target_minus_q[iq, ik] = grid.index_of(folded)
        for ig, g in enumerate(g_channels):
            if not channel_in_disk[iq, ig]:
                continue
            for ik in range(grid.size):
                for iv in range(2):
                    start = iv * active.n_active
                    stop = start + active.n_active
                    physical_source = (
                        int(active.source_index[ik, iv])
                        if active.source_index is not None
                        else int(ik)
                    )
                    lambdas[iq, ig, ik, start:stop, start:stop] = hole_form_factor(
                        iv,
                        physical_source,
                        q,
                        g,
                    )

    q_vectors_nm_inv = geometry.mesh_q_vectors_nm_inv(grid, q_list, g_channels)
    if interaction.coulomb_kind == "dual_gate":
        v_q, v_over_a, q_zero_channels = _physical_v_over_a(
            geometry,
            grid,
            q_vectors_nm_inv,
            channel_in_disk,
            interaction,
        )
    else:
        v_q, v_over_a, q_zero_channels = _dimensionless_v_over_a(q_list, g_channels, interaction)
        v_over_a = np.where(channel_in_disk, v_over_a, 0.0)
        v_q = np.where(channel_in_disk, v_q, 0.0)

    return DensityVertices(
        q_shifts=q_list,
        target_minus_q=target_minus_q,
        q_is_zero=np.any(q_zero_channels, axis=-1),
        lambda_blocks=lambdas,
        v_over_a=v_over_a,
        g_channels=g_channels,
        channel_in_disk=channel_in_disk,
        q_vectors_nm_inv=q_vectors_nm_inv,
        q_norm_nm_inv=np.linalg.norm(q_vectors_nm_inv, axis=-1),
        v_q=v_q,
    )


def _normalized_overlap(overlap: complex) -> complex:
    mag = abs(overlap)
    return 1.0 + 0.0j if mag < 1e-14 else overlap / mag


def unit_link(u: np.ndarray, v: np.ndarray) -> complex:
    return _normalized_overlap(np.vdot(u, v))


def _sewn_unit_link(
    u: np.ndarray,
    v: np.ndarray,
    *,
    shell: tuple[GridCoord, ...],
    shift: GridCoord,
    shell_index: dict[GridCoord, int],
) -> complex:
    left = np.asarray(u, dtype=complex)
    right = np.asarray(v, dtype=complex)
    if shift == (0, 0):
        return unit_link(left, right)
    src, tgt = _shift_gather(shell, shell_index, shift)
    if src.size == 0:
        return 1.0 + 0.0j
    n_internal = left.size // len(shell)
    left_blocks = left.reshape(n_internal, len(shell))
    right_blocks = right.reshape(n_internal, len(shell))
    overlap = sum(
        np.vdot(left_blocks[component, src], right_blocks[component, tgt])
        for component in range(n_internal)
    )
    return _normalized_overlap(overlap)


def chern_number_on_grid(
    grid: MomentumGrid,
    vectors: np.ndarray,
    band: int,
    *,
    shell: tuple[GridCoord, ...],
) -> float:
    """Compute a single-band Fukui Chern number with reciprocal sewing."""

    vecs = np.asarray(vectors, dtype=complex)
    shell_index = {g: i for i, g in enumerate(shell)}
    total = 0.0
    for i in range(grid.n1):
        for j in range(grid.n2):
            u00 = vecs[grid.index_of((i, j)), :, band]
            c10, s10 = grid.shift_plus_q((i, j), (1, 0))
            c01, s01 = grid.shift_plus_q((i, j), (0, 1))
            c11_y, s11_y = grid.shift_plus_q(c10, (0, 1))
            c11_x, s11_x = grid.shift_plus_q(c01, (1, 0))
            u10 = vecs[grid.index_of(c10), :, band]
            u01 = vecs[grid.index_of(c01), :, band]
            u11 = vecs[grid.index_of(c11_y), :, band]
            link_x_00 = _sewn_unit_link(u00, u10, shell=shell, shift=s10, shell_index=shell_index)
            link_y_10 = _sewn_unit_link(u10, u11, shell=shell, shift=s11_y, shell_index=shell_index)
            link_x_01 = _sewn_unit_link(u01, u11, shell=shell, shift=s11_x, shell_index=shell_index)
            link_y_00 = _sewn_unit_link(u00, u01, shell=shell, shift=s01, shell_index=shell_index)
            total += np.angle(link_x_00 * link_y_10 * np.conj(link_x_01) * np.conj(link_y_00))
            if c11_x != c11_y:
                raise RuntimeError("inconsistent periodic plaquette corner")
    return float(total / (2.0 * np.pi))


def chern_number_table(
    bands: TaigeBandStructure,
    *,
    bases: tuple[Literal["electron", "hole"], ...] = ("electron", "hole"),
    band_indices: tuple[int, ...] | None = None,
) -> list[ChernNumberRow]:
    """Return Chern numbers for selected non-interacting bands."""

    indices = tuple(range(bands.n_bands)) if band_indices is None else tuple(int(i) for i in band_indices)
    rows: list[ChernNumberRow] = []
    for basis in bases:
        vectors = bands.electron_vectors if basis == "electron" else bands.hole_vectors
        for valley in bands.valley_order:
            iv = bands.valley_index(valley)
            for band in indices:
                rows.append(
                    ChernNumberRow(
                        basis=basis,
                        valley=valley,
                        band=band,
                        chern=chern_number_on_grid(
                            bands.grid,
                            vectors[:, iv],
                            band,
                            shell=bands.shell,
                        ),
                    )
                )
    return rows


def taige_path_nodes() -> list[tuple[str, tuple[float, float]]]:
    return [
        ("Gamma", (0.0, 0.0)),
        ("kappa+", (2.0 / 3.0, -1.0 / 3.0)),
        ("m", (0.5, 0.0)),
        ("kappa-", (1.0 / 3.0, 1.0 / 3.0)),
        ("Gamma", (0.0, 0.0)),
        ("m", (0.5, 0.0)),
    ]


def taige_momentum_path(
    *,
    n_per_segment: int = 24,
    model: ContinuumModelParams | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], list[str]]:
    """Build the Taige Gamma-kappa+-m-kappa--Gamma-m momentum path."""

    geometry = MoireGeometry(model or taige_model_params())
    nodes = taige_path_nodes()
    points: list[np.ndarray] = []
    distances: list[float] = []
    ticks: list[int] = []
    labels: list[str] = []
    cumulative = 0.0
    for idx, (start_label, start) in enumerate(nodes[:-1]):
        end_label, end = nodes[idx + 1]
        start_arr = np.asarray(start, dtype=float)
        end_arr = np.asarray(end, dtype=float)
        if idx == 0:
            ticks.append(0)
            labels.append(start_label)
        for step in range(n_per_segment):
            t = step / float(n_per_segment)
            point = (1.0 - t) * start_arr + t * end_arr
            if points:
                delta = geometry.k_from_fractional(point - points[-1])
                cumulative += float(np.linalg.norm(delta))
            points.append(point)
            distances.append(cumulative)
        ticks.append(len(points) - 1)
        labels.append(end_label)
    points.append(np.asarray(nodes[-1][1], dtype=float))
    delta = geometry.k_from_fractional(points[-1] - points[-2])
    cumulative += float(np.linalg.norm(delta))
    distances.append(cumulative)
    ticks[-1] = len(points) - 1
    return np.asarray(points), np.asarray(distances), ticks, labels


def compute_taige_path_spectrum(
    model: ContinuumModelParams,
    *,
    n_per_segment: int = 24,
    track_bands: bool = True,
) -> dict[str, np.ndarray | list[int] | list[str]]:
    """Evaluate non-interacting Taige bands along the standard path.

    The Kprime path is generated from the K valley at folded ``-k``. This is
    the same non-Kramers T-prime convention used for the coarse active-space
    bandstructure and avoids comparing two different continuum gauges.
    """

    continuum = TaigeContinuumModel(model)
    path, distances, ticks, labels = taige_momentum_path(n_per_segment=n_per_segment, model=model)
    n_bands = int(model.n_bands)
    electron = np.empty((path.shape[0], 2, n_bands), dtype=float)
    vectors = np.empty((path.shape[0], 2, continuum.dim, n_bands), dtype=complex)
    for ip, k_frac in enumerate(path):
        for iv, valley in enumerate(VALLEY_ORDER):
            if valley == VALLEY_K:
                physical_k = np.asarray(k_frac, dtype=float)
            else:
                physical_k = _fold_fractional(-np.asarray(k_frac, dtype=float))
            evals, evecs = np.linalg.eigh(continuum.hamiltonian(physical_k, VALLEY_K))
            order = np.argsort(evals)[::-1][:n_bands]
            electron[ip, iv] = evals[order]
            vectors[ip, iv] = evecs[:, order]
    if track_bands and path.shape[0] > 1 and n_bands > 1:
        electron, vectors = _track_continuous_path_bands(electron, vectors)
    return {
        "k_path": path,
        "distances": distances,
        "ticks": ticks,
        "labels": labels,
        "electron_energies": electron,
        "hole_energies": -electron,
    }


def _fold_fractional(k_frac: np.ndarray) -> np.ndarray:
    """Fold a fractional reciprocal coordinate into the unit parallelogram."""

    k = np.asarray(k_frac, dtype=float)
    return k - np.floor(k)


def _track_continuous_path_bands(
    energies: np.ndarray,
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Track nearby path bands by energy continuity and wavefunction overlap."""

    from itertools import permutations

    tracked_energies = np.asarray(energies, dtype=float).copy()
    tracked_vectors = np.asarray(vectors, dtype=complex).copy()
    n_bands = tracked_energies.shape[-1]
    for iv in range(tracked_energies.shape[1]):
        previous_energies = tracked_energies[0, iv]
        previous = tracked_vectors[0, iv]
        for ip in range(1, tracked_energies.shape[0]):
            energy_cost = np.abs(previous_energies[:, None] - energies[ip, iv][None, :])
            overlaps = np.abs(previous.conj().T @ vectors[ip, iv]) ** 2
            best_perm = min(
                permutations(range(n_bands)),
                key=lambda perm: (
                    sum(energy_cost[i, perm[i]] for i in range(n_bands)),
                    -sum(overlaps[i, perm[i]] for i in range(n_bands)),
                ),
            )
            perm = np.asarray(best_perm, dtype=int)
            tracked_energies[ip, iv] = energies[ip, iv, perm]
            tracked_vectors[ip, iv] = vectors[ip, iv][:, perm]
            previous_energies = tracked_energies[ip, iv]
            previous = tracked_vectors[ip, iv]
    return tracked_energies, tracked_vectors
