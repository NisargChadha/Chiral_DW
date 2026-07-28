"""Taige-parameter tMoTe2 continuum, topology, and Coulomb helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from chiral_dw.config import ContinuumFiniteQParams, ContinuumInteractionParams, ContinuumModelParams
from chiral_dw.continuum.momentum_channels import (
    c3_radial_channel_mask,
    hexagonal_q_shell,
    reciprocal_box as _shared_reciprocal_box,
)
from chiral_dw.continuum.models import (
    ContinuumActiveSpace,
    DensityVertices,
    MomentumGrid,
    VALLEY_K,
    VALLEY_KPRIME,
    VALLEY_ORDER,
    dense_lambdas_from_compact,
    hermitize,
)
from chiral_dw.continuum.taige_sewing import reciprocal_shift_gather

GridCoord = tuple[int, int]
_MAX_TAIGE_VERTEX_ROLL_Q_POINTS = 8

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
TAIGE_MOTE2_MATERIAL = "MoTe2_Taige"

TAIGE_WSE2_A0_ANGSTROM = 3.32
TAIGE_WSE2_M_EFF = 0.43
TAIGE_WSE2_V_MEV = 9.0
# Table SI uses phi=+128 deg; this C3-gauge code convention flips the sign,
# as checked by the WSe2 K-valley first-band Chern number at u_D=0.
TAIGE_WSE2_PHI_DEG = -128.0
TAIGE_WSE2_W_MEV = 18.0
TAIGE_WSE2_MATERIAL = "WSe2_Taige"
TaigeMaterial = Literal["mote2", "wse2"]

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
    material: TaigeMaterial = "mote2",
    theta_deg: float = TAIGE_THETA_DEG,
    u_D: float = 0.0,
    plane_wave_shell: int = 1,
    n_bands: int = 2,
    n_active_bands_per_valley: int = 1,
) -> ContinuumModelParams:
    """Return Chiral_DW-native Taige model parameters."""

    material_key = _normalize_taige_material(material)
    if material_key == "mote2":
        material_label = TAIGE_MOTE2_MATERIAL
        a0_angstrom = TAIGE_A0_ANGSTROM
        m_eff = TAIGE_M_EFF
        moire_potential_mev = TAIGE_V_MEV
        phi_deg = TAIGE_PHI_DEG
        tunneling_mev = TAIGE_W_MEV
    elif material_key == "wse2":
        material_label = TAIGE_WSE2_MATERIAL
        a0_angstrom = TAIGE_WSE2_A0_ANGSTROM
        m_eff = TAIGE_WSE2_M_EFF
        moire_potential_mev = TAIGE_WSE2_V_MEV
        phi_deg = TAIGE_WSE2_PHI_DEG
        tunneling_mev = TAIGE_WSE2_W_MEV
    else:
        raise ValueError(f"unknown Taige material {material!r}")
    return ContinuumModelParams(
        material=material_label,
        theta_deg=float(theta_deg),
        a0_angstrom=a0_angstrom,
        m_eff=m_eff,
        moire_potential_mev=moire_potential_mev,
        phi_deg=phi_deg,
        tunneling_mev=tunneling_mev,
        displacement_mev=float(u_D),
        plane_wave_shell=int(plane_wave_shell),
        n_bands=int(n_bands),
        n_active_bands_per_valley=int(n_active_bands_per_valley),
        active_model="taige",
    )


def taige_wse2_model_params(
    *,
    theta_deg: float = TAIGE_THETA_DEG,
    u_D: float = 0.0,
    plane_wave_shell: int = 1,
    n_bands: int = 2,
    n_active_bands_per_valley: int = 1,
) -> ContinuumModelParams:
    """Return Taige Table-SI WSe2 continuum parameters in the local C3 gauge."""

    return taige_model_params(
        material="wse2",
        theta_deg=theta_deg,
        u_D=u_D,
        plane_wave_shell=plane_wave_shell,
        n_bands=n_bands,
        n_active_bands_per_valley=n_active_bands_per_valley,
    )


def _normalize_taige_material(material: str) -> TaigeMaterial:
    key = str(material).strip().lower().replace("-", "").replace("_", "")
    if key in {"mote2", "mo"}:
        return "mote2"
    if key in {"wse2", "wse"}:
        return "wse2"
    raise ValueError("Taige material must be 'mote2' or 'wse2'")


def taige_material_label(material: str) -> str:
    """Return the artifact label for a supported Taige material preset."""

    key = _normalize_taige_material(material)
    return TAIGE_MOTE2_MATERIAL if key == "mote2" else TAIGE_WSE2_MATERIAL


def taige_material_smear_length_nm(material: str) -> float:
    """Return the default Taige Gaussian smear length for a material."""

    key = _normalize_taige_material(material)
    a0 = TAIGE_A0_ANGSTROM if key == "mote2" else TAIGE_WSE2_A0_ANGSTROM
    return float(a0 / 10.0)


def taige_interaction_params(
    *,
    material: TaigeMaterial = "mote2",
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
    vertex_workers: int = 1,
    exchange_workers: int = 1,
    density_vertex_retention: Literal["full", "hartree_only"] = "full",
    density_vertex_layout: Literal["auto", "dense", "valley_compact"] = "auto",
    exchange_representation: Literal["auto", "dense", "valley_sector"] = "auto",
    form_factor_backend: Literal["auto", "scalar", "cached_gather", "vectorized"] = "auto",
) -> ContinuumInteractionParams:
    """Return dual-gated smeared Coulomb parameters for a Taige TMD preset."""

    smear = (
        taige_material_smear_length_nm(material)
        if smear_length_nm is None
        else float(smear_length_nm)
    )
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
        vertex_workers=int(vertex_workers),
        exchange_workers=int(exchange_workers),
        density_vertex_retention=density_vertex_retention,
        density_vertex_layout=density_vertex_layout,
        exchange_representation=exchange_representation,
        form_factor_backend=form_factor_backend,
    )


def taige_wse2_interaction_params(
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
    vertex_workers: int = 1,
    exchange_workers: int = 1,
    density_vertex_retention: Literal["full", "hartree_only"] = "full",
    density_vertex_layout: Literal["auto", "dense", "valley_compact"] = "auto",
    exchange_representation: Literal["auto", "dense", "valley_sector"] = "auto",
    form_factor_backend: Literal["auto", "scalar", "cached_gather", "vectorized"] = "auto",
) -> ContinuumInteractionParams:
    """Return dual-gated smeared Coulomb parameters for Taige WSe2."""

    return taige_interaction_params(
        material="wse2",
        include_q0=include_q0,
        q_mesh=q_mesh,
        q_shell=q_shell,
        local_field_cutoff=local_field_cutoff,
        epsilon=epsilon,
        gate_distance_nm=gate_distance_nm,
        smear_length_nm=smear_length_nm,
        interaction_strength_scale=interaction_strength_scale,
        hartree_scale=hartree_scale,
        exchange_scale=exchange_scale,
        vertex_workers=vertex_workers,
        exchange_workers=exchange_workers,
        density_vertex_retention=density_vertex_retention,
        density_vertex_layout=density_vertex_layout,
        exchange_representation=exchange_representation,
        form_factor_backend=form_factor_backend,
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
    return _shared_reciprocal_box(g_cutoff)


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
    bands: TaigeBandStructure | None = None,
) -> tuple[ContinuumActiveSpace, TaigeBandStructure]:
    """Build the Taige active hole basis, optionally from a cached eigensystem."""

    resolved_bands = bands if bands is not None else compute_taige_bandstructure(model, grid)
    active = active_space_from_taige_bands(grid, model, resolved_bands, finite_q)
    return active, resolved_bands


def active_space_from_taige_bands(
    grid: MomentumGrid,
    model: ContinuumModelParams,
    bands: TaigeBandStructure,
    finite_q: ContinuumFiniteQParams | None = None,
) -> ContinuumActiveSpace:
    """Slice a cached Taige eigensystem into one HF active space."""

    if bands.grid != grid:
        raise ValueError("cached bands use a different momentum grid")
    if bands.model != model:
        differing = {
            key
            for key, value in model.model_dump().items()
            if key != "n_active_bands_per_valley"
            and value != bands.model.model_dump().get(key)
        }
        if differing:
            raise ValueError(f"cached bands use incompatible model fields: {sorted(differing)}")
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
    return active


def q_transfers(grid: MomentumGrid, interaction: ContinuumInteractionParams) -> tuple[GridCoord, ...]:
    if interaction.q_mesh == "full":
        return centered_mesh_transfers(grid)
    return hexagonal_q_shell(interaction.q_shell)


def _shift_gather(
    shell: tuple[GridCoord, ...],
    shell_index: dict[GridCoord, int],
    shift: GridCoord,
) -> tuple[np.ndarray, np.ndarray]:
    # ``shell_index`` remains in the private signature for compatibility with
    # the density-vertex call sites; the canonical mapping lives in the public
    # sewing module used by both topology and orbital magnetization.
    del shell_index
    return reciprocal_shift_gather(shell, shift)


GatherCache = dict[GridCoord, tuple[np.ndarray, np.ndarray]]


def _taige_form_factor_shifts(
    *,
    q_list: tuple[GridCoord, ...],
    g_channels: tuple[GridCoord, ...],
    n_k: int,
) -> tuple[GridCoord, ...]:
    shifts: set[GridCoord] = set()
    grid_size = int(n_k) * int(n_k)
    for q in q_list:
        for ik in range(grid_size):
            k_coord = _grid_coord_of(ik, n_k)
            _forward, rec_shift = _fold_grid_coord(
                (k_coord[0] + int(q[0]), k_coord[1] + int(q[1])),
                n_k,
            )
            for g in g_channels:
                shifts.add((rec_shift[0] + int(g[0]), rec_shift[1] + int(g[1])))
    return tuple(sorted(shifts))


def _build_shift_gather_cache(
    *,
    shell: tuple[GridCoord, ...],
    shell_index: dict[GridCoord, int],
    shifts: tuple[GridCoord, ...],
) -> GatherCache:
    return {shift: _shift_gather(shell, shell_index, shift) for shift in shifts}


def _overlap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    shell: tuple[GridCoord, ...],
    shell_index: dict[GridCoord, int],
    shift: GridCoord,
    gather_cache: GatherCache | None = None,
) -> np.ndarray:
    key = (int(shift[0]), int(shift[1]))
    if gather_cache is None:
        src, tgt = _shift_gather(shell, shell_index, key)
    else:
        src, tgt = gather_cache[key]
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


def _overlap_batch(
    left: np.ndarray,
    right: np.ndarray,
    *,
    src: np.ndarray,
    tgt: np.ndarray,
    n_plane_waves: int,
) -> np.ndarray:
    na_l = left.shape[-1]
    na_r = right.shape[-1]
    out = np.zeros((left.shape[0], na_l, na_r), dtype=complex)
    if src.size == 0:
        return out
    left_blocks = left.reshape(left.shape[0], 2, n_plane_waves, na_l)
    right_blocks = right.reshape(right.shape[0], 2, n_plane_waves, na_r)
    for layer in range(2):
        out += np.einsum(
            "kxa,kxb->kab",
            np.conj(left_blocks[:, layer, src, :]),
            right_blocks[:, layer, tgt, :],
            optimize=True,
        )
    return out


def _channel_mask(
    geometry: MoireGeometry,
    grid: MomentumGrid,
    q_list: tuple[GridCoord, ...],
    g_channels: tuple[GridCoord, ...],
    local_field_cutoff: int,
) -> np.ndarray:
    del geometry
    return c3_radial_channel_mask(
        grid,
        q_list,
        g_channels,
        local_field_cutoff,
    )


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


def _grid_coord_of(index: int, n_k: int) -> GridCoord:
    idx = int(index)
    n = int(n_k)
    return idx // n, idx % n


def _grid_index_of(coord: GridCoord, n_k: int) -> int:
    n = int(n_k)
    i, j = int(coord[0]) % n, int(coord[1]) % n
    return i * n + j


def _fold_grid_coord(coord: GridCoord, n_k: int) -> tuple[GridCoord, GridCoord]:
    n = int(n_k)
    i, j = int(coord[0]), int(coord[1])
    fi = i % n
    fj = j % n
    return (fi, fj), ((i - fi) // n, (j - fj) // n)


def _taige_density_vertex_q_slab_compact(
    *,
    q_start: int,
    q_stop: int,
    q_list: tuple[GridCoord, ...],
    g_channels: tuple[GridCoord, ...],
    channel_in_disk: np.ndarray,
    n_k: int,
    n_active: int,
    shell: tuple[GridCoord, ...],
    shell_index: dict[GridCoord, int],
    gather_cache: GatherCache | None,
    electron_vectors: np.ndarray,
    source_index: np.ndarray | None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Build one contiguous q-slab of valley-compact Taige density vertices."""

    q_slab = q_list[int(q_start) : int(q_stop)]
    n_q_slab = len(q_slab)
    n_g = len(g_channels)
    grid_size = int(n_k) * int(n_k)
    target_minus_q = np.empty((n_q_slab, grid_size), dtype=int)
    q_is_zero = np.zeros(n_q_slab, dtype=bool)
    lambdas = np.zeros((n_q_slab, n_g, grid_size, 2, n_active, n_active), dtype=complex)

    def electron_form_factor(
        iv: int,
        ik: int,
        q_mesh: GridCoord,
        g_channel: GridCoord,
    ) -> np.ndarray:
        k_coord = _grid_coord_of(ik, n_k)
        forward, rec_shift = _fold_grid_coord(
            (k_coord[0] + int(q_mesh[0]), k_coord[1] + int(q_mesh[1])),
            n_k,
        )
        ikq = _grid_index_of(forward, n_k)
        shift = (rec_shift[0] + int(g_channel[0]), rec_shift[1] + int(g_channel[1]))
        left = electron_vectors[ik, iv, :, :n_active]
        right = electron_vectors[ikq, iv, :, :n_active]
        return _overlap(
            left,
            right,
            shell=shell,
            shell_index=shell_index,
            shift=shift,
            gather_cache=gather_cache,
        )

    def hole_form_factor(
        iv: int,
        ik: int,
        q_mesh: GridCoord,
        g_channel: GridCoord,
    ) -> np.ndarray:
        k_coord = _grid_coord_of(ik, n_k)
        ik_minus_q = _grid_index_of(
            _fold_grid_coord(
                (k_coord[0] - int(q_mesh[0]), k_coord[1] - int(q_mesh[1])),
                n_k,
            )[0],
            n_k,
        )
        return electron_form_factor(iv, ik_minus_q, q_mesh, g_channel).T

    for local_iq, q in enumerate(q_slab):
        q_is_zero[local_iq] = q == (0, 0)
        for ik in range(grid_size):
            k_coord = _grid_coord_of(ik, n_k)
            folded, _shift = _fold_grid_coord(
                (k_coord[0] - int(q[0]), k_coord[1] - int(q[1])),
                n_k,
            )
            target_minus_q[local_iq, ik] = _grid_index_of(folded, n_k)
        for ig, g in enumerate(g_channels):
            if not channel_in_disk[int(q_start) + local_iq, ig]:
                continue
            for ik in range(grid_size):
                for iv in range(2):
                    physical_source = (
                        int(source_index[ik, iv]) if source_index is not None else int(ik)
                    )
                    lambdas[local_iq, ig, ik, iv] = hole_form_factor(
                        iv,
                        physical_source,
                        q,
                        g,
                    )

    return int(q_start), target_minus_q, q_is_zero, lambdas


def _taige_density_vertex_q_slab_compact_vectorized(
    *,
    q_start: int,
    q_stop: int,
    q_list: tuple[GridCoord, ...],
    g_channels: tuple[GridCoord, ...],
    channel_in_disk: np.ndarray,
    n_k: int,
    n_active: int,
    shell: tuple[GridCoord, ...],
    shell_index: dict[GridCoord, int],
    gather_cache: GatherCache | None,
    electron_vectors: np.ndarray,
    source_index: np.ndarray | None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Build one q-slab with batched k-point form-factor contractions."""

    if gather_cache is None:
        raise ValueError("vectorized Taige form factors require a gather cache")
    q_slab = q_list[int(q_start) : int(q_stop)]
    n_q_slab = len(q_slab)
    n_g = len(g_channels)
    grid_size = int(n_k) * int(n_k)
    n_plane_waves = len(shell)
    target_minus_q = np.empty((n_q_slab, grid_size), dtype=int)
    q_is_zero = np.zeros(n_q_slab, dtype=bool)
    lambdas = np.zeros((n_q_slab, n_g, grid_size, 2, n_active, n_active), dtype=complex)

    def physical_source_for(iv: int) -> np.ndarray:
        if source_index is None:
            return np.arange(grid_size, dtype=int)
        return np.asarray(source_index[:, int(iv)], dtype=int)

    for local_iq, q in enumerate(q_slab):
        q_is_zero[local_iq] = q == (0, 0)
        for ik in range(grid_size):
            k_coord = _grid_coord_of(ik, n_k)
            folded, _shift = _fold_grid_coord(
                (k_coord[0] - int(q[0]), k_coord[1] - int(q[1])),
                n_k,
            )
            target_minus_q[local_iq, ik] = _grid_index_of(folded, n_k)

        for iv in range(2):
            sources = physical_source_for(iv)
            left_indices = np.empty(grid_size, dtype=int)
            right_indices = np.empty(grid_size, dtype=int)
            rec_shifts = np.empty((grid_size, 2), dtype=int)
            for ik, physical_source in enumerate(sources):
                source_coord = _grid_coord_of(int(physical_source), n_k)
                left_coord, _minus_shift = _fold_grid_coord(
                    (source_coord[0] - int(q[0]), source_coord[1] - int(q[1])),
                    n_k,
                )
                left_index = _grid_index_of(left_coord, n_k)
                forward, rec_shift = _fold_grid_coord(
                    (left_coord[0] + int(q[0]), left_coord[1] + int(q[1])),
                    n_k,
                )
                left_indices[ik] = left_index
                right_indices[ik] = _grid_index_of(forward, n_k)
                rec_shifts[ik] = rec_shift

            for ig, g in enumerate(g_channels):
                if not channel_in_disk[int(q_start) + local_iq, ig]:
                    continue
                shifts = rec_shifts + np.asarray(g, dtype=int)[None, :]
                groups: dict[GridCoord, list[int]] = {}
                for ik, shift in enumerate(shifts):
                    key = (int(shift[0]), int(shift[1]))
                    groups.setdefault(key, []).append(ik)
                for shift, indices in groups.items():
                    src, tgt = gather_cache[shift]
                    idx = np.asarray(indices, dtype=int)
                    overlaps = _overlap_batch(
                        electron_vectors[left_indices[idx], iv, :, :n_active],
                        electron_vectors[right_indices[idx], iv, :, :n_active],
                        src=src,
                        tgt=tgt,
                        n_plane_waves=n_plane_waves,
                    )
                    lambdas[local_iq, ig, idx, iv] = np.swapaxes(overlaps, -1, -2)

    return int(q_start), target_minus_q, q_is_zero, lambdas


def _taige_density_vertex_q_slab(
    *,
    q_start: int,
    q_stop: int,
    q_list: tuple[GridCoord, ...],
    g_channels: tuple[GridCoord, ...],
    channel_in_disk: np.ndarray,
    n_k: int,
    n_active: int,
    dim: int,
    shell: tuple[GridCoord, ...],
    shell_index: dict[GridCoord, int],
    gather_cache: GatherCache | None,
    electron_vectors: np.ndarray,
    source_index: np.ndarray | None,
    form_factor_backend: str,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    """Build one contiguous q-slab of dense Taige density vertices."""

    compact_builder = (
        _taige_density_vertex_q_slab_compact_vectorized
        if form_factor_backend == "vectorized"
        else _taige_density_vertex_q_slab_compact
    )
    q_start_out, target_minus_q, q_is_zero, compact = compact_builder(
        q_start=q_start,
        q_stop=q_stop,
        q_list=q_list,
        g_channels=g_channels,
        channel_in_disk=channel_in_disk,
        n_k=n_k,
        n_active=n_active,
        shell=shell,
        shell_index=shell_index,
        gather_cache=gather_cache,
        electron_vectors=electron_vectors,
        source_index=source_index,
    )
    dense = dense_lambdas_from_compact(compact)
    if dense.shape[-2:] != (int(dim), int(dim)):
        raise ValueError("expanded Taige density vertices have incompatible active dimension")
    return q_start_out, target_minus_q, q_is_zero, dense


def _q_slab_ranges(n_q: int, vertex_workers: int) -> tuple[tuple[int, int], ...]:
    n_jobs = max(1, min(int(vertex_workers), int(n_q)))
    n_slabs = max(1, min(int(n_q), 4 * n_jobs))
    bounds = np.linspace(0, int(n_q), n_slabs + 1, dtype=int)
    return tuple(
        (int(start), int(stop))
        for start, stop in zip(bounds[:-1], bounds[1:])
        if int(start) < int(stop)
    )


def _resolve_taige_form_factor_backend(interaction: ContinuumInteractionParams) -> str:
    requested = str(getattr(interaction, "form_factor_backend", "auto"))
    if requested == "auto":
        return "vectorized"
    if requested in {"scalar", "cached_gather", "vectorized"}:
        return requested
    raise ValueError("form_factor_backend must be 'auto', 'scalar', 'cached_gather', or 'vectorized'")


def _taige_gather_cache_for_backend(
    *,
    form_factor_backend: str,
    q_list: tuple[GridCoord, ...],
    g_channels: tuple[GridCoord, ...],
    grid: MomentumGrid,
    shell: tuple[GridCoord, ...],
    shell_index: dict[GridCoord, int],
) -> GatherCache | None:
    if form_factor_backend == "scalar":
        return None
    shifts = _taige_form_factor_shifts(
        q_list=q_list,
        g_channels=g_channels,
        n_k=grid.n_k,
    )
    return _build_shift_gather_cache(shell=shell, shell_index=shell_index, shifts=shifts)


def _build_taige_density_vertex_arrays_serial(
    *,
    q_list: tuple[GridCoord, ...],
    g_channels: tuple[GridCoord, ...],
    channel_in_disk: np.ndarray,
    grid: MomentumGrid,
    active: ContinuumActiveSpace,
    shell_index: dict[GridCoord, int],
    gather_cache: GatherCache | None,
    electron_vectors: np.ndarray,
    compact: bool,
    form_factor_backend: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if compact:
        slab_builder = (
            _taige_density_vertex_q_slab_compact_vectorized
            if form_factor_backend == "vectorized"
            else _taige_density_vertex_q_slab_compact
        )
    else:
        slab_builder = _taige_density_vertex_q_slab
    kwargs = {
        "q_start": 0,
        "q_stop": len(q_list),
        "q_list": q_list,
        "g_channels": g_channels,
        "channel_in_disk": channel_in_disk,
        "n_k": grid.n_k,
        "n_active": active.n_active,
        "shell": active.shell,
        "shell_index": shell_index,
        "gather_cache": gather_cache,
        "electron_vectors": electron_vectors,
        "source_index": active.source_index,
    }
    if not compact:
        kwargs["dim"] = active.dim
        kwargs["form_factor_backend"] = form_factor_backend
    _q_start, target_minus_q, q_is_zero, lambdas = slab_builder(
        **kwargs,
    )
    return target_minus_q, q_is_zero, lambdas


def _build_taige_density_vertex_arrays_parallel(
    *,
    q_list: tuple[GridCoord, ...],
    g_channels: tuple[GridCoord, ...],
    channel_in_disk: np.ndarray,
    grid: MomentumGrid,
    active: ContinuumActiveSpace,
    shell_index: dict[GridCoord, int],
    gather_cache: GatherCache | None,
    electron_vectors: np.ndarray,
    vertex_workers: int,
    compact: bool,
    form_factor_backend: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from joblib import Parallel, delayed

    n_q = len(q_list)
    n_g = len(g_channels)
    target_minus_q = np.empty((n_q, grid.size), dtype=int)
    q_is_zero = np.zeros(n_q, dtype=bool)
    if compact:
        lambdas = np.zeros((n_q, n_g, grid.size, 2, active.n_active, active.n_active), dtype=complex)
        slab_builder = (
            _taige_density_vertex_q_slab_compact_vectorized
            if form_factor_backend == "vectorized"
            else _taige_density_vertex_q_slab_compact
        )
    else:
        lambdas = np.zeros((n_q, n_g, grid.size, active.dim, active.dim), dtype=complex)
        slab_builder = _taige_density_vertex_q_slab
    n_jobs = max(1, min(int(vertex_workers), n_q))
    ranges = _q_slab_ranges(n_q, n_jobs)

    def _task(start: int, stop: int):
        kwargs = {
            "q_start": start,
            "q_stop": stop,
            "q_list": q_list,
            "g_channels": g_channels,
            "channel_in_disk": channel_in_disk,
            "n_k": grid.n_k,
            "n_active": active.n_active,
            "shell": active.shell,
            "shell_index": shell_index,
            "gather_cache": gather_cache,
            "electron_vectors": electron_vectors,
            "source_index": active.source_index,
        }
        if not compact:
            kwargs["dim"] = active.dim
            kwargs["form_factor_backend"] = form_factor_backend
        return slab_builder(**kwargs)

    tasks = (delayed(_task)(start, stop) for start, stop in ranges)
    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        return_as="generator",
        mmap_mode="r",
        max_nbytes="32M",
    )(tasks)
    for q_start, target_slab, q_zero_slab, lambda_slab in results:
        q_stop = q_start + target_slab.shape[0]
        target_minus_q[q_start:q_stop] = target_slab
        q_is_zero[q_start:q_stop] = q_zero_slab
        lambdas[q_start:q_stop] = lambda_slab
    return target_minus_q, q_is_zero, lambdas


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
    layout = (
        "valley_compact"
        if interaction.density_vertex_layout == "auto"
        else str(interaction.density_vertex_layout)
    )
    if layout not in {"dense", "valley_compact"}:
        raise ValueError("density_vertex_layout must be 'auto', 'dense', or 'valley_compact'")
    compact = layout == "valley_compact"
    form_factor_backend = _resolve_taige_form_factor_backend(interaction)
    gather_cache = _taige_gather_cache_for_backend(
        form_factor_backend=form_factor_backend,
        q_list=q_list,
        g_channels=g_channels,
        grid=grid,
        shell=shell,
        shell_index=shell_index,
    )
    if interaction.vertex_workers <= 1:
        target_minus_q, q_is_zero, lambdas = _build_taige_density_vertex_arrays_serial(
            q_list=q_list,
            g_channels=g_channels,
            channel_in_disk=channel_in_disk,
            grid=grid,
            active=active,
            shell_index=shell_index,
            gather_cache=gather_cache,
            electron_vectors=electron_vectors,
            compact=compact,
            form_factor_backend=form_factor_backend,
        )
    else:
        target_minus_q, q_is_zero, lambdas = _build_taige_density_vertex_arrays_parallel(
            q_list=q_list,
            g_channels=g_channels,
            channel_in_disk=channel_in_disk,
            grid=grid,
            active=active,
            shell_index=shell_index,
            gather_cache=gather_cache,
            electron_vectors=electron_vectors,
            vertex_workers=interaction.vertex_workers,
            compact=compact,
            form_factor_backend=form_factor_backend,
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

    lambda_blocks = (
        np.zeros((0, 0, grid.size, active.dim, active.dim), dtype=complex)
        if compact
        else lambdas
    )
    lambda_compact = lambdas if compact else None

    return DensityVertices(
        q_shifts=q_list,
        target_minus_q=target_minus_q,
        q_is_zero=np.logical_or(q_is_zero, np.any(q_zero_channels, axis=-1)),
        lambda_blocks=lambda_blocks,
        v_over_a=v_over_a,
        g_channels=g_channels,
        vertex_layout=layout,
        lambda_compact=lambda_compact,
        channel_in_disk=channel_in_disk,
        q_vectors_nm_inv=q_vectors_nm_inv,
        q_norm_nm_inv=np.linalg.norm(q_vectors_nm_inv, axis=-1),
        v_q=v_q,
    )


def roll_taige_density_vertices(
    q0_vertices: DensityVertices,
    finite_q_active: ContinuumActiveSpace,
) -> DensityVertices:
    """Gather raw q=0 Taige vertices into a finite-Q active frame.

    The active-frame valley shifts cancel from the intravalley transfer, so all
    transfer and interaction metadata are unchanged. Only the source momentum
    of each valley block is gathered. Returned arrays never alias the q=0
    vertex arrays.
    """

    if not finite_q_active.finite_q_enabled:
        raise ValueError("finite_q_active must use an enabled finite-Q frame")
    source_index = finite_q_active.source_index
    if source_index is None:
        raise ValueError("finite-Q Taige active space requires source_index")
    source = np.asarray(source_index, dtype=int)
    n_k = int(finite_q_active.n_k)
    n_active = int(finite_q_active.n_active)
    if source.shape != (n_k, 2):
        raise ValueError("source_index must have shape (n_k, 2)")
    if np.any(source < 0) or np.any(source >= n_k):
        raise ValueError("source_index contains an out-of-range momentum index")
    if np.asarray(q0_vertices.target_minus_q).shape[1] != n_k:
        raise ValueError("q=0 vertices and finite-Q active space use different grids")

    layout = str(q0_vertices.vertex_layout)
    n_q = len(q0_vertices.q_shifts)
    q_slabs = tuple(
        (start, min(start + _MAX_TAIGE_VERTEX_ROLL_Q_POINTS, n_q))
        for start in range(0, n_q, _MAX_TAIGE_VERTEX_ROLL_Q_POINTS)
    )
    if layout == "valley_compact":
        if q0_vertices.lambda_compact is None:
            raise ValueError("valley_compact DensityVertices require lambda_compact")
        q0_compact = np.asarray(q0_vertices.lambda_compact)
        if q0_compact.shape[2:] != (n_k, 2, n_active, n_active):
            raise ValueError("q=0 compact vertices are incompatible with the active space")
        rolled_compact = np.empty_like(q0_compact)
        for q_start, q_stop in q_slabs:
            for valley in range(2):
                rolled_compact[q_start:q_stop, :, :, valley] = q0_compact[
                    q_start:q_stop, :, source[:, valley], valley
                ]
        rolled_dense = np.asarray(q0_vertices.lambda_blocks).copy()
        lambda_compact = rolled_compact
    elif layout == "dense":
        q0_dense = np.asarray(q0_vertices.lambda_blocks)
        dim = 2 * n_active
        if q0_dense.shape[2:] != (n_k, dim, dim):
            raise ValueError("q=0 dense vertices are incompatible with the active space")
        rolled_dense = np.zeros_like(q0_dense)
        for q_start, q_stop in q_slabs:
            for valley in range(2):
                block = slice(valley * n_active, (valley + 1) * n_active)
                rolled_dense[q_start:q_stop, :, :, block, block] = q0_dense[
                    q_start:q_stop, :, source[:, valley], block, block
                ]
        lambda_compact = None
    else:
        raise ValueError(f"unknown density vertex layout {layout!r}")

    def _copy_optional(array: np.ndarray | None) -> np.ndarray | None:
        return None if array is None else np.asarray(array).copy()

    return DensityVertices(
        q_shifts=tuple(q0_vertices.q_shifts),
        target_minus_q=np.asarray(q0_vertices.target_minus_q, dtype=int).copy(),
        q_is_zero=np.asarray(q0_vertices.q_is_zero, dtype=bool).copy(),
        lambda_blocks=rolled_dense,
        v_over_a=np.asarray(q0_vertices.v_over_a, dtype=float).copy(),
        g_channels=tuple(q0_vertices.g_channels),
        vertex_layout=layout,
        lambda_compact=lambda_compact,
        channel_in_disk=_copy_optional(q0_vertices.channel_in_disk),
        q_vectors_nm_inv=_copy_optional(q0_vertices.q_vectors_nm_inv),
        q_norm_nm_inv=_copy_optional(q0_vertices.q_norm_nm_inv),
        v_q=_copy_optional(q0_vertices.v_q),
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
