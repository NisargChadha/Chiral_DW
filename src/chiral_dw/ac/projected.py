"""Projected two-valley HF bundle for finite-LL Aharonov-Casher bands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chiral_dw.ac.nonideal import ACParams, NonIdealACLLModel
from chiral_dw.config import (
    ACProjectedHFParams,
    ContinuumGridParams,
    ContinuumInteractionParams,
    ContinuumModelParams,
)
from chiral_dw.continuum.hf import ContinuumHFBackend, EnergyComponents
from chiral_dw.continuum.momentum_channels import (
    C3_RADIAL_Q_PLUS_G_V2,
    c3_channel_index_map,
    c3_mesh_index_map,
    c3_radial_channel_mask,
    hexagonal_q_shell,
    inversion_channel_index_map,
    reciprocal_box,
)
from chiral_dw.continuum.models import (
    ContinuumActiveSpace,
    ContinuumBundle,
    DensityVertices,
    MomentumGrid,
    block_trace_product,
    hermitize,
)
from chiral_dw.continuum.symmetry import mesh_inversion_map

E2_MEV_NM = 1439.96454784255
AC_DENSITY_FORM_FACTOR_CONVENTION = (
    "magnetic_bloch_source_to_target_folded_q_plus_g_v1"
)


class C3SymmetrizedACBackend:
    """C3 orbit average of an AC HF functional without constraining its density."""

    def __init__(self, base: ContinuumHFBackend, grid: MomentumGrid) -> None:
        self.base = base
        self.grid = grid
        self.c3_partner = c3_mesh_index_map(grid)
        self.h0 = self._average_covariant_operator(base.h0)

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def _rotate_blocks(self, blocks: np.ndarray, turns: int = 1) -> np.ndarray:
        out = np.asarray(blocks, dtype=complex)
        for _ in range(int(turns) % 3):
            rotated = np.empty_like(out)
            rotated[self.c3_partner] = out
            out = rotated
        return out

    def _pull_back_blocks(self, blocks: np.ndarray, turns: int) -> np.ndarray:
        return self._rotate_blocks(blocks, (-int(turns)) % 3)

    def _average_covariant_operator(self, blocks: np.ndarray) -> np.ndarray:
        return hermitize(
            sum(self._rotate_blocks(blocks, turn) for turn in range(3)) / 3.0
        )

    def _average_hamiltonian_map(self, method_name: str, density: np.ndarray) -> np.ndarray:
        terms = []
        for turn in range(3):
            rotated_density = self._rotate_blocks(density, turn)
            rotated_hamiltonian = getattr(self.base, method_name)(rotated_density)
            terms.append(self._pull_back_blocks(rotated_hamiltonian, turn))
        return hermitize(sum(terms) / 3.0)

    def hartree_hamiltonian(self, density: np.ndarray) -> np.ndarray:
        return self._average_hamiltonian_map("hartree_hamiltonian", density)

    def fock_hamiltonian(self, density: np.ndarray) -> np.ndarray:
        return self._average_hamiltonian_map("fock_hamiltonian", density)

    def self_energy(self, Q: np.ndarray) -> np.ndarray:
        """Return the C3-averaged linear Hartree-plus-Fock field."""

        density = self.as_block_density(Q)
        return hermitize(
            self.hartree_hamiltonian(density) + self.fock_hamiltonian(density)
        )

    def hf_hamiltonian(self, P: np.ndarray) -> np.ndarray:
        density = self.as_block_density(P)
        Q = density - self.p_ref
        return hermitize(self.h0 + self.self_energy(Q))

    def hartree_energy(self, Q: np.ndarray) -> float:
        density = self.as_block_density(Q)
        return float(
            np.mean(
                [
                    self.base.hartree_energy(self._rotate_blocks(density, turn))
                    for turn in range(3)
                ]
            )
        )

    def total_energy_from_fields(
        self,
        P: np.ndarray,
        hartree_field: np.ndarray,
        fock_field: np.ndarray,
    ) -> float:
        density = self.as_block_density(P)
        direct_field = hermitize(np.asarray(hartree_field, dtype=complex))
        exchange_field = hermitize(np.asarray(fock_field, dtype=complex))
        if direct_field.shape != self.h0.shape:
            raise ValueError("hartree_field has the wrong shape")
        if exchange_field.shape != self.h0.shape:
            raise ValueError("fock_field has the wrong shape")
        relative_density = density - self.p_ref
        one_body = block_trace_product(self.h0, density)
        hartree = self.hartree_energy(relative_density)
        fock = 0.5 * block_trace_product(exchange_field, relative_density)
        return float(one_body + hartree + fock)

    def interaction_components(self, Q: np.ndarray) -> tuple[float, float]:
        density = self.as_block_density(Q)
        components = [
            self.base.interaction_components(self._rotate_blocks(density, turn))
            for turn in range(3)
        ]
        return (
            float(np.mean([row[0] for row in components])),
            float(np.mean([row[1] for row in components])),
        )

    def interaction_energy(self, Q: np.ndarray) -> float:
        hartree, fock = self.interaction_components(Q)
        return float(hartree + fock)

    def uniform_hartree_energy(self, Q: np.ndarray) -> float:
        density = self.as_block_density(Q)
        return float(
            np.mean(
                [
                    self.base.uniform_hartree_energy(self._rotate_blocks(density, turn))
                    for turn in range(3)
                ]
            )
        )

    def energy(self, P: np.ndarray) -> EnergyComponents:
        density = self.as_block_density(P)
        rows = [self.base.energy(self._rotate_blocks(density, turn)) for turn in range(3)]
        return EnergyComponents(
            total=float(np.mean([row.total for row in rows])),
            one_body=float(np.mean([row.one_body for row in rows])),
            hartree=float(np.mean([row.hartree for row in rows])),
            fock=float(np.mean([row.fock for row in rows])),
        )

    def total_energy(self, P: np.ndarray) -> float:
        return self.energy(P).total


@dataclass(frozen=True)
class ACProjectedBandStructure:
    """Single-particle finite-LL AC band data on the HF momentum mesh."""

    k_fractional: np.ndarray
    k_points: np.ndarray
    k_eigenvalues: np.ndarray
    k_active_vectors: np.ndarray
    direct_gaps: np.ndarray
    tprime_partner_index: np.ndarray
    diagnostics: dict[str, float]


def _reciprocal_box(g_cutoff: int) -> tuple[tuple[int, int], ...]:
    return reciprocal_box(g_cutoff)


def _centered_mesh_transfers(grid: MomentumGrid) -> tuple[tuple[int, int], ...]:
    def axis(n: int) -> list[int]:
        return [i if i <= n // 2 else i - n for i in range(n)]

    return tuple((i, j) for i in axis(grid.n1) for j in axis(grid.n2))


def _q_transfers(
    grid: MomentumGrid,
    interaction: ContinuumInteractionParams,
) -> tuple[tuple[int, int], ...]:
    if interaction.q_mesh == "full":
        return _centered_mesh_transfers(grid)
    return hexagonal_q_shell(interaction.q_shell)


def _mesh_fractional(grid: MomentumGrid) -> np.ndarray:
    frac = np.zeros((grid.size, 2), dtype=float)
    for ik in range(grid.size):
        i, j = grid.coord_of(ik)
        frac[ik] = (i / grid.n1, j / grid.n2)
    return frac


def _reciprocal_basis(model: NonIdealACLLModel) -> tuple[np.ndarray, np.ndarray]:
    return model.fields.G_shell[0], model.fields.G_shell[1]


def _cart_from_fractional(model: NonIdealACLLModel, frac: np.ndarray) -> np.ndarray:
    b1, b2 = _reciprocal_basis(model)
    arr = np.asarray(frac, dtype=float)
    return arr[..., 0, None] * b1 + arr[..., 1, None] * b2


def _cart_from_coord(model: NonIdealACLLModel, coord: tuple[float, float]) -> np.ndarray:
    b1, b2 = _reciprocal_basis(model)
    return float(coord[0]) * b1 + float(coord[1]) * b2


def build_ac_active_space(
    model: NonIdealACLLModel,
    grid: MomentumGrid,
    *,
    active_band: int,
    diagnostics_n_k: int,
) -> tuple[ContinuumActiveSpace, ACProjectedBandStructure]:
    """Build the two-valley AC active space before adding interactions."""

    frac = _mesh_fractional(grid)
    k_points = _cart_from_fractional(model, frac)
    n_ll = model.n_ll
    if int(active_band) >= n_ll:
        raise ValueError("active_band must be smaller than n_ll")

    all_evals = np.empty((grid.size, n_ll), dtype=float)
    active_vectors = np.empty((grid.size, n_ll), dtype=complex)
    gaps = np.empty(grid.size, dtype=float)
    for ik, k in enumerate(k_points):
        sol = model.solve(k, active_band=int(active_band))
        all_evals[ik] = sol.eigenvalues
        active_vectors[ik] = sol.eigenvectors[:, int(active_band)]
        gaps[ik] = sol.direct_gap

    partner = mesh_inversion_map(grid)
    n_active = 1
    h0 = np.zeros((grid.size, 2 * n_active, 2 * n_active), dtype=complex)
    hole_energies = np.zeros((grid.size, 2, n_active), dtype=float)
    band_vectors = np.zeros((grid.size, 2, n_ll, n_active), dtype=complex)
    for ik in range(grid.size):
        e_k = all_evals[ik, int(active_band)]
        e_kp = all_evals[int(partner[ik]), int(active_band)]
        hole_energies[ik, 0, 0] = e_k
        hole_energies[ik, 1, 0] = e_kp
        band_vectors[ik, 0, :, 0] = active_vectors[ik]
        band_vectors[ik, 1, :, 0] = np.conj(active_vectors[int(partner[ik])])
        h0[ik] = np.diag([e_k, e_kp]).astype(complex)

    model_params = ContinuumModelParams(
        n_bands=1,
        n_active_bands_per_valley=1,
        active_model="ac_finite_ll",
    )
    diagnostics = model.band_diagnostics(
        n_k=int(diagnostics_n_k),
        active_band=int(active_band),
    )
    bands = ACProjectedBandStructure(
        k_fractional=frac,
        k_points=k_points,
        k_eigenvalues=all_evals,
        k_active_vectors=active_vectors,
        direct_gaps=gaps,
        tprime_partner_index=partner,
        diagnostics=diagnostics,
    )
    active = ContinuumActiveSpace(
        grid=grid,
        n_active=n_active,
        h0=hermitize(h0),
        hole_energies=hole_energies,
        band_vectors=band_vectors,
        model=model_params,
        n_plane_waves=n_ll,
        electron_energies=-hole_energies,
        electron_vectors=np.conj(band_vectors),
        geometry=None,
        bands=bands,
    )
    return active, bands


def _dimensionless_interaction_arrays(
    model: NonIdealACLLModel,
    grid: MomentumGrid,
    q_list: tuple[tuple[int, int], ...],
    g_channels: tuple[tuple[int, int], ...],
    interaction: ContinuumInteractionParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_vectors = np.zeros((len(q_list), len(g_channels), 2), dtype=float)
    q_norm = np.zeros((len(q_list), len(g_channels)), dtype=float)
    v_q = np.zeros_like(q_norm)
    for iq, q in enumerate(q_list):
        q_cart = _cart_from_coord(model, (q[0] / grid.n1, q[1] / grid.n2))
        for ig, g in enumerate(g_channels):
            g_cart = _cart_from_coord(model, g)
            Q = q_cart + g_cart
            norm = float(np.linalg.norm(Q))
            q_vectors[iq, ig] = Q
            q_norm[iq, ig] = norm
            v_q[iq, ig] = float(interaction.v0) / (
                1.0 + (norm * float(interaction.gate_distance)) ** 2
            )
    return q_vectors, q_norm, v_q, v_q.copy()


def _dimensionless_dual_gate_interaction_arrays(
    model: NonIdealACLLModel,
    grid: MomentumGrid,
    q_list: tuple[tuple[int, int], ...],
    g_channels: tuple[tuple[int, int], ...],
    interaction: ContinuumInteractionParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``2*pi*v0*tanh(|Q|d)/|Q|`` in AC dimensionless units."""

    q_vectors = np.zeros((len(q_list), len(g_channels), 2), dtype=float)
    q_norm = np.zeros((len(q_list), len(g_channels)), dtype=float)
    v_q = np.zeros_like(q_norm)
    v0 = float(interaction.v0)
    gate_distance = float(interaction.gate_distance)
    q0_limit = 2.0 * np.pi * v0 * gate_distance
    for iq, q in enumerate(q_list):
        q_cart = _cart_from_coord(model, (q[0] / grid.n1, q[1] / grid.n2))
        for ig, g in enumerate(g_channels):
            g_cart = _cart_from_coord(model, g)
            Q = q_cart + g_cart
            norm = float(np.linalg.norm(Q))
            q_vectors[iq, ig] = Q
            q_norm[iq, ig] = norm
            if norm < 1e-12:
                if interaction.include_q0:
                    v_q[iq, ig] = q0_limit
            else:
                v_q[iq, ig] = 2.0 * np.pi * v0 * np.tanh(norm * gate_distance) / norm
    return q_vectors, q_norm, v_q, v_q.copy()


def _dual_gate_interaction_arrays(
    model: NonIdealACLLModel,
    grid: MomentumGrid,
    q_list: tuple[tuple[int, int], ...],
    g_channels: tuple[tuple[int, int], ...],
    interaction: ContinuumInteractionParams,
    *,
    moire_length_nm: float,
    energy_unit_mev: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a_m_model = float(model.fields.params.a_m)
    q_scale = a_m_model / float(moire_length_nm)
    cell_area_nm2 = model.fields.unit_cell_area * (float(moire_length_nm) / a_m_model) ** 2
    total_area_nm2 = float(grid.size) * cell_area_nm2
    q_vectors = np.zeros((len(q_list), len(g_channels), 2), dtype=float)
    q_norm = np.zeros((len(q_list), len(g_channels)), dtype=float)
    v_q = np.zeros_like(q_norm)
    v_over_a = np.zeros_like(q_norm)
    for iq, q in enumerate(q_list):
        q_cart = _cart_from_coord(model, (q[0] / grid.n1, q[1] / grid.n2))
        for ig, g in enumerate(g_channels):
            g_cart = _cart_from_coord(model, g)
            Q_nm = (q_cart + g_cart) * q_scale
            norm = float(np.linalg.norm(Q_nm))
            q_vectors[iq, ig] = Q_nm
            q_norm[iq, ig] = norm
            if norm < 1e-12:
                if interaction.include_q0:
                    v_q[iq, ig] = (
                        float(interaction.v0)
                        * 2.0
                        * np.pi
                        * E2_MEV_NM
                        * float(interaction.gate_distance_nm)
                        / float(interaction.epsilon)
                    )
            else:
                value = (
                    float(interaction.v0)
                    * 2.0
                    * np.pi
                    * E2_MEV_NM
                    * np.tanh(norm * float(interaction.gate_distance_nm))
                    / (float(interaction.epsilon) * norm)
                )
                if interaction.smear_length_nm > 0.0:
                    value *= np.exp(-0.5 * (norm * float(interaction.smear_length_nm)) ** 2)
                v_q[iq, ig] = value
            v_over_a[iq, ig] = v_q[iq, ig] / (total_area_nm2 * float(energy_unit_mev))
    return q_vectors, q_norm, v_q, v_over_a


def _interaction_arrays(
    model: NonIdealACLLModel,
    grid: MomentumGrid,
    q_list: tuple[tuple[int, int], ...],
    g_channels: tuple[tuple[int, int], ...],
    interaction: ContinuumInteractionParams,
    *,
    moire_length_nm: float,
    energy_unit_mev: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if interaction.coulomb_kind == "dual_gate":
        return _dual_gate_interaction_arrays(
            model,
            grid,
            q_list,
            g_channels,
            interaction,
            moire_length_nm=moire_length_nm,
            energy_unit_mev=energy_unit_mev,
        )
    if interaction.coulomb_kind == "dimensionless_dual_gate":
        return _dimensionless_dual_gate_interaction_arrays(
            model,
            grid,
            q_list,
            g_channels,
            interaction,
        )
    return _dimensionless_interaction_arrays(model, grid, q_list, g_channels, interaction)


def _up_form_factor(
    model: NonIdealACLLModel,
    active: ContinuumActiveSpace,
    *,
    source: int,
    target: int,
    source_shift: tuple[int, int],
    g_channel: tuple[int, int],
) -> complex:
    """Return the scalar hole vertex for physical momentum ``q+G``.

    ``density_form_factor_matrix(k, p, G)`` uses the source-to-target
    magnetic-Bloch convention

    ``<u_k|exp(i G.r)|u_p>``.

    If ``target-q = source+source_shift`` on the finite momentum torus, the
    unfolded target is obtained from the stored target by the reciprocal
    shift ``-source_shift``.  In the LL convention the physical local-field
    image ``q+G`` therefore enters the stored source/target overlap as the
    reciprocal argument ``source_shift-G``.  Keeping this folding shift is
    essential: in the ideal LLL it gives
    ``|Lambda(q,G)| = exp(-l_B^2 |q+G|^2 / 4)`` and preserves
    ``rho(q+G)^dagger = rho(-q-G)`` across Brillouin-zone boundaries.
    """

    c_source = active.band_vectors[int(source), 0, :, 0]
    c_target = active.band_vectors[int(target), 0, :, 0]
    k_source = active.bands.k_points[int(source)]
    k_target = active.bands.k_points[int(target)]
    g_effective = (
        int(source_shift[0]) - int(g_channel[0]),
        int(source_shift[1]) - int(g_channel[1]),
    )
    matrix = model.density_form_factor_matrix(
        k_source,
        k_target,
        _cart_from_coord(model, g_effective),
    )
    return complex(c_source.conj() @ matrix @ c_target)


def _ac_vertex_q_slab(
    *,
    model: NonIdealACLLModel,
    active: ContinuumActiveSpace,
    q_list: tuple[tuple[int, int], ...],
    g_channels: tuple[tuple[int, int], ...],
    channel_in_disk: np.ndarray,
    q_start: int,
    q_stop: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Build one independent q-slab of finite-LL AC density vertices."""

    grid = active.grid
    count = int(q_stop) - int(q_start)
    target_minus_q = np.empty((count, grid.size), dtype=int)
    lambdas = np.zeros(
        (count, len(g_channels), grid.size, active.dim, active.dim),
        dtype=complex,
    )
    for local_iq, iq in enumerate(range(int(q_start), int(q_stop))):
        q = q_list[iq]
        for ik in range(grid.size):
            source_coord, source_shift = grid.shift_minus_q(grid.coord_of(ik), q)
            source = grid.index_of(source_coord)
            target_minus_q[local_iq, ik] = source
            for ig, g in enumerate(g_channels):
                if not channel_in_disk[iq, ig]:
                    continue
                lambdas[local_iq, ig, ik, 0, 0] = _up_form_factor(
                    model,
                    active,
                    source=source,
                    target=ik,
                    source_shift=source_shift,
                    g_channel=g,
                )
    return int(q_start), target_minus_q, lambdas


def build_ac_density_vertices(
    model: NonIdealACLLModel,
    active: ContinuumActiveSpace,
    interaction: ContinuumInteractionParams | None = None,
    *,
    moire_length_nm: float = 1.0,
    energy_unit_mev: float = 1.0,
) -> DensityVertices:
    """Build projected density vertices for the finite-LL AC active band."""

    controls = interaction or ContinuumInteractionParams()
    grid = active.grid
    q_list = _q_transfers(grid, controls)
    g_channels = _reciprocal_box(controls.local_field_cutoff)
    n_q = len(q_list)
    n_g = len(g_channels)
    target_minus_q = np.empty((n_q, grid.size), dtype=int)
    q_is_zero = np.zeros(n_q, dtype=bool)
    lambdas = np.zeros((n_q, n_g, grid.size, active.dim, active.dim), dtype=complex)
    channel_in_disk = c3_radial_channel_mask(
        grid,
        q_list,
        g_channels,
        controls.local_field_cutoff,
    )
    c3_channel_index_map(grid, q_list, g_channels, channel_in_disk)

    for iq, q in enumerate(q_list):
        q_is_zero[iq] = q == (0, 0)

    n_workers = max(1, min(int(controls.vertex_workers), n_q))
    if n_workers == 1:
        slab_results = (
            _ac_vertex_q_slab(
                model=model,
                active=active,
                q_list=q_list,
                g_channels=g_channels,
                channel_in_disk=channel_in_disk,
                q_start=0,
                q_stop=n_q,
            ),
        )
    else:
        from joblib import Parallel, delayed

        bounds = np.linspace(0, n_q, n_workers + 1, dtype=int)
        tasks = (
            delayed(_ac_vertex_q_slab)(
                model=model,
                active=active,
                q_list=q_list,
                g_channels=g_channels,
                channel_in_disk=channel_in_disk,
                q_start=int(start),
                q_stop=int(stop),
            )
            for start, stop in zip(bounds[:-1], bounds[1:])
            if int(start) < int(stop)
        )
        slab_results = Parallel(
            n_jobs=n_workers,
            backend="loky",
            return_as="generator",
            mmap_mode="r",
            max_nbytes="32M",
        )(tasks)

    for q_start, target_slab, lambda_slab in slab_results:
        q_stop = q_start + target_slab.shape[0]
        target_minus_q[q_start:q_stop] = target_slab
        lambdas[q_start:q_stop] = lambda_slab

    inversion_partner = inversion_channel_index_map(
        grid,
        q_list,
        g_channels,
        channel_in_disk,
    )
    momentum_inversion = mesh_inversion_map(grid)
    for iq, ig in np.argwhere(channel_in_disk):
        jq, jg = inversion_partner[iq, ig]
        lambdas[iq, ig, :, 1, 1] = np.conj(
            lambdas[jq, jg, momentum_inversion, 0, 0]
        )

    q_vectors, q_norm, v_q, v_over_a = _interaction_arrays(
        model,
        grid,
        q_list,
        g_channels,
        controls,
        moire_length_nm=moire_length_nm,
        energy_unit_mev=energy_unit_mev,
    )
    v_q = np.where(channel_in_disk, v_q, 0.0)
    v_over_a = np.where(channel_in_disk, v_over_a, 0.0)
    return DensityVertices(
        q_shifts=q_list,
        target_minus_q=target_minus_q,
        q_is_zero=q_is_zero,
        lambda_blocks=lambdas,
        v_over_a=v_over_a,
        g_channels=g_channels,
        channel_in_disk=channel_in_disk,
        q_vectors_nm_inv=q_vectors if controls.coulomb_kind == "dual_gate" else None,
        q_norm_nm_inv=q_norm if controls.coulomb_kind == "dual_gate" else None,
        v_q=v_q,
    )


def build_ac_projected_bundle(
    params: ACProjectedHFParams | None = None,
    *,
    ac_params: ACParams | None = None,
    grid: ContinuumGridParams | None = None,
    interaction: ContinuumInteractionParams | None = None,
) -> ContinuumBundle:
    """Build a two-valley projected HF bundle from finite-LL AC bands."""

    controls = params or ACProjectedHFParams()
    if controls.density_vertex_scheme != C3_RADIAL_Q_PLUS_G_V2:
        raise ValueError(
            f"unsupported AC density vertex scheme {controls.density_vertex_scheme!r}"
        )
    ac_controls = ac_params or controls.ac
    grid_controls = grid or controls.grid
    interaction_controls = interaction or controls.interaction
    model = NonIdealACLLModel(ac_controls)
    momentum_grid = MomentumGrid(grid_controls.n_k)
    active, bands = build_ac_active_space(
        model,
        momentum_grid,
        active_band=controls.active_band,
        diagnostics_n_k=controls.band_diagnostics_n_k,
    )
    vertices = build_ac_density_vertices(
        model,
        active,
        interaction_controls,
        moire_length_nm=controls.moire_length_nm,
        energy_unit_mev=controls.energy_unit_mev,
    )
    raw_backend = ContinuumHFBackend(active.h0, vertices, interaction_controls)
    backend = C3SymmetrizedACBackend(raw_backend, momentum_grid)
    return ContinuumBundle(
        grid=active.grid,
        active=active,
        vertices=vertices,
        backend=backend,
        params=active.model,
        interaction=interaction_controls,
        bands=bands,
        geometry=None,
        form_factors=model,
    )


__all__ = [
    "AC_DENSITY_FORM_FACTOR_CONVENTION",
    "ACProjectedBandStructure",
    "build_ac_active_space",
    "build_ac_density_vertices",
    "build_ac_projected_bundle",
]
