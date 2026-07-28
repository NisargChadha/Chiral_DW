"""Build native continuum active spaces and density vertices."""

from __future__ import annotations

import numpy as np

from chiral_dw.config import (
    ContinuumFiniteQParams,
    ContinuumGridParams,
    ContinuumInteractionParams,
    ContinuumModelParams,
)
from chiral_dw.continuum.models import (
    ContinuumActiveSpace,
    ContinuumBundle,
    DensityVertices,
    MomentumGrid,
    hermitize,
)


def _qiwuzhang_hole_vector(u: float, v: float, mass: float = 0.6) -> tuple[float, np.ndarray]:
    """Return one smooth active-band spinor for a C=1 two-component model."""

    kx = 2.0 * np.pi * float(u)
    ky = 2.0 * np.pi * float(v)
    d = np.array(
        [
            np.sin(kx),
            np.sin(ky),
            float(mass) + np.cos(kx) + np.cos(ky),
        ],
        dtype=float,
    )
    norm = float(np.linalg.norm(d))
    dhat = d / max(norm, 1e-15)
    h = np.array(
        [
            [dhat[2], dhat[0] - 1j * dhat[1]],
            [dhat[0] + 1j * dhat[1], -dhat[2]],
        ],
        dtype=complex,
    )
    evals, evecs = np.linalg.eigh(h)
    idx = int(np.argmin(evals))
    return float(-norm), evecs[:, idx]


def build_active_space(
    grid_params: ContinuumGridParams | None = None,
    model_params: ContinuumModelParams | None = None,
    finite_q: ContinuumFiniteQParams | None = None,
) -> ContinuumActiveSpace:
    """Build the native two-valley active space."""

    grid_controls = grid_params or ContinuumGridParams()
    model = model_params or ContinuumModelParams()
    finite_q_params = finite_q or ContinuumFiniteQParams()
    grid = MomentumGrid(grid_controls.n_k)
    if model.active_model == "taige":
        from chiral_dw.continuum.taige import build_taige_active_space

        active, _bands = build_taige_active_space(grid, model, finite_q_params)
        return active
    if finite_q_params.enabled:
        raise NotImplementedError("finite-Q active frames are implemented for Taige models only")
    if model.active_model != "qiwuzhang":
        raise ValueError(f"unknown active_model {model.active_model!r}")
    n_active = int(model.n_active_bands_per_valley)
    if n_active != 1:
        raise NotImplementedError("native continuum v1 supports one active band per valley")

    frac = grid.fractional_coords()
    h0 = np.zeros((grid.size, 2 * n_active, 2 * n_active), dtype=complex)
    hole_energies = np.zeros((grid.size, 2, n_active), dtype=float)
    band_vectors = np.zeros((grid.size, 2, 2, n_active), dtype=complex)
    for ik, (u, v) in enumerate(frac):
        e_k, vec_k = _qiwuzhang_hole_vector(float(u), float(v))
        e_kp, vec_kp = _qiwuzhang_hole_vector(float(-u), float(-v))
        hole_energies[ik, 0, 0] = e_k + 0.5 * model.displacement_mev
        hole_energies[ik, 1, 0] = e_kp - 0.5 * model.displacement_mev
        band_vectors[ik, 0, :, 0] = vec_k
        band_vectors[ik, 1, :, 0] = np.conj(vec_kp)
        h0[ik] = np.diag(hole_energies[ik].reshape(-1).astype(complex))
    return ContinuumActiveSpace(
        grid=grid,
        n_active=n_active,
        h0=hermitize(h0),
        hole_energies=hole_energies,
        band_vectors=band_vectors,
        model=model,
    )


def q_shifts_for_shell(shell: int) -> tuple[tuple[int, int], ...]:
    """Return integer mesh momentum shifts within a small square shell."""

    radius = int(shell)
    shifts: list[tuple[int, int]] = []
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            if max(abs(di), abs(dj)) <= radius:
                shifts.append((di, dj))
    if (0, 0) not in shifts:
        shifts.insert(0, (0, 0))
    shifts = sorted(set(shifts), key=lambda x: (abs(x[0]) + abs(x[1]), x[0], x[1]))
    return tuple(shifts)


def build_density_vertices(
    active: ContinuumActiveSpace,
    interaction: ContinuumInteractionParams | None = None,
) -> DensityVertices:
    """Build a simple screened-Coulomb projected density vertex table."""

    controls = interaction or ContinuumInteractionParams()
    if active.model.active_model == "taige":
        from chiral_dw.continuum.taige import build_taige_density_vertices

        return build_taige_density_vertices(active, controls)
    shifts = q_shifts_for_shell(controls.q_shell)
    n_q = len(shifts)
    n_g = 1
    target_minus_q = np.zeros((n_q, active.n_k), dtype=int)
    q_is_zero = np.zeros(n_q, dtype=bool)
    lambdas = np.zeros((n_q, n_g, active.n_k, active.dim, active.dim), dtype=complex)
    v_over_a = np.zeros((n_q, n_g), dtype=float)
    eye = np.eye(active.dim, dtype=complex)

    for iq, (di, dj) in enumerate(shifts):
        q_is_zero[iq] = di == 0 and dj == 0
        q2 = float(di * di + dj * dj)
        screened = controls.v0 / (1.0 + q2 * controls.gate_distance * controls.gate_distance)
        v_over_a[iq, 0] = screened
        for ik in range(active.n_k):
            i, j = active.grid.coord_of(ik)
            target_minus_q[iq, ik] = active.grid.index_of((i - di, j - dj))
            lambdas[iq, 0, ik] = eye

    return DensityVertices(
        q_shifts=shifts,
        target_minus_q=target_minus_q,
        q_is_zero=q_is_zero,
        lambda_blocks=lambdas,
        v_over_a=v_over_a,
    )


def build_continuum_bundle(
    model: ContinuumModelParams | None = None,
    grid: ContinuumGridParams | None = None,
    interaction: ContinuumInteractionParams | None = None,
    finite_q: ContinuumFiniteQParams | None = None,
) -> ContinuumBundle:
    """Build a self-contained native continuum/HF bundle."""

    model_params = model or ContinuumModelParams()
    interaction_params = interaction or ContinuumInteractionParams()
    finite_q_params = finite_q or ContinuumFiniteQParams()
    if model_params.active_model == "taige" and finite_q_params.enabled:
        _q0_bundle, finite_q_bundle = build_taige_q_sector_bundles(
            model=model_params,
            grid=grid,
            interaction=interaction_params,
            finite_q=finite_q_params,
        )
        return finite_q_bundle
    active = build_active_space(grid, model_params, finite_q_params)
    vertices = build_density_vertices(active, interaction_params)
    from chiral_dw.continuum.hf import ContinuumHFBackend

    backend = ContinuumHFBackend(active.h0, vertices, interaction_params)
    return ContinuumBundle(
        grid=active.grid,
        active=active,
        vertices=backend.vertices,
        backend=backend,
        params=model_params,
        interaction=interaction_params,
        finite_q=finite_q_params,
        bands=active.bands,
        geometry=active.geometry,
        form_factors=None,
    )


def build_taige_q_sector_bundles(
    model: ContinuumModelParams,
    grid: ContinuumGridParams | None,
    interaction: ContinuumInteractionParams | None,
    finite_q: ContinuumFiniteQParams,
) -> tuple[ContinuumBundle, ContinuumBundle]:
    """Build q=0 and finite-Q Taige bundles from shared bands and raw vertices."""

    if model.active_model != "taige":
        raise ValueError("shared Taige q-sector construction requires active_model='taige'")
    if not finite_q.enabled:
        raise ValueError("finite_q must be enabled for paired q-sector construction")

    from chiral_dw.continuum.hf import ContinuumHFBackend
    from chiral_dw.continuum.taige import (
        build_taige_active_space,
        build_taige_density_vertices,
        compute_taige_bandstructure,
        roll_taige_density_vertices,
    )

    grid_params = grid or ContinuumGridParams()
    interaction_params = interaction or ContinuumInteractionParams()
    momentum_grid = MomentumGrid(grid_params.n_k)
    bands = compute_taige_bandstructure(model, momentum_grid)
    q0_params = ContinuumFiniteQParams()
    q0_active, _ = build_taige_active_space(
        momentum_grid,
        model,
        q0_params,
        bands=bands,
    )
    finite_q_active, _ = build_taige_active_space(
        momentum_grid,
        model,
        finite_q,
        bands=bands,
    )

    raw_q0_vertices = build_taige_density_vertices(q0_active, interaction_params)
    raw_finite_q_vertices = roll_taige_density_vertices(
        raw_q0_vertices,
        finite_q_active,
    )
    q0_backend = ContinuumHFBackend(
        q0_active.h0,
        raw_q0_vertices,
        interaction_params,
    )
    finite_q_backend = ContinuumHFBackend(
        finite_q_active.h0,
        raw_finite_q_vertices,
        interaction_params,
    )

    def _bundle(
        active: ContinuumActiveSpace,
        backend: ContinuumHFBackend,
        controls: ContinuumFiniteQParams,
    ) -> ContinuumBundle:
        return ContinuumBundle(
            grid=active.grid,
            active=active,
            vertices=backend.vertices,
            backend=backend,
            params=model,
            interaction=interaction_params,
            finite_q=controls,
            bands=bands,
            geometry=active.geometry,
            form_factors=None,
        )

    return (
        _bundle(q0_active, q0_backend, q0_params),
        _bundle(finite_q_active, finite_q_backend, finite_q),
    )
