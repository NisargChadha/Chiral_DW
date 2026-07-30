from types import SimpleNamespace

import numpy as np
import pytest

from chiral_dw.config import (
    ContinuumFiniteQParams,
    ContinuumGridParams,
    ContinuumHFParams,
    ContinuumInteractionParams,
    ContinuumModelParams,
)
from chiral_dw.continuum import (
    ContinuumHFBackend,
    DensityVertices,
    TPrimeConstraint,
    ValleyU1Constraint,
    active_basis_frames,
    build_continuum_bundle,
    build_symmetric_hf_references,
    build_taige_q_sector_bundles,
    convex_weights,
    evaluate_hf_high_symmetry_path,
    linear_interaction_hamiltonian,
    linear_interaction_path,
    linear_interaction_weights,
    mesh_inversion_map,
    projector_path_for_interpolation,
    projector_maps,
    reference_diagnostics,
    rotate_valley_u1,
    solve_hf,
    solve_reference_hf,
    symmetric_convex_hamiltonian,
    symmetric_convex_path,
    symmetric_convex_projector,
    taige_ivc_minus_half_shift_coord,
    taige_ivc_minus_q_coord,
    taige_model_params,
)
from chiral_dw.continuum.models import SymmetricHFReferences, block_trace_product, hermitize
from chiral_dw.continuum.models import dense_lambdas_from_compact
from chiral_dw.continuum.hf import (
    _choose_oda_lambda,
    _exchange_q_slab_ranges,
    _hermitize_dense_in_place,
)
from chiral_dw.continuum.seeds import ivc_seed, valley_polarized_seed
from chiral_dw.continuum.taige import build_taige_density_vertices
from chiral_dw.response import compute_cG, k_theta_from_projectors_with_basis


def _small_bundle():
    return build_continuum_bundle(
        model=ContinuumModelParams(displacement_mev=0.0),
        grid=ContinuumGridParams(n_k=3),
        interaction=ContinuumInteractionParams(v0=0.2, q_shell=0, gate_distance=1.0),
    )


def _synthetic_refs() -> SymmetricHFReferences:
    h_vp_plus = np.asarray([[[0.0, 0.0], [0.0, 5.0]]], dtype=complex)
    h_vp_minus = np.asarray([[[5.0, 0.0], [0.0, 0.0]]], dtype=complex)
    h_ivc = np.asarray([[[2.5, -1.0], [-1.0, 2.5]]], dtype=complex)
    return SymmetricHFReferences(
        vp_plus=SimpleNamespace(H_hf=h_vp_plus),
        vp_minus=SimpleNamespace(H_hf=h_vp_minus),
        ivc=SimpleNamespace(H_hf=h_ivc),
        n_occ_per_k=1,
    )


def test_linear_interaction_path_uses_h0_once_and_preserves_legacy_mode():
    refs = _synthetic_refs()
    h0 = np.asarray([[[1.0, 0.2], [0.2, 1.8]]], dtype=complex)
    theta = np.asarray([0.0, np.pi / 3.0, np.pi / 2.0, np.pi])

    projectors, diagnostics = linear_interaction_path(refs, theta, h0)
    old_projectors, old_diagnostics = projector_path_for_interpolation(
        refs,
        theta,
        trial_interpolation="convex_full_hf",
    )
    direct_old_projectors, direct_old_diagnostics = symmetric_convex_path(refs, theta)

    assert np.allclose(projectors[0], symmetric_convex_projector(refs, 0.0)[0])
    assert np.allclose(projectors[2], symmetric_convex_projector(refs, np.pi / 2.0)[0])
    assert np.allclose(projectors[3], symmetric_convex_projector(refs, np.pi)[0])
    assert np.allclose(old_projectors, direct_old_projectors)
    assert [row.w_ivc for row in old_diagnostics] == pytest.approx(
        [row.w_ivc for row in direct_old_diagnostics]
    )

    w_plus, w_minus, w_ivc = linear_interaction_weights(np.pi / 3.0)
    expected = h0 + w_plus * (refs.H_vp_plus - h0) + w_minus * (
        refs.H_vp_minus - h0
    ) + w_ivc * (refs.H_ivc - h0)
    actual = linear_interaction_hamiltonian(refs, np.pi / 3.0, h0)
    assert np.allclose(actual, hermitize(expected))
    assert diagnostics[1].w_vp_plus == pytest.approx(0.5)
    assert diagnostics[1].w_vp_minus == pytest.approx(0.0)
    assert diagnostics[1].w_ivc == pytest.approx(np.sin(np.pi / 3.0))


def _slow_hartree(backend: ContinuumHFBackend, Q: np.ndarray) -> np.ndarray:
    out = np.zeros_like(Q, dtype=complex)
    for iq, ig, v in backend.hartree_channels:
        if backend.vertex_layout == "valley_compact":
            lam = dense_lambdas_from_compact(backend.lambda_compact[iq, ig])
        else:
            lam = backend.lambda_blocks[iq, ig]
        density = np.einsum("kab,kba->", lam, Q, optimize=True)
        out += 0.5 * v * (
            np.conj(density) * lam + density * np.swapaxes(lam.conj(), -1, -2)
        )
    return hermitize(out)


def _slow_fock(backend: ContinuumHFBackend, Q: np.ndarray) -> np.ndarray:
    out = np.zeros_like(Q, dtype=complex)
    scale = float(backend.interaction.exchange_scale)
    for iq in range(backend.n_q):
        targets = backend.target_minus_q[iq]
        for ig in range(backend.n_g):
            v = scale * float(backend.v_over_a[iq, ig])
            if v == 0.0:
                continue
            if backend.vertex_layout == "valley_compact":
                lam = dense_lambdas_from_compact(backend.lambda_compact[iq, ig])
            else:
                lam = backend.lambda_blocks[iq, ig]
            for ik in range(backend.n_blocks):
                jk = int(targets[ik])
                out[ik] -= 0.5 * v * lam[ik] @ Q[jk] @ lam[ik].conj().T
                out[jk] -= 0.5 * v * lam[ik].conj().T @ Q[ik] @ lam[ik]
    return hermitize(out)


def test_tiled_dense_hermitization_matches_full_expression():
    rng = np.random.default_rng(17)
    raw = rng.normal(size=(13, 13)) + 1j * rng.normal(size=(13, 13))
    expected = 0.5 * (raw + raw.conj().T)
    actual = raw.copy()

    returned = _hermitize_dense_in_place(actual, tile_size=4)

    assert returned is actual
    assert np.allclose(actual, expected)


def test_parallel_exchange_q_slabs_are_bounded_for_production_mesh():
    ranges = _exchange_q_slab_ranges(900, exchange_workers=4)

    assert ranges[0] == (0, 8)
    assert ranges[-1] == (896, 900)
    assert max(stop - start for start, stop in ranges) <= 8
    assert [iq for start, stop in ranges for iq in range(start, stop)] == list(
        range(900)
    )


def test_optimized_backend_matches_slow_hartree_fock_reference():
    rng = np.random.default_rng(11)
    h0 = np.stack(
        [
            np.diag([-0.3, 0.4]),
            np.array([[0.1, 0.03], [0.03, 0.7]], dtype=complex),
            np.diag([0.2, 0.8]),
        ]
    )
    lambdas = rng.normal(size=(2, 2, 3, 2, 2)) + 1j * rng.normal(size=(2, 2, 3, 2, 2))
    lambdas[0, 0] = np.eye(2)
    target_minus_q = np.asarray([[0, 1, 2], [2, 0, 1]], dtype=int)
    vertices = DensityVertices(
        q_shifts=((0, 0), (1, 0)),
        target_minus_q=target_minus_q,
        q_is_zero=np.asarray([True, False]),
        lambda_blocks=lambdas,
        v_over_a=np.asarray([[0.2, 0.05], [0.07, 0.03]], dtype=float),
        g_channels=((0, 0), (1, 0)),
    )
    backend = ContinuumHFBackend(h0, vertices, ContinuumInteractionParams())
    parallel_backend = ContinuumHFBackend(
        h0,
        vertices,
        ContinuumInteractionParams(exchange_workers=2),
    )
    raw = rng.normal(size=h0.shape) + 1j * rng.normal(size=h0.shape)
    Q = hermitize(raw)

    assert backend.tVE.shape == (backend.n_blocks * backend.dim**2, backend.n_blocks * backend.dim**2)
    assert np.allclose(parallel_backend.tVE, backend.tVE)
    assert np.allclose(parallel_backend.fock_hamiltonian(Q), backend.fock_hamiltonian(Q))
    assert np.allclose(parallel_backend.hf_hamiltonian(Q), backend.hf_hamiltonian(Q))
    assert np.allclose(backend.hartree_hamiltonian(Q), _slow_hartree(backend, Q))
    assert np.allclose(backend.fock_hamiltonian(Q), _slow_fock(backend, Q))
    assert np.allclose(
        backend.hf_hamiltonian(Q),
        hermitize(backend.h0 + _slow_hartree(backend, Q) + _slow_fock(backend, Q)),
    )
    slow_hartree = 0.5 * block_trace_product(_slow_hartree(backend, Q), Q)
    slow_fock = 0.5 * block_trace_product(_slow_fock(backend, Q), Q)
    assert backend.energy(Q).hartree == pytest.approx(slow_hartree)
    assert backend.energy(Q).fock == pytest.approx(slow_fock)

    P, _evals, _direct, _indirect = backend.update_density(backend.h0, 3)
    assert np.real(np.trace(P, axis1=-2, axis2=-1).sum()) == pytest.approx(3.0)


def test_working_field_oda_linearity_matches_direct_recomputation():
    bundle = _small_bundle()
    backend = bundle.backend
    P_vp = valley_polarized_seed(bundle.active, "K")
    P_ivc = ivc_seed(bundle.active, n_occ_per_k=1)
    P = hermitize(0.63 * P_vp + 0.37 * P_ivc)
    relative_density = P - backend.p_ref
    hartree = backend.hartree_hamiltonian(relative_density)
    fock = backend.fock_hamiltonian(relative_density)
    H = hermitize(backend.h0 + hartree + fock)
    P_aufbau, _evals, _direct, _indirect = backend.update_density_per_k(H, 1)
    delta = hermitize(P_aufbau - P)
    trial_relative_density = P_aufbau - backend.p_ref
    trial_hartree = backend.hartree_hamiltonian(trial_relative_density)
    trial_fock = backend.fock_hamiltonian(trial_relative_density)
    delta_hartree = hermitize(trial_hartree - hartree)
    delta_fock = hermitize(trial_fock - fock)

    slope = block_trace_product(H, delta)
    curvature_from_linearity = (
        2.0 * backend.hartree_energy(delta)
        + block_trace_product(delta_fock, delta)
    )
    curvature_direct = 2.0 * backend.interaction_energy(delta)
    lambda_linear, reason_linear = _choose_oda_lambda(
        slope,
        curvature_from_linearity,
        1e-4,
    )
    lambda_direct, reason_direct = _choose_oda_lambda(
        slope,
        curvature_direct,
        1e-4,
    )

    assert curvature_from_linearity == pytest.approx(curvature_direct, abs=1e-12)
    assert lambda_linear == pytest.approx(lambda_direct, abs=1e-12)
    assert reason_linear == reason_direct

    P_mixed = hermitize(P + lambda_linear * delta)
    hartree_mixed = hermitize(hartree + lambda_linear * delta_hartree)
    fock_mixed = hermitize(fock + lambda_linear * delta_fock)
    assert np.allclose(
        hartree_mixed,
        backend.hartree_hamiltonian(P_mixed - backend.p_ref),
        atol=1e-12,
    )
    assert np.allclose(
        fock_mixed,
        backend.fock_hamiltonian(P_mixed - backend.p_ref),
        atol=1e-12,
    )
    assert backend.total_energy_from_fields(
        P_mixed,
        hartree_mixed,
        fock_mixed,
    ) == pytest.approx(backend.energy(P_mixed).total, abs=1e-12)


def test_hf_oda_applies_one_trial_fock_field_per_iteration(monkeypatch):
    bundle = _small_bundle()
    backend = bundle.backend
    original_fock_hamiltonian = backend.fock_hamiltonian
    calls = 0

    def counted_fock_hamiltonian(Q):
        nonlocal calls
        calls += 1
        return original_fock_hamiltonian(Q)

    monkeypatch.setattr(backend, "fock_hamiltonian", counted_fock_hamiltonian)
    result = solve_hf(
        backend,
        ivc_seed(bundle.active, n_occ_per_k=1),
        ContinuumHFParams(
            max_iter=4,
            min_iter=4,
            mixing_method="oda",
            tolerance=1e-30,
            energy_tolerance=1e-30,
        ),
    )

    # One initial field, one trial field per iteration, and one final
    # idempotent-projector field.
    assert calls == result.n_iter + 2


def test_working_field_oda_matches_direct_recomputation_trajectory():
    bundle = _small_bundle()
    backend = bundle.backend
    controls = ContinuumHFParams(
        max_iter=4,
        min_iter=4,
        mixing_method="oda",
        tolerance=1e-30,
        energy_tolerance=1e-30,
    )
    P_direct = hermitize(
        0.57 * valley_polarized_seed(bundle.active, "K")
        + 0.43 * ivc_seed(bundle.active, n_occ_per_k=1)
    )
    direct_lambdas = []
    direct_energies = []
    for _iteration in range(1, controls.max_iter + 1):
        H_direct = backend.hf_hamiltonian(P_direct)
        P_aufbau, _evals, _direct, _indirect = backend.update_density_per_k(
            H_direct,
            controls.n_occ_per_k,
        )
        delta = hermitize(P_aufbau - P_direct)
        slope = block_trace_product(H_direct, delta)
        curvature = 2.0 * backend.interaction_energy(delta)
        lambda_value, _reason = _choose_oda_lambda(
            slope,
            curvature,
            controls.oda_lambda_min,
        )
        P_direct = hermitize(P_direct + lambda_value * delta)
        direct_lambdas.append(lambda_value)
        direct_energies.append(backend.energy(P_direct).total)
    H_direct = backend.hf_hamiltonian(P_direct)
    P_final_direct, _evals, _direct, _indirect = backend.update_density_per_k(
        H_direct,
        controls.n_occ_per_k,
    )

    result = solve_hf(
        backend,
        hermitize(
            0.57 * valley_polarized_seed(bundle.active, "K")
            + 0.43 * ivc_seed(bundle.active, n_occ_per_k=1)
        ),
        controls,
    )

    assert [row.lambda_value for row in result.history] == pytest.approx(
        direct_lambdas,
        abs=1e-12,
    )
    assert [row.energy for row in result.history] == pytest.approx(
        direct_energies,
        abs=1e-12,
    )
    assert np.allclose(result.P, P_final_direct, atol=1e-12)


def test_finite_q_taige_backend_matches_slow_hartree_fock_reference():
    interaction = ContinuumInteractionParams(
        coulomb_kind="dimensionless_screened",
        v0=0.03,
        q_shell=1,
        local_field_cutoff=0,
    )
    bundle = build_continuum_bundle(
        model=taige_model_params(theta_deg=3.5, u_D=0.0, plane_wave_shell=1, n_bands=1),
        grid=ContinuumGridParams(n_k=6),
        finite_q=ContinuumFiniteQParams(
            enabled=True,
            q_coord=taige_ivc_minus_q_coord(6),
            half_shift_coord=taige_ivc_minus_half_shift_coord(6),
        ),
        interaction=interaction,
    )
    parallel_bundle = build_continuum_bundle(
        model=taige_model_params(theta_deg=3.5, u_D=0.0, plane_wave_shell=1, n_bands=1),
        grid=ContinuumGridParams(n_k=6),
        finite_q=ContinuumFiniteQParams(
            enabled=True,
            q_coord=taige_ivc_minus_q_coord(6),
            half_shift_coord=taige_ivc_minus_half_shift_coord(6),
        ),
        interaction=interaction.model_copy(update={"exchange_workers": 2}),
    )
    rng = np.random.default_rng(13)
    Q = hermitize(
        rng.normal(size=bundle.active.h0.shape) + 1j * rng.normal(size=bundle.active.h0.shape)
    )

    assert bundle.backend.exchange_representation == "valley_sector"
    assert parallel_bundle.backend.exchange_representation == "valley_sector"
    assert np.allclose(
        parallel_bundle.backend.dense_exchange_tve_for_debug(),
        bundle.backend.dense_exchange_tve_for_debug(),
    )
    assert np.allclose(
        parallel_bundle.backend.fock_hamiltonian(Q),
        bundle.backend.fock_hamiltonian(Q),
    )
    assert np.allclose(
        parallel_bundle.backend.hf_hamiltonian(Q),
        bundle.backend.hf_hamiltonian(Q),
    )
    parallel_energy = parallel_bundle.backend.energy(Q)
    serial_energy = bundle.backend.energy(Q)
    assert parallel_energy.total == pytest.approx(serial_energy.total, abs=1e-12)
    assert parallel_energy.one_body == pytest.approx(serial_energy.one_body, abs=1e-12)
    assert parallel_energy.hartree == pytest.approx(serial_energy.hartree, abs=1e-12)
    assert parallel_energy.fock == pytest.approx(serial_energy.fock, abs=1e-12)
    assert np.allclose(bundle.backend.hartree_hamiltonian(Q), _slow_hartree(bundle.backend, Q))
    assert np.allclose(bundle.backend.fock_hamiltonian(Q), _slow_fock(bundle.backend, Q))


@pytest.mark.parametrize("layout", ["valley_compact", "dense"])
def test_rolled_finite_q_backend_matches_direct_reconstruction(layout):
    interaction = ContinuumInteractionParams(
        coulomb_kind="dimensionless_screened",
        v0=0.04,
        q_shell=1,
        local_field_cutoff=1,
        density_vertex_layout=layout,
        exchange_representation="auto",
    )
    finite_q = ContinuumFiniteQParams(
        enabled=True,
        q_coord=taige_ivc_minus_q_coord(6),
        half_shift_coord=taige_ivc_minus_half_shift_coord(6),
    )
    _q0_bundle, rolled_bundle = build_taige_q_sector_bundles(
        taige_model_params(
            theta_deg=3.5,
            u_D=0.0,
            plane_wave_shell=1,
            n_bands=2,
        ),
        ContinuumGridParams(n_k=6),
        interaction,
        finite_q,
    )
    direct_vertices = build_taige_density_vertices(
        rolled_bundle.active,
        interaction,
    )
    direct_backend = ContinuumHFBackend(
        rolled_bundle.active.h0,
        direct_vertices,
        interaction,
    )
    rolled_backend = rolled_bundle.backend
    rng = np.random.default_rng(71)
    density = hermitize(
        rng.normal(size=rolled_bundle.active.h0.shape)
        + 1j * rng.normal(size=rolled_bundle.active.h0.shape)
    )

    assert direct_backend.exchange_representation == rolled_backend.exchange_representation
    assert direct_backend.exchange_representation == (
        "valley_sector" if layout == "valley_compact" else "dense"
    )
    assert np.allclose(
        direct_backend.dense_exchange_tve_for_debug(),
        rolled_backend.dense_exchange_tve_for_debug(),
        atol=1e-12,
    )
    assert np.allclose(
        direct_backend.hartree_hamiltonian(density),
        rolled_backend.hartree_hamiltonian(density),
        atol=1e-12,
    )
    assert np.allclose(
        direct_backend.fock_hamiltonian(density),
        rolled_backend.fock_hamiltonian(density),
        atol=1e-12,
    )
    assert np.allclose(
        direct_backend.hf_hamiltonian(density),
        rolled_backend.hf_hamiltonian(density),
        atol=1e-12,
    )
    direct_energy = direct_backend.energy(density)
    rolled_energy = rolled_backend.energy(density)
    assert direct_energy.one_body == pytest.approx(rolled_energy.one_body, abs=1e-12)
    assert direct_energy.hartree == pytest.approx(rolled_energy.hartree, abs=1e-12)
    assert direct_energy.fock == pytest.approx(rolled_energy.fock, abs=1e-12)
    assert direct_energy.total == pytest.approx(rolled_energy.total, abs=1e-12)

    hf_params = ContinuumHFParams(
        n_occ_per_k=1,
        max_iter=4,
        min_iter=1,
        mixing_method="linear",
        mixing=0.4,
        seed_random_weight=0.0,
    )
    seed = ivc_seed(rolled_bundle.active)
    direct_result = solve_hf(direct_backend, seed, hf_params)
    rolled_result = solve_hf(rolled_backend, seed, hf_params)
    assert direct_result.energy == pytest.approx(rolled_result.energy, abs=1e-12)
    assert np.allclose(direct_result.H_hf, rolled_result.H_hf, atol=1e-12)
    assert np.allclose(direct_result.P, rolled_result.P, atol=1e-12)


def test_shared_q_sector_backends_are_isolated_before_and_after_retention():
    interaction = ContinuumInteractionParams(
        coulomb_kind="dimensionless_screened",
        v0=0.04,
        q_shell=1,
        local_field_cutoff=1,
        density_vertex_retention="hartree_only",
    )
    finite_q = ContinuumFiniteQParams(
        enabled=True,
        q_coord=taige_ivc_minus_q_coord(6),
        half_shift_coord=taige_ivc_minus_half_shift_coord(6),
    )
    q0_bundle, finite_bundle = build_taige_q_sector_bundles(
        taige_model_params(
            theta_deg=3.5,
            u_D=0.0,
            plane_wave_shell=1,
            n_bands=1,
        ),
        ContinuumGridParams(n_k=6),
        interaction,
        finite_q,
    )
    q0_backend = q0_bundle.backend
    finite_backend = finite_bundle.backend

    assert q0_backend is not finite_backend
    assert q0_backend.valley_sector_exchange is not finite_backend.valley_sector_exchange
    assert not np.shares_memory(q0_backend.lambda_compact, finite_backend.lambda_compact)
    assert not np.shares_memory(q0_backend.target_minus_q, finite_backend.target_minus_q)
    assert not np.shares_memory(
        q0_backend.valley_sector_exchange.sectors,
        finite_backend.valley_sector_exchange.sectors,
    )
    finite_lambdas = finite_backend.lambda_compact.copy()
    finite_targets = finite_backend.target_minus_q.copy()
    finite_exchange = finite_backend.valley_sector_exchange.sectors.copy()

    q0_backend.lambda_compact[...] = 0.0
    q0_backend.target_minus_q[...] = 0
    q0_backend.valley_sector_exchange.sectors[...] = 0.0

    assert np.array_equal(finite_backend.lambda_compact, finite_lambdas)
    assert np.array_equal(finite_backend.target_minus_q, finite_targets)
    assert np.array_equal(
        finite_backend.valley_sector_exchange.sectors,
        finite_exchange,
    )


def test_taige_hartree_only_retention_preserves_backend_physics():
    base_interaction = ContinuumInteractionParams(
        coulomb_kind="dimensionless_screened",
        v0=0.04,
        q_mesh="full",
        q_shell=0,
        local_field_cutoff=1,
        density_vertex_retention="full",
    )
    model = taige_model_params(theta_deg=3.5, u_D=0.0, plane_wave_shell=1, n_bands=1)
    grid = ContinuumGridParams(n_k=3)
    dense_bundle = build_continuum_bundle(
        model=model,
        grid=grid,
        interaction=base_interaction.model_copy(update={"density_vertex_layout": "dense"}),
    )
    compact_bundle = build_continuum_bundle(
        model=model,
        grid=grid,
        interaction=base_interaction,
    )
    retained_bundle = build_continuum_bundle(
        model=model,
        grid=grid,
        interaction=base_interaction.model_copy(
            update={"density_vertex_retention": "hartree_only"}
        ),
    )
    rng = np.random.default_rng(19)
    Q = hermitize(
        rng.normal(size=dense_bundle.active.h0.shape)
        + 1j * rng.normal(size=dense_bundle.active.h0.shape)
    )

    assert dense_bundle.vertices.vertex_layout == "dense"
    assert compact_bundle.vertices.vertex_layout == "valley_compact"
    assert retained_bundle.vertices.vertex_layout == "valley_compact"
    assert dense_bundle.vertices.lambda_blocks.shape[:2] == (9, 9)
    assert compact_bundle.vertices.lambda_blocks.shape[:2] == (0, 0)
    assert compact_bundle.vertices.lambda_compact.shape[:3] == (9, 9, 9)
    assert np.allclose(
        dense_lambdas_from_compact(compact_bundle.vertices.lambda_compact),
        dense_bundle.vertices.lambda_blocks,
    )
    assert retained_bundle.vertices.lambda_blocks.shape[:2] == (0, 0)
    assert retained_bundle.vertices.q_shifts == dense_bundle.vertices.q_shifts
    assert retained_bundle.vertices.g_channels == dense_bundle.vertices.g_channels
    assert np.array_equal(
        retained_bundle.vertices.target_minus_q,
        dense_bundle.vertices.target_minus_q,
    )
    assert retained_bundle.backend.lambda_compact.shape[0] == len(
        retained_bundle.backend.full_hartree_channels
    )
    retained_channels = int(np.prod(retained_bundle.backend.lambda_compact.shape[:2]))
    full_channels = int(np.prod(dense_bundle.backend.lambda_blocks.shape[:2]))
    assert retained_channels < full_channels

    assert dense_bundle.backend.exchange_representation == "dense"
    assert compact_bundle.backend.exchange_representation == "valley_sector"
    assert retained_bundle.backend.exchange_representation == "valley_sector"
    assert compact_bundle.backend.tVE is None
    assert retained_bundle.backend.tVE is None
    assert compact_bundle.backend.valley_sector_exchange is not None
    assert retained_bundle.backend.valley_sector_exchange is not None
    assert np.allclose(
        compact_bundle.backend.dense_exchange_tve_for_debug(),
        dense_bundle.backend.dense_exchange_tve_for_debug(),
    )
    assert np.allclose(
        retained_bundle.backend.dense_exchange_tve_for_debug(),
        dense_bundle.backend.dense_exchange_tve_for_debug(),
    )
    assert np.allclose(
        retained_bundle.backend.fock_hamiltonian(Q),
        dense_bundle.backend.fock_hamiltonian(Q),
    )
    assert np.allclose(
        retained_bundle.backend.hartree_hamiltonian(Q),
        dense_bundle.backend.hartree_hamiltonian(Q),
    )
    assert np.allclose(
        retained_bundle.backend.hf_hamiltonian(Q),
        dense_bundle.backend.hf_hamiltonian(Q),
    )
    full_energy = dense_bundle.backend.energy(Q)
    retained_energy = retained_bundle.backend.energy(Q)
    assert retained_energy.total == pytest.approx(full_energy.total)
    assert retained_energy.one_body == pytest.approx(full_energy.one_body)
    assert retained_energy.hartree == pytest.approx(full_energy.hartree)
    assert retained_energy.fock == pytest.approx(full_energy.fock)

    full_H = dense_bundle.backend.hf_hamiltonian(Q)
    retained_H = retained_bundle.backend.hf_hamiltonian(Q)
    _P_full, _evals_full, full_direct, full_indirect = dense_bundle.backend.update_density_per_k(
        full_H,
        1,
    )
    _P_retained, _evals_retained, retained_direct, retained_indirect = (
        retained_bundle.backend.update_density_per_k(retained_H, 1)
    )
    assert retained_direct == pytest.approx(full_direct)
    assert retained_indirect == pytest.approx(full_indirect)

    hf_params = ContinuumHFParams(
        n_occ_per_k=1,
        max_iter=3,
        min_iter=1,
        mixing_method="linear",
        mixing=0.4,
        seed_random_weight=0.0,
    )
    full_seed = valley_polarized_seed(dense_bundle.active, valley="K")
    retained_seed = valley_polarized_seed(retained_bundle.active, valley="K")
    full_result = solve_hf(
        dense_bundle.backend,
        full_seed,
        hf_params,
        constraint=ValleyU1Constraint(dense_bundle.active),
    )
    retained_result = solve_hf(
        retained_bundle.backend,
        retained_seed,
        hf_params,
        constraint=ValleyU1Constraint(retained_bundle.active),
    )
    assert retained_result.energy == pytest.approx(full_result.energy)
    assert np.allclose(retained_result.H_hf, full_result.H_hf)

    full_spectrum = evaluate_hf_high_symmetry_path(
        dense_bundle,
        full_result.P,
        n_per_segment=1,
        reference="full",
    )
    retained_spectrum = evaluate_hf_high_symmetry_path(
        retained_bundle,
        retained_result.P,
        n_per_segment=1,
        reference="retained",
    )
    assert np.allclose(retained_spectrum.energies, full_spectrum.energies)


def test_valley_u1_constraint_projects_intervalley_blocks_and_preserves_vp_seed():
    bundle = _small_bundle()
    active = bundle.active
    constraint = ValleyU1Constraint(active)
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(active.n_k, active.dim, active.dim)) + 1j * rng.normal(
        size=(active.n_k, active.dim, active.dim)
    )
    blocks = hermitize(raw)
    projected = constraint.project_density(blocks)
    n = active.n_active

    assert np.allclose(projected[:, :n, n:], 0.0)
    assert np.allclose(projected[:, n:, :n], 0.0)
    assert np.allclose(projected, projected.conj().swapaxes(-1, -2))
    assert np.allclose(np.trace(projected, axis1=-2, axis2=-1), np.trace(blocks, axis1=-2, axis2=-1))
    assert np.allclose(constraint.project_density(projected), projected)

    vp = valley_polarized_seed(active, "K")
    assert np.allclose(constraint.project_density(vp), vp)

    pinned = ValleyU1Constraint(active, pinned_valley="K")
    P_pinned, _evals, _direct, _indirect = pinned.update_density(blocks, 1)
    assert np.allclose(P_pinned[:, n:, n:], 0.0)
    assert np.allclose(np.trace(P_pinned, axis1=-2, axis2=-1), 1.0)


def test_tprime_constraint_is_involutive_and_final_aufbau_is_idempotent():
    bundle = _small_bundle()
    active = bundle.active
    constraint = TPrimeConstraint(active)
    partner = mesh_inversion_map(active.grid)

    assert np.all(partner[partner] == np.arange(active.n_k))

    rng = np.random.default_rng(3)
    raw = rng.normal(size=(active.n_k, active.dim, active.dim)) + 1j * rng.normal(
        size=(active.n_k, active.dim, active.dim)
    )
    blocks = hermitize(raw)
    assert np.allclose(constraint.transform(constraint.transform(blocks)), blocks)

    projected = constraint.project_operator(blocks)
    assert constraint.symmetry_error(projected) < 1e-12

    P, _evals, _direct, _indirect = constraint.update_density(projected, 1)
    assert np.linalg.norm(P @ P - P) < 1e-12
    assert constraint.symmetry_error(P) < 1e-12


def test_tprime_global_aufbau_preserves_requested_integer_trace():
    bundle = _small_bundle()
    active = bundle.active
    constraint = TPrimeConstraint(active)
    rng = np.random.default_rng(31)
    raw = rng.normal(size=(active.n_k, active.dim, active.dim)) + 1j * rng.normal(
        size=(active.n_k, active.dim, active.dim)
    )
    projected = constraint.project_operator(hermitize(raw))

    for target in (1, 5, active.n_k):
        density, _evals, _direct, _indirect = constraint.update_density_global(
            projected,
            target,
        )
        assert np.isclose(np.trace(density, axis1=-2, axis2=-1).sum().real, target)
        assert np.linalg.norm(density @ density - density) < 1e-12
        assert constraint.symmetry_error(density) < 1e-12


def test_hf_solver_reports_idempotent_final_projectors_for_reference_states():
    bundle = _small_bundle()
    params = ContinuumHFParams(max_iter=6, min_iter=1, mixing=0.6, tolerance=1e-9)

    vp_plus = solve_reference_hf(
        bundle,
        "vp_plus",
        params,
        constraint=ValleyU1Constraint(bundle.active),
    )
    vp_minus = solve_reference_hf(
        bundle,
        "vp_minus",
        params,
        constraint=ValleyU1Constraint(bundle.active),
    )
    ivc = solve_reference_hf(
        bundle,
        "ivc",
        params,
        constraint=TPrimeConstraint(bundle.active),
    )

    for result in (vp_plus, vp_minus, ivc):
        assert result.diagnostics.density_kind == "final_idempotent"
        assert result.diagnostics.idempotency_error_fro < 1e-8
        assert result.diagnostics.idempotency_error_max < 1e-8
        assert result.diagnostics.trace_error < 1e-8
        assert np.isfinite(result.diagnostics.aufbau_residual_norm)

    vp_plus_k = np.real(np.trace(vp_plus.P[:, 0:1, 0:1], axis1=-2, axis2=-1).sum())
    vp_plus_kp = np.real(np.trace(vp_plus.P[:, 1:2, 1:2], axis1=-2, axis2=-1).sum())
    vp_minus_k = np.real(np.trace(vp_minus.P[:, 0:1, 0:1], axis1=-2, axis2=-1).sum())
    vp_minus_kp = np.real(np.trace(vp_minus.P[:, 1:2, 1:2], axis1=-2, axis2=-1).sum())
    assert vp_plus_k > vp_plus_kp
    assert vp_minus_kp > vp_minus_k
    assert ivc.diagnostics.constraint_name == "tprime"


def test_hf_solver_iteration_callback_receives_copies_and_snapshot_flags():
    bundle = _small_bundle()
    active = bundle.active
    params = ContinuumHFParams(
        max_iter=4,
        min_iter=4,
        mixing_method="linear",
        mixing=0.5,
        tolerance=1e-30,
        store_projector_snapshots=True,
        first_iteration_snapshot=True,
        snapshot_interval=2,
    )
    rows = []

    def callback(iteration, P_iter, energy, diagnostics, is_snapshot):
        rows.append(
            {
                "iteration": int(iteration),
                "energy": float(energy),
                "diagnostics_iteration": int(diagnostics.iteration),
                "is_snapshot": bool(is_snapshot),
                "trace_before_mutation": float(np.real(np.trace(P_iter, axis1=-2, axis2=-1).sum())),
            }
        )
        P_iter[:] = np.nan

    result = solve_hf(
        bundle.backend,
        valley_polarized_seed(active, "K"),
        params,
        constraint=ValleyU1Constraint(active, pinned_valley="K"),
        on_iteration=callback,
    )

    assert [row["iteration"] for row in rows] == [diag.iteration for diag in result.history]
    assert [row["diagnostics_iteration"] for row in rows] == [1, 2, 3, 4]
    assert [row["is_snapshot"] for row in rows] == [True, True, False, True]
    assert [snapshot.iteration for snapshot in result.snapshots] == [1, 2, 4]
    assert all(np.isfinite(row["energy"]) for row in rows)
    assert all(diag.lambda_value == pytest.approx(0.5) for diag in result.history)
    assert np.all(np.isfinite(result.P))


def test_projector_maps_include_valley_matrix_entries_and_preserve_existing_maps():
    bundle = _small_bundle()
    active = bundle.active
    vp = valley_polarized_seed(active, "K")
    vp_maps = projector_maps(vp, active)

    assert np.allclose(vp_maps["P_KK"], vp_maps["K"])
    assert np.allclose(vp_maps["P_KprimeKprime"], vp_maps["Kprime"])
    assert np.allclose(vp_maps["P_KK"], 1.0)
    assert np.allclose(vp_maps["P_KprimeKprime"], 0.0)
    assert np.allclose(vp_maps["P_KKprime_abs"], 0.0)
    assert np.allclose(vp_maps["P_KprimeK_abs"], 0.0)

    ivc = ivc_seed(active, n_occ_per_k=1)
    ivc_maps = projector_maps(ivc, active)
    assert np.allclose(ivc_maps["P_KKprime_abs"], ivc_maps["IVC_abs"])
    assert np.allclose(ivc_maps["P_KprimeK_abs"], ivc_maps["IVC_abs"])
    assert np.max(ivc_maps["P_KKprime_abs"]) > 0.0


def test_active_basis_frames_are_orthonormal_direct_sum_frames():
    bundle = _small_bundle()
    active = bundle.active
    frames = active_basis_frames(active)

    assert frames.shape == (active.n_k, 4, 2)
    overlap = frames.conj().swapaxes(-1, -2) @ frames
    assert np.allclose(overlap, np.eye(active.dim), atol=1e-12)


def _toy_refs_with_scalar_terms(n_blocks: int = 4) -> SymmetricHFReferences:
    h0 = np.zeros((n_blocks, 2, 2), dtype=complex)
    scalar = np.broadcast_to(3.0 * np.eye(2, dtype=complex), h0.shape).copy()
    H_plus = scalar - np.broadcast_to(np.diag([1.0, -1.0]).astype(complex), h0.shape)
    H_minus = scalar + np.broadcast_to(np.diag([1.0, -1.0]).astype(complex), h0.shape)
    H_ivc = scalar - np.broadcast_to(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex), h0.shape)

    def result(H):
        return SimpleNamespace(H_hf=H, P=np.zeros_like(H), diagnostics=None)

    return SymmetricHFReferences(
        vp_plus=result(H_plus),
        vp_minus=result(H_minus),
        ivc=result(H_ivc),
        n_occ_per_k=1,
    )


def test_symmetric_convex_path_preserves_scalar_terms_and_rotates_ivc():
    refs = _toy_refs_with_scalar_terms()
    weights = np.asarray([convex_weights(x) for x in np.linspace(0.0, np.pi, 7)])

    assert np.all(weights >= -1e-15)
    assert np.allclose(np.sum(weights, axis=1), 1.0)
    assert convex_weights(0.0) == pytest.approx((1.0, 0.0, 0.0))
    assert convex_weights(0.5 * np.pi) == pytest.approx((0.0, 0.0, 1.0))
    assert convex_weights(np.pi) == pytest.approx((0.0, 1.0, 0.0))

    H0 = symmetric_convex_hamiltonian(refs, 0.0)
    Hpi = symmetric_convex_hamiltonian(refs, np.pi)
    Hmid = symmetric_convex_hamiltonian(refs, 0.5 * np.pi)
    assert np.allclose(np.trace(H0, axis1=-2, axis2=-1), 6.0)
    assert np.allclose(np.trace(Hpi, axis1=-2, axis2=-1), 6.0)
    assert np.allclose(np.trace(Hmid, axis1=-2, axis2=-1), 6.0)
    assert np.allclose(H0 - 3.0 * np.eye(2), -(Hpi - 3.0 * np.eye(2)))

    phi = np.pi / 4.0
    P_phi, H_phi, diag = symmetric_convex_projector(refs, 0.5 * np.pi, phi)
    P_zero, _H_zero, _diag_zero = symmetric_convex_projector(refs, 0.5 * np.pi, 0.0)
    assert np.allclose(P_phi, rotate_valley_u1(P_zero, phi), atol=1e-12)
    assert np.allclose(H_phi, rotate_valley_u1(Hmid, phi), atol=1e-12)
    assert diag.projector_idempotency_error_fro < 1e-12


def test_tiny_end_to_end_continuum_response_smoke():
    bundle = _small_bundle()
    params = ContinuumHFParams(max_iter=5, min_iter=1, mixing=0.7)
    refs = build_symmetric_hf_references(bundle, params)
    theta = np.linspace(0.0, np.pi, 5)
    projectors, diagnostics = symmetric_convex_path(refs, theta)
    response_projectors = projectors.reshape(5, 3, 3, 2, 2)
    basis = active_basis_frames(bundle.active).reshape(3, 3, -1, bundle.active.dim)
    response = k_theta_from_projectors_with_basis(response_projectors, theta, basis)

    assert projectors.shape == (5, bundle.active.n_k, 2, 2)
    assert len(diagnostics) == 5
    assert np.all(np.isfinite(response.K))
    assert np.isfinite(compute_cG(response.theta, response.K))
    assert refs.vp_plus.diagnostics.idempotency_error_fro < 1e-8
    assert refs.vp_minus.diagnostics.idempotency_error_fro < 1e-8
    assert refs.ivc.diagnostics.idempotency_error_fro < 1e-8

    ref_diag = reference_diagnostics(refs)
    assert ref_diag["ivc"].intervalley_norm >= 0.0
