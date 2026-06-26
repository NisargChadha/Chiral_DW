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
    convex_weights,
    mesh_inversion_map,
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
from chiral_dw.continuum.seeds import ivc_seed, valley_polarized_seed
from chiral_dw.response import compute_cG, k_theta_from_projectors_with_basis


def _small_bundle():
    return build_continuum_bundle(
        model=ContinuumModelParams(displacement_mev=0.0),
        grid=ContinuumGridParams(n_k=3),
        interaction=ContinuumInteractionParams(v0=0.2, q_shell=0, gate_distance=1.0),
    )


def _slow_hartree(backend: ContinuumHFBackend, Q: np.ndarray) -> np.ndarray:
    out = np.zeros_like(Q, dtype=complex)
    for iq, ig, v in backend.hartree_channels:
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
            lam = backend.lambda_blocks[iq, ig]
            for ik in range(backend.n_blocks):
                jk = int(targets[ik])
                out[ik] -= 0.5 * v * lam[ik] @ Q[jk] @ lam[ik].conj().T
                out[jk] -= 0.5 * v * lam[ik].conj().T @ Q[ik] @ lam[ik]
    return hermitize(out)


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
    raw = rng.normal(size=h0.shape) + 1j * rng.normal(size=h0.shape)
    Q = hermitize(raw)

    assert backend.tVE.shape == (backend.n_blocks * backend.dim**2, backend.n_blocks * backend.dim**2)
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


def test_finite_q_taige_backend_matches_slow_hartree_fock_reference():
    bundle = build_continuum_bundle(
        model=taige_model_params(theta_deg=3.5, u_D=0.0, plane_wave_shell=1, n_bands=1),
        grid=ContinuumGridParams(n_k=6),
        finite_q=ContinuumFiniteQParams(
            enabled=True,
            q_coord=taige_ivc_minus_q_coord(6),
            half_shift_coord=taige_ivc_minus_half_shift_coord(6),
        ),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.03,
            q_shell=1,
            local_field_cutoff=0,
        ),
    )
    rng = np.random.default_rng(13)
    Q = hermitize(
        rng.normal(size=bundle.active.h0.shape) + 1j * rng.normal(size=bundle.active.h0.shape)
    )

    assert np.allclose(bundle.backend.hartree_hamiltonian(Q), _slow_hartree(bundle.backend, Q))
    assert np.allclose(bundle.backend.fock_hamiltonian(Q), _slow_fock(bundle.backend, Q))


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
