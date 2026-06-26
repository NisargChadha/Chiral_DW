import numpy as np
import pytest

from chiral_dw.config import (
    ContinuumFiniteQParams,
    ContinuumGridParams,
    ContinuumHFParams,
    ContinuumInteractionParams,
)
from chiral_dw.continuum import (
    TPrimeConstraint,
    ValleyU1Constraint,
    active_basis_frames,
    build_continuum_bundle,
    build_symmetric_hf_references,
    chern_number_table,
    compute_taige_path_spectrum,
    finite_q_shift_metadata,
    random_projector_like_seed,
    symmetric_convex_path,
    taige_interaction_params,
    taige_ivc_minus_half_shift_coord,
    taige_ivc_minus_q_coord,
    taige_model_params,
)
from chiral_dw.continuum.seeds import build_seed, mix_projector_seeds
from chiral_dw.continuum.taige import TaigeContinuumModel, coulomb_potential_mev_nm2
from chiral_dw.response import compute_cG, k_theta_from_projectors_with_basis


def _tiny_taige_bundle(interaction=None):
    model = taige_model_params(theta_deg=3.5, u_D=0.0, plane_wave_shell=1, n_bands=1)
    return build_continuum_bundle(
        model=model,
        grid=ContinuumGridParams(n_k=2),
        interaction=interaction
        or ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.05,
            q_shell=0,
            local_field_cutoff=0,
        ),
    )


def test_taige_ivc_minus_finite_q_helpers_and_metadata():
    assert taige_ivc_minus_q_coord(18) == (6, 6)
    assert taige_ivc_minus_half_shift_coord(18) == (3, 12)

    with pytest.raises(ValueError, match="divisible by 6"):
        taige_ivc_minus_q_coord(15)
    with pytest.raises(ValueError, match="divisible by 6"):
        taige_ivc_minus_half_shift_coord(15)

    finite_q = ContinuumFiniteQParams(
        enabled=True,
        q_coord=taige_ivc_minus_q_coord(18),
        half_shift_coord=taige_ivc_minus_half_shift_coord(18),
    )
    grid = build_continuum_bundle(
        model=taige_model_params(theta_deg=3.5, u_D=0.0, plane_wave_shell=1, n_bands=1),
        grid=ContinuumGridParams(n_k=18),
        finite_q=finite_q,
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.0,
            q_shell=0,
        ),
    ).grid
    half = grid.assert_half_q_on_mesh(finite_q.q_coord, finite_q.half_shift_coord)
    assert (2 * half[0] - finite_q.q_coord[0]) % grid.n1 == 0
    assert (2 * half[1] - finite_q.q_coord[1]) % grid.n2 == 0

    metadata = finite_q_shift_metadata(finite_q, grid)
    assert metadata["enabled"] is True
    assert metadata["q_coord"] == [6, 6]
    assert metadata["half_shift_coord"] == [3, 12]
    assert np.allclose(metadata["half_shift_centered_fractional"], [1 / 6, -1 / 3])


def test_taige_continuum_hamiltonian_and_active_space_are_well_formed():
    model = taige_model_params(theta_deg=3.5, u_D=0.0, plane_wave_shell=1, n_bands=1)
    continuum = TaigeContinuumModel(model)
    H = continuum.hamiltonian(np.array([0.0, 0.0]), "K")

    assert np.allclose(H, H.conj().T)
    assert continuum.n_plane_waves == 7

    bundle = _tiny_taige_bundle()
    active = bundle.active
    assert active.h0.shape == (4, 2, 2)
    assert active.band_vectors.shape == (4, 2, 14, 1)
    frames = active_basis_frames(active)
    assert frames.shape == (4, 28, 2)
    assert np.allclose(frames.conj().swapaxes(-1, -2) @ frames, np.eye(active.dim), atol=1e-10)
    assert bundle.bands is not None
    assert bundle.geometry is not None


def test_taige_finite_q_active_space_uses_symmetric_physical_sources():
    model = taige_model_params(theta_deg=3.5, u_D=0.0, plane_wave_shell=1, n_bands=1)
    q0_bundle = build_continuum_bundle(
        model=model,
        grid=ContinuumGridParams(n_k=6),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.05,
            q_shell=0,
        ),
    )
    q0_explicit_bundle = build_continuum_bundle(
        model=model,
        grid=ContinuumGridParams(n_k=6),
        finite_q=ContinuumFiniteQParams(enabled=False, q_coord=(0, 0)),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.05,
            q_shell=0,
        ),
    )
    assert np.allclose(q0_bundle.active.h0, q0_explicit_bundle.active.h0)
    assert np.allclose(q0_bundle.vertices.lambda_blocks, q0_explicit_bundle.vertices.lambda_blocks)
    q0_sources = np.repeat(np.arange(q0_bundle.active.n_k)[:, None], 2, axis=1)
    assert np.array_equal(q0_bundle.active.source_index, q0_sources)
    assert np.count_nonzero(q0_bundle.active.source_shift) == 0

    finite_q = ContinuumFiniteQParams(
        enabled=True,
        q_coord=taige_ivc_minus_q_coord(6),
        half_shift_coord=taige_ivc_minus_half_shift_coord(6),
    )
    finite_bundle = build_continuum_bundle(
        model=model,
        grid=ContinuumGridParams(n_k=6),
        finite_q=finite_q,
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.05,
            q_shell=0,
        ),
    )
    active = finite_bundle.active
    grid = active.grid

    assert active.finite_q_enabled is True
    assert active.q_coord == finite_q.q_coord
    assert active.half_shift_coord == finite_q.half_shift_coord
    assert active.h0.shape == q0_bundle.active.h0.shape
    assert np.any(active.source_index != np.arange(grid.size)[:, None])
    assert np.any(active.source_shift != 0)

    partner = TPrimeConstraint(active).partner_index
    for ik in range(grid.size):
        k_source = int(active.source_index[ik, 0])
        k_source_coord = grid.coord_of(k_source)
        inverted_source = grid.index_of((-k_source_coord[0], -k_source_coord[1]))
        assert int(active.source_index[int(partner[ik]), 1]) == inverted_source


def test_taige_chern_table_returns_finite_values_on_tiny_grid():
    bundle = _tiny_taige_bundle()
    rows = chern_number_table(bundle.bands, band_indices=(0,))

    assert len(rows) == 4
    assert all(np.isfinite(row.chern) for row in rows)


def test_taige_path_spectrum_uses_tprime_kprime_convention():
    model = taige_model_params(theta_deg=3.5, u_D=0.0, plane_wave_shell=3, n_bands=2)
    data = compute_taige_path_spectrum(model, n_per_segment=6)
    hole = data["hole_energies"]

    assert np.max(np.abs(hole[:, 0, :] - hole[:, 1, :])) < 1e-2
    assert int(np.argmax(hole[:, 0, 0])) == data["ticks"][0]


def test_taige_density_vertices_have_q0_identity_and_smeared_dual_gate_weights():
    interaction = taige_interaction_params(include_q0=True, q_shell=1, local_field_cutoff=0)
    bundle = _tiny_taige_bundle(interaction)
    vertices = bundle.vertices
    iq0 = vertices.q_shifts.index((0, 0))

    assert vertices.lambda_blocks.shape[:3] == (9, 1, 4)
    assert np.allclose(vertices.lambda_blocks[iq0, 0], np.eye(bundle.active.dim), atol=1e-10)
    assert vertices.v_over_a.shape == (9, 1)
    assert vertices.v_over_a[iq0, 0] > 0.0

    unsmeared = interaction.model_copy(update={"smear_length_nm": 0.0})
    assert coulomb_potential_mev_nm2(5.0, interaction) < coulomb_potential_mev_nm2(5.0, unsmeared)


def test_taige_interaction_params_accept_screening_overrides():
    interaction = taige_interaction_params(
        include_q0=False,
        q_mesh="full",
        q_shell=0,
        local_field_cutoff=4,
        epsilon=12.5,
        gate_distance_nm=18.0,
        smear_length_nm=0.2,
        interaction_strength_scale=0.7,
        hartree_scale=0.9,
        exchange_scale=0.8,
    )

    assert interaction.coulomb_kind == "dual_gate"
    assert interaction.include_q0 is False
    assert interaction.q_mesh == "full"
    assert interaction.q_shell == 0
    assert interaction.local_field_cutoff == 4
    assert interaction.epsilon == 12.5
    assert interaction.gate_distance_nm == 18.0
    assert interaction.smear_length_nm == 0.2
    assert interaction.v0 == 0.7
    assert interaction.hartree_scale == 0.9
    assert interaction.exchange_scale == 0.8


def test_taige_finite_q_density_vertices_use_shifted_physical_sources():
    model = taige_model_params(theta_deg=3.5, u_D=0.0, plane_wave_shell=1, n_bands=1)
    interaction = ContinuumInteractionParams(
        coulomb_kind="dimensionless_screened",
        v0=0.05,
        q_shell=1,
        local_field_cutoff=0,
    )
    q0_bundle = build_continuum_bundle(
        model=model,
        grid=ContinuumGridParams(n_k=6),
        interaction=interaction,
    )
    finite_bundle = build_continuum_bundle(
        model=model,
        grid=ContinuumGridParams(n_k=6),
        finite_q=ContinuumFiniteQParams(
            enabled=True,
            q_coord=taige_ivc_minus_q_coord(6),
            half_shift_coord=taige_ivc_minus_half_shift_coord(6),
        ),
        interaction=interaction,
    )
    active = finite_bundle.active
    vertices = finite_bundle.vertices
    iq0 = vertices.q_shifts.index((0, 0))
    iq = vertices.q_shifts.index((1, 0))

    assert np.allclose(vertices.lambda_blocks[iq0, 0], np.eye(active.dim), atol=1e-10)
    for ik in range(active.n_k):
        physical = int(active.source_index[ik, 0])
        if physical == ik:
            continue
        finite_block = vertices.lambda_blocks[iq, 0, ik, 0:1, 0:1]
        shifted_block = q0_bundle.vertices.lambda_blocks[iq, 0, physical, 0:1, 0:1]
        unshifted_block = q0_bundle.vertices.lambda_blocks[iq, 0, ik, 0:1, 0:1]
        assert np.allclose(finite_block, shifted_block)
        assert not np.allclose(finite_block, unshifted_block)
        break
    else:
        raise AssertionError("finite-Q source map did not shift any K-valley source")


def test_projector_like_seed_mix_preserves_trace_and_hf_snapshots_are_recorded():
    bundle = _tiny_taige_bundle()
    active = bundle.active
    ordered = build_seed("vp_plus", active)
    vp_constraint = ValleyU1Constraint(active)
    noise = vp_constraint.project_density(random_projector_like_seed(ordered, seed=4))
    mixed = mix_projector_seeds(ordered, noise, ordered_weight=0.8, random_weight=0.2)
    mixed = vp_constraint.project_density(mixed)

    assert np.allclose(mixed, mixed.conj().swapaxes(-1, -2))
    assert np.allclose(np.trace(mixed, axis1=-2, axis2=-1), 1.0)
    assert np.allclose(mixed[:, : active.n_active, active.n_active :], 0.0)
    assert np.real(np.trace(noise[:, active.n_active :, active.n_active :], axis1=-2, axis2=-1).sum()) > 0.0

    ivc_constraint = TPrimeConstraint(active)
    ivc_noise = ivc_constraint.project_density(random_projector_like_seed(build_seed("ivc", active), seed=5))
    assert ivc_constraint.symmetry_error(ivc_noise) < 1e-12
    assert np.allclose(np.trace(ivc_noise, axis1=-2, axis2=-1), 1.0)

    params = ContinuumHFParams(
        max_iter=3,
        min_iter=1,
        mixing_method="oda",
        mixing=0.7,
        seed_ordered_weight=0.8,
        seed_random_weight=0.2,
        store_projector_snapshots=True,
        snapshot_interval=1,
    )
    result = build_symmetric_hf_references(bundle, params).vp_plus

    assert result.diagnostics.idempotency_error_fro < 1e-8
    assert result.diagnostics.trace_error < 1e-8
    assert len(result.snapshots) >= 1
    assert result.snapshots[0].P.shape == active.h0.shape


def test_taige_finite_q_ivc_seed_and_tprime_hf_smoke():
    q0_bundle = _tiny_taige_bundle()
    with pytest.raises(ValueError, match="finite_q"):
        build_seed("finite_q_ivc", q0_bundle.active)

    finite_bundle = build_continuum_bundle(
        model=taige_model_params(theta_deg=3.5, u_D=0.0, plane_wave_shell=1, n_bands=1),
        grid=ContinuumGridParams(n_k=6),
        finite_q=ContinuumFiniteQParams(
            enabled=True,
            q_coord=taige_ivc_minus_q_coord(6),
            half_shift_coord=taige_ivc_minus_half_shift_coord(6),
        ),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.02,
            q_shell=0,
            local_field_cutoff=0,
        ),
    )
    P0 = build_seed("finite_q_ivc", finite_bundle.active)
    maps = active_basis_frames(finite_bundle.active)
    assert maps.shape[0] == finite_bundle.active.n_k
    assert np.allclose(np.trace(P0, axis1=-2, axis2=-1), 1.0)
    assert np.max(np.abs(P0[:, :1, 1:])) > 0.0

    result = build_seed("finite_q_ivc", finite_bundle.active)
    assert TPrimeConstraint(finite_bundle.active).symmetry_error(result) < 1e-12

    params = ContinuumHFParams(max_iter=2, min_iter=0, mixing=0.7)
    hf = build_symmetric_hf_references(finite_bundle, params).ivc
    assert np.isfinite(hf.energy)
    assert hf.diagnostics.idempotency_error_fro < 1e-8
    assert hf.diagnostics.trace_error < 1e-8


def test_tiny_taige_symmetric_response_smoke():
    bundle = _tiny_taige_bundle()
    params = ContinuumHFParams(max_iter=3, min_iter=1, mixing=0.7)
    refs = build_symmetric_hf_references(bundle, params)
    theta = np.linspace(0.0, np.pi, 5)
    projectors, diagnostics = symmetric_convex_path(refs, theta)
    response_projectors = projectors.reshape(5, 2, 2, 2, 2)
    basis = active_basis_frames(bundle.active).reshape(2, 2, -1, bundle.active.dim)
    response = k_theta_from_projectors_with_basis(response_projectors, theta, basis)

    assert refs.vp_plus.constraint_name == ValleyU1Constraint(bundle.active).name
    assert refs.ivc.constraint_name == TPrimeConstraint(bundle.active).name
    assert len(diagnostics) == 5
    assert np.all(np.isfinite(response.K))
    assert np.isfinite(compute_cG(response.theta, response.K))
