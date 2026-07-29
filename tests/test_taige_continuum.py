import numpy as np
import pytest
import weakref

from chiral_dw.config import (
    ContinuumFiniteQParams,
    ContinuumGridParams,
    ContinuumHFParams,
    ContinuumInteractionParams,
    ContinuumWorkflowParams,
    ResponseParams,
)
from chiral_dw.continuum import (
    ContinuumSymmetricHFBranch,
    ContinuumHFDiagnostics,
    ContinuumHFResult,
    MomentumGrid,
    SymmetricHFReferences,
    TPrimeConstraint,
    ValleyU1Constraint,
    active_basis_frames,
    build_continuum_bundle,
    build_taige_q_sector_bundles,
    build_symmetric_hf_references,
    chern_number_table,
    compute_taige_path_spectrum,
    density_vertices_dense_lambdas,
    finite_q_shift_metadata,
    evaluate_hf_high_symmetry_path,
    hf_band_chern_table,
    hf_hamiltonian_at_k,
    order_diagnostics,
    random_projector_like_seed,
    run_taige_branch_selected_symmetric_hf_workflow,
    select_ivc_branch_by_energy,
    symmetric_convex_path,
    taige_active_fine_frame,
    taige_interaction_params,
    taige_ivc_minus_half_shift_coord,
    taige_ivc_minus_q_coord,
    taige_ivc_minus_shift_choice,
    taige_ivc_plus_half_shift_coord,
    taige_ivc_plus_q_coord,
    taige_ivc_plus_shift_choice,
    taige_ivc_shift_choices,
    taige_model_params,
)
import chiral_dw.continuum.taige as taige_mod
import chiral_dw.continuum.hf as hf_mod
import chiral_dw.continuum.workflow as workflow_mod
from chiral_dw.continuum.seeds import build_seed, mix_projector_seeds
from chiral_dw.continuum.taige import (
    TaigeContinuumModel,
    build_taige_active_space,
    build_taige_density_vertices,
    compute_taige_bandstructure,
    coulomb_potential_mev_nm2,
    roll_taige_density_vertices,
)
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


def _dummy_hf_result(H: np.ndarray, seed: str, *, energy: float = 0.0) -> ContinuumHFResult:
    P = np.zeros_like(H, dtype=complex)
    P[:, 0, 0] = 1.0
    diagnostics = ContinuumHFDiagnostics(
        energy=float(energy),
        delta_energy=0.0,
        delta_P=0.0,
        idempotency_error_fro=0.0,
        idempotency_error_max=0.0,
        constraint_error=0.0,
        aufbau_residual_norm=0.0,
        commutator_norm=0.0,
        trace_error=0.0,
        direct_gap_min=1.0,
        indirect_gap=1.0,
        iteration=0,
        constraint_name=None,
        density_kind="final_idempotent",
    )
    return ContinuumHFResult(
        P=P,
        H_hf=H,
        energy=float(energy),
        converged=True,
        n_iter=0,
        diagnostics=diagnostics,
        seed=seed,
        constraint_name=None,
    )


def _dummy_refs_for_bundle(
    bundle,
    *,
    ivc_energy: float,
    vp_plus_energy: float = 0.0,
    vp_minus_energy: float = 0.1,
) -> SymmetricHFReferences:
    H = np.asarray(bundle.active.h0, dtype=complex)
    return SymmetricHFReferences(
        vp_plus=_dummy_hf_result(H, "vp_plus", energy=vp_plus_energy),
        vp_minus=_dummy_hf_result(H, "vp_minus", energy=vp_minus_energy),
        ivc=_dummy_hf_result(H, "finite_q_ivc" if bundle.active.finite_q_enabled else "ivc", energy=ivc_energy),
        n_occ_per_k=1,
    )


def _branch_for_selection_test(bundle, name: str, *, ivc_energy: float) -> ContinuumSymmetricHFBranch:
    return ContinuumSymmetricHFBranch(
        name=name,
        bundle=bundle,
        references=_dummy_refs_for_bundle(bundle, ivc_energy=ivc_energy),
        metadata=finite_q_shift_metadata(bundle.finite_q, bundle.grid),
    )


def test_taige_ivc_minus_finite_q_helpers_and_metadata():
    assert taige_ivc_minus_q_coord(18) == (6, 6)
    assert taige_ivc_minus_half_shift_coord(18) == (3, 12)

    assert taige_ivc_minus_q_coord(15) == (5, 5)
    assert taige_ivc_minus_half_shift_coord(15) == (10, 10)
    with pytest.raises(ValueError, match="divisible by 3"):
        taige_ivc_minus_q_coord(13)
    with pytest.raises(ValueError, match="divisible by 3"):
        taige_ivc_minus_half_shift_coord(13)

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


def test_taige_ivc_policy_keeps_both_opposite_q_sectors():
    choices = taige_ivc_shift_choices(18)

    assert choices["minus"] == taige_ivc_minus_shift_choice(18)
    assert choices["plus"] == taige_ivc_plus_shift_choice(18)
    assert taige_ivc_minus_q_coord(18) == (6, 6)
    assert taige_ivc_plus_q_coord(18) == (12, 12)
    assert taige_ivc_minus_half_shift_coord(18) == (3, 12)
    assert taige_ivc_plus_half_shift_coord(18) == (15, 6)
    assert tuple(
        (choices["plus"].q_coord[axis] + choices["minus"].q_coord[axis]) % 18
        for axis in range(2)
    ) == (0, 0)
    assert tuple(
        (
            choices["plus"].half_shift_coord[axis]
            + choices["minus"].half_shift_coord[axis]
        )
        % 18
        for axis in range(2)
    ) == (0, 0)
    assert choices["minus"].sector == "minus"
    assert choices["plus"].sector == "plus"


def test_taige_ivc_minus_nearest_half_shift_choices_for_finite_size_meshes():
    expected = {
        12: ((2, 8), (4, 4), True),
        13: ((2, 9), (4, 5), False),
        14: ((2, 9), (4, 4), False),
        15: ((10, 10), (5, 5), True),
        16: ((2, 11), (4, 6), False),
        17: ((3, 11), (6, 5), False),
        18: ((3, 12), (6, 6), True),
        19: ((3, 13), (6, 7), False),
        20: ((3, 13), (6, 6), False),
    }
    for n_k, (half_shift, q_coord, exact) in expected.items():
        choice = taige_ivc_minus_shift_choice(n_k, policy="nearest_half")
        assert choice.half_shift_coord == half_shift
        assert choice.q_coord == q_coord
        assert choice.exact is exact
        assert choice.policy == "nearest_half"

    exact_choice = taige_ivc_minus_shift_choice(18, policy="exact")
    assert exact_choice.half_shift_coord == (3, 12)
    assert exact_choice.q_coord == (6, 6)
    assert exact_choice.exact is True

    exact_odd_choice = taige_ivc_minus_shift_choice(21, policy="exact")
    assert exact_odd_choice.q_coord == (7, 7)
    assert exact_odd_choice.half_shift_coord == (14, 14)
    assert exact_odd_choice.exact is True
    assert exact_odd_choice.half_shift_error_fractional_norm > 0.0
    exact_odd_plus = taige_ivc_plus_shift_choice(21, policy="exact")
    assert exact_odd_plus.q_coord == (14, 14)
    assert exact_odd_plus.half_shift_coord == (7, 7)
    assert exact_odd_plus.exact is True
    assert tuple(
        (exact_odd_choice.q_coord[axis] + exact_odd_plus.q_coord[axis]) % 21
        for axis in range(2)
    ) == (0, 0)
    assert tuple(
        (
            exact_odd_choice.half_shift_coord[axis]
            + exact_odd_plus.half_shift_coord[axis]
        )
        % 21
        for axis in range(2)
    ) == (0, 0)

    with pytest.raises(ValueError, match="divisible by 3"):
        taige_ivc_minus_shift_choice(13, policy="exact")


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


def test_taige_order_diagnostics_distinguish_vp_and_ivc_seeds():
    bundle = _tiny_taige_bundle()
    active = bundle.active
    vp = order_diagnostics(build_seed("vp_plus", active), active, n_occ_per_k=1)
    ivc = order_diagnostics(build_seed("ivc", active), active, n_occ_per_k=1)

    assert np.isclose(vp.Nz_block, 1.0)
    assert np.isclose(vp.Nz_abs, 1.0)
    assert np.isclose(vp.C_IVC_block, 0.0)
    assert np.isclose(vp.IVC_amplitude_block, 0.0)
    assert np.isclose(vp.C_IVC_scalar, 0.0)
    assert np.isclose(vp.IVC_amplitude_scalar, 0.0)
    assert np.isclose(ivc.Nz_block, 0.0, atol=1e-12)
    assert np.isclose(ivc.Nz_abs, 0.0, atol=1e-12)
    assert np.isclose(ivc.C_IVC_block, 0.25)
    assert np.isclose(ivc.IVC_amplitude_block, 0.5)
    assert np.isclose(ivc.C_IVC_scalar, 0.25)
    assert np.isclose(ivc.IVC_amplitude_scalar, 1.0)


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
    assert np.allclose(
        density_vertices_dense_lambdas(q0_bundle.vertices),
        density_vertices_dense_lambdas(q0_explicit_bundle.vertices),
    )
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


def test_taige_ivc_branch_selector_prefers_lower_energy_and_q0_ties():
    q0_bundle = build_continuum_bundle(
        model=taige_model_params(theta_deg=3.5, u_D=0.0, plane_wave_shell=0, n_bands=1),
        grid=ContinuumGridParams(n_k=6),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.0,
            q_shell=0,
            local_field_cutoff=0,
        ),
    )
    finite_bundle = build_continuum_bundle(
        model=q0_bundle.params,
        grid=ContinuumGridParams(n_k=6),
        finite_q=ContinuumFiniteQParams(
            enabled=True,
            q_coord=taige_ivc_minus_q_coord(6),
            half_shift_coord=taige_ivc_minus_half_shift_coord(6),
        ),
        interaction=q0_bundle.interaction,
    )
    q0_branch = _branch_for_selection_test(q0_bundle, "q0", ivc_energy=-1.0)
    finite_branch = _branch_for_selection_test(finite_bundle, "finite_q", ivc_energy=-1.5)

    selected, metadata = select_ivc_branch_by_energy(
        q0_branch=q0_branch,
        finite_q_branch=finite_branch,
        ivc_branch_policy="lower_energy",
        tie_atol=1e-9,
    )
    assert selected == "finite_q"
    assert metadata["finite_q_minus_q0_ivc_energy_per_cell"] < 0.0

    tie_branch = _branch_for_selection_test(finite_bundle, "finite_q", ivc_energy=-1.0)
    selected, metadata = select_ivc_branch_by_energy(
        q0_branch=q0_branch,
        finite_q_branch=tie_branch,
        ivc_branch_policy="lower_energy",
        tie_atol=1e-9,
    )
    assert selected == "q0"
    assert metadata["selected_ivc_branch"] == "q0"


def test_taige_branch_selected_workflow_uses_whole_finite_q_frame(monkeypatch):
    def fake_references(bundle, params):
        return _dummy_refs_for_bundle(
            bundle,
            ivc_energy=-2.0 if bundle.active.finite_q_enabled else -1.0,
            vp_plus_energy=-0.4 if bundle.active.finite_q_enabled else -0.2,
            vp_minus_energy=-0.3 if bundle.active.finite_q_enabled else -0.1,
        )

    monkeypatch.setattr(workflow_mod, "build_symmetric_hf_references", fake_references)
    params = ContinuumWorkflowParams(
        model=taige_model_params(theta_deg=3.5, u_D=0.0, plane_wave_shell=0, n_bands=1),
        grid=ContinuumGridParams(n_k=6),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.0,
            q_shell=0,
            local_field_cutoff=0,
        ),
        hf=ContinuumHFParams(max_iter=1, min_iter=0),
        response=ResponseParams(n_theta=5, theta_min=1e-4, theta_max=np.pi - 1e-4),
    )

    result = run_taige_branch_selected_symmetric_hf_workflow(
        params,
        finite_q_enabled=True,
        ivc_branch_policy="lower_energy",
        write_outputs=False,
    )

    assert result.selected_ivc_branch == "finite_q"
    assert result.bundle.active.finite_q_enabled is True
    assert result.references is result.finite_q_branch.references
    assert result.q0_branch.bundle.active.finite_q_enabled is False
    assert result.projectors.shape[-1] == result.bundle.active.dim
    assert result.branch_selection["finite_q_minus_q0_ivc_energy_per_cell"] < 0.0
    assert set(result.branch_selection["finite_q_sector_energy_per_cell"]) == {
        "plus",
        "minus",
    }
    assert result.branch_selection["selected_finite_q_sector"] in {
        "plus",
        "minus",
    }
    assert set(result.finite_q_branches) == {"plus", "minus"}
    assert (
        result.finite_q_branch
        is result.finite_q_branches[result.branch_selection["selected_finite_q_sector"]]
    )


def test_taige_branch_selected_workflow_can_force_q0(monkeypatch):
    def fake_references(bundle, params):
        return _dummy_refs_for_bundle(
            bundle,
            ivc_energy=-2.0 if bundle.active.finite_q_enabled else -1.0,
        )

    monkeypatch.setattr(workflow_mod, "build_symmetric_hf_references", fake_references)
    params = ContinuumWorkflowParams(
        model=taige_model_params(theta_deg=3.5, u_D=0.0, plane_wave_shell=0, n_bands=1),
        grid=ContinuumGridParams(n_k=6),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.0,
            q_shell=0,
            local_field_cutoff=0,
        ),
        hf=ContinuumHFParams(max_iter=1, min_iter=0),
        response=ResponseParams(n_theta=5, theta_min=1e-4, theta_max=np.pi - 1e-4),
    )

    result = run_taige_branch_selected_symmetric_hf_workflow(
        params,
        finite_q_enabled=True,
        ivc_branch_policy="q0",
        write_outputs=False,
    )

    assert result.selected_ivc_branch == "q0"
    assert result.bundle.active.finite_q_enabled is False
    assert result.finite_q_branch.bundle.active.finite_q_enabled is True


def test_taige_branch_selected_workflow_suppresses_texture_when_ivc_wins(monkeypatch):
    def fake_references(bundle, params):
        return _dummy_refs_for_bundle(
            bundle,
            ivc_energy=-1.0,
            vp_plus_energy=0.0,
            vp_minus_energy=0.1,
        )

    monkeypatch.setattr(workflow_mod, "build_symmetric_hf_references", fake_references)
    params = ContinuumWorkflowParams(
        model=taige_model_params(theta_deg=3.5, u_D=20.0, plane_wave_shell=0, n_bands=1),
        grid=ContinuumGridParams(n_k=3),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.0,
            q_shell=0,
            local_field_cutoff=0,
        ),
        hf=ContinuumHFParams(max_iter=1, min_iter=0),
        response=ResponseParams(n_theta=5, theta_min=1e-4, theta_max=np.pi - 1e-4),
    )

    result = run_taige_branch_selected_symmetric_hf_workflow(
        params,
        finite_q_enabled=False,
        ivc_branch_policy="q0",
        suppress_texture_when_ivc_below_vp=True,
        write_outputs=False,
    )

    assert result.branch_selection["texture_valid"] is False
    assert result.branch_selection["texture_invalid_reason"] == "ivc_energy_below_vp_reference"
    assert result.branch_selection["hf_ground_state"] == "IVC_0"
    assert np.isnan(result.response.cG)
    assert np.all(np.isnan(result.response.K))
    assert np.isnan(result.summary.gap_min)
    assert result.summary.valid_local_gap is False
    assert np.all(np.isnan(result.projectors.real))


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

    assert vertices.vertex_layout == "valley_compact"
    assert vertices.lambda_blocks.shape[:3] == (0, 0, 4)
    assert vertices.lambda_compact.shape[:4] == (7, 1, 4, 2)
    dense_lambdas = density_vertices_dense_lambdas(vertices)
    assert dense_lambdas.shape[:3] == (7, 1, 4)
    assert np.allclose(dense_lambdas[iq0, 0], np.eye(bundle.active.dim), atol=1e-10)
    assert vertices.v_over_a.shape == (7, 1)
    assert vertices.v_over_a[iq0, 0] > 0.0

    dense = _tiny_taige_bundle(interaction.model_copy(update={"density_vertex_layout": "dense"}))
    assert dense.vertices.vertex_layout == "dense"
    assert np.allclose(dense.vertices.lambda_blocks, dense_lambdas, atol=1e-12)

    unsmeared = interaction.model_copy(update={"smear_length_nm": 0.0})
    assert coulomb_potential_mev_nm2(5.0, interaction) < coulomb_potential_mev_nm2(5.0, unsmeared)


def _assert_matching_density_vertices(serial, parallel):
    assert serial.q_shifts == parallel.q_shifts
    assert serial.g_channels == parallel.g_channels
    assert serial.vertex_layout == parallel.vertex_layout
    assert np.array_equal(serial.target_minus_q, parallel.target_minus_q)
    assert np.array_equal(serial.q_is_zero, parallel.q_is_zero)
    assert np.array_equal(serial.channel_in_disk, parallel.channel_in_disk)
    assert np.allclose(
        density_vertices_dense_lambdas(serial),
        density_vertices_dense_lambdas(parallel),
        atol=1e-12,
    )
    if serial.vertex_layout == "valley_compact":
        assert np.allclose(serial.lambda_compact, parallel.lambda_compact, atol=1e-12)
        assert serial.lambda_blocks.shape[:2] == (0, 0)
    else:
        assert np.allclose(serial.lambda_blocks, parallel.lambda_blocks, atol=1e-12)
    assert np.allclose(serial.v_over_a, parallel.v_over_a, atol=1e-14)
    assert np.allclose(serial.q_vectors_nm_inv, parallel.q_vectors_nm_inv, atol=1e-14)
    assert np.allclose(serial.v_q, parallel.v_q, atol=1e-14)


@pytest.mark.parametrize(
    ("q_mesh", "q_shell", "local_field_cutoff", "n_k"),
    [
        ("shell", 1, 1, 3),
        ("full", 0, 0, 3),
    ],
)
def test_taige_density_vertices_parallel_q_slabs_match_serial(
    q_mesh,
    q_shell,
    local_field_cutoff,
    n_k,
):
    model = taige_model_params(theta_deg=3.5, u_D=0.0, plane_wave_shell=1, n_bands=1)
    interaction = ContinuumInteractionParams(
        coulomb_kind="dimensionless_screened",
        v0=0.05,
        q_mesh=q_mesh,
        q_shell=q_shell,
        local_field_cutoff=local_field_cutoff,
        vertex_workers=1,
    )
    bundle = build_continuum_bundle(
        model=model,
        grid=ContinuumGridParams(n_k=n_k),
        interaction=interaction,
    )

    serial = build_taige_density_vertices(bundle.active, interaction)
    parallel = build_taige_density_vertices(
        bundle.active,
        interaction.model_copy(update={"vertex_workers": 2}),
    )

    _assert_matching_density_vertices(serial, parallel)


@pytest.mark.parametrize("finite_q_enabled", [False, True])
def test_taige_density_vertices_cached_gather_matches_scalar(finite_q_enabled):
    n_k = 6 if finite_q_enabled else 3
    model = taige_model_params(theta_deg=3.5, u_D=0.0, plane_wave_shell=1, n_bands=1)
    finite_q = (
        ContinuumFiniteQParams(
            enabled=True,
            q_coord=taige_ivc_minus_q_coord(n_k),
            half_shift_coord=taige_ivc_minus_half_shift_coord(n_k),
        )
        if finite_q_enabled
        else ContinuumFiniteQParams()
    )
    interaction = ContinuumInteractionParams(
        coulomb_kind="dimensionless_screened",
        v0=0.05,
        q_mesh="full",
        q_shell=0,
        local_field_cutoff=1,
        vertex_workers=1,
        form_factor_backend="scalar",
    )
    bundle = build_continuum_bundle(
        model=model,
        grid=ContinuumGridParams(n_k=n_k),
        finite_q=finite_q,
        interaction=interaction,
    )

    scalar = build_taige_density_vertices(bundle.active, interaction)
    cached = build_taige_density_vertices(
        bundle.active,
        interaction.model_copy(update={"form_factor_backend": "cached_gather"}),
    )
    cached_parallel = build_taige_density_vertices(
        bundle.active,
        interaction.model_copy(update={"form_factor_backend": "cached_gather", "vertex_workers": 2}),
    )
    vectorized = build_taige_density_vertices(
        bundle.active,
        interaction.model_copy(update={"form_factor_backend": "vectorized"}),
    )
    vectorized_parallel = build_taige_density_vertices(
        bundle.active,
        interaction.model_copy(update={"form_factor_backend": "vectorized", "vertex_workers": 2}),
    )

    _assert_matching_density_vertices(scalar, cached)
    _assert_matching_density_vertices(scalar, cached_parallel)
    _assert_matching_density_vertices(scalar, vectorized)
    _assert_matching_density_vertices(scalar, vectorized_parallel)


def test_taige_interaction_params_accept_screening_overrides():
    interaction = taige_interaction_params(
        include_q0=False,
        q_mesh="full",
        q_shell=0,
        local_field_cutoff=4,
        momentum_transfer_cutoff_km=3.0,
        epsilon=12.5,
        gate_distance_nm=18.0,
        smear_length_nm=0.2,
        interaction_strength_scale=0.7,
        hartree_scale=0.9,
        exchange_scale=0.8,
        vertex_workers=2,
        exchange_workers=3,
        density_vertex_retention="hartree_only",
        density_vertex_layout="dense",
        exchange_representation="dense",
        form_factor_backend="cached_gather",
    )

    assert interaction.coulomb_kind == "dual_gate"
    assert interaction.include_q0 is False
    assert interaction.q_mesh == "full"
    assert interaction.q_shell == 0
    assert interaction.local_field_cutoff == 4
    assert interaction.momentum_transfer_cutoff_km == 3.0
    assert interaction.epsilon == 12.5
    assert interaction.gate_distance_nm == 18.0
    assert interaction.smear_length_nm == 0.2
    assert interaction.v0 == 0.7
    assert interaction.hartree_scale == 0.9
    assert interaction.exchange_scale == 0.8
    assert interaction.vertex_workers == 2
    assert interaction.exchange_workers == 3
    assert interaction.density_vertex_retention == "hartree_only"
    assert interaction.density_vertex_layout == "dense"
    assert interaction.exchange_representation == "dense"
    assert interaction.form_factor_backend == "cached_gather"


def test_taige_explicit_momentum_disk_matches_strict_three_gm_policy():
    model = taige_model_params(
        theta_deg=3.9,
        u_D=0.0,
        plane_wave_shell=1,
        n_bands=1,
        n_active_bands_per_valley=1,
    )
    grid = MomentumGrid(6)
    active, _bands = build_taige_active_space(
        grid,
        model,
        ContinuumFiniteQParams(),
    )
    interaction = taige_interaction_params(
        q_mesh="full",
        local_field_cutoff=4,
        momentum_transfer_cutoff_km=3.0,
        smear_length_nm=0.0,
    )

    vertices = build_taige_density_vertices(active, interaction)
    expected = np.asarray(vertices.q_norm_nm_inv) < 3.0 * active.geometry.kM_inv_nm

    assert np.array_equal(vertices.channel_in_disk, expected)
    assert np.any(expected)
    assert np.any(~expected)


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
    dense_vertices = density_vertices_dense_lambdas(vertices)
    dense_q0 = density_vertices_dense_lambdas(q0_bundle.vertices)
    iq0 = vertices.q_shifts.index((0, 0))
    iq = vertices.q_shifts.index((1, 0))

    assert vertices.vertex_layout == "valley_compact"
    assert np.allclose(dense_vertices[iq0, 0], np.eye(active.dim), atol=1e-10)
    for ik in range(active.n_k):
        physical = int(active.source_index[ik, 0])
        if physical == ik:
            continue
        finite_block = dense_vertices[iq, 0, ik, 0:1, 0:1]
        shifted_block = dense_q0[iq, 0, physical, 0:1, 0:1]
        unshifted_block = dense_q0[iq, 0, ik, 0:1, 0:1]
        assert np.allclose(finite_block, shifted_block)
        assert not np.allclose(finite_block, unshifted_block)
        break
    else:
        raise AssertionError("finite-Q source map did not shift any K-valley source")


def test_shared_taige_q_sector_builder_computes_bands_once(monkeypatch):
    calls = 0
    original = taige_mod.compute_taige_bandstructure

    def counted(model, grid):
        nonlocal calls
        calls += 1
        return original(model, grid)

    monkeypatch.setattr(taige_mod, "compute_taige_bandstructure", counted)
    model = taige_model_params(
        theta_deg=3.5,
        u_D=0.0,
        plane_wave_shell=1,
        n_bands=2,
    )
    finite_q = ContinuumFiniteQParams(
        enabled=True,
        q_coord=taige_ivc_minus_q_coord(6),
        half_shift_coord=taige_ivc_minus_half_shift_coord(6),
    )
    q0_bundle, finite_bundle = build_taige_q_sector_bundles(
        model,
        ContinuumGridParams(n_k=6),
        ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.02,
            q_shell=0,
            local_field_cutoff=0,
        ),
        finite_q,
    )

    assert calls == 1
    assert q0_bundle.bands is finite_bundle.bands
    assert q0_bundle.active.bands is finite_bundle.active.bands
    assert q0_bundle.active.finite_q_enabled is False
    assert finite_bundle.active.finite_q_enabled is True

    standalone_q0 = build_continuum_bundle(
        model=model,
        grid=ContinuumGridParams(n_k=6),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.02,
            q_shell=0,
            local_field_cutoff=0,
        ),
    )
    assert calls == 2
    assert np.allclose(q0_bundle.active.h0, standalone_q0.active.h0)
    _assert_matching_density_vertices(q0_bundle.vertices, standalone_q0.vertices)
    assert np.allclose(
        q0_bundle.backend.dense_exchange_tve_for_debug(),
        standalone_q0.backend.dense_exchange_tve_for_debug(),
    )


def test_single_finite_q_bundle_builds_only_finite_q_exchange_backend(monkeypatch):
    calls = 0
    original = hf_mod.ContinuumHFBackend

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(hf_mod, "ContinuumHFBackend", counted)
    bundle = build_continuum_bundle(
        model=taige_model_params(
            theta_deg=3.5,
            u_D=0.0,
            plane_wave_shell=1,
            n_bands=1,
        ),
        grid=ContinuumGridParams(n_k=6),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.02,
            q_shell=0,
            local_field_cutoff=0,
        ),
        finite_q=ContinuumFiniteQParams(
            enabled=True,
            q_coord=taige_ivc_minus_q_coord(6),
            half_shift_coord=taige_ivc_minus_half_shift_coord(6),
        ),
    )

    assert calls == 1
    assert bundle.active.finite_q_enabled is True


def test_single_finite_q_bundle_releases_q0_vertices_before_backend(monkeypatch):
    source_ref = None
    original_vertex_builder = taige_mod.build_taige_density_vertices
    original_backend = hf_mod.ContinuumHFBackend

    def tracked_vertex_builder(active, interaction):
        nonlocal source_ref
        vertices = original_vertex_builder(active, interaction)
        if not active.finite_q_enabled:
            source_ref = weakref.ref(vertices)
        return vertices

    def checking_backend(*args, **kwargs):
        assert source_ref is not None
        assert source_ref() is None
        return original_backend(*args, **kwargs)

    monkeypatch.setattr(
        taige_mod,
        "build_taige_density_vertices",
        tracked_vertex_builder,
    )
    monkeypatch.setattr(hf_mod, "ContinuumHFBackend", checking_backend)
    bundle = build_continuum_bundle(
        model=taige_model_params(
            theta_deg=3.5,
            u_D=0.0,
            plane_wave_shell=1,
            n_bands=1,
        ),
        grid=ContinuumGridParams(n_k=6),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.02,
            q_shell=0,
            local_field_cutoff=0,
        ),
        finite_q=ContinuumFiniteQParams(
            enabled=True,
            q_coord=taige_ivc_minus_q_coord(6),
            half_shift_coord=taige_ivc_minus_half_shift_coord(6),
        ),
    )

    assert bundle.active.finite_q_enabled is True


@pytest.mark.parametrize("layout", ["valley_compact", "dense"])
def test_taige_rolled_vertices_exhaustively_match_direct_reconstruction(layout):
    model = taige_model_params(
        theta_deg=3.5,
        u_D=0.0,
        plane_wave_shell=1,
        n_bands=2,
    )
    grid_params = ContinuumGridParams(n_k=6)
    grid = MomentumGrid(grid_params.n_k)
    finite_q = ContinuumFiniteQParams(
        enabled=True,
        q_coord=taige_ivc_minus_q_coord(6),
        half_shift_coord=taige_ivc_minus_half_shift_coord(6),
    )
    bands = compute_taige_bandstructure(model, grid)
    q0_active, _ = build_taige_active_space(grid, model, bands=bands)
    finite_active, _ = build_taige_active_space(
        grid,
        model,
        finite_q,
        bands=bands,
    )
    baseline_q0 = None
    baseline_direct = None

    for backend in ("scalar", "cached_gather", "vectorized"):
        serial_interaction = ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.05,
            q_shell=1,
            local_field_cutoff=1,
            density_vertex_layout=layout,
            form_factor_backend=backend,
            vertex_workers=1,
        )
        parallel_interaction = serial_interaction.model_copy(
            update={"vertex_workers": 2}
        )
        q0_serial = build_taige_density_vertices(q0_active, serial_interaction)
        q0_parallel = build_taige_density_vertices(q0_active, parallel_interaction)
        rolled_serial = roll_taige_density_vertices(q0_serial, finite_active)
        rolled_parallel = roll_taige_density_vertices(q0_parallel, finite_active)
        direct_serial = build_taige_density_vertices(
            finite_active,
            serial_interaction,
        )
        direct_parallel = build_taige_density_vertices(
            finite_active,
            parallel_interaction,
        )

        _assert_matching_density_vertices(q0_serial, q0_parallel)
        _assert_matching_density_vertices(rolled_serial, rolled_parallel)
        _assert_matching_density_vertices(rolled_serial, direct_serial)
        _assert_matching_density_vertices(rolled_serial, direct_parallel)
        if baseline_q0 is None:
            baseline_q0 = q0_serial
            baseline_direct = direct_serial
        else:
            _assert_matching_density_vertices(baseline_q0, q0_serial)
            _assert_matching_density_vertices(baseline_direct, direct_serial)

        if layout == "valley_compact":
            q0_lambda = q0_serial.lambda_compact
            rolled_lambda = rolled_serial.lambda_compact
        else:
            q0_lambda = q0_serial.lambda_blocks
            rolled_lambda = rolled_serial.lambda_blocks
        assert q0_lambda is not None
        assert rolled_lambda is not None
        for iq in range(len(q0_serial.q_shifts)):
            for ig in range(len(q0_serial.g_channels)):
                for ik in range(finite_active.n_k):
                    for valley in range(2):
                        source = int(finite_active.source_index[ik, valley])
                        if layout == "valley_compact":
                            expected = q0_lambda[iq, ig, source, valley]
                            actual = rolled_lambda[iq, ig, ik, valley]
                        else:
                            start = valley * finite_active.n_active
                            stop = start + finite_active.n_active
                            expected = q0_lambda[
                                iq, ig, source, start:stop, start:stop
                            ]
                            actual = rolled_lambda[
                                iq, ig, ik, start:stop, start:stop
                            ]
                        assert np.allclose(actual, expected, atol=1e-12)


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


def test_tiny_taige_multi_active_response_smoke():
    model = taige_model_params(
        theta_deg=3.5,
        u_D=0.0,
        plane_wave_shell=1,
        n_bands=2,
        n_active_bands_per_valley=2,
    )
    bundle = build_continuum_bundle(
        model=model,
        grid=ContinuumGridParams(n_k=2),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.0,
            q_shell=0,
            local_field_cutoff=0,
        ),
    )
    active = bundle.active
    assert active.dim == 4

    H_plus = np.broadcast_to(np.diag([-2.0, -1.0, 2.0, 3.0]), active.h0.shape).astype(complex)
    H_minus = np.broadcast_to(np.diag([2.0, 3.0, -2.0, -1.0]), active.h0.shape).astype(complex)
    H_ivc = np.zeros_like(active.h0, dtype=complex)
    H_ivc[:, 0, 2] = H_ivc[:, 2, 0] = -1.0
    H_ivc[:, 1, 3] = H_ivc[:, 3, 1] = -0.5
    refs = SymmetricHFReferences(
        vp_plus=_dummy_hf_result(H_plus, "vp_plus"),
        vp_minus=_dummy_hf_result(H_minus, "vp_minus"),
        ivc=_dummy_hf_result(H_ivc, "ivc"),
        n_occ_per_k=1,
    )

    theta = np.linspace(0.1, np.pi - 0.1, 5)
    projectors, diagnostics = symmetric_convex_path(refs, theta)
    response_projectors = projectors.reshape(5, 2, 2, active.dim, active.dim)
    basis = active_basis_frames(active).reshape(2, 2, -1, active.dim)
    response = k_theta_from_projectors_with_basis(response_projectors, theta, basis)

    assert response_projectors.shape[-2:] == (4, 4)
    assert len(diagnostics) == 5
    assert np.all(np.isfinite(response.K))
    assert np.isfinite(response.cG)


def test_taige_fixed_density_hf_path_and_chern_smoke():
    model = taige_model_params(
        theta_deg=3.5,
        u_D=0.0,
        plane_wave_shell=1,
        n_bands=2,
        n_active_bands_per_valley=2,
    )
    bundle = build_continuum_bundle(
        model=model,
        grid=ContinuumGridParams(n_k=3),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.0,
            q_shell=0,
            local_field_cutoff=0,
        ),
    )
    active = bundle.active
    P = build_seed("vp_plus", active, n_occ_per_k=1)

    coarse_index = active.grid.index_of((1, 2))
    k_frac = np.array((1 / active.grid.n1, 2 / active.grid.n2), dtype=float)
    fine_h = hf_hamiltonian_at_k(bundle, P, k_frac)
    assert np.allclose(np.linalg.eigvalsh(fine_h), np.linalg.eigvalsh(active.h0[coarse_index]), atol=1e-8)

    spectrum = evaluate_hf_high_symmetry_path(bundle, P, n_per_segment=2, reference="VP")
    assert spectrum.energies.shape == (11, active.dim)
    assert len(spectrum.rows) == 11 * active.dim
    assert np.all(np.isfinite(spectrum.energies))
    assert np.allclose(np.sum(spectrum.valley_weights, axis=-1), 1.0)

    chern_rows = hf_band_chern_table(active, active.h0, reference="h0")
    assert len(chern_rows) == active.dim
    assert all(np.isfinite(row.chern) for row in chern_rows)


def test_taige_fixed_density_hf_path_supports_finite_q_frame():
    finite_q = ContinuumFiniteQParams(
        enabled=True,
        q_coord=taige_ivc_minus_q_coord(6),
        half_shift_coord=taige_ivc_minus_half_shift_coord(6),
    )
    bundle = build_continuum_bundle(
        model=taige_model_params(theta_deg=3.5, u_D=0.0, plane_wave_shell=1, n_bands=1),
        grid=ContinuumGridParams(n_k=6),
        finite_q=finite_q,
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.0,
            q_shell=0,
            local_field_cutoff=0,
        ),
    )
    P = build_seed("finite_q_ivc", bundle.active)

    frame = taige_active_fine_frame(bundle.active, np.array([0.0, 0.0]))
    assert np.any(frame.physical_shift != 0) or np.any(np.abs(frame.physical_k_frac) > 0.0)

    spectrum = evaluate_hf_high_symmetry_path(bundle, P, n_per_segment=1, reference="finite-Q IVC")
    assert spectrum.energies.shape == (6, bundle.active.dim)
    assert len(spectrum.rows) == 6 * bundle.active.dim
    assert np.all(np.isfinite(spectrum.energies))
