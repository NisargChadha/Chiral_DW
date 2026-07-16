import numpy as np
import pytest

from chiral_dw.config import ContinuumHFParams, ContinuumInteractionParams
from chiral_dw.config import ContinuumGridParams
from chiral_dw.continuum import (
    ContinuumHFBackend,
    DensityVertices,
    SETFillingEnergyRow,
    TaigeSETWorkflowParams,
    chemical_potential_rows,
    gaussian_dos,
    hf_band_validity_summary,
    inverse_compressibility_rows,
    run_taige_set_hysteresis_branch_point,
    select_set_hysteresis_envelope,
    set_gap_summary,
    solve_global_hf,
    run_taige_set_point,
    taige_interaction_params,
    taige_model_params,
)


def _noninteracting_backend() -> ContinuumHFBackend:
    h0 = np.asarray(
        [
            np.diag([0.0, 3.0]),
            np.diag([1.0, 4.0]),
        ],
        dtype=complex,
    )
    lambdas = np.eye(2, dtype=complex)[None, None, None, :, :]
    lambdas = np.repeat(lambdas, h0.shape[0], axis=2)
    vertices = DensityVertices(
        q_shifts=((0, 0),),
        target_minus_q=np.asarray([[0, 1]], dtype=int),
        q_is_zero=np.asarray([True]),
        lambda_blocks=lambdas,
        v_over_a=np.zeros((1, 1), dtype=float),
        g_channels=((0, 0),),
    )
    return ContinuumHFBackend(h0, vertices, ContinuumInteractionParams())


def _energy_row(n_particles: int, energy: float, intrinsic: float | None = None):
    intrinsic_energy = energy if intrinsic is None else intrinsic
    return SETFillingEnergyRow(
        n_particles=n_particles,
        filling_holes=n_particles / 2.0,
        energy_total_mev=energy,
        energy_per_cell_mev=energy / 2.0,
        uniform_hartree_energy_mev=energy - intrinsic_energy,
        intrinsic_energy_total_mev=intrinsic_energy,
        intrinsic_energy_per_cell_mev=intrinsic_energy / 2.0,
        one_body_energy_mev=intrinsic_energy,
        hartree_energy_mev=0.0,
        fock_energy_mev=0.0,
        converged=True,
        n_iter=1,
        trace_error=0.0,
        aufbau_residual_norm=0.0,
        commutator_norm=0.0,
    )


def test_global_hf_fills_lowest_states_and_reports_global_diagnostics():
    backend = _noninteracting_backend()
    P0, _evals, _direct, _indirect = backend.update_density(backend.h0, 2)
    result = solve_global_hf(
        backend,
        P0,
        2,
        ContinuumHFParams(max_iter=3, min_iter=1, mixing_method="oda"),
    )

    assert result.converged
    assert result.energy == pytest.approx(1.0)
    assert np.trace(result.P, axis1=-2, axis2=-1).sum().real == pytest.approx(2.0)
    assert result.diagnostics.occupation_mode == "global"
    assert result.diagnostics.indirect_gap == pytest.approx(2.0)
    assert result.diagnostics.trace_error < 1e-12


def test_zero_interaction_set_gap_matches_global_single_particle_gap():
    rows = [_energy_row(1, 0.0), _energy_row(2, 1.0), _energy_row(3, 4.0)]

    gap = set_gap_summary(rows, n_particles_filling_one=2)

    assert gap.mu_minus_hole_raw_mev == pytest.approx(1.0)
    assert gap.mu_plus_hole_raw_mev == pytest.approx(3.0)
    assert gap.charge_gap_raw_mev == pytest.approx(2.0)


def test_set_finite_differences_and_electron_sign_convention():
    rows = [
        _energy_row(1, 1.0, intrinsic=0.5),
        _energy_row(2, 4.0, intrinsic=2.0),
        _energy_row(3, 9.0, intrinsic=4.5),
    ]

    mu = chemical_potential_rows(rows, n_cells=2)
    kappa = inverse_compressibility_rows(rows, n_cells=2, moire_cell_area_nm2=10.0)

    assert [row.mu_hole_raw_mev for row in mu] == pytest.approx([3.0, 5.0])
    assert [row.mu_electron_raw_mev for row in mu] == pytest.approx([-3.0, -5.0])
    assert kappa[0].dmu_dnu_raw_mev == pytest.approx(4.0)
    assert kappa[0].dmu_dnu_intrinsic_mev == pytest.approx(2.0)
    assert kappa[0].dmu_dn_raw_mev_nm2 == pytest.approx(40.0)


def test_set_hysteresis_envelope_selects_each_particle_number_independently():
    up = [_energy_row(1, 1.0), _energy_row(2, 4.0), _energy_row(3, 9.0)]
    down = [_energy_row(1, 0.0), _energy_row(2, 5.0), _energy_row(3, 8.0)]

    envelope = select_set_hysteresis_envelope(
        up,
        down,
        n_particles_filling_one=2,
    )

    assert envelope.selected_direction_by_particles == {1: "down", 2: "up", 3: "down"}
    assert envelope.down_minus_up_intrinsic_energy_mev == pytest.approx(
        {1: -1.0, 2: 1.0, 3: -1.0}
    )
    assert [row.energy_total_mev for row in envelope.selected_energy_rows] == pytest.approx(
        [0.0, 4.0, 8.0]
    )
    assert envelope.set_gap.charge_gap_raw_mev == pytest.approx(0.0)


def test_set_hysteresis_envelope_rejects_two_unconverged_candidates():
    up = [_energy_row(1, 0.0), _energy_row(2, 1.0), _energy_row(3, 3.0)]
    down = [_energy_row(1, 0.0), _energy_row(2, 0.5), _energy_row(3, 3.0)]
    up[1] = up[1].model_copy(update={"converged": False})
    down[1] = down[1].model_copy(update={"converged": False})

    with pytest.raises(ValueError, match="both hysteresis branches are unconverged"):
        select_set_hysteresis_envelope(up, down, n_particles_filling_one=2)


def test_negative_indirect_gap_invalidates_fixed_per_k_insulator():
    H = np.asarray([np.diag([0.0, 5.0]), np.diag([6.0, 7.0])], dtype=complex)

    summary = hf_band_validity_summary(H)

    assert summary.direct_gap_mev == pytest.approx(1.0)
    assert summary.indirect_gap_mev == pytest.approx(-1.0)
    assert not summary.valid_fixed_per_k_insulator
    assert summary.chern_resolved_by_direct_gap
    assert not summary.chern_physically_interpretable
    assert summary.invalid_reason == "nonpositive_indirect_gap"


def test_uniform_hartree_energy_reconstructs_omitted_capacitive_channel():
    h0 = np.zeros((1, 2, 2), dtype=complex)
    vertices = DensityVertices(
        q_shifts=((0, 0),),
        target_minus_q=np.asarray([[0]], dtype=int),
        q_is_zero=np.asarray([True]),
        lambda_blocks=np.eye(2, dtype=complex)[None, None, None, :, :],
        v_over_a=np.asarray([[0.2]], dtype=float),
        g_channels=((0, 0),),
    )
    backend = ContinuumHFBackend(
        h0,
        vertices,
        ContinuumInteractionParams(density_vertex_retention="hartree_only"),
    )
    P = np.eye(2, dtype=complex)[None, :, :]

    assert backend.n_q == 0
    assert backend.energy(P).hartree == pytest.approx(0.0)
    assert backend.uniform_hartree_energy(P) == pytest.approx(0.4)


def test_gaussian_dos_is_normalized_per_momentum_block():
    evals = np.asarray([[0.0, 1.0], [0.5, 1.5]])
    energy = np.linspace(-4.0, 5.0, 20001)

    dos = gaussian_dos(evals, energy, sigma_mev=0.1)

    assert np.trapezoid(dos, energy) == pytest.approx(2.0, abs=2e-4)


def test_small_taige_set_point_runs_fixed_and_global_paths():
    params = TaigeSETWorkflowParams(
        model=taige_model_params(
            theta_deg=3.0,
            u_D=5.75,
            plane_wave_shell=1,
            n_bands=1,
            n_active_bands_per_valley=1,
        ),
        grid=ContinuumGridParams(n_k=2),
        interaction=taige_interaction_params(
            q_mesh="shell",
            q_shell=0,
            local_field_cutoff=0,
            interaction_strength_scale=0.0,
        ),
        hf=ContinuumHFParams(
            max_iter=3,
            min_iter=1,
            mixing_method="oda",
            seed_ordered_weight=1.0,
            seed_random_weight=0.0,
        ),
        particle_offsets=(-1, 0, 1),
        dos_energy_points=101,
    )

    result = run_taige_set_point(params)

    assert result.summary.n_cells == 4
    assert result.summary.n_fillings == 3
    assert sorted(result.filling_results) == [3, 4, 5]
    assert len(result.chemical_potential_rows) == 2
    assert len(result.inverse_compressibility_rows) == 1
    assert len(result.dos_rows) == 101
    assert all(row.converged for row in result.filling_energy_rows)

    branch = run_taige_set_hysteresis_branch_point(
        params,
        {n_particles: hf_result.P for n_particles, hf_result in result.filling_results.items()},
        direction="up",
    )
    assert branch.summary.direction == "up"
    assert branch.summary.all_fillings_converged
    assert sorted(branch.filling_results) == [3, 4, 5]
    assert branch.summary.neutral_topology.one_state_per_k_max_error >= 0.0
