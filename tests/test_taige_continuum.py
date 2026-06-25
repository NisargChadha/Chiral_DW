import numpy as np

from chiral_dw.config import ContinuumGridParams, ContinuumHFParams, ContinuumInteractionParams
from chiral_dw.continuum import (
    TPrimeConstraint,
    ValleyU1Constraint,
    build_continuum_bundle,
    build_symmetric_hf_references,
    chern_number_table,
    compute_taige_path_spectrum,
    random_projector_like_seed,
    symmetric_convex_path,
    taige_interaction_params,
    taige_model_params,
)
from chiral_dw.continuum.seeds import build_seed, mix_projector_seeds
from chiral_dw.continuum.taige import TaigeContinuumModel, coulomb_potential_mev_nm2
from chiral_dw.response import compute_cG, k_theta_from_projectors


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
    assert bundle.bands is not None
    assert bundle.geometry is not None


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


def test_projector_like_seed_mix_preserves_trace_and_hf_snapshots_are_recorded():
    bundle = _tiny_taige_bundle()
    active = bundle.active
    ordered = build_seed("vp_plus", active)
    noise = random_projector_like_seed(ordered, seed=4)
    mixed = mix_projector_seeds(ordered, noise, ordered_weight=0.8, random_weight=0.2)

    assert np.allclose(mixed, mixed.conj().swapaxes(-1, -2))
    assert np.allclose(np.trace(mixed, axis1=-2, axis2=-1), 1.0)

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


def test_tiny_taige_symmetric_response_smoke():
    bundle = _tiny_taige_bundle()
    params = ContinuumHFParams(max_iter=3, min_iter=1, mixing=0.7)
    refs = build_symmetric_hf_references(bundle, params)
    theta = np.linspace(0.0, np.pi, 5)
    projectors, diagnostics = symmetric_convex_path(refs, theta)
    response = k_theta_from_projectors(projectors.reshape(5, 2, 2, 2, 2), theta)

    assert refs.vp_plus.constraint_name == ValleyU1Constraint(bundle.active, pinned_valley="K").name
    assert refs.ivc.constraint_name == TPrimeConstraint(bundle.active).name
    assert len(diagnostics) == 5
    assert np.all(np.isfinite(response.K))
    assert np.isfinite(compute_cG(response.theta, response.K))
