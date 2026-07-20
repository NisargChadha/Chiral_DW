from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from chiral_dw.config import (
    ContinuumGridParams,
    ContinuumHFParams,
    OrbitalMagnetizationParams,
    TaigeOrbitalMagnetizationWorkflowParams,
)
from chiral_dw.continuum.models import MomentumGrid
from chiral_dw.continuum.orbital_magnetization import evaluate_projector_orbital_magnetization
from chiral_dw.continuum.orbital_magnetization_workflow import (
    build_frozen_hole_subspaces,
    load_taige_band_cache,
    save_taige_band_cache,
    run_taige_orbital_magnetization_workflow,
    taige_band_cache_signature,
    taige_transport_factory,
)
from chiral_dw.continuum.taige import (
    active_space_from_taige_bands,
    compute_taige_bandstructure,
    taige_model_params,
    taige_interaction_params,
)


def _bare_problem(n_k: int = 2, n_bands: int = 4):
    grid = MomentumGrid(n_k)
    model = taige_model_params(
        theta_deg=3.7,
        u_D=0.0,
        plane_wave_shell=1,
        n_bands=n_bands,
        n_active_bands_per_valley=2,
    )
    bands = compute_taige_bandstructure(model, grid)
    active = active_space_from_taige_bands(grid, model, bands)
    projector = np.zeros_like(active.h0)
    projector[:, 0, 0] = 1.0
    hf = SimpleNamespace(P=projector, H_hf=active.h0.copy())
    return grid, model, bands, active, hf


def test_band_cache_round_trip_and_signature_validation(tmp_path) -> None:
    grid, model, bands, _active, _hf = _bare_problem()
    path = tmp_path / "bands.npz"
    saved = save_taige_band_cache(path, bands)
    loaded, manifest = load_taige_band_cache(
        path, expected_model=model, expected_grid=grid
    )

    assert saved.cache_hash == manifest.cache_hash
    assert manifest.signature == taige_band_cache_signature(model, grid)
    assert np.array_equal(loaded.electron_energies, bands.electron_energies)
    assert np.array_equal(loaded.electron_vectors, bands.electron_vectors)
    assert np.array_equal(loaded.hole_vectors, np.conj(loaded.electron_vectors))

    wrong_model = model.model_copy(update={"theta_deg": 3.8})
    with pytest.raises(ValueError, match="expected model"):
        load_taige_band_cache(path, expected_model=wrong_model)


def test_corrupt_band_cache_is_rejected(tmp_path) -> None:
    path = tmp_path / "corrupt.npz"
    path.write_bytes(b"not-an-npz")
    with pytest.raises(ValueError, match="could not load"):
        load_taige_band_cache(path)


def test_cached_max_band_slicing_is_nested_and_preserves_two_band_energies() -> None:
    grid, model, bands, active, _hf = _bare_problem(n_bands=4)
    two_band_model = model.model_copy(update={"n_bands": 2})
    independent = compute_taige_bandstructure(two_band_model, grid)
    cached_active = active_space_from_taige_bands(grid, model, bands)

    assert np.allclose(cached_active.hole_energies, independent.hole_energies)
    assert np.array_equal(cached_active.band_vectors, bands.hole_vectors[..., :2])
    # K is diagonalized directly and is cutoff independent.  Kprime is generated
    # by a finite-shell T-prime map followed by whole-frame Loewdin
    # orthonormalization, so max-cutoff slicing is intentionally the common
    # nested convention rather than recomputing a different frame at each N.
    assert np.allclose(
        cached_active.band_vectors[:, 0], independent.hole_vectors[:, 0], atol=2e-12
    )
    assert active.n_active == 2


def test_frozen_construction_keeps_remote_bands_out_of_self_energy() -> None:
    _grid, _model, bands, active, hf = _bare_problem()
    frozen = build_frozen_hole_subspaces(
        active=active,
        bands=bands,
        hf=hf,
        n_remote_bands_per_valley=1,
    )

    assert frozen.occupied_frames.shape[-1] == 1
    assert frozen.empty_frames.shape[-1] == 5
    assert frozen.diagnostics.n_remote_bands_per_valley == 1
    assert frozen.diagnostics.max_active_remote_overlap < 2e-12
    assert frozen.diagnostics.max_active_remote_self_energy_mev == pytest.approx(0.0)
    assert frozen.diagnostics.max_projector_occupation_error < 2e-12

    valley_dim = bands.hole_vectors.shape[2]
    assert np.allclose(
        frozen.empty_frames[:, :valley_dim, 3], bands.hole_vectors[:, 0, :, 2]
    )
    assert np.allclose(
        frozen.empty_frames[:, valley_dim:, 4], bands.hole_vectors[:, 1, :, 2]
    )


def test_zero_self_energy_frozen_cutoff_runs_common_basis_observable() -> None:
    grid, _model, bands, active, hf = _bare_problem(n_k=3)
    frozen = build_frozen_hole_subspaces(
        active=active,
        bands=bands,
        hf=hf,
        n_remote_bands_per_valley=1,
    )
    geometry = bands.geometry
    reciprocal_basis = geometry.kM_inv_nm * np.column_stack((geometry.b1, geometry.b2))
    result = evaluate_projector_orbital_magnetization(
        grid=grid,
        occupied_frames=frozen.occupied_frames,
        empty_frames=frozen.empty_frames,
        hamiltonian_on_occupied=frozen.hamiltonian_on_occupied,
        hamiltonian_on_empty=frozen.hamiltonian_on_empty,
        reciprocal_basis_nm_inv=reciprocal_basis,
        chemical_potential_hole_mev=frozen.gap.hole_mu_midgap_mev,
        transport=taige_transport_factory(bands.shell),
    )
    assert np.isfinite(result.summary.orbital_magnetization_mu_b_per_cell)
    assert result.summary.moire_cell_area_nm2 == pytest.approx(
        geometry.moire_cell_area_nm2
    )


def test_hole_gap_edges_map_to_reversed_electron_edges() -> None:
    _grid, _model, bands, active, hf = _bare_problem()
    frozen = build_frozen_hole_subspaces(
        active=active,
        bands=bands,
        hf=hf,
        n_remote_bands_per_valley=0,
    )
    gap = frozen.gap
    assert gap.electron_vbm_mev == pytest.approx(-gap.hole_empty_min_mev)
    assert gap.electron_cbm_mev == pytest.approx(-gap.hole_occupied_max_mev)
    assert gap.hole_mu_at_electron_vbm_mev == pytest.approx(gap.hole_empty_min_mev)
    assert gap.hole_mu_at_electron_cbm_mev == pytest.approx(gap.hole_occupied_max_mev)


def test_tiny_workflow_smoke_writes_complete_restartable_artifacts(tmp_path) -> None:
    model = taige_model_params(
        theta_deg=3.7,
        u_D=0.0,
        plane_wave_shell=1,
        n_bands=3,
        n_active_bands_per_valley=2,
    )
    params = TaigeOrbitalMagnetizationWorkflowParams(
        model=model,
        grid=ContinuumGridParams(n_k=2),
        interaction=taige_interaction_params(
            q_mesh="shell",
            q_shell=0,
            local_field_cutoff=0,
            interaction_strength_scale=0.0,
        ),
        hf=ContinuumHFParams(
            n_occ_per_k=1,
            max_iter=4,
            min_iter=0,
            mixing_method="linear",
            mixing=1.0,
            tolerance=1e-9,
            energy_tolerance=1e-11,
        ),
        orbital=OrbitalMagnetizationParams(
            remote_cutoffs_per_valley=(0, 1),
            enlarged_hf_bands_per_valley=(2,),
            benchmark_repeats=2,
            store_k_resolved_terms=True,
        ),
        output_dir=str(tmp_path / "smoke"),
    )
    first = run_taige_orbital_magnetization_workflow(params)
    assert first.frozen_rows == 6
    assert first.enlarged_hf_rows == 3
    assert first.matched_rows == 1
    assert first.all_hf_converged
    assert first.manifest_passed

    expected = {
        "continuum_bands_max.npz",
        "hf_active_2.npz",
        "remote_convergence.csv",
        "hf_active_space_convergence.csv",
        "matched_cutoff_comparison.csv",
        "benchmarks.csv",
        "benchmarks.json",
        "k_resolved_terms.npz",
        "summary.json",
        "run_manifest.json",
    }
    assert expected.issubset({path.name for path in (tmp_path / "smoke").iterdir()})

    second = run_taige_orbital_magnetization_workflow(params)
    assert second.band_cache_hash == first.band_cache_hash
    benchmark_text = (tmp_path / "smoke" / "benchmarks.csv").read_text()
    assert "continuum_band_cache_load" in benchmark_text
    assert "hf_state_cache_load" in benchmark_text
