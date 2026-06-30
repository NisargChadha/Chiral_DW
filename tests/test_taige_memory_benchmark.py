import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from chiral_dw.config import ContinuumGridParams, ContinuumInteractionParams
from chiral_dw.continuum.builder import build_continuum_bundle
from chiral_dw.continuum.memory_benchmark import (
    PackedHermitianExchange,
    TaigeMemoryBenchmarkInput,
    compact_lambdas_from_dense,
    dense_lambdas_from_compact,
    estimate_taige_array_bytes,
    hartree_only_backend_from_dense,
    matrix_free_backend_from_dense,
    run_taige_memory_benchmark_worker,
)
from chiral_dw.continuum.models import MomentumGrid, hermitize
from chiral_dw.continuum.taige import q_transfers, reciprocal_box, taige_model_params

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_taige_memory_backends.py"


def _tiny_params(**updates):
    params = TaigeMemoryBenchmarkInput(
        n_k=2,
        plane_wave_shell=1,
        n_bands=1,
        n_active_bands_per_valley=1,
        local_field_cutoff=0,
        fock_repeats=2,
    )
    return params.model_copy(update=updates)


def _tiny_taige_backend():
    return build_continuum_bundle(
        model=taige_model_params(
            theta_deg=3.5,
            u_D=0.0,
            plane_wave_shell=1,
            n_bands=1,
            n_active_bands_per_valley=1,
        ),
        grid=ContinuumGridParams(n_k=2),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.05,
            q_mesh="full",
            q_shell=0,
            local_field_cutoff=0,
        ),
    ).backend


def test_taige_memory_byte_estimator_matches_shapes():
    params = _tiny_params(n_k=3, local_field_cutoff=1)
    estimates = estimate_taige_array_bytes(params)
    grid_size = params.n_k * params.n_k
    interaction = params.interaction_params()
    n_q = len(q_transfers(MomentumGrid(params.n_k), interaction))
    n_g = len(reciprocal_box(params.local_field_cutoff))
    dim = 2 * params.n_active_bands_per_valley
    expected_lambda = n_q * n_g * grid_size * dim * dim * np.dtype(np.complex128).itemsize
    expected_tve = (grid_size * dim * dim) ** 2 * np.dtype(np.complex128).itemsize
    expected_sector_tve = 4 * (grid_size * params.n_active_bands_per_valley**2) ** 2 * (
        np.dtype(np.complex128).itemsize
    )

    assert estimates.n_blocks == grid_size
    assert estimates.n_q == n_q
    assert estimates.n_g == n_g
    assert estimates.lambda_blocks_mb == expected_lambda / 1024**2
    assert estimates.dense_tve_mb == expected_tve / 1024**2
    assert estimates.sector_tve_mb == expected_sector_tve / 1024**2


def test_compact_lambdas_round_trip_valley_diagonal_blocks():
    rng = np.random.default_rng(1)
    dense = np.zeros((2, 3, 4, 4, 4), dtype=complex)
    dense[..., :2, :2] = rng.normal(size=(2, 3, 4, 2, 2))
    dense[..., 2:, 2:] = rng.normal(size=(2, 3, 4, 2, 2))

    compact = compact_lambdas_from_dense(dense, n_active=2)
    restored = dense_lambdas_from_compact(compact)

    assert compact.shape == (2, 3, 4, 2, 2, 2)
    assert np.allclose(restored, dense)


def test_packed_hermitian_exchange_matvec_matches_dense():
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(7, 7)) + 1j * rng.normal(size=(7, 7))
    dense = 0.5 * (raw + raw.conj().T)
    x = rng.normal(size=7) + 1j * rng.normal(size=7)

    packed = PackedHermitianExchange.from_dense(dense)

    assert np.allclose(packed.matvec(x), dense @ x)


def test_hartree_only_and_matrix_free_backends_match_dense_backend():
    backend = _tiny_taige_backend()
    rng = np.random.default_rng(3)
    Q = hermitize(rng.normal(size=backend.h0.shape) + 1j * rng.normal(size=backend.h0.shape))
    hartree_only = hartree_only_backend_from_dense(backend)
    matrix_free = matrix_free_backend_from_dense(backend)

    assert np.allclose(hartree_only.fock_hamiltonian(Q), backend.fock_hamiltonian(Q))
    assert np.allclose(hartree_only.hf_hamiltonian(Q), backend.hf_hamiltonian(Q))
    assert np.allclose(matrix_free.fock_hamiltonian(Q), backend.fock_hamiltonian(Q))
    assert np.allclose(matrix_free.hf_hamiltonian(Q), backend.hf_hamiltonian(Q))


def test_benchmark_worker_variants_match_baseline_reference(tmp_path):
    params = _tiny_params()
    reference = tmp_path / "baseline_reference.npz"
    baseline = run_taige_memory_benchmark_worker(
        params=params,
        variant="baseline",
        reference_output=reference,
    )
    assert baseline.summary.skipped is False
    for variant in (
        "hartree_only",
        "fused",
        "compact",
        "fused_compact",
        "compact_dense_exchange",
        "sector_exchange",
        "sector_cached_gather",
        "packed",
        "matrix_free",
    ):
        result = run_taige_memory_benchmark_worker(
            params=params,
            variant=variant,
            reference_input=reference,
        )
        assert result.summary.skipped is False
        assert result.correctness is not None
        assert result.correctness.passed


def test_benchmark_script_smoke_writes_outputs(tmp_path):
    output = tmp_path / "bench"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(output),
            "--n-k-list",
            "2",
            "--variants",
            "baseline,hartree_only",
            "--plane-wave-shell",
            "1",
            "--n-bands",
            "1",
            "--n-active-bands-per-valley",
            "1",
            "--local-field-cutoff",
            "0",
            "--fock-repeats",
            "2",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )

    assert (output / "benchmark_results.json").exists()
    assert (output / "variant_summary.csv").exists()
    assert (output / "stage_measurements.csv").exists()
    assert (output / "correctness.csv").exists()
    assert (output / "benchmark_report.md").exists()
