from __future__ import annotations

import numpy as np
import pytest

from chiral_dw.continuum.gauge_adapter import ActiveSpaceGaugeAdapter


def _random_hermitian_blocks(seed: int, n_k: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n_k, dim, dim)) + 1j * rng.normal(
        size=(n_k, dim, dim)
    )
    return 0.5 * (raw + raw.conj().swapaxes(-1, -2))


def test_active_space_gauge_adapter_round_trips_general_unitaries():
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))
    unitaries = np.stack([np.linalg.qr(block)[0] for block in raw])
    adapter = ActiveSpaceGaugeAdapter(
        target_index_of_source=np.asarray([2, 0, 1]),
        source_index_in_target_order=np.asarray([2, 3, 0, 1]),
        unitary_target_from_reordered_source=unitaries,
        transpose_source_matrices=True,
    )
    source = _random_hermitian_blocks(7, 3, 4)

    target = adapter.to_target(source)
    recovered = adapter.to_source(target)

    assert np.allclose(recovered, source, atol=1.0e-12, rtol=0.0)
    assert np.allclose(target, target.conj().swapaxes(-1, -2), atol=1.0e-12)


def test_diagonal_phase_adapter_matches_explicit_formula():
    phases = np.asarray(
        [
            [1.0, 1.0j],
            [np.exp(0.2j), np.exp(-0.3j)],
        ]
    )
    adapter = ActiveSpaceGaugeAdapter.from_diagonal_phases(
        target_index_of_source=np.asarray([1, 0]),
        source_index_in_target_order=np.asarray([1, 0]),
        phases_target_order_by_source=phases,
        transpose_source_matrices=True,
    )
    source = _random_hermitian_blocks(2, 2, 2)
    target = adapter.to_target(source)

    for source_index, target_index in enumerate((1, 0)):
        reordered = source[source_index].T[np.ix_([1, 0], [1, 0])]
        expected = (
            phases[source_index, :, None]
            * reordered
            * phases[source_index, None, :].conj()
        )
        assert np.allclose(target[target_index], expected)


def test_active_space_gauge_adapter_rejects_nonunitary_input():
    with pytest.raises(ValueError, match="unitary"):
        ActiveSpaceGaugeAdapter(
            target_index_of_source=np.asarray([0]),
            source_index_in_target_order=np.asarray([0, 1]),
            unitary_target_from_reordered_source=np.ones((1, 2, 2)),
        )
