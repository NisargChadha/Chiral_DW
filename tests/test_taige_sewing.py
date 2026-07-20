from __future__ import annotations

import numpy as np
import pytest

from chiral_dw.continuum.models import MomentumGrid
from chiral_dw.continuum.taige_sewing import TaigeReciprocalTransport


SHELL = ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))


def test_folded_to_raw_maps_g_plus_shift_without_moving_internal_blocks() -> None:
    transport = TaigeReciprocalTransport(SHELL, (1, 0))
    frame = np.arange(2 * len(SHELL), dtype=float).reshape(2 * len(SHELL), 1)

    moved = transport.folded_to_raw_vectors(frame).reshape(2, len(SHELL))
    original = frame.reshape(2, len(SHELL))

    shell_index = {g: i for i, g in enumerate(SHELL)}
    for block in range(2):
        for source, g in enumerate(SHELL):
            target = shell_index.get((g[0] + 1, g[1]))
            expected = 0.0 if target is None else original[block, target]
            assert moved[block, source] == expected


def test_transport_preserves_physical_momentum_identity() -> None:
    shift = (1, -1)
    transport = TaigeReciprocalTransport(SHELL, shift)
    k_fold = np.array([0.2, 0.3])
    k_raw = k_fold + np.asarray(shift)
    for source, target in zip(transport.source_indices, transport.target_indices, strict=True):
        assert np.allclose(k_raw + SHELL[source], k_fold + SHELL[target])


def test_sewn_overlap_matches_explicit_gather_and_conjugation_commutes() -> None:
    rng = np.random.default_rng(12)
    transport = TaigeReciprocalTransport(SHELL, (0, 1))
    left = rng.normal(size=2 * len(SHELL)) + 1j * rng.normal(size=2 * len(SHELL))
    right = rng.normal(size=2 * len(SHELL)) + 1j * rng.normal(size=2 * len(SHELL))

    expected = np.vdot(left, transport.folded_to_raw_vectors(right))
    assert transport.sewn_overlap(left, right) == pytest.approx(expected)
    assert np.allclose(
        transport.folded_to_raw_vectors(right.conj()),
        transport.folded_to_raw_vectors(right).conj(),
    )


def test_operator_transport_preserves_hermiticity_and_hole_convention() -> None:
    rng = np.random.default_rng(13)
    transport = TaigeReciprocalTransport(SHELL, (-1, 0))
    raw = rng.normal(size=(2 * len(SHELL), 2 * len(SHELL)))
    h_electron = raw + raw.T
    h_hole = -h_electron.conj()

    moved_electron = transport.folded_to_raw_operator(h_electron)
    moved_hole = transport.folded_to_raw_operator(h_hole)

    assert np.allclose(moved_electron, moved_electron.conj().T)
    assert np.allclose(moved_hole, -moved_electron.conj())


def test_direct_sum_operator_blocks_are_not_swapped() -> None:
    transport = TaigeReciprocalTransport(SHELL, (1, 0))
    dimension = 4 * len(SHELL)  # two valleys times two layers
    operator = np.zeros((dimension, dimension), dtype=complex)
    for block in range(4):
        sl = slice(block * len(SHELL), (block + 1) * len(SHELL))
        operator[sl, sl] = (block + 1) * np.eye(len(SHELL))

    moved = transport.folded_to_raw_operator(operator)
    for block in range(4):
        sl = slice(block * len(SHELL), (block + 1) * len(SHELL))
        diagonal = np.diag(moved[sl, sl])
        assert set(np.unique(diagonal)).issubset({0.0, float(block + 1)})
        assert np.count_nonzero(diagonal) == transport.matched_plane_waves


def test_matched_projector_is_retained_but_edge_weight_loss_is_reported() -> None:
    transport = TaigeReciprocalTransport(SHELL, (1, 0))
    matched_target = int(transport.target_indices[0])
    matched = np.zeros((len(SHELL), 1), dtype=complex)
    matched[matched_target] = 1.0
    diagnostics = transport.frame_diagnostics(matched)
    assert diagnostics.min_retained_state_weight == pytest.approx(1.0)
    assert diagnostics.max_gram_loss == pytest.approx(0.0)

    unmatched = next(index for index in range(len(SHELL)) if index not in transport.target_indices)
    edge = np.zeros((len(SHELL), 1), dtype=complex)
    edge[unmatched] = 1.0
    diagnostics = transport.frame_diagnostics(edge)
    assert diagnostics.min_retained_state_weight == pytest.approx(0.0)
    assert diagnostics.max_gram_loss == pytest.approx(1.0)


def test_mesh_edge_shift_orientations() -> None:
    grid = MomentumGrid(4)
    assert grid.shift_plus_q((3, 2), (1, 0)) == ((0, 2), (1, 0))
    assert grid.shift_plus_q((0, 2), (-1, 0)) == ((3, 2), (-1, 0))
    assert grid.shift_plus_q((2, 3), (0, 1)) == ((2, 0), (0, 1))
    assert grid.shift_plus_q((2, 0), (0, -1)) == ((2, 3), (0, -1))


def test_invalid_dimensions_are_rejected() -> None:
    transport = TaigeReciprocalTransport(SHELL, (0, 0))
    with pytest.raises(ValueError, match="multiple of the shell size"):
        transport.folded_to_raw_vectors(np.zeros(3))
    with pytest.raises(ValueError, match="square"):
        transport.folded_to_raw_operator(np.zeros((5, 4)))
