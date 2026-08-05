from __future__ import annotations

import numpy as np
import pytest

from chiral_dw.ac.nonideal import NonIdealACLLModel
from chiral_dw.ac.projected import build_ac_active_space
from chiral_dw.ac.response import ACBandOverlapProvider
from chiral_dw.ac.sewing import ACReciprocalTransport, reciprocal_parity
from chiral_dw.config import FirstShellACParams
from chiral_dw.continuum.models import MomentumGrid


def _model(n_ll: int = 4) -> NonIdealACLLModel:
    return NonIdealACLLModel(
        FirstShellACParams(b1=0.07, u1=-0.03, n_ll=n_ll)
    )


def test_reciprocal_parity_matches_magnetic_lattice_parity() -> None:
    assert reciprocal_parity((0, 0)) == 1.0
    assert reciprocal_parity((2, 0)) == 1.0
    assert reciprocal_parity((-2, 4)) == 1.0
    assert reciprocal_parity((1, 0)) == -1.0
    assert reciprocal_parity((0, -1)) == -1.0
    assert reciprocal_parity((1, 1)) == -1.0


def test_active_sewing_is_unitary_with_conjugate_valley_phases() -> None:
    model = _model()
    transport = ACReciprocalTransport(model)
    b1, b2 = model.fields.G_shell[:2]
    k = 0.23 * b1 + 0.31 * b2

    sewing = transport.active_sewing_matrix(k, (1, -1))

    assert np.allclose(sewing.conj().T @ sewing, np.eye(2), atol=1e-14)
    assert sewing[0, 1] == 0.0
    assert sewing[1, 0] == 0.0
    assert sewing[1, 1] == pytest.approx(np.conj(sewing[0, 0]))


def test_fractional_folding_handles_positive_negative_and_exact_boundaries() -> None:
    transport = ACReciprocalTransport(_model())

    folded, shift = transport.fold_fractional(np.array([1.0, -0.25]))
    assert np.allclose(folded, [0.0, 0.75])
    assert shift == (1, -1)

    folded, shift = transport.fold_fractional(np.array([-1e-14, 1.0 + 1e-14]))
    assert np.allclose(folded, [0.0, 0.0])
    assert shift == (0, 1)


@pytest.mark.parametrize(
    ("first", "second"),
    [((1, 0), (0, 1)), ((0, 1), (1, 0)), ((-1, 1), (2, -1))],
)
def test_active_sewing_obeys_magnetic_cocycle(
    first: tuple[int, int],
    second: tuple[int, int],
) -> None:
    model = _model()
    transport = ACReciprocalTransport(model)
    b1, b2 = model.fields.G_shell[:2]
    k = 0.17 * b1 + 0.29 * b2

    assert transport.cocycle_residual(k, first, second) < 2e-14


def test_projector_transport_composes_and_preserves_projector_properties() -> None:
    model = _model()
    transport = ACReciprocalTransport(model)
    b1, b2 = model.fields.G_shell[:2]
    k = 0.19 * b1 + 0.37 * b2
    spinor = np.array([np.sqrt(0.4), np.sqrt(0.6) * np.exp(0.31j)])
    projector = np.outer(spinor, spinor.conj())

    first = (1, 0)
    second = (0, 1)
    once = transport.folded_to_raw_projector(projector, k, first)
    twice = transport.folded_to_raw_projector(
        once,
        k + transport.reciprocal_vector(first),
        second,
    )
    direct = transport.folded_to_raw_projector(projector, k, (1, 1))

    assert np.allclose(twice, direct, atol=2e-14)
    assert np.allclose(direct, direct.conj().T, atol=1e-14)
    assert np.allclose(direct @ direct, direct, atol=1e-14)
    assert np.trace(direct) == pytest.approx(1.0)


@pytest.mark.parametrize("n_ll", [1, 2, 5])
def test_ll_form_factor_is_covariant_when_right_endpoint_is_shifted(n_ll: int) -> None:
    model = _model(n_ll=n_ll)
    transport = ACReciprocalTransport(model)
    b1, b2 = model.fields.G_shell[:2]
    k = 0.23 * b1 + 0.31 * b2
    p = 0.61 * b1 + 0.17 * b2
    operator_G = b1 - b2
    shift = (1, 0)
    reciprocal = transport.reciprocal_vector(shift)

    raw = model.density_form_factor_matrix(k, p + reciprocal, operator_G)
    expected = transport.valley_phase(p, shift, valley=1) * model.density_form_factor_matrix(
        k,
        p,
        operator_G - reciprocal,
    )

    assert np.allclose(raw, expected, atol=2e-14)


@pytest.mark.parametrize("n_ll", [1, 2, 5])
def test_ll_form_factor_is_covariant_when_left_endpoint_is_shifted(n_ll: int) -> None:
    model = _model(n_ll=n_ll)
    transport = ACReciprocalTransport(model)
    b1, b2 = model.fields.G_shell[:2]
    k = 0.23 * b1 + 0.31 * b2
    p = 0.61 * b1 + 0.17 * b2
    operator_G = b1 - b2
    shift = (0, -1)
    reciprocal = transport.reciprocal_vector(shift)

    raw = model.density_form_factor_matrix(k + reciprocal, p, operator_G)
    expected = np.conj(
        transport.valley_phase(k, shift, valley=1)
    ) * model.density_form_factor_matrix(
        k,
        p,
        operator_G + reciprocal,
    )

    assert np.allclose(raw, expected, atol=2e-14)


def test_sewn_active_overlap_is_invariant_under_translated_edge_representations() -> None:
    model = _model(n_ll=4)
    grid = MomentumGrid(6)
    active, _bands = build_ac_active_space(
        model,
        grid,
        active_band=0,
        diagnostics_n_k=3,
    )
    provider = ACBandOverlapProvider(model, active=active)

    for j in range(grid.n_k):
        start = np.array([0.0, j / grid.n_k])
        stop = np.array([0.0, (j + 1) / grid.n_k])
        baseline = provider.sewn_active_overlap_fractional(start, stop)
        translated = provider.sewn_active_overlap_fractional(
            start + np.array([1.0, 0.0]),
            stop + np.array([1.0, 0.0]),
        )
        assert np.allclose(translated, baseline, atol=2e-14)

    for i in range(grid.n_k):
        start = np.array([i / grid.n_k, 0.0])
        stop = np.array([(i + 1) / grid.n_k, 0.0])
        baseline = provider.sewn_active_overlap_fractional(start, stop)
        translated = provider.sewn_active_overlap_fractional(
            start + np.array([0.0, 1.0]),
            stop + np.array([0.0, 1.0]),
        )
        assert np.allclose(translated, baseline, atol=2e-14)


def test_sewn_active_overlap_requires_the_stored_active_frame() -> None:
    provider = ACBandOverlapProvider(_model())
    with pytest.raises(ValueError, match="active-space band frame"):
        provider.sewn_active_overlap_fractional((0.0, 0.0), (0.1, 0.0))
