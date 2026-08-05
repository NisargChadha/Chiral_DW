from __future__ import annotations

import numpy as np
import pytest

from chiral_dw.ac.nonideal import NonIdealACLLModel
from chiral_dw.ac.sewing import ACReciprocalTransport, reciprocal_parity
from chiral_dw.config import FirstShellACParams


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
