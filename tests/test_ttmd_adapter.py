from types import SimpleNamespace

import numpy as np
import pytest

from chiral_dw.config import SourceInterpolationParams, TMDHFReferenceParams
from chiral_dw.ttmd_adapter import (
    TMDHFUnavailableError,
    TTMDReferenceProjectors,
    diagnose_reference_projectors,
    endpoint_diagnostics,
    flavor_u1_rotation,
    projector_distance,
    references_from_tmd_hf_bundle,
    require_tmd_hf,
    rotate_flavor_blocks,
    source_interpolation_hamiltonian,
    source_interpolation_path,
    source_interpolation_projector,
)

SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


def _toy_refs(n_blocks: int = 4) -> TTMDReferenceProjectors:
    h0 = np.zeros((n_blocks, 2, 2), dtype=complex)
    z_vp = np.array([1.0, 0.0], dtype=complex)
    z_ivc = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    P_vp = np.broadcast_to(np.outer(z_vp, z_vp.conj()), h0.shape).copy()
    P_ivc = np.broadcast_to(np.outer(z_ivc, z_ivc.conj()), h0.shape).copy()
    delta_vp = np.broadcast_to(-SIGMA_Z, h0.shape).copy()
    delta_ivc = np.broadcast_to(-SIGMA_X, h0.shape).copy()
    return TTMDReferenceProjectors(
        h0=h0,
        P_vp=P_vp,
        P_ivc=P_ivc,
        delta_vp=delta_vp,
        delta_ivc=delta_ivc,
        n_occ_per_block=1,
    )


def test_import_guard_reports_editable_tmd_hf_hint():
    def missing_importer(name: str):
        raise ModuleNotFoundError(name)

    with pytest.raises(TMDHFUnavailableError) as excinfo:
        require_tmd_hf(importer=missing_importer)

    message = str(excinfo.value)
    assert "/Users/nisargchadha/Documents/TMD_HF" in message
    assert "pip install -e" in message


def test_reference_shape_validation_and_raw_field_diagnostics():
    refs = _toy_refs()
    diagnostics = diagnose_reference_projectors(refs)

    assert diagnostics.h0_hermiticity_error == 0.0
    assert diagnostics.vp_projector_idempotency_error < 1e-14
    assert diagnostics.ivc_projector_idempotency_error < 1e-14
    assert diagnostics.delta_vp_scalar_norm == pytest.approx(0.0)
    assert diagnostics.delta_vp_traceless_norm > 0.0
    assert diagnostics.delta_ivc_offdiag_norm > 0.0

    with pytest.raises(ValueError):
        TTMDReferenceProjectors(
            h0=refs.h0,
            P_vp=refs.P_vp[:, :1, :1],
            P_ivc=refs.P_ivc,
            delta_vp=refs.delta_vp,
            delta_ivc=refs.delta_ivc,
        )


def test_source_interpolation_endpoints_match_reference_projectors():
    refs = _toy_refs()
    controls = SourceInterpolationParams(source_scale=1.0, n_occ_per_block=1)

    vp = source_interpolation_projector(refs, 0.0, 0.0, controls)
    ivc = source_interpolation_projector(refs, 0.5 * np.pi, 0.0, controls)
    endpoints = endpoint_diagnostics(refs, controls)

    assert projector_distance(vp.P, refs.P_vp)["relative_frobenius"] < 1e-12
    assert projector_distance(ivc.P, refs.P_ivc)["relative_frobenius"] < 1e-12
    assert endpoints.theta0_vs_vp_relative_frobenius < 1e-12
    assert endpoints.theta_pi_over_2_vs_ivc_relative_frobenius < 1e-12
    assert vp.diagnostics.direct_gap_min == pytest.approx(2.0)
    assert ivc.diagnostics.trace_mean == pytest.approx(1.0)


def test_phi_rotation_covariance_for_ivc_source():
    refs = _toy_refs()
    phi = np.pi / 3.0

    result_phi = source_interpolation_projector(refs, 0.5 * np.pi, phi)
    result_zero = source_interpolation_projector(refs, 0.5 * np.pi, 0.0)
    rotated_zero = rotate_flavor_blocks(result_zero.P, phi)
    periodic = source_interpolation_projector(refs, 0.5 * np.pi, 2.0 * np.pi)

    assert np.allclose(result_phi.P, rotated_zero, atol=1e-12)
    assert np.allclose(periodic.P, result_zero.P, atol=1e-12)

    U = flavor_u1_rotation(2, phi)
    assert np.allclose(U.conj().T @ U, np.eye(2))


def test_references_from_tmd_hf_bundle_use_raw_hf_fields():
    refs = _toy_refs()

    class FakeBackend:
        h0 = refs.h0

        def as_block_density(self, P):
            return np.asarray(P, dtype=complex)

        def hf_hamiltonian(self, P):
            if np.allclose(P, refs.P_vp):
                return self.h0 + refs.delta_vp
            if np.allclose(P, refs.P_ivc):
                return self.h0 + refs.delta_ivc
            raise AssertionError("unexpected projector")

    bundle = SimpleNamespace(active=SimpleNamespace(n_k=refs.n_blocks), backend=FakeBackend())
    built = references_from_tmd_hf_bundle(
        bundle,
        P_vp=refs.P_vp,
        P_ivc=refs.P_ivc,
        params=TMDHFReferenceParams(n_occ_per_block=1),
    )

    assert np.allclose(built.delta_vp, refs.delta_vp)
    assert np.allclose(built.delta_ivc, refs.delta_ivc)
    assert built.metadata["source_convention"] == "Delta=H_HF(P)-H0"


def test_small_source_interpolation_path_smoke():
    refs = _toy_refs(n_blocks=3)
    theta = np.linspace(0.0, np.pi, 5)
    projectors, diagnostics = source_interpolation_path(refs, theta)
    H_mid = source_interpolation_hamiltonian(refs, theta[2])

    assert projectors.shape == (5, 3, 2, 2)
    assert len(diagnostics) == 5
    assert np.all(np.isfinite([row.direct_gap_min for row in diagnostics]))
    assert np.allclose(H_mid, H_mid.conj().swapaxes(-1, -2))
    assert max(row.projector_idempotency_error for row in diagnostics) < 1e-12
