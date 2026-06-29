import json
from pathlib import Path

import numpy as np

from chiral_dw.ac.kahler import IdealACKahlerModel
from chiral_dw.ac.nonideal import NonIdealACLLModel
from chiral_dw.ac.projected import build_ac_projected_bundle
from chiral_dw.config import (
    ACProjectedHFParams,
    ContinuumGridParams,
    ContinuumHFParams,
    ContinuumInteractionParams,
    FirstShellACParams,
)
from chiral_dw.continuum import (
    TPrimeConstraint,
    ValleyU1Constraint,
    active_basis_frames,
    build_symmetric_hf_references,
    symmetric_convex_path,
)
from chiral_dw.response import k_theta_from_projectors_with_basis


def _small_params() -> ACProjectedHFParams:
    return ACProjectedHFParams(
        grid=ContinuumGridParams(n_k=3),
        ac=FirstShellACParams(b1=0.12, u1=0.04, n_ll=3),
        interaction=ContinuumInteractionParams(
            coulomb_kind="dimensionless_screened",
            v0=0.03,
            q_shell=1,
            local_field_cutoff=1,
            gate_distance=1.5,
        ),
        hf=ContinuumHFParams(max_iter=4, min_iter=1, mixing=0.6, tolerance=1e-9),
        band_diagnostics_n_k=3,
    )


def test_kahler_chi_solves_periodic_poisson_equation():
    ll_model = NonIdealACLLModel(FirstShellACParams(b1=0.15, u1=0.0, n_ll=3))
    G, b_coeff, _A = ll_model.vector_potential_coefficients()
    kahler = IdealACKahlerModel(ll_model.fields)
    sol = kahler.solve_chi_from_fourier(G, b_coeff, n_grid=32)

    G2 = np.sum(sol.g_vectors**2, axis=-1)
    lap_coeff = -G2 * sol.chi_coeffs
    nonzero = G2 > 1e-14

    assert abs(float(np.mean(sol.chi))) < 1e-12
    assert np.allclose(lap_coeff[nonzero], sol.b_coeffs[nonzero], atol=1e-11)
    assert abs(sol.chi_coeffs[0, 0]) < 1e-12


def test_kahler_uniform_limit_has_unit_dimensionless_berry_curvature():
    ll_model = NonIdealACLLModel(FirstShellACParams(b1=0.0, u1=0.0, n_ll=2))
    kahler = IdealACKahlerModel(ll_model.fields)
    sol = kahler.solve_chi_from_fourier(np.empty((0, 2)), np.empty(0), n_grid=16)
    G, coeffs, _phi = kahler.exp2chi_fourier_coeffs(sol)
    b1, b2 = ll_model.fields.G_shell[0], ll_model.fields.G_shell[1]
    pts = np.linspace(0.0, 1.0, 5, endpoint=False)
    uu, vv = np.meshgrid(pts, pts, indexing="ij")
    k = uu[..., None] * b1 + vv[..., None] * b2
    omega = kahler.dimensionless_berry_curvature(k, G, coeffs)

    assert np.allclose(omega, 1.0, atol=1e-12)


def test_ac_projected_bundle_shapes_tprime_and_density_identities():
    bundle = build_ac_projected_bundle(_small_params())
    active = bundle.active
    vertices = bundle.vertices

    assert active.h0.shape == (9, 2, 2)
    assert active.band_vectors.shape == (9, 2, 3, 1)
    assert vertices.lambda_blocks.shape[:3] == (9, 9, 9)
    assert active_basis_frames(active).shape == (9, 6, 2)
    assert TPrimeConstraint(active).symmetry_error(active.h0) < 1e-12
    assert ValleyU1Constraint(active).symmetry_error(vertices.lambda_blocks[0, 0]) < 1e-12

    q0 = vertices.q_shifts.index((0, 0))
    g0 = vertices.g_channels.index((0, 0))
    q0_lambda = vertices.lambda_blocks[q0, g0]
    assert np.allclose(q0_lambda[:, 0, 0], 1.0, atol=1e-12)
    assert np.allclose(q0_lambda[:, 1, 1], 1.0, atol=1e-12)
    assert np.allclose(q0_lambda[:, 0, 1], 0.0, atol=1e-12)
    assert np.allclose(q0_lambda[:, 1, 0], 0.0, atol=1e-12)

    q_plus = vertices.q_shifts.index((1, 0))
    q_minus = vertices.q_shifts.index((-1, 0))
    for ik in range(active.n_k):
        jk = int(vertices.target_minus_q[q_plus, ik])
        lhs = vertices.lambda_blocks[q_plus, g0, ik]
        rhs = vertices.lambda_blocks[q_minus, g0, jk].conj().T
        assert np.allclose(lhs, rhs, atol=1e-10)


def test_ac_projected_bundle_runs_symmetric_hf_and_embedded_response():
    params = _small_params()
    bundle = build_ac_projected_bundle(params)
    refs = build_symmetric_hf_references(bundle, params.hf)

    assert refs.vp_plus.constraint_name == "valley_u1"
    assert refs.vp_minus.constraint_name == "valley_u1"
    assert refs.ivc.constraint_name == "tprime"

    theta = np.linspace(1e-4, np.pi - 1e-4, 5)
    projectors, _diag = symmetric_convex_path(refs, theta)
    n_k = bundle.grid.n_k
    frames = active_basis_frames(bundle.active).reshape(n_k, n_k, -1, bundle.active.dim)
    response = k_theta_from_projectors_with_basis(
        projectors.reshape(len(theta), n_k, n_k, bundle.active.dim, bundle.active.dim),
        theta,
        frames,
    )

    assert np.all(np.isfinite(response.K))
    assert np.isfinite(response.cG)


def test_ac_projected_notebook_is_paired_when_present():
    root = Path(__file__).resolve().parents[1]
    notebook = root / "notebooks" / "ac_adiabatic_projected_hf.ipynb"
    script = root / "notebooks" / "ac_adiabatic_projected_hf.py"
    if not notebook.exists():
        return
    data = json.loads(notebook.read_text())
    assert data["metadata"]["jupytext"]["formats"] == "ipynb,py:percent"
    assert script.exists()
    text = script.read_text()
    assert "build_ac_projected_bundle" in text
    assert "TPrimeConstraint" in text
    assert "ValleyU1Constraint" in text
    assert "k_theta_from_projectors_with_basis" in text
