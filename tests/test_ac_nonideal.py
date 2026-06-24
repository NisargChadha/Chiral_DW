import numpy as np

from chiral_dw.ac.adiabatic import AdiabaticMoireFields
from chiral_dw.ac.nonideal import (
    NonIdealACLLModel,
    first_shell_magnetic_coefficients,
    first_shell_potential_coefficients,
    fourier_params_from_first_shell,
    landau_polynomial,
)
from chiral_dw.config import FirstShellACParams, TMoTe2ACParams


def test_landau_polynomial_small_values():
    x = 0.3 + 0.2j
    y = -0.1 + 0.5j

    assert landau_polynomial(0, 0, x, y) == 1
    assert np.allclose(landau_polynomial(1, 0, x, y), -y)
    assert np.allclose(landau_polynomial(0, 1, x, y), -x)
    assert np.allclose(landau_polynomial(1, 1, x, y), x * y - 1)


def test_first_shell_coefficients_are_real_c3_and_break_c6():
    coeffs = first_shell_potential_coefficients(u1=0.02, u1_c3=0.11)

    for idx, partner in [(0, 3), (1, 4), (2, 5)]:
        assert np.allclose(coeffs[partner], coeffs[idx].conjugate())
    assert np.allclose(coeffs[[0, 2, 4]], coeffs[0])
    assert np.allclose(coeffs[[1, 3, 5]], coeffs[1])
    assert not np.allclose(coeffs[0], coeffs[1])
    assert np.allclose(first_shell_magnetic_coefficients(0.02, 0.11), coeffs)


def test_first_shell_field_has_zero_average_and_coulomb_gauge():
    model = NonIdealACLLModel(FirstShellACParams(b1=0.2, u1=0.0, n_ll=5))
    G, B_coeff, A_coeff = model.vector_potential_coefficients()

    divergence = 1j * np.sum(G * A_coeff, axis=1)
    curl = 1j * (G[:, 0] * A_coeff[:, 1] - G[:, 1] * A_coeff[:, 0])
    assert np.allclose(divergence, 0.0, atol=1e-12)
    assert np.allclose(curl, B_coeff, atol=1e-12)
    assert np.isclose(np.sum(B_coeff).real, -12.0 * np.pi / model.fields.unit_cell_area * 0.2)


def test_hamiltonian_is_hermitian():
    params = FirstShellACParams(b1=0.15, b1_c3=0.03, u1=-0.08, u1_c3=0.04, n_ll=7)
    model = NonIdealACLLModel(params)
    H = model.hamiltonian(np.array([0.13, -0.27]))

    assert np.allclose(H, H.conj().T, atol=1e-12)


def test_fourier_params_reproduce_first_shell_hamiltonian():
    first_shell = FirstShellACParams(b1=0.15, b1_c3=0.03, u1=-0.08, u1_c3=0.04, n_ll=5)
    model_first = NonIdealACLLModel(first_shell)
    model_fourier = NonIdealACLLModel(fourier_params_from_first_shell(first_shell))
    k = np.array([0.13, -0.27])

    assert np.allclose(model_first.hamiltonian(k), model_fourier.hamiltonian(k), atol=1e-12)


def test_density_form_factor_identities():
    model = NonIdealACLLModel(FirstShellACParams(b1=0.1, u1=0.05, n_ll=5))
    k = np.array([0.17, -0.11])
    p = np.array([-0.07, 0.23])
    G = model.fields.G_shell[0] - model.fields.G_shell[1]

    assert np.allclose(
        model.density_form_factor_matrix(k, p, np.zeros(2)),
        model.basis_overlap_matrix(k, p),
        atol=1e-12,
    )
    lhs = model.density_form_factor_matrix(k, p, G)
    rhs = model.density_form_factor_matrix(p, k, -G).conj().T
    assert np.allclose(lhs, rhs, atol=1e-12)


def test_zero_harmonics_flat_uniform_chern_one():
    model = NonIdealACLLModel(FirstShellACParams(b1=0.0, u1=0.0, n_ll=5))
    diag = model.band_diagnostics(n_k=8)

    assert diag["bandwidth"] < 1e-12
    assert diag["min_direct_gap"] > 0.9
    assert np.isclose(diag["chern"], 1.0, atol=5e-3)
    assert diag["berry_std"] < 1e-2


def test_projectors_and_time_reversal_partner():
    model = NonIdealACLLModel(FirstShellACParams(b1=0.1, u1=0.05, n_ll=5))
    k = np.array([0.2, 0.1])
    P = model.solve(k).projector
    P_down = model.projector_down_from_up(-k)
    P_expected = model.solve(k).projector.conj()

    assert np.allclose(P, P.conj().T, atol=1e-12)
    assert np.allclose(P @ P, P, atol=1e-12)
    assert np.isclose(np.trace(P), 1.0)
    assert np.allclose(P_down, P_expected, atol=1e-12)


def test_tmote2_adiabatic_fields_and_fourier_backend_smoke():
    fields = AdiabaticMoireFields(TMoTe2ACParams(grid_size=32, n_ll=3))
    _, _, rr = fields.primitive_grid(16)
    dimless_b = fields.dimensionless_effective_magnetic_field(rr)
    assert np.isclose(float(np.mean(dimless_b)), 1.0, atol=2e-2)
    assert np.isclose(fields.omega_c_mev(1.0), 2.27, rtol=0.03)

    model = NonIdealACLLModel(TMoTe2ACParams(theta_deg=3.7, n_ll=3, grid_size=32, g_shell_cutoff=1))
    G, U, B = model.fourier_coefficients()
    diag = model.band_diagnostics(n_k=4)

    assert G.shape[1] == 2
    assert len(G) > 0
    assert np.all(np.isfinite(U))
    assert np.all(np.isfinite(B))
    assert np.isfinite(diag["chern"])
    assert diag["min_direct_gap"] > 0.0
