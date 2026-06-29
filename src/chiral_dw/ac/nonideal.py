"""Nonideal finite-Landau-level Aharonov-Casher backend."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, factorial

import numpy as np
import scipy.linalg

from chiral_dw.ac.adiabatic import AdiabaticMoireFields
from chiral_dw.config import FirstShellACParams, FourierACParams, FourierCoefficient, TMoTe2ACParams

ACParams = FirstShellACParams | FourierACParams | TMoTe2ACParams


@dataclass(frozen=True)
class BandSolution:
    """Eigenvalues/eigenvectors and a rank-one projector at one k point."""

    k: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    active_band: int
    projector: np.ndarray
    direct_gap: float


def _complex_from_xy(v: np.ndarray) -> complex:
    return complex(float(v[0]), float(v[1]))


def cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def landau_polynomial(n: int, nprime: int, x: complex, y: complex) -> complex:
    """Return L_{n,n'}(x,y)=exp(xy) d_x^n d_y^n' exp(-xy)."""
    if n < 0 or nprime < 0:
        return 0.0 + 0.0j
    total = 0.0 + 0.0j
    for j in range(min(n, nprime) + 1):
        coeff = ((-1) ** (n + nprime - j)) * comb(n, j) * factorial(nprime) / factorial(nprime - j)
        total += coeff * (x ** (nprime - j)) * (y ** (n - j))
    return total


def first_shell_potential_coefficients(u1: float, u1_c3: float = 0.0) -> np.ndarray:
    """Return real-space-real C3 first-shell potential coefficients."""
    coeff = np.empty(6, dtype=complex)
    coeff[[0, 2, 4]] = complex(float(u1), float(u1_c3))
    coeff[[1, 3, 5]] = complex(float(u1), -float(u1_c3))
    return coeff


def first_shell_magnetic_coefficients(b1: float, b1_c3: float = 0.0) -> np.ndarray:
    """Return coefficients for -B'(r) A_M/(2*pi)."""
    return first_shell_potential_coefficients(b1, b1_c3)


def second_harmonic_vectors(fields: AdiabaticMoireFields) -> np.ndarray:
    """Return the six second-harmonic reciprocal vectors 2*G_j."""
    return 2.0 * fields.G_shell.copy()


def second_harmonic_potential_coefficients(u2: float) -> np.ndarray:
    """Return real-space-real C3 second-harmonic potential coefficients."""
    return first_shell_potential_coefficients(u2, 0.0)


def second_harmonic_magnetic_coefficients(b2: float) -> np.ndarray:
    """Return second-harmonic coefficients for -B'(r) A_M/(2*pi)."""
    return first_shell_potential_coefficients(b2, 0.0)


def _low_harmonic_dimensionless_coefficients(
    fields: AdiabaticMoireFields, params: FirstShellACParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    G_parts = [fields.G_shell.copy()]
    U_parts = [first_shell_potential_coefficients(params.u1, params.u1_c3)]
    b_parts = [first_shell_magnetic_coefficients(params.b1, params.b1_c3)]
    if abs(params.u2) > 0.0 or abs(params.b2) > 0.0:
        G_parts.append(second_harmonic_vectors(fields))
        U_parts.append(second_harmonic_potential_coefficients(params.u2))
        b_parts.append(second_harmonic_magnetic_coefficients(params.b2))
    return np.concatenate(G_parts), np.concatenate(U_parts), np.concatenate(b_parts)


def fourier_params_from_first_shell(params: FirstShellACParams) -> FourierACParams:
    """Convert first-shell parameters to explicit physical Fourier coefficients."""
    fields = AdiabaticMoireFields(TMoTe2ACParams(a_m=params.a_m, n_ll=params.n_ll))
    G, U_coeff, b_dimless_coeff = _low_harmonic_dimensionless_coefficients(fields, params)
    B_coeff = -(2.0 * np.pi / fields.unit_cell_area) * b_dimless_coeff
    return FourierACParams(
        g_vectors=tuple((float(g[0]), float(g[1])) for g in G),
        u_coefficients=tuple(FourierCoefficient.from_complex(z) for z in U_coeff),
        b_coefficients=tuple(FourierCoefficient.from_complex(z) for z in B_coeff),
        n_ll=params.n_ll,
        material=params.material,
        a_m=params.a_m,
    )


class NonIdealACLLModel:
    """Single-particle nonideal AC model in the average-field LL basis."""

    def __init__(self, params: ACParams | None = None) -> None:
        self.params: ACParams = params or FirstShellACParams()
        if self.params.n_ll < 1:
            raise ValueError("n_ll must be >= 1")
        if isinstance(self.params, TMoTe2ACParams):
            self.fields = AdiabaticMoireFields(self.params)
        else:
            self.fields = AdiabaticMoireFields(TMoTe2ACParams(a_m=self.params.a_m, n_ll=self.params.n_ll))
        self.l2 = self.fields.unit_cell_area / (2.0 * np.pi)
        self.bz_area = 4.0 * np.pi**2 / self.fields.unit_cell_area
        self.first_shell = self.fields.G_shell.copy()
        self._fourier_cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    @property
    def n_ll(self) -> int:
        return int(self.params.n_ll)

    def first_shell_fourier_coefficients(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not isinstance(self.params, FirstShellACParams):
            raise TypeError("first_shell_fourier_coefficients requires FirstShellACParams")
        G, U_coeff, b_dimless_coeff = _low_harmonic_dimensionless_coefficients(
            self.fields, self.params
        )
        B_over_2m_coeff = -0.5 * b_dimless_coeff
        return G, U_coeff, B_over_2m_coeff

    @staticmethod
    def _shell_index(m: np.ndarray, n: np.ndarray) -> np.ndarray:
        return np.maximum.reduce([np.abs(m), np.abs(n), np.abs(m + n)])

    def _tmote2_fourier_coefficients(
        self, params: TMoTe2ACParams
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_grid = int(params.grid_size)
        _, _, rr = self.fields.primitive_grid(n_grid)
        b_field = self.fields.effective_magnetic_field(rr)
        b_average = float(np.mean(b_field))
        b0_abs = abs(b_average)
        if b0_abs < 1e-14:
            raise ValueError("average effective magnetic field is too small")
        xi = self.fields.dimensionless_xi(rr)
        omega_c = self.fields.omega_c_mev(params.theta_deg)
        U_mev = self.fields.delta_plus_aligned(rr) - omega_c * xi
        b_coeffs = np.fft.fft2(b_field - b_average) / n_grid**2
        U_coeffs = np.fft.fft2(U_mev - float(np.mean(U_mev))) / n_grid**2 / omega_c
        mm, nn, G = self.fields.reciprocal_fft_data(n_grid)
        shells = self._shell_index(mm, nn)
        nonzero = np.sum(G**2, axis=-1) > 1e-14
        in_shell = shells <= int(params.g_shell_cutoff)
        B_over_2m = b_coeffs / (2.0 * b0_abs)
        combined_scale = max(float(np.max(np.abs(U_coeffs))), float(np.max(np.abs(B_over_2m))), 1.0)
        large_enough = (np.abs(U_coeffs) + np.abs(B_over_2m)) > float(params.coefficient_cutoff) * combined_scale
        keep = nonzero & in_shell & large_enough
        return G[keep], U_coeffs[keep], B_over_2m[keep], b_coeffs[keep]

    def fourier_coefficients(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return G, U_G/omega_c, and B'_G/(2m omega_c)."""
        if isinstance(self.params, FirstShellACParams):
            return self.first_shell_fourier_coefficients()
        if self._fourier_cache is None:
            if isinstance(self.params, FourierACParams):
                G = np.asarray(self.params.g_vectors, dtype=float)
                U = np.asarray([z.as_complex() for z in self.params.u_coefficients], dtype=complex)
                B = np.asarray([z.as_complex() for z in self.params.b_coefficients], dtype=complex)
                b0_abs = 2.0 * np.pi / self.fields.unit_cell_area
                self._fourier_cache = (G, U, B / (2.0 * b0_abs), B)
            elif isinstance(self.params, TMoTe2ACParams):
                self._fourier_cache = self._tmote2_fourier_coefficients(self.params)
            else:  # pragma: no cover
                raise TypeError(f"unsupported AC parameter type: {type(self.params)!r}")
        G, U, B_over_2m, _ = self._fourier_cache
        return G, U, B_over_2m

    def vector_potential_coefficients(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return G, B'_G, and Coulomb-gauge A'_G."""
        if isinstance(self.params, FirstShellACParams):
            G, _, b_dimless_coeff = _low_harmonic_dimensionless_coefficients(
                self.fields, self.params
            )
            B_coeff = -(2.0 * np.pi / self.fields.unit_cell_area) * b_dimless_coeff
        else:
            self.fourier_coefficients()
            assert self._fourier_cache is not None
            G, _, _, B_coeff = self._fourier_cache
        G2 = np.sum(G**2, axis=1)
        A = np.zeros((len(G), 2), dtype=complex)
        A[:, 0] = 1j * B_coeff * G[:, 1] / G2
        A[:, 1] = -1j * B_coeff * G[:, 0] / G2
        return G, B_coeff, A

    def _eta_bar(self, G: np.ndarray) -> float:
        basis = np.column_stack([self.fields.G_shell[0], self.fields.G_shell[1]])
        coords = np.linalg.solve(basis, G)
        rounded = np.rint(coords).astype(int)
        if np.allclose(coords, rounded, atol=1e-8) and np.all(rounded % 2 == 0):
            return 1.0
        return -1.0

    def _hprime_element(
        self,
        n: int,
        nprime: int,
        G: np.ndarray,
        U_G: complex,
        B_G_over_2m: complex,
        k: np.ndarray,
    ) -> complex:
        gamma = np.sqrt(self.l2 / 2.0) * _complex_from_xy(G)
        gamma_star = np.conj(gamma)
        L = landau_polynomial(n, nprime, gamma_star, gamma)
        B_term = 0.0 + 0.0j
        if n > 0:
            B_term += n * landau_polynomial(n - 1, nprime, gamma_star, gamma) / gamma_star
        if nprime > 0:
            B_term += nprime * landau_polynomial(n, nprime - 1, gamma_star, gamma) / gamma
        B_term *= B_G_over_2m
        pref = ((-1) ** n) / np.sqrt(float(factorial(n) * factorial(nprime)))
        lamb = np.exp(-self.l2 * np.dot(G, G) / 4.0)
        phase = np.exp(1j * self.l2 * cross2(G, k))
        return pref * self._eta_bar(G) * lamb * (L * U_G + B_term) * phase

    def hamiltonian(self, k: np.ndarray) -> np.ndarray:
        """Build the dimensionless LL-basis Hamiltonian at k."""
        k = np.asarray(k, dtype=float)
        H = np.diag(np.arange(self.n_ll, dtype=float)).astype(complex)
        Gs, U_coeffs, B_coeffs = self.fourier_coefficients()
        for G, U_G, B_G in zip(Gs, U_coeffs, B_coeffs):
            for n in range(self.n_ll):
                for np_ in range(self.n_ll):
                    H[n, np_] += self._hprime_element(n, np_, G, U_G, B_G, k)
        return 0.5 * (H + H.conj().T)

    def solve(self, k: np.ndarray, active_band: int = 0) -> BandSolution:
        H = self.hamiltonian(k)
        vals, vecs = scipy.linalg.eigh(H)
        v = vecs[:, active_band]
        projector = np.outer(v, v.conj())
        gap = float(vals[active_band + 1] - vals[active_band]) if active_band + 1 < len(vals) else np.inf
        return BandSolution(
            k=np.asarray(k, dtype=float),
            eigenvalues=vals,
            eigenvectors=vecs,
            active_band=int(active_band),
            projector=projector,
            direct_gap=gap,
        )

    def density_form_factor_matrix(self, k: np.ndarray, p: np.ndarray, G: np.ndarray) -> np.ndarray:
        """Return <u_k^n|exp(i G.r)|u_p^n'> in the LL basis."""
        k = np.asarray(k, dtype=float)
        p = np.asarray(p, dtype=float)
        G = np.asarray(G, dtype=float)
        delta = G + k - p
        gamma = np.sqrt(self.l2 / 2.0) * _complex_from_xy(delta)
        gamma_star = np.conj(gamma)
        kc = _complex_from_xy(k)
        pc = _complex_from_xy(p)
        Gc = _complex_from_xy(G)
        exponent = self.l2 * 0.5 * ((kc + Gc).conjugate() * pc - kc.conjugate() * Gc)
        exponent -= self.l2 * 0.25 * (np.dot(G, G) + np.dot(k, k) + np.dot(p, p))
        base = np.exp(exponent)
        S = np.zeros((self.n_ll, self.n_ll), dtype=complex)
        for n in range(self.n_ll):
            for np_ in range(self.n_ll):
                pref = ((-1) ** n) / np.sqrt(float(factorial(n) * factorial(np_)))
                S[n, np_] = (
                    self._eta_bar(G)
                    * pref
                    * landau_polynomial(n, np_, gamma_star, gamma)
                    * base
                )
        return S

    def basis_overlap_matrix(self, k: np.ndarray, p: np.ndarray) -> np.ndarray:
        return self.density_form_factor_matrix(k, p, np.zeros(2))

    def state_overlap(
        self, k: np.ndarray, coeffs_k: np.ndarray, p: np.ndarray, coeffs_p: np.ndarray
    ) -> complex:
        return coeffs_k.conj() @ self.basis_overlap_matrix(k, p) @ coeffs_p

    def projector_down_from_up(self, k: np.ndarray, active_band: int = 0) -> np.ndarray:
        """Time-reversed K' projector P_down(k)=P_up(-k)^*."""
        return self.solve(-np.asarray(k, dtype=float), active_band=active_band).projector.conj()

    def fold_to_reciprocal_wigner_seitz(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        original_shape = k.shape
        flat = k.reshape(-1, 2)
        b1 = self.fields.G_shell[0]
        b2 = self.fields.G_shell[1]
        shifts = np.array([i * b1 + j * b2 for i in range(-2, 3) for j in range(-2, 3)])
        candidates = flat[:, None, :] - shifts[None, :, :]
        norms = np.sum(candidates**2, axis=-1)
        folded = candidates[np.arange(len(flat)), np.argmin(norms, axis=1)]
        return folded.reshape(original_shape)

    def berry_curvature_fukui(
        self, n_k: int = 21, active_band: int = 0
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Return plaquette Berry curvature in units of Omega/(2*pi/A_BZ)."""
        b1 = self.fields.G_shell[0]
        b2 = self.fields.G_shell[1]
        du = 1.0 / n_k
        dv = 1.0 / n_k
        area_plaq = abs(cross2(b1 * du, b2 * dv))
        coeffs = [[None for _ in range(n_k + 1)] for _ in range(n_k + 1)]
        ks = [[None for _ in range(n_k + 1)] for _ in range(n_k + 1)]
        for i in range(n_k + 1):
            for j in range(n_k + 1):
                k = (i * du) * b1 + (j * dv) * b2
                sol = self.solve(k, active_band=active_band)
                ks[i][j] = k
                coeffs[i][j] = sol.eigenvectors[:, active_band]
        phases = np.zeros((n_k, n_k), dtype=float)
        centers = np.zeros((n_k, n_k, 2), dtype=float)
        for i in range(n_k):
            for j in range(n_k):
                k00, k10, k11, k01 = ks[i][j], ks[i + 1][j], ks[i + 1][j + 1], ks[i][j + 1]
                c00, c10, c11, c01 = coeffs[i][j], coeffs[i + 1][j], coeffs[i + 1][j + 1], coeffs[i][j + 1]
                U1 = self.state_overlap(k00, c00, k10, c10)
                U2 = self.state_overlap(k10, c10, k11, c11)
                U3 = self.state_overlap(k11, c11, k01, c01)
                U4 = self.state_overlap(k01, c01, k00, c00)
                phases[i, j] = np.angle(U1 * U2 * U3 * U4)
                centers[i, j] = (i + 0.5) * du * b1 + (j + 0.5) * dv * b2
        omega = phases / area_plaq
        omega_dimless = omega / (2.0 * np.pi / self.bz_area)
        chern = float(np.sum(phases) / (2.0 * np.pi))
        return centers, omega_dimless, chern

    def band_diagnostics(self, n_k: int = 15, active_band: int = 0) -> dict[str, float]:
        energies = []
        gaps = []
        b1 = self.fields.G_shell[0]
        b2 = self.fields.G_shell[1]
        for i in range(n_k):
            for j in range(n_k):
                k = ((i + 0.5) / n_k) * b1 + ((j + 0.5) / n_k) * b2
                sol = self.solve(k, active_band=active_band)
                energies.append(sol.eigenvalues[active_band])
                gaps.append(sol.direct_gap)
        _, omega_dimless, chern = self.berry_curvature_fukui(n_k=n_k, active_band=active_band)
        energies_arr = np.asarray(energies)
        gaps_arr = np.asarray(gaps)
        return {
            "bandwidth": float(np.max(energies_arr) - np.min(energies_arr)),
            "min_direct_gap": float(np.min(gaps_arr)),
            "chern": chern,
            "berry_min": float(np.min(omega_dimless)),
            "berry_max": float(np.max(omega_dimless)),
            "berry_std": float(np.std(omega_dimless)),
        }
