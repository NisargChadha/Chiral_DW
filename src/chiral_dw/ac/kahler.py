"""Ideal Aharonov-Casher Kahler diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chiral_dw.ac.adiabatic import AdiabaticMoireFields


@dataclass(frozen=True)
class ChiSolution:
    """Periodic Poisson solution for the ideal AC dressing field chi."""

    u: np.ndarray
    v: np.ndarray
    r: np.ndarray
    b_periodic: np.ndarray
    b_average: float
    chi: np.ndarray
    chi_coeffs: np.ndarray
    b_coeffs: np.ndarray
    m_indices: np.ndarray
    n_indices: np.ndarray
    g_vectors: np.ndarray


class IdealACKahlerModel:
    """Compute ideal AC chi, norm, and Berry curvature from Fourier data."""

    def __init__(self, fields: AdiabaticMoireFields | None = None) -> None:
        self.fields = fields or AdiabaticMoireFields()
        self.l2 = self.fields.unit_cell_area / (2.0 * np.pi)
        self.bz_area = 4.0 * np.pi**2 / self.fields.unit_cell_area

    def solve_chi(self, n_grid: int = 96) -> ChiSolution:
        """Solve nabla^2 chi = B(r)-<B> for the adiabatic field model."""

        _, _, rr = self.fields.primitive_grid(int(n_grid))
        b_field = self.fields.effective_magnetic_field(rr)
        return self.solve_chi_from_values(b_field, n_grid=int(n_grid))

    def solve_chi_from_fourier(
        self,
        g_vectors: np.ndarray,
        b_coefficients: np.ndarray,
        *,
        n_grid: int = 96,
    ) -> ChiSolution:
        """Solve chi for a periodic field specified by B'_G coefficients."""

        n = int(n_grid)
        _, _, rr = self.fields.primitive_grid(n)
        G = np.asarray(g_vectors, dtype=float)
        coeff = np.asarray(b_coefficients, dtype=complex)
        if G.shape != (coeff.size, 2):
            raise ValueError("g_vectors must have shape (n_coeff, 2)")
        phase = np.tensordot(rr, G, axes=([-1], [-1]))
        b_periodic = np.real(np.sum(coeff[None, None, :] * np.exp(1j * phase), axis=-1))
        return self.solve_chi_from_values(b_periodic, n_grid=n)

    def solve_chi_from_values(self, b_values: np.ndarray, *, n_grid: int | None = None) -> ChiSolution:
        """Solve chi from samples of a real periodic magnetic field."""

        b = np.asarray(b_values, dtype=float)
        if b.ndim != 2 or b.shape[0] != b.shape[1]:
            raise ValueError("b_values must be a square primitive-cell grid")
        n = int(b.shape[0] if n_grid is None else n_grid)
        if b.shape != (n, n):
            raise ValueError("n_grid does not match b_values")
        uu, vv, rr = self.fields.primitive_grid(n)
        b_average = float(np.mean(b))
        b_periodic = b - b_average
        b_coeffs = np.fft.fft2(b_periodic) / n**2
        mm, nn, G = self.fields.reciprocal_fft_data(n)
        G2 = np.sum(G**2, axis=-1)

        chi_coeffs = np.zeros_like(b_coeffs, dtype=complex)
        nonzero = G2 > 1e-14
        chi_coeffs[nonzero] = -b_coeffs[nonzero] / G2[nonzero]
        chi = np.real(np.fft.ifft2(chi_coeffs) * n**2)
        chi -= float(np.mean(chi))
        return ChiSolution(
            u=uu,
            v=vv,
            r=rr,
            b_periodic=b_periodic,
            b_average=b_average,
            chi=chi,
            chi_coeffs=chi_coeffs,
            b_coeffs=b_coeffs,
            m_indices=mm,
            n_indices=nn,
            g_vectors=G,
        )

    def exp2chi_fourier_coeffs(
        self,
        chi_solution: ChiSolution,
        *,
        coefficient_cutoff: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return G, eta_G lambda_G Phi_G, and raw Phi_G for exp(2 chi)."""

        chi = np.asarray(chi_solution.chi, dtype=float)
        n = chi.shape[0]
        phi = np.fft.fft2(np.exp(2.0 * chi)) / n**2
        G = chi_solution.g_vectors.reshape(-1, 2)
        m = chi_solution.m_indices.reshape(-1)
        n_idx = chi_solution.n_indices.reshape(-1)
        phi_flat = phi.reshape(-1)
        parity = np.where((m % 2 == 0) & (n_idx % 2 == 0), 1.0, -1.0)
        lambdas = np.exp(-self.l2 * np.sum(G**2, axis=1) / 4.0)
        weighted = parity * lambdas * phi_flat
        scale = max(float(np.max(np.abs(weighted))), 1e-300)
        keep = np.abs(weighted) > float(coefficient_cutoff) * scale
        return G[keep], weighted[keep], phi_flat[keep]

    def ac_norm_and_derivatives(
        self,
        k: np.ndarray,
        g_vectors: np.ndarray,
        weighted_coeffs: np.ndarray,
        *,
        chunk_size: int = 512,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate S(k), grad_k S(k), and laplacian_k S(k)."""

        k_arr = np.asarray(k, dtype=float)
        flat_k = k_arr.reshape(-1, 2)
        G = np.asarray(g_vectors, dtype=float)
        coeff = np.asarray(weighted_coeffs, dtype=complex)
        if G.shape != (coeff.size, 2):
            raise ValueError("g_vectors and weighted_coeffs have incompatible shapes")

        S = np.zeros(flat_k.shape[0], dtype=complex)
        grad = np.zeros((flat_k.shape[0], 2), dtype=complex)
        lap = np.zeros(flat_k.shape[0], dtype=complex)
        q_vectors = self.l2 * np.stack([-G[:, 1], G[:, 0]], axis=1)
        q2 = np.sum(q_vectors**2, axis=1)
        for start in range(0, len(G), int(chunk_size)):
            stop = min(start + int(chunk_size), len(G))
            phase = flat_k @ q_vectors[start:stop].T
            cexp = np.exp(1j * phase) * coeff[start:stop][None, :]
            S += np.sum(cexp, axis=1)
            grad += 1j * (cexp @ q_vectors[start:stop])
            lap -= cexp @ q2[start:stop]
        return (
            np.real_if_close(S, tol=1000).real.reshape(k_arr.shape[:-1]),
            np.real_if_close(grad, tol=1000).real.reshape(k_arr.shape),
            np.real_if_close(lap, tol=1000).real.reshape(k_arr.shape[:-1]),
        )

    def berry_curvature(
        self,
        k: np.ndarray,
        g_vectors: np.ndarray,
        weighted_coeffs: np.ndarray,
    ) -> np.ndarray:
        """Return the ideal AC Berry curvature Omega(k)."""

        S, grad, lap = self.ac_norm_and_derivatives(k, g_vectors, weighted_coeffs)
        lap_log = lap / S - np.sum(grad**2, axis=-1) / S**2
        return 2.0 * np.pi / self.bz_area + 0.5 * lap_log

    def dimensionless_berry_curvature(
        self,
        k: np.ndarray,
        g_vectors: np.ndarray,
        weighted_coeffs: np.ndarray,
    ) -> np.ndarray:
        """Return Omega(k) divided by its Brillouin-zone average."""

        return self.berry_curvature(k, g_vectors, weighted_coeffs) / (
            2.0 * np.pi / self.bz_area
        )


__all__ = ["ChiSolution", "IdealACKahlerModel"]
