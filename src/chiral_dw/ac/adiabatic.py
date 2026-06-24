"""Adiabatic real-space fields used by the nonideal AC backend."""

from __future__ import annotations

import numpy as np
import scipy.constants as sc

from chiral_dw.config import TMoTe2ACParams

_HBAR2_OVER_2ME_MEV_AA2 = (sc.hbar**2 / (2.0 * sc.m_e)) / sc.e * 1e23


class AdiabaticMoireFields:
    """Evaluate tMoTe2 adiabatic fields in dimensionless moire coordinates."""

    def __init__(self, params: TMoTe2ACParams | None = None) -> None:
        self.params = params or TMoTe2ACParams()
        self._setup_geometry()

    def _setup_geometry(self) -> None:
        p = self.params
        self.q_mag = 4.0 * np.pi / (3.0 * p.a_m)
        self.unit_cell_area = np.sqrt(3.0) * p.a_m**2 / 2.0
        self.q_vecs = np.array(
            [
                self.q_mag
                * np.array([np.sin(2.0 * j * np.pi / 3.0), -np.cos(2.0 * j * np.pi / 3.0)])
                for j in range(3)
            ]
        )
        self.G_shell = np.array(
            [
                np.sqrt(3.0)
                * self.q_mag
                * np.array([np.cos(j * np.pi / 3.0), np.sin(j * np.pi / 3.0)])
                for j in range(6)
            ]
        )
        self.G_even = self.G_shell[[0, 2, 4]]
        reciprocal_matrix = np.column_stack([self.G_shell[0], self.G_shell[1]])
        self.real_lattice = 2.0 * np.pi * np.linalg.inv(reciprocal_matrix).T
        self.a1 = self.real_lattice[:, 0]
        self.a2 = self.real_lattice[:, 1]
        lattice_vectors = []
        for m1 in range(-1, 2):
            for m2 in range(-1, 2):
                if m1 == 0 and m2 == 0:
                    continue
                lattice_vectors.append(m1 * self.a1 + m2 * self.a2)
        lattice_vectors = np.array(lattice_vectors)
        min_norm = np.min(np.linalg.norm(lattice_vectors, axis=1))
        self.nearest_real_vectors = lattice_vectors[
            np.isclose(np.linalg.norm(lattice_vectors, axis=1), min_norm)
        ]

    @property
    def psi(self) -> float:
        return float(np.deg2rad(self.params.phi_deg))

    def primitive_grid(self, n_grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords = np.arange(n_grid) / n_grid
        uu, vv = np.meshgrid(coords, coords, indexing="ij")
        rr = uu[..., None] * self.a1 + vv[..., None] * self.a2
        return uu, vv, rr

    def reciprocal_fft_data(self, n_grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        freqs = np.fft.fftfreq(n_grid, d=1.0 / n_grid).astype(int)
        mm, nn = np.meshgrid(freqs, freqs, indexing="ij")
        G = mm[..., None] * self.G_shell[0] + nn[..., None] * self.G_shell[1]
        return mm, nn, G

    def delta(self, r: np.ndarray) -> np.ndarray:
        """Return Delta=(Delta_x, Delta_y, Delta_z) at positions r."""
        p = self.params
        r = np.asarray(r, dtype=float)
        q_phase = np.tensordot(r, self.q_vecs, axes=([-1], [-1]))
        tunneling = p.folded_ac_w_mev * np.sum(np.exp(1j * q_phase), axis=-1)
        G_phase = np.tensordot(r, self.G_even, axes=([-1], [-1]))
        delta_plus = 2.0 * p.V_mev * np.sum(np.cos(G_phase - self.psi), axis=-1)
        delta_minus = 2.0 * p.V_mev * np.sum(np.cos(G_phase + self.psi), axis=-1)
        delta_z = 0.5 * (delta_plus - delta_minus) + 0.5 * p.uD_mev
        return np.stack([np.real(tunneling), np.imag(tunneling), delta_z], axis=-1)

    def delta0(self, r: np.ndarray) -> np.ndarray:
        """Return scalar moire potential Delta_0(r)."""
        p = self.params
        r = np.asarray(r, dtype=float)
        G_phase = np.tensordot(r, self.G_even, axes=([-1], [-1]))
        delta_plus = 2.0 * p.V_mev * np.sum(np.cos(G_phase - self.psi), axis=-1)
        delta_minus = 2.0 * p.V_mev * np.sum(np.cos(G_phase + self.psi), axis=-1)
        return 0.5 * (delta_plus + delta_minus)

    def delta_plus_aligned(self, r: np.ndarray) -> np.ndarray:
        """Return Delta_+(r)=Delta_0(r)+|Delta(r)| in meV."""
        delta = self.delta(r)
        return self.delta0(r) + np.linalg.norm(delta, axis=-1)

    def delta_derivatives(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return partial_x Delta and partial_y Delta."""
        p = self.params
        r = np.asarray(r, dtype=float)
        q_phase = np.tensordot(r, self.q_vecs, axes=([-1], [-1]))
        exp_q = np.exp(1j * q_phase)
        d_tunnel = 1j * p.folded_ac_w_mev * np.einsum("...j,ja->...a", exp_q, self.q_vecs)
        G_phase = np.tensordot(r, self.G_even, axes=([-1], [-1]))
        d_plus = -2.0 * p.V_mev * np.einsum(
            "...j,ja->...a", np.sin(G_phase - self.psi), self.G_even
        )
        d_minus = -2.0 * p.V_mev * np.einsum(
            "...j,ja->...a", np.sin(G_phase + self.psi), self.G_even
        )
        d_z = 0.5 * (d_plus - d_minus)
        d_delta = np.stack([np.real(d_tunnel), np.imag(d_tunnel), d_z], axis=-1)
        return d_delta[..., 0, :], d_delta[..., 1, :]

    def n_derivatives(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        delta = self.delta(r)
        d_delta_x, d_delta_y = self.delta_derivatives(r)
        norm = np.linalg.norm(delta, axis=-1, keepdims=True)
        dn_x = d_delta_x / norm - delta * np.sum(delta * d_delta_x, axis=-1, keepdims=True) / norm**3
        dn_y = d_delta_y / norm - delta * np.sum(delta * d_delta_y, axis=-1, keepdims=True) / norm**3
        return dn_x, dn_y

    def effective_magnetic_field(self, r: np.ndarray) -> np.ndarray:
        """Return B(r)=n dot (partial_x n cross partial_y n)/2."""
        delta = self.delta(r)
        n = delta / np.linalg.norm(delta, axis=-1, keepdims=True)
        dn_x, dn_y = self.n_derivatives(r)
        return 0.5 * np.sum(n * np.cross(dn_x, dn_y), axis=-1)

    def quantum_metric_trace(self, r: np.ndarray) -> np.ndarray:
        dn_x, dn_y = self.n_derivatives(r)
        return 0.25 * (np.sum(dn_x**2, axis=-1) + np.sum(dn_y**2, axis=-1))

    def dimensionless_xi(self, r: np.ndarray) -> np.ndarray:
        """Return xi=(D-B)/(2|<B>|), the dimensionless residual field."""
        b = self.effective_magnetic_field(r)
        b0_abs = abs(float(np.mean(b)))
        if b0_abs < 1e-14:
            raise ValueError("average effective magnetic field is too small")
        return (self.quantum_metric_trace(r) - b) / (2.0 * b0_abs)

    def omega_c_mev(self, theta_deg: float | None = None) -> float:
        """Return average-field Landau-level spacing in meV."""
        theta = np.deg2rad(float(self.params.theta_deg if theta_deg is None else theta_deg))
        if theta <= 0.0:
            raise ValueError("theta_deg must be positive")
        a_m_angstrom = self.params.a0_angstrom / (2.0 * np.sin(0.5 * theta))
        area = np.sqrt(3.0) * a_m_angstrom**2 / 2.0
        l2_angstrom = area / (2.0 * np.pi)
        return float(2.0 * (_HBAR2_OVER_2ME_MEV_AA2 / self.params.m_eff) / l2_angstrom)

    def dimensionless_effective_magnetic_field(self, r: np.ndarray) -> np.ndarray:
        """Return -B(r) A_M / (2 pi), whose average is one."""
        return -self.effective_magnetic_field(r) * self.unit_cell_area / (2.0 * np.pi)

    def reciprocal_wigner_seitz_mask(self, k: np.ndarray, atol: float = 1e-12) -> np.ndarray:
        nearest = self.G_shell
        dots = np.tensordot(k, nearest, axes=([-1], [-1]))
        bounds = 0.5 * np.sum(nearest**2, axis=1)
        return np.all(dots <= bounds + atol, axis=-1)
