"""Projected physical energy for two-flavor AC projectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chiral_dw.ac.source import FlavorSourceProjector
from chiral_dw.config import GatedInteractionParams


def real_scalar(value: complex, tol: float = 1e-9) -> float:
    z = complex(value)
    if abs(z.imag) > tol:
        raise ValueError(f"expected real scalar, got imaginary part {z.imag}")
    return float(z.real)


@dataclass(frozen=True)
class EnergyBreakdown:
    """Physical energy components per occupied k state."""

    total: float
    band: float
    hartree: float
    fock: float


class ProjectedPhysicalEnergy:
    """Band, Hartree, and Fock energy for arbitrary two-flavor projectors."""

    def __init__(
        self,
        source: FlavorSourceProjector,
        interaction: GatedInteractionParams | None = None,
    ) -> None:
        self.source = source
        self.model = source.model
        self.interaction = interaction or GatedInteractionParams()
        self.q_mesh = self.source.k_mesh.copy()
        self.reciprocal_images = self._build_reciprocal_images()
        self._active_coefficients: dict[tuple[float, float], np.ndarray] = {}
        self._exchange_terms: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] | None = None
        self._hartree_terms: list[tuple[float, np.ndarray, np.ndarray]] | None = None

    def _build_reciprocal_images(self) -> np.ndarray:
        shell = self.interaction.interaction_shell
        b1 = self.source.b1
        b2 = self.source.b2
        return np.asarray(
            [m * b1 + n * b2 for m in range(-shell, shell + 1) for n in range(-shell, shell + 1)],
            dtype=float,
        )

    @staticmethod
    def _key(v: np.ndarray, decimals: int = 12) -> tuple[float, float]:
        rounded = np.round(np.asarray(v, dtype=float), decimals=decimals)
        return float(rounded[0]), float(rounded[1])

    def interaction_value(self, Q: np.ndarray) -> float:
        """Return V(Q)=2*pi*v0*tanh(|Q|d)/|Q|, with Q=0 omitted."""
        q_norm = float(np.linalg.norm(Q))
        if q_norm < 1e-12:
            return 0.0
        p = self.interaction
        return float(2.0 * np.pi * p.v0 * np.tanh(q_norm * p.gate_distance) / q_norm)

    def _coefficients(self, k: np.ndarray) -> np.ndarray:
        key = self._key(k)
        cached = self._active_coefficients.get(key)
        if cached is not None:
            return cached
        sol = self.model.solve(k, active_band=self.source.active_band)
        coeffs = sol.eigenvectors[:, self.source.active_band]
        self._active_coefficients[key] = coeffs
        return coeffs

    def active_up_form_factor(self, k: np.ndarray, q: np.ndarray, G: np.ndarray) -> complex:
        p = np.asarray(k, dtype=float) + np.asarray(q, dtype=float)
        c_k = self._coefficients(k)
        c_p = self._coefficients(p)
        matrix = self.model.density_form_factor_matrix(k, p, G)
        return complex(c_k.conj() @ matrix @ c_p)

    def active_down_form_factor(self, k: np.ndarray, q: np.ndarray, G: np.ndarray) -> complex:
        return np.conj(
            self.active_up_form_factor(
                -np.asarray(k, dtype=float),
                -np.asarray(q, dtype=float),
                -np.asarray(G, dtype=float),
            )
        )

    def _flavor_form_factors(self, q: np.ndarray, G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        up = np.asarray([self.active_up_form_factor(k, q, G) for k in self.source.k_mesh])
        down = np.asarray([self.active_down_form_factor(k, q, G) for k in self.source.k_mesh])
        return up, down

    def exchange_terms(self) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray]]:
        if self._exchange_terms is not None:
            return self._exchange_terms
        terms: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        for q_idx, q in enumerate(self.q_mesh):
            target = self.source.target_indices_for_q(q_idx)
            for G in self.reciprocal_images:
                V_Q = self.interaction_value(q + G)
                if V_Q == 0.0:
                    continue
                up, down = self._flavor_form_factors(q, G)
                terms.append((V_Q, target, up, down))
        self._exchange_terms = terms
        return terms

    def hartree_terms(self) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """Return direct q=0 terms with the uniform G=0 Hartree term omitted."""
        if self._hartree_terms is not None:
            return self._hartree_terms
        terms: list[tuple[float, np.ndarray, np.ndarray]] = []
        q0 = np.zeros(2)
        for G in self.reciprocal_images:
            V_G = self.interaction_value(G)
            if V_G == 0.0:
                continue
            up, down = self._flavor_form_factors(q0, G)
            terms.append((V_G, up, down))
        self._hartree_terms = terms
        return terms

    def band_energy(self, P: np.ndarray) -> float:
        value = np.einsum("kab,kba->", self.source.h0, P, optimize=True) / self.source.n_total
        return real_scalar(value)

    def hartree_energy(self, P: np.ndarray) -> float:
        total = 0.0 + 0.0j
        for V_G, up, down in self.hartree_terms():
            density = np.sum(up * P[:, 0, 0] + down * P[:, 1, 1])
            total += V_G * density.conj() * density
        return real_scalar(0.5 * total / (self.source.n_total**2))

    def fock_energy(self, P: np.ndarray) -> float:
        total = 0.0 + 0.0j
        for V_Q, target, up, down in self.exchange_terms():
            lambdas = np.zeros((self.source.n_total, 2, 2), dtype=complex)
            lambdas[:, 0, 0] = up
            lambdas[:, 1, 1] = down
            total += V_Q * np.einsum(
                "kab,kbc,kcd,kda->",
                P,
                lambdas,
                P[target],
                lambdas.conj().swapaxes(-1, -2),
                optimize=True,
            )
        return real_scalar(-0.5 * total / (self.source.n_total**2))

    def energy(self, P: np.ndarray) -> EnergyBreakdown:
        band = self.band_energy(P)
        hartree = self.hartree_energy(P)
        fock = self.fock_energy(P)
        return EnergyBreakdown(total=band + hartree + fock, band=band, hartree=hartree, fock=fock)
