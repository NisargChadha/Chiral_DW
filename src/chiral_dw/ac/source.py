"""Source-field projectors in the two-flavor AC active-band basis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from chiral_dw.ac.nonideal import NonIdealACLLModel

SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
PAULI = np.stack([SIGMA_X, SIGMA_Y, SIGMA_Z], axis=0)


def unit_vector(v: tuple[float, float, float] | np.ndarray, eps: float = 1e-14) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        raise ValueError("vector must have nonzero norm")
    return arr / norm


def spinor_from_angles(theta: float, phi: float = 0.0) -> np.ndarray:
    """Return z=(cos(theta/2), exp(i phi) sin(theta/2))."""
    return np.array(
        [np.cos(0.5 * float(theta)), np.exp(1j * float(phi)) * np.sin(0.5 * float(theta))],
        dtype=complex,
    )


def projector_from_spinor(z: np.ndarray) -> np.ndarray:
    spinor = np.asarray(z, dtype=complex)
    return spinor[:, None] * spinor[None, :].conj()


@dataclass(frozen=True)
class SourceProjectorResult:
    """Projectors and diagnostics produced by a source field."""

    projector: np.ndarray
    min_gap: float
    polarization: np.ndarray
    parallel_polarization: float


class FlavorSourceProjector:
    """Generate rank-one two-flavor projectors from h0(k)-m0 n.sigma."""

    def __init__(
        self,
        ac_model: NonIdealACLLModel,
        n_k: int,
        n_vec: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 1.0),
        active_band: int = 0,
        occupy: Literal["lowest", "highest"] = "lowest",
    ) -> None:
        if n_k < 1:
            raise ValueError("n_k must be >= 1")
        if occupy not in {"lowest", "highest"}:
            raise ValueError("occupy must be 'lowest' or 'highest'")
        self.model = ac_model
        self.n_k = int(n_k)
        self.n_vec = unit_vector(n_vec)
        self.active_band = int(active_band)
        self.occupy = occupy
        self.b1 = self.model.fields.G_shell[0]
        self.b2 = self.model.fields.G_shell[1]
        self.k_mesh = self._build_mesh()
        self.k_indices = np.asarray([(i, j) for i in range(self.n_k) for j in range(self.n_k)])
        self.h0 = self._build_h0()

    @property
    def n_total(self) -> int:
        return self.n_k * self.n_k

    def _build_mesh(self) -> np.ndarray:
        return np.asarray(
            [
                (i / self.n_k) * self.b1 + (j / self.n_k) * self.b2
                for i in range(self.n_k)
                for j in range(self.n_k)
            ],
            dtype=float,
        )

    def _build_h0(self) -> np.ndarray:
        h0 = np.zeros((self.n_total, 2, 2), dtype=complex)
        for idx, k in enumerate(self.k_mesh):
            eps_up = self.model.solve(k, active_band=self.active_band).eigenvalues[self.active_band]
            eps_down = self.model.solve(-k, active_band=self.active_band).eigenvalues[self.active_band]
            h0[idx, 0, 0] = eps_up
            h0[idx, 1, 1] = eps_down
        return h0

    def target_indices_for_q(self, q_flat_index: int) -> np.ndarray:
        qi, qj = self.k_indices[int(q_flat_index)]
        target_i = (self.k_indices[:, 0] + qi) % self.n_k
        target_j = (self.k_indices[:, 1] + qj) % self.n_k
        return target_i * self.n_k + target_j

    def trial_hamiltonian(self, m0: float) -> np.ndarray:
        source = np.tensordot(self.n_vec, PAULI, axes=(0, 0))
        return self.h0 - float(m0) * source[None, :, :]

    def projector(self, m0: float) -> np.ndarray:
        vals, vecs = np.linalg.eigh(self.trial_hamiltonian(m0))
        band = 0 if self.occupy == "lowest" else 1
        spinors = vecs[:, :, band]
        return spinors[:, :, None] * spinors[:, None, :].conj()

    def projector_and_diagnostics(self, m0: float) -> SourceProjectorResult:
        H = self.trial_hamiltonian(m0)
        vals, vecs = np.linalg.eigh(H)
        band = 0 if self.occupy == "lowest" else 1
        spinors = vecs[:, :, band]
        P = spinors[:, :, None] * spinors[:, None, :].conj()
        gap = float(np.min(vals[:, 1] - vals[:, 0])) if vals.shape[1] > 1 else np.inf
        pol, parallel = self.polarization(P)
        return SourceProjectorResult(P, gap, pol, parallel)

    def fixed_spinor_projector(self, theta: float, phi: float = 0.0) -> np.ndarray:
        P0 = projector_from_spinor(spinor_from_angles(theta, phi))
        return np.broadcast_to(P0, (self.n_total, 2, 2)).copy()

    def polarization(self, P: np.ndarray) -> tuple[np.ndarray, float]:
        arr = np.asarray(P, dtype=complex)
        if arr.shape != (self.n_total, 2, 2):
            raise ValueError(f"P must have shape {(self.n_total, 2, 2)}")
        local = np.einsum("kab,mba->km", arr, PAULI, optimize=True)
        m_vec = np.real_if_close(np.mean(local, axis=0), tol=1000).real
        return m_vec, float(np.dot(m_vec, self.n_vec))
