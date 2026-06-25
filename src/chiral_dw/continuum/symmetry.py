"""Native valley U(1) and non-Kramers T-prime constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chiral_dw.continuum.models import ContinuumActiveSpace, MomentumGrid, hermitize


def mesh_inversion_map(grid: MomentumGrid) -> np.ndarray:
    """Return partner indices for k -> -k on the fractional momentum mesh."""

    partner = np.empty(grid.size, dtype=int)
    for ik in range(grid.size):
        i, j = grid.coord_of(ik)
        partner[ik] = grid.index_of((-i, -j))
    return partner


def valley_swap_matrix(n_active: int) -> np.ndarray:
    """Return the active-space unitary swapping K and Kprime valleys."""

    n = int(n_active)
    out = np.zeros((2 * n, 2 * n), dtype=complex)
    out[:n, n:] = np.eye(n, dtype=complex)
    out[n:, :n] = np.eye(n, dtype=complex)
    return out


def valley_u1_rotation(dim: int, phi: float) -> np.ndarray:
    """Return U_phi=diag(exp(-i phi/2), exp(+i phi/2)) in valley blocks."""

    if int(dim) <= 0 or int(dim) % 2:
        raise ValueError("dim must be a positive even integer")
    n = int(dim) // 2
    half = 0.5 * float(phi)
    phases = np.concatenate(
        [np.exp(-1j * half) * np.ones(n), np.exp(1j * half) * np.ones(n)]
    )
    return np.diag(phases.astype(complex))


def rotate_valley_u1(blocks: np.ndarray, phi: float) -> np.ndarray:
    """Rotate block operators or projectors by the valley U(1)."""

    arr = np.asarray(blocks, dtype=complex)
    U = valley_u1_rotation(arr.shape[-1], phi)
    return hermitize(np.einsum("ab,...bc,dc->...ad", U, arr, U.conj(), optimize=True))


def _fixed_per_k_aufbau(H: np.ndarray, n_occ_per_k: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fill a fixed number of states in every momentum block."""

    blocks = hermitize(np.asarray(H, dtype=complex))
    n_blocks, dim, _ = blocks.shape
    n_occ = int(n_occ_per_k)
    if n_occ < 1 or n_occ > dim:
        raise ValueError("n_occ_per_k must be between 1 and the block dimension")
    evals = np.empty((n_blocks, dim), dtype=float)
    P = np.zeros_like(blocks, dtype=complex)
    direct = np.full(n_blocks, np.inf, dtype=float)
    for ik in range(n_blocks):
        vals, vecs = np.linalg.eigh(blocks[ik])
        evals[ik] = vals
        occ = vecs[:, :n_occ]
        P[ik] = occ @ occ.conj().T
        if n_occ < dim:
            direct[ik] = vals[n_occ] - vals[n_occ - 1]
    if n_occ < dim:
        indirect = float(np.min(evals[:, n_occ]) - np.max(evals[:, n_occ - 1]))
    else:
        indirect = float("inf")
    return hermitize(P), evals, float(np.min(direct)), indirect


def _tprime_real_basis(n_active: int) -> np.ndarray:
    """Return columns whose coefficients are real under tau_x K."""

    n = int(n_active)
    dim = 2 * n
    basis = np.zeros((dim, dim), dtype=complex)
    root = 1.0 / np.sqrt(2.0)
    for a in range(n):
        basis[a, 2 * a] = root
        basis[n + a, 2 * a] = root
        basis[a, 2 * a + 1] = 1j * root
        basis[n + a, 2 * a + 1] = -1j * root
    return basis


def _tprime_self_projector(H: np.ndarray, n_active: int, n_occ_per_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize one self-inversion block in a T-prime-real basis."""

    basis = _tprime_real_basis(n_active)
    real_h = np.real_if_close(basis.conj().T @ hermitize(H) @ basis, tol=1000).real
    evals, real_vecs = np.linalg.eigh(0.5 * (real_h + real_h.T))
    occ = basis @ real_vecs[:, : int(n_occ_per_k)]
    return hermitize(occ @ occ.conj().T), evals


@dataclass(frozen=True)
class ValleyU1Constraint:
    """Continuous valley U(1) density/operator constraint."""

    active: ContinuumActiveSpace
    name: str = "valley_u1"

    def project_density(self, P: np.ndarray) -> np.ndarray:
        return self.project_operator(P)

    def project_operator(self, H: np.ndarray) -> np.ndarray:
        arr = np.asarray(H, dtype=complex)
        if arr.shape != (self.active.n_k, self.active.dim, self.active.dim):
            raise ValueError("blocks have incompatible active-space shape")
        n = self.active.n_active
        out = arr.copy()
        out[:, :n, n:] = 0.0
        out[:, n:, :n] = 0.0
        return hermitize(out)

    def symmetry_error(self, blocks: np.ndarray) -> float:
        arr = np.asarray(blocks, dtype=complex)
        return float(np.linalg.norm(arr - self.project_operator(arr)))

    def update_density(
        self, H: np.ndarray, n_occ_per_k: int
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        return _fixed_per_k_aufbau(self.project_operator(H), n_occ_per_k)


@dataclass(frozen=True)
class TPrimeConstraint:
    """Non-Kramers T' = tau_x K active-space density/operator constraint."""

    active: ContinuumActiveSpace
    name: str = "tprime"

    def __post_init__(self) -> None:
        object.__setattr__(self, "partner_index", mesh_inversion_map(self.active.grid))
        object.__setattr__(self, "swap", valley_swap_matrix(self.active.n_active))

    def transform(self, blocks: np.ndarray) -> np.ndarray:
        arr = np.asarray(blocks, dtype=complex)
        if arr.shape != (self.active.n_k, self.active.dim, self.active.dim):
            raise ValueError("blocks have incompatible active-space shape")
        out = np.empty_like(arr)
        x = self.swap
        for ik, jk in enumerate(self.partner_index):
            out[ik] = x @ arr[jk].conj() @ x.conj().T
        return out

    def project_density(self, P: np.ndarray) -> np.ndarray:
        return hermitize(0.5 * (np.asarray(P, dtype=complex) + self.transform(P)))

    def project_operator(self, H: np.ndarray) -> np.ndarray:
        return hermitize(0.5 * (np.asarray(H, dtype=complex) + self.transform(H)))

    def symmetry_error(self, blocks: np.ndarray) -> float:
        arr = np.asarray(blocks, dtype=complex)
        return float(np.linalg.norm(arr - self.transform(arr)))

    def update_density(
        self, H: np.ndarray, n_occ_per_k: int
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        projected = self.project_operator(H)
        P, evals, direct, indirect = _fixed_per_k_aufbau(projected, n_occ_per_k)
        out = np.empty_like(P)
        seen: set[int] = set()
        x = self.swap
        for ik, jk in enumerate(self.partner_index):
            if ik in seen:
                continue
            if ik == jk:
                out[ik], evals[ik] = _tprime_self_projector(
                    projected[ik], self.active.n_active, n_occ_per_k
                )
            else:
                out[ik] = P[ik]
                out[jk] = x @ P[ik].conj() @ x.conj().T
            seen.add(ik)
            seen.add(int(jk))
        return hermitize(out), evals, direct, indirect


def build_constraint(kind: str | None, active: ContinuumActiveSpace):
    """Build a native continuum HF constraint."""

    key = "none" if kind is None else str(kind).strip().lower()
    if key in {"", "none", "off", "false"}:
        return None
    if key in {"valley_u1", "u1", "valley"}:
        return ValleyU1Constraint(active)
    if key in {"tprime", "t_prime", "t'", "non_kramers_time_reversal"}:
        return TPrimeConstraint(active)
    raise ValueError("constraint kind must be 'none', 'valley_u1', or 'tprime'")
