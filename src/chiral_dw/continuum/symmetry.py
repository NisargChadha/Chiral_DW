"""Native valley U(1) and non-Kramers T-prime constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chiral_dw.continuum.models import (
    ContinuumActiveSpace,
    MomentumGrid,
    VALLEY_K,
    VALLEY_KPRIME,
    hermitize,
)


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


def _global_aufbau_occupations(
    evals_flat: np.ndarray, n_particles: float, tol: float = 1e-10
) -> np.ndarray:
    """Return global zero-temperature occupations with degenerate-shell averaging."""

    evals = np.asarray(evals_flat, dtype=float)
    n_states = evals.size
    n = float(n_particles)
    if n < -tol or n > n_states + tol:
        raise ValueError(f"n_particles must be in [0, {n_states}]; got {n_particles}")
    n = min(float(n_states), max(0.0, n))
    occ = np.zeros(n_states, dtype=float)
    if n <= tol:
        return occ
    if n >= n_states - tol:
        occ[:] = 1.0
        return occ

    order = np.argsort(evals, kind="mergesort")
    n_int = int(round(n))
    if abs(n - n_int) <= tol:
        occ[order[:n_int]] = 1.0
        return occ

    rank = int(np.floor(n))
    frac = n - rank
    fermi_rank = rank if frac > tol else max(rank - 1, 0)
    fermi = evals[order[fermi_rank]]
    lower = evals < fermi - tol
    shell = np.abs(evals - fermi) <= tol
    occ[lower] = 1.0
    remaining = n - float(np.count_nonzero(lower))
    shell_count = int(np.count_nonzero(shell))
    if shell_count:
        occ[shell] = np.clip(remaining / shell_count, 0.0, 1.0)
    return occ


def _occupation_gap(evals_flat: np.ndarray, occ_flat: np.ndarray, tol: float = 1e-10) -> float:
    """Return the global occupied-empty gap for a block eigensystem."""

    evals = np.asarray(evals_flat, dtype=float)
    occ = np.asarray(occ_flat, dtype=float)
    if np.any((occ > tol) & (occ < 1.0 - tol)):
        return 0.0
    occupied = evals[occ > 1.0 - tol]
    empty = evals[occ < tol]
    if occupied.size == 0 or empty.size == 0:
        return float("nan")
    return float(np.min(empty) - np.max(occupied))


def _direct_gap(evals: np.ndarray, occupations: np.ndarray, tol: float = 1e-10) -> float:
    """Return the smallest blockwise occupied-empty gap."""

    gaps: list[float] = []
    for vals, occ in zip(np.asarray(evals, dtype=float), np.asarray(occupations, dtype=float)):
        if np.any((occ > tol) & (occ < 1.0 - tol)):
            gaps.append(0.0)
            continue
        occupied = vals[occ > 1.0 - tol]
        empty = vals[occ < tol]
        if occupied.size and empty.size:
            gaps.append(float(np.min(empty) - np.max(occupied)))
    if not gaps:
        return float("nan")
    return float(np.min(gaps))


def _global_aufbau(H: np.ndarray, n_particles: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fill the lowest states globally across all momentum blocks."""

    blocks = hermitize(np.asarray(H, dtype=complex))
    n_blocks, dim, _ = blocks.shape
    evals = np.empty((n_blocks, dim), dtype=float)
    evecs = np.empty((n_blocks, dim, dim), dtype=complex)
    for ik in range(n_blocks):
        evals[ik], evecs[ik] = np.linalg.eigh(blocks[ik])
    occ = _global_aufbau_occupations(evals.reshape(-1), n_particles).reshape(n_blocks, dim)
    P = np.einsum("kai,ki,kbi->kab", evecs, occ, evecs.conj(), optimize=True)
    indirect = _occupation_gap(evals.reshape(-1), occ.reshape(-1))
    direct = _direct_gap(evals, occ)
    return hermitize(P), evals, direct, indirect


def _fixed_valley_aufbau(
    H: np.ndarray,
    n_occ_per_k: int,
    *,
    n_active: int,
    valley_index: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fill a fixed number of states from one valley block at every momentum."""

    blocks = hermitize(np.asarray(H, dtype=complex))
    n_blocks, dim, _ = blocks.shape
    n_occ = int(n_occ_per_k)
    n = int(n_active)
    if n_occ < 1 or n_occ > n:
        raise ValueError("pinned-valley Aufbau needs n_occ_per_k <= n_active")
    start = int(valley_index) * n
    stop = start + n
    evals = np.empty((n_blocks, dim), dtype=float)
    P = np.zeros_like(blocks, dtype=complex)
    direct = np.full(n_blocks, np.inf, dtype=float)
    for ik in range(n_blocks):
        vals, vecs = np.linalg.eigh(blocks[ik, start:stop, start:stop])
        full_vals = np.linalg.eigvalsh(blocks[ik])
        evals[ik] = np.sort(full_vals)
        occ = vecs[:, :n_occ]
        P[ik, start:stop, start:stop] = occ @ occ.conj().T
        if n_occ < n:
            direct[ik] = vals[n_occ] - vals[n_occ - 1]
    if n_occ < n:
        selected_evals = []
        for ik in range(n_blocks):
            vals = np.linalg.eigvalsh(blocks[ik, start:stop, start:stop])
            selected_evals.append(vals)
        selected = np.asarray(selected_evals)
        indirect = float(np.min(selected[:, n_occ]) - np.max(selected[:, n_occ - 1]))
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

    evals, vectors = _tprime_self_eigensystem(H, n_active)
    occ = vectors[:, : int(n_occ_per_k)]
    return hermitize(occ @ occ.conj().T), evals


def _tprime_self_eigensystem(H: np.ndarray, n_active: int) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize one self-inversion block in a T-prime-real basis."""

    basis = _tprime_real_basis(n_active)
    real_h = np.real_if_close(basis.conj().T @ hermitize(H) @ basis, tol=1000).real
    evals, real_vecs = np.linalg.eigh(0.5 * (real_h + real_h.T))
    return evals, basis @ real_vecs


@dataclass(frozen=True)
class ValleyU1Constraint:
    """Continuous valley U(1) density/operator constraint."""

    active: ContinuumActiveSpace
    name: str = "valley_u1"
    pinned_valley: str | None = None

    def __post_init__(self) -> None:
        if self.pinned_valley is None:
            return
        if self.pinned_valley not in {VALLEY_K, VALLEY_KPRIME}:
            raise ValueError("pinned_valley must be None, 'K', or 'Kprime'")
        object.__setattr__(self, "name", f"valley_u1_{self.pinned_valley}")

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
        projected = self.project_operator(H)
        if self.pinned_valley is None:
            return _fixed_per_k_aufbau(projected, n_occ_per_k)
        return _fixed_valley_aufbau(
            projected,
            n_occ_per_k,
            n_active=self.active.n_active,
            valley_index=self.active.valley_index(self.pinned_valley),
        )

    def update_density_global(
        self, H: np.ndarray, n_particles: float
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        projected = self.project_operator(H)
        return _global_aufbau(projected, n_particles)


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

    def update_density_global(
        self, H: np.ndarray, n_particles: float
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        projected = self.project_operator(H)
        n_blocks, dim, _ = projected.shape
        P = np.zeros_like(projected, dtype=complex)
        evals = np.empty((n_blocks, dim), dtype=float)
        records: list[tuple[float, float, tuple[tuple[int, np.ndarray], ...]]] = []
        seen: set[int] = set()
        x = self.swap
        for ik, jk in enumerate(self.partner_index):
            jk_int = int(jk)
            if ik in seen:
                continue
            if ik == jk_int:
                vals, vecs = _tprime_self_eigensystem(projected[ik], self.active.n_active)
                evals[ik] = vals
                for band, value in enumerate(vals):
                    v = vecs[:, band]
                    records.append((float(value), 1.0, ((ik, np.outer(v, v.conj())),)))
            else:
                vals, vecs = np.linalg.eigh(projected[ik])
                evals[ik] = vals
                evals[jk_int] = vals
                for band, value in enumerate(vals):
                    v = vecs[:, band]
                    block = np.outer(v, v.conj())
                    partner = x @ block.conj() @ x.conj().T
                    records.append((float(value), 2.0, ((ik, block), (jk_int, partner))))
            seen.add(ik)
            seen.add(jk_int)

        target = int(round(float(n_particles)))
        occ_values: list[tuple[float, float]] = []
        if abs(float(n_particles) - target) <= 1e-10:
            costs = [float(value) * float(weight) for value, weight, _blocks in records]
            weights = [int(round(weight)) for _value, weight, _blocks in records]
            inf = float("inf")
            dp = [inf] * (target + 1)
            parent: list[tuple[int, int] | None] = [None] * (target + 1)
            dp[0] = 0.0
            for idx, (weight, cost) in enumerate(zip(weights, costs)):
                if weight <= 0:
                    continue
                for count in range(target - weight, -1, -1):
                    candidate = dp[count] + cost
                    if candidate < dp[count + weight]:
                        dp[count + weight] = candidate
                        parent[count + weight] = (count, idx)
            selected: set[int] = set()
            if np.isfinite(dp[target]):
                count = target
                while count > 0 and parent[count] is not None:
                    prev, idx = parent[count]
                    selected.add(idx)
                    count = prev
            else:
                selected = set()
            if selected:
                for idx, (value, _weight, blocks) in enumerate(records):
                    fraction = 1.0 if idx in selected else 0.0
                    for ik, block in blocks:
                        P[ik] += fraction * block
                    occ_values.append((value, fraction))
            else:
                target = -1

        if target < 0 or abs(float(n_particles) - int(round(float(n_particles)))) > 1e-10:
            remaining = float(n_particles)
            for value, weight, blocks in sorted(records, key=lambda item: item[0]):
                if remaining <= 1e-10:
                    occ_values.append((value, 0.0))
                    continue
                if remaining >= weight - 1e-10:
                    fraction = 1.0
                    remaining -= weight
                else:
                    fraction = max(0.0, remaining / weight)
                    remaining = 0.0
                for ik, block in blocks:
                    P[ik] += fraction * block
                occ_values.append((value, fraction))

        values = np.asarray([row[0] for row in occ_values], dtype=float)
        occ = np.asarray([row[1] for row in occ_values], dtype=float)
        indirect = _occupation_gap(values, occ)
        return hermitize(P), evals, indirect, indirect


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
