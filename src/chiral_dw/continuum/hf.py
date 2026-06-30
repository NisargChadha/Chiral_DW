"""Native optimized continuum Hartree-Fock backend and solver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from chiral_dw.config import ContinuumHFParams, ContinuumInteractionParams
from chiral_dw.continuum.models import (
    ContinuumHFDiagnostics,
    ContinuumHFIterationSnapshot,
    ContinuumHFResult,
    DensityVertices,
    ReferenceHamiltonianDiagnostics,
    block_trace_product,
    hermitize,
    projector_idempotency_errors,
)
from chiral_dw.continuum.symmetry import _fixed_per_k_aufbau, _global_aufbau

HFIterationCallback = Callable[[int, np.ndarray, float, ContinuumHFDiagnostics, bool], None]


@dataclass(frozen=True)
class EnergyComponents:
    """One-body, Hartree, Fock, and total HF energy."""

    total: float
    one_body: float
    hartree: float
    fock: float


def _exchange_q_slab_ranges(n_q: int, exchange_workers: int) -> tuple[tuple[int, int], ...]:
    n_jobs = max(1, min(int(exchange_workers), int(n_q)))
    n_slabs = max(1, min(int(n_q), 16 * n_jobs))
    bounds = np.linspace(0, int(n_q), n_slabs + 1, dtype=int)
    return tuple(
        (int(start), int(stop))
        for start, stop in zip(bounds[:-1], bounds[1:])
        if int(start) < int(stop)
    )


def _exchange_tve_q_slab(
    *,
    q_start: int,
    q_stop: int,
    lambda_blocks: np.ndarray,
    v_over_a: np.ndarray,
    exchange_scale: float,
) -> tuple[tuple[int, np.ndarray, np.ndarray], ...]:
    """Return dense block-exchange contributions for one q-slab."""

    rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    scale = float(exchange_scale)
    for iq in range(int(q_start), int(q_stop)):
        v = 0.5 * scale * v_over_a[iq]
        if not np.any(v):
            continue
        lam_q = lambda_blocks[iq]
        forward = np.einsum(
            "g,gkac,gkbd->kabcd",
            v,
            lam_q,
            np.conj(lam_q),
            optimize=True,
        )
        reverse = np.einsum(
            "g,gkca,gkdb->kabcd",
            v,
            np.conj(lam_q),
            lam_q,
            optimize=True,
        )
        rows.append((iq, forward, reverse))
    return tuple(rows)


def _hermitize_dense_in_place(matrix: np.ndarray, *, tile_size: int = 1024) -> np.ndarray:
    """Replace ``matrix`` by ``0.5 * (matrix + matrix.conj().T)`` using tiles."""

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    n = matrix.shape[0]
    tile = max(1, int(tile_size))
    for row_start in range(0, n, tile):
        row_stop = min(row_start + tile, n)
        diagonal = matrix[row_start:row_stop, row_start:row_stop]
        diagonal[...] = 0.5 * (diagonal + diagonal.conj().T)
        for col_start in range(row_stop, n, tile):
            col_stop = min(col_start + tile, n)
            upper = matrix[row_start:row_stop, col_start:col_stop].copy()
            lower = matrix[col_start:col_stop, row_start:row_stop]
            sym = 0.5 * (upper + lower.conj().T)
            matrix[row_start:row_stop, col_start:col_stop] = sym
            matrix[col_start:col_stop, row_start:row_stop] = sym.conj().T
    return matrix


class ContinuumHFBackend:
    """TMD_HF-style optimized block HF backend for native continuum vertices."""

    def __init__(
        self,
        h0: np.ndarray,
        vertices: DensityVertices,
        interaction: ContinuumInteractionParams | None = None,
    ) -> None:
        self.h0 = hermitize(np.asarray(h0, dtype=complex))
        if self.h0.ndim != 3 or self.h0.shape[-1] != self.h0.shape[-2]:
            raise ValueError("h0 must have shape (n_blocks, dim, dim)")
        self.n_blocks, self.dim, _ = self.h0.shape
        self.n_total = self.n_blocks * self.dim
        self.vertices = vertices
        self.interaction = interaction or ContinuumInteractionParams()
        self.lambda_blocks = np.asarray(vertices.lambda_blocks, dtype=complex)
        if self.lambda_blocks.ndim != 5:
            raise ValueError("lambda_blocks must have shape (n_q, n_g, n_blocks, dim, dim)")
        if self.lambda_blocks.shape[2:] != self.h0.shape:
            raise ValueError("density vertices and h0 have incompatible shapes")
        self.n_q, self.n_g = self.lambda_blocks.shape[:2]
        self.target_minus_q = np.asarray(vertices.target_minus_q, dtype=int)
        if self.target_minus_q.shape != (self.n_q, self.n_blocks):
            raise ValueError("target_minus_q must have shape (n_q, n_blocks)")
        self.q_is_zero = np.asarray(vertices.q_is_zero, dtype=bool)
        if self.q_is_zero.shape != (self.n_q,):
            raise ValueError("q_is_zero must have shape (n_q,)")
        self.v_over_a = np.asarray(vertices.v_over_a, dtype=float)
        if self.v_over_a.shape != (self.n_q, self.n_g):
            raise ValueError("v_over_a must have shape (n_q, n_g)")
        self.p_ref = np.zeros_like(self.h0, dtype=complex)
        self.hartree_channels = self._find_hartree_channels()
        self.full_hartree_channels = tuple(self.hartree_channels)
        self._hartree_channel_lookup = {
            (int(iq), int(ig)): (int(iq), int(ig), float(v))
            for iq, ig, v in self.hartree_channels
        }
        self.tVE = self._build_exchange_tve()
        self._apply_density_vertex_retention()

    def as_block_density(self, P: np.ndarray) -> np.ndarray:
        arr = np.asarray(P, dtype=complex)
        if arr.shape == self.h0.shape:
            return hermitize(arr)
        if arr.shape == (self.n_total, self.n_total):
            out = np.empty_like(self.h0)
            for ik in range(self.n_blocks):
                start = ik * self.dim
                out[ik] = arr[start : start + self.dim, start : start + self.dim]
            return hermitize(out)
        raise ValueError(
            f"density must have shape {self.h0.shape} or {(self.n_total, self.n_total)}; "
            f"got {arr.shape}"
        )

    def _find_hartree_channels(self) -> list[tuple[int, int, float]]:
        channels: list[tuple[int, int, float]] = []
        scale = float(self.interaction.hartree_scale)
        if scale == 0.0:
            return channels
        for iq in range(self.n_q):
            if not bool(self.q_is_zero[iq]):
                continue
            for ig in range(self.n_g):
                if (
                    self.interaction.q0_hartree == "omit_uniform"
                    and self._is_uniform_channel(iq, ig)
                ):
                    continue
                v = scale * float(self.v_over_a[iq, ig])
                if v != 0.0:
                    channels.append((iq, ig, v))
        return channels

    def _is_uniform_channel(self, iq: int, ig: int) -> bool:
        if self.vertices.q_norm_nm_inv is not None:
            return bool(float(self.vertices.q_norm_nm_inv[int(iq), int(ig)]) < 1e-12)
        q = self.vertices.q_shifts[int(iq)]
        g = self.vertices.g_channels[int(ig)]
        return int(q[0]) + int(g[0]) == 0 and int(q[1]) + int(g[1]) == 0

    def _build_exchange_tve(self) -> np.ndarray:
        block_dim = self.dim * self.dim
        size = self.n_blocks * block_dim
        tVE = np.zeros((size, size), dtype=complex)
        scale = float(self.interaction.exchange_scale)
        if scale == 0.0:
            return tVE
        local = np.arange(block_dim)
        block_rows = np.arange(self.n_blocks)[:, None] * block_dim + local[None, :]
        if self.interaction.exchange_workers <= 1:
            self._build_exchange_tve_serial(tVE, block_rows, block_dim, scale)
        else:
            self._build_exchange_tve_parallel(tVE, block_rows, block_dim, scale)
        return _hermitize_dense_in_place(tVE)

    def _scatter_exchange_tve_q_contribution(
        self,
        tVE: np.ndarray,
        block_rows: np.ndarray,
        block_dim: int,
        iq: int,
        forward: np.ndarray,
        reverse: np.ndarray,
    ) -> None:
        local = np.arange(block_dim)
        target_rows = self.target_minus_q[int(iq), :, None] * block_dim + local[None, :]
        tVE[block_rows[:, :, None], target_rows[:, None, :]] += forward.reshape(
            self.n_blocks,
            block_dim,
            block_dim,
        )
        tVE[target_rows[:, :, None], block_rows[:, None, :]] += reverse.reshape(
            self.n_blocks,
            block_dim,
            block_dim,
        )

    def _build_exchange_tve_serial(
        self,
        tVE: np.ndarray,
        block_rows: np.ndarray,
        block_dim: int,
        scale: float,
    ) -> None:
        for iq in range(self.n_q):
            for q_index, forward, reverse in _exchange_tve_q_slab(
                q_start=iq,
                q_stop=iq + 1,
                lambda_blocks=self.lambda_blocks,
                v_over_a=self.v_over_a,
                exchange_scale=scale,
            ):
                self._scatter_exchange_tve_q_contribution(
                    tVE,
                    block_rows,
                    block_dim,
                    q_index,
                    forward,
                    reverse,
                )

    def _build_exchange_tve_parallel(
        self,
        tVE: np.ndarray,
        block_rows: np.ndarray,
        block_dim: int,
        scale: float,
    ) -> None:
        from joblib import Parallel, delayed

        n_jobs = max(1, min(int(self.interaction.exchange_workers), self.n_q))
        ranges = _exchange_q_slab_ranges(self.n_q, n_jobs)
        tasks = (
            delayed(_exchange_tve_q_slab)(
                q_start=start,
                q_stop=stop,
                lambda_blocks=self.lambda_blocks,
                v_over_a=self.v_over_a,
                exchange_scale=scale,
            )
            for start, stop in ranges
        )
        results = Parallel(
            n_jobs=n_jobs,
            backend="loky",
            return_as="generator",
            mmap_mode="r",
            max_nbytes="32M",
        )(tasks)
        for slab_rows in results:
            for q_index, forward, reverse in slab_rows:
                self._scatter_exchange_tve_q_contribution(
                    tVE,
                    block_rows,
                    block_dim,
                    q_index,
                    forward,
                    reverse,
                )

    def _empty_retained_lambdas(self) -> np.ndarray:
        return np.zeros((0, 0, self.n_blocks, self.dim, self.dim), dtype=complex)

    def _vertices_without_lambda_blocks(self) -> DensityVertices:
        return DensityVertices(
            q_shifts=self.vertices.q_shifts,
            target_minus_q=np.asarray(self.vertices.target_minus_q, dtype=int).copy(),
            q_is_zero=np.asarray(self.vertices.q_is_zero, dtype=bool).copy(),
            lambda_blocks=self._empty_retained_lambdas(),
            v_over_a=np.asarray(self.vertices.v_over_a, dtype=float).copy(),
            g_channels=self.vertices.g_channels,
            channel_in_disk=(
                None
                if self.vertices.channel_in_disk is None
                else np.asarray(self.vertices.channel_in_disk, dtype=bool).copy()
            ),
            q_vectors_nm_inv=(
                None
                if self.vertices.q_vectors_nm_inv is None
                else np.asarray(self.vertices.q_vectors_nm_inv, dtype=float).copy()
            ),
            q_norm_nm_inv=(
                None
                if self.vertices.q_norm_nm_inv is None
                else np.asarray(self.vertices.q_norm_nm_inv, dtype=float).copy()
            ),
            v_q=(
                None
                if self.vertices.v_q is None
                else np.asarray(self.vertices.v_q, dtype=float).copy()
            ),
        )

    def _apply_density_vertex_retention(self) -> None:
        policy = str(self.interaction.density_vertex_retention)
        if policy == "full":
            return
        if policy != "hartree_only":
            raise ValueError("density_vertex_retention must be 'full' or 'hartree_only'")

        original_lambdas = self.lambda_blocks
        original_targets = self.target_minus_q
        original_v_over_a = self.v_over_a
        retained_count = len(self.full_hartree_channels)
        if retained_count == 0:
            self.lambda_blocks = self._empty_retained_lambdas()
            self.target_minus_q = np.zeros((0, self.n_blocks), dtype=int)
            self.q_is_zero = np.zeros(0, dtype=bool)
            self.v_over_a = np.zeros((0, 0), dtype=float)
            self.hartree_channels = []
            self._hartree_channel_lookup = {}
            self.n_q, self.n_g = 0, 0
            self.vertices = self._vertices_without_lambda_blocks()
            return

        retained_lambdas = np.empty(
            (retained_count, 1, self.n_blocks, self.dim, self.dim),
            dtype=complex,
        )
        retained_targets = np.empty((retained_count, self.n_blocks), dtype=int)
        retained_v_over_a = np.empty((retained_count, 1), dtype=float)
        retained_q_is_zero = np.ones(retained_count, dtype=bool)
        retained_channels: list[tuple[int, int, float]] = []
        lookup: dict[tuple[int, int], tuple[int, int, float]] = {}
        for new_iq, (old_iq, old_ig, v) in enumerate(self.full_hartree_channels):
            retained_lambdas[new_iq, 0] = original_lambdas[int(old_iq), int(old_ig)]
            retained_targets[new_iq] = original_targets[int(old_iq)]
            retained_v_over_a[new_iq, 0] = original_v_over_a[int(old_iq), int(old_ig)]
            retained_channels.append((new_iq, 0, float(v)))
            lookup[(int(old_iq), int(old_ig))] = (new_iq, 0, float(v))

        self.lambda_blocks = retained_lambdas
        self.target_minus_q = retained_targets
        self.q_is_zero = retained_q_is_zero
        self.v_over_a = retained_v_over_a
        self.hartree_channels = retained_channels
        self._hartree_channel_lookup = lookup
        self.n_q, self.n_g = self.lambda_blocks.shape[:2]
        self.vertices = self._vertices_without_lambda_blocks()

    def hartree_lambda_for_channel(self, iq: int, ig: int) -> np.ndarray | None:
        retained = self._hartree_channel_lookup.get((int(iq), int(ig)))
        if retained is None:
            return None
        retained_iq, retained_ig, _v = retained
        return self.lambda_blocks[int(retained_iq), int(retained_ig)]

    def hartree_weight_for_channel(self, iq: int, ig: int) -> float | None:
        retained = self._hartree_channel_lookup.get((int(iq), int(ig)))
        if retained is None:
            return None
        return float(retained[2])

    def hartree_hamiltonian(self, Q: np.ndarray) -> np.ndarray:
        density = self.as_block_density(Q)
        out = np.zeros_like(density, dtype=complex)
        for iq, ig, v in self.hartree_channels:
            lam = self.lambda_blocks[iq, ig]
            rho = np.einsum("kab,kba->", lam, density, optimize=True)
            out += 0.5 * v * (np.conj(rho) * lam + rho * np.swapaxes(lam.conj(), -1, -2))
        return hermitize(out)

    def fock_hamiltonian(self, Q: np.ndarray) -> np.ndarray:
        density = self.as_block_density(Q)
        out = -np.reshape(self.tVE @ np.reshape(density, (-1,)), density.shape)
        return hermitize(out)

    def hf_hamiltonian(self, P: np.ndarray) -> np.ndarray:
        density = self.as_block_density(P)
        Q = density - self.p_ref
        return hermitize(self.h0 + self.hartree_hamiltonian(Q) + self.fock_hamiltonian(Q))

    def interaction_components(self, Q: np.ndarray) -> tuple[float, float]:
        density = self.as_block_density(Q)
        hartree = 0.0
        for iq, ig, v in self.hartree_channels:
            lam = self.lambda_blocks[iq, ig]
            rho = np.einsum("kab,kba->", lam, density, optimize=True)
            hartree += 0.5 * v * float(np.real(rho * np.conj(rho)))
        fock = 0.5 * block_trace_product(self.fock_hamiltonian(density), density)
        return hartree, fock

    def interaction_energy(self, Q: np.ndarray) -> float:
        hartree, fock = self.interaction_components(Q)
        return float(hartree + fock)

    def energy(self, P: np.ndarray) -> EnergyComponents:
        density = self.as_block_density(P)
        one_body = block_trace_product(self.h0, density)
        hartree, fock = self.interaction_components(density - self.p_ref)
        return EnergyComponents(
            total=float(one_body + hartree + fock),
            one_body=float(one_body),
            hartree=float(hartree),
            fock=float(fock),
        )

    def total_energy(self, P: np.ndarray) -> float:
        return self.energy(P).total

    def update_density(
        self,
        H: np.ndarray,
        n_particles: float,
        constraint=None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        blocks = hermitize(np.asarray(H, dtype=complex))
        if blocks.shape != self.h0.shape:
            raise ValueError("H has the wrong shape")
        if constraint is not None and hasattr(constraint, "update_density_global"):
            return constraint.update_density_global(blocks, n_particles)
        if constraint is not None:
            blocks = constraint.project_operator(blocks)
        P, evals, direct, indirect = _global_aufbau(blocks, n_particles)
        if constraint is not None:
            P = constraint.project_density(P)
        return P, evals, direct, indirect

    def update_density_per_k(
        self,
        H: np.ndarray,
        n_occ_per_k: int,
        constraint=None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        blocks = hermitize(np.asarray(H, dtype=complex))
        if blocks.shape != self.h0.shape:
            raise ValueError("H has the wrong shape")
        if constraint is not None and hasattr(constraint, "update_density"):
            return constraint.update_density(blocks, n_occ_per_k)
        if constraint is not None:
            blocks = constraint.project_operator(blocks)
        return _fixed_per_k_aufbau(blocks, n_occ_per_k)


def _commutator_norm(H: np.ndarray, P: np.ndarray) -> float:
    comm = np.einsum("kab,kbc->kac", H, P, optimize=True) - np.einsum(
        "kab,kbc->kac", P, H, optimize=True
    )
    return float(np.linalg.norm(comm))


def _expected_trace(backend: ContinuumHFBackend, params: ContinuumHFParams) -> float:
    return float(backend.n_blocks * params.n_occ_per_k)


def _choose_oda_lambda(s: float, c: float, lambda_min: float) -> tuple[float, str | None]:
    """Choose the ODA damping lambda for E(lambda)=E0+s lambda+c lambda^2/2."""

    if c <= 0.0:
        return 1.0, "nonpositive_curvature_full_step"
    lam = float(np.clip(-float(s) / float(c), 0.0, 1.0))
    if lam <= float(lambda_min):
        return 1.0, "zero_lambda_full_step"
    return lam, None


def compute_hf_diagnostics(
    backend: ContinuumHFBackend,
    P: np.ndarray,
    params: ContinuumHFParams,
    *,
    constraint=None,
    P_prev: np.ndarray | None = None,
    energy_prev: float | None = None,
    iteration: int = 0,
    density_kind: Literal["mixed", "final_idempotent"] = "mixed",
    lambda_value: float | None = None,
    fallback_reason: str | None = None,
) -> ContinuumHFDiagnostics:
    """Compute scalar diagnostics for one density."""

    density = backend.as_block_density(P)
    H = backend.hf_hamiltonian(density)
    H_projected = constraint.project_operator(H) if constraint is not None else H
    expected_trace = _expected_trace(backend, params)
    P_aufbau, _evals, direct, indirect = backend.update_density_per_k(
        H_projected,
        params.n_occ_per_k,
        constraint,
    )
    energy = backend.energy(density).total
    idem_fro, idem_max = projector_idempotency_errors(density)
    trace = np.trace(density, axis1=-2, axis2=-1)
    residual = float(np.linalg.norm(density - P_aufbau))
    constraint_error = (
        float(constraint.symmetry_error(density)) if constraint is not None else 0.0
    )
    return ContinuumHFDiagnostics(
        energy=float(energy),
        delta_energy=float("nan") if energy_prev is None else float(energy - energy_prev),
        delta_P=float("nan")
        if P_prev is None
        else float(np.linalg.norm(density - backend.as_block_density(P_prev))),
        idempotency_error_fro=idem_fro,
        idempotency_error_max=idem_max,
        constraint_error=constraint_error,
        aufbau_residual_norm=residual,
        commutator_norm=_commutator_norm(H_projected, density),
        trace_error=float(abs(np.real(np.sum(trace)) - expected_trace)),
        direct_gap_min=float(direct),
        indirect_gap=float(indirect),
        iteration=int(iteration),
        constraint_name=getattr(constraint, "name", None) if constraint is not None else None,
        lambda_value=lambda_value,
        fallback_reason=fallback_reason,
        density_kind=density_kind,
        self_consistency_warning=bool(residual > params.final_residual_tolerance),
    )


def solve_hf(
    backend: ContinuumHFBackend,
    P_init: np.ndarray,
    params: ContinuumHFParams | None = None,
    *,
    constraint=None,
    seed: str = "",
    on_iteration: HFIterationCallback | None = None,
) -> ContinuumHFResult:
    """Run zero-temperature fixed-per-k HF with TMD_HF-style ODA mixing."""

    controls = params or ContinuumHFParams()
    energy_tol = controls.energy_tolerance
    P = backend.as_block_density(P_init)
    if constraint is not None:
        P = constraint.project_density(P)
    history: list[ContinuumHFDiagnostics] = []
    snapshots: list[ContinuumHFIterationSnapshot] = []
    converged = False
    diagnostics = compute_hf_diagnostics(
        backend,
        P,
        controls,
        constraint=constraint,
        iteration=0,
    )
    energy = diagnostics.energy
    n_iter = 0
    for iteration in range(1, controls.max_iter + 1):
        n_iter = iteration
        P_prev = P
        energy_prev = energy
        H_prev = backend.hf_hamiltonian(P_prev)
        H_projected = constraint.project_operator(H_prev) if constraint is not None else H_prev
        P_aufbau, _evals, _direct, _indirect = backend.update_density_per_k(
            H_projected,
            controls.n_occ_per_k,
            constraint,
        )
        delta = hermitize(P_aufbau - P_prev)
        fallback_reason = None
        if controls.mixing_method == "oda":
            s = block_trace_product(H_projected, delta)
            c = 2.0 * backend.interaction_energy(delta)
            mix, fallback_reason = _choose_oda_lambda(s, c, controls.oda_lambda_min)
        else:
            mix = float(controls.mixing)
        P = hermitize(P_prev + mix * delta)
        if constraint is not None:
            P = constraint.project_density(P)
        energy = backend.energy(P).total
        diagnostics = compute_hf_diagnostics(
            backend,
            P,
            controls,
            constraint=constraint,
            P_prev=P_prev,
            energy_prev=energy_prev,
            iteration=iteration,
            lambda_value=float(mix),
            fallback_reason=fallback_reason,
        )
        history.append(diagnostics)
        should_snapshot = controls.store_projector_snapshots and (
            (controls.first_iteration_snapshot and iteration == 1)
            or (iteration % controls.snapshot_interval == 0)
        )
        if should_snapshot:
            snapshots.append(
                ContinuumHFIterationSnapshot(
                    iteration=iteration,
                    P=P.copy(),
                    energy=float(diagnostics.energy),
                    diagnostics=diagnostics,
                )
            )
        if on_iteration is not None:
            on_iteration(
                iteration,
                P.copy(),
                float(diagnostics.energy),
                diagnostics,
                bool(should_snapshot),
            )
        if (
            iteration >= controls.min_iter
            and diagnostics.commutator_norm < controls.tolerance
            and diagnostics.aufbau_residual_norm < controls.tolerance
            and diagnostics.constraint_error < controls.tolerance
            and diagnostics.trace_error < controls.tolerance
            and abs(diagnostics.delta_energy) < energy_tol
        ):
            converged = True
            break

    H_mixed = backend.hf_hamiltonian(P)
    H_projected = constraint.project_operator(H_mixed) if constraint is not None else H_mixed
    P_final, _evals, _direct, _indirect = backend.update_density_per_k(
        H_projected,
        controls.n_occ_per_k,
        constraint,
    )
    final_H_raw = backend.hf_hamiltonian(P_final)
    final_H = constraint.project_operator(final_H_raw) if constraint is not None else final_H_raw
    final_diagnostics = compute_hf_diagnostics(
        backend,
        P_final,
        controls,
        constraint=constraint,
        P_prev=P,
        energy_prev=diagnostics.energy,
        iteration=n_iter,
        density_kind="final_idempotent",
    )
    if final_diagnostics.self_consistency_warning:
        converged = False
    return ContinuumHFResult(
        P=P_final,
        H_hf=final_H,
        energy=backend.energy(P_final).total,
        converged=converged,
        n_iter=n_iter,
        diagnostics=final_diagnostics,
        history=tuple(history),
        snapshots=tuple(snapshots),
        seed=seed,
        constraint_name=getattr(constraint, "name", None) if constraint is not None else None,
    )


def scalar_channel(blocks: np.ndarray) -> np.ndarray:
    arr = np.asarray(blocks, dtype=complex)
    dim = arr.shape[-1]
    trace = np.trace(arr, axis1=-2, axis2=-1) / float(dim)
    return trace[..., None, None] * np.eye(dim, dtype=complex)


def intervalley_channel(blocks: np.ndarray) -> np.ndarray:
    arr = np.asarray(blocks, dtype=complex)
    dim = arr.shape[-1]
    if dim % 2:
        raise ValueError("active-space dimension must be even")
    n = dim // 2
    out = np.zeros_like(arr)
    out[..., :n, n:] = arr[..., :n, n:]
    out[..., n:, :n] = arr[..., n:, :n]
    return hermitize(out)


def valley_diagonal_channel(blocks: np.ndarray) -> np.ndarray:
    arr = np.asarray(blocks, dtype=complex)
    return hermitize(arr - intervalley_channel(arr))


def reference_hamiltonian_diagnostics(H: np.ndarray) -> ReferenceHamiltonianDiagnostics:
    arr = np.asarray(H, dtype=complex)
    scalar = scalar_channel(arr)
    return ReferenceHamiltonianDiagnostics(
        scalar_norm=float(np.linalg.norm(scalar)),
        traceless_norm=float(np.linalg.norm(arr - scalar)),
        valley_diagonal_norm=float(np.linalg.norm(valley_diagonal_channel(arr))),
        intervalley_norm=float(np.linalg.norm(intervalley_channel(arr))),
        hermiticity_error=float(np.max(np.abs(arr - arr.conj().swapaxes(-1, -2)))),
    )
