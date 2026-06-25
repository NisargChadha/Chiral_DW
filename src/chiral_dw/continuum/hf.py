"""Native zero-temperature fixed-occupation continuum Hartree-Fock."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from chiral_dw.config import ContinuumHFParams, ContinuumInteractionParams
from chiral_dw.continuum.models import (
    ContinuumHFDiagnostics,
    ContinuumHFResult,
    DensityVertices,
    ReferenceHamiltonianDiagnostics,
    block_trace_product,
    hermitize,
    projector_idempotency_errors,
)
from chiral_dw.continuum.symmetry import _fixed_per_k_aufbau


@dataclass(frozen=True)
class EnergyComponents:
    """One-body, Hartree, Fock, and total HF energy."""

    total: float
    one_body: float
    hartree: float
    fock: float


class ContinuumHFBackend:
    """Small block HF backend using native projected density vertices."""

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
        self.vertices = vertices
        self.interaction = interaction or ContinuumInteractionParams()
        self.lambda_blocks = np.asarray(vertices.lambda_blocks, dtype=complex)
        self.target_minus_q = np.asarray(vertices.target_minus_q, dtype=int)
        self.q_is_zero = np.asarray(vertices.q_is_zero, dtype=bool)
        self.v_over_a = np.asarray(vertices.v_over_a, dtype=float)
        if self.lambda_blocks.shape[2:] != self.h0.shape:
            raise ValueError("density vertices and h0 have incompatible shapes")
        self.n_q, self.n_g = self.lambda_blocks.shape[:2]

    def as_block_density(self, P: np.ndarray) -> np.ndarray:
        arr = np.asarray(P, dtype=complex)
        if arr.shape != self.h0.shape:
            raise ValueError(f"density must have shape {self.h0.shape}, got {arr.shape}")
        return hermitize(arr)

    def hartree_hamiltonian(self, P: np.ndarray) -> np.ndarray:
        Q = self.as_block_density(P)
        out = np.zeros_like(Q, dtype=complex)
        if self.interaction.hartree_scale == 0.0:
            return out
        for iq in range(self.n_q):
            if self.q_is_zero[iq] and self.interaction.q0_hartree == "omit_uniform":
                continue
            for ig in range(self.n_g):
                v = float(self.v_over_a[iq, ig]) * float(self.interaction.hartree_scale)
                if v == 0.0:
                    continue
                lam = self.lambda_blocks[iq, ig]
                density = np.einsum("kab,kba->", lam, Q, optimize=True)
                out += 0.5 * v * (
                    np.conj(density) * lam + density * np.swapaxes(lam.conj(), -1, -2)
                )
        return hermitize(out)

    def fock_hamiltonian(self, P: np.ndarray) -> np.ndarray:
        Q = self.as_block_density(P)
        out = np.zeros_like(Q, dtype=complex)
        if self.interaction.exchange_scale == 0.0:
            return out
        for iq in range(self.n_q):
            targets = self.target_minus_q[iq]
            for ig in range(self.n_g):
                v = float(self.v_over_a[iq, ig]) * float(self.interaction.exchange_scale)
                if v == 0.0:
                    continue
                lam = self.lambda_blocks[iq, ig]
                for ik in range(self.n_blocks):
                    jk = int(targets[ik])
                    out[ik] -= v * lam[ik] @ Q[jk] @ lam[ik].conj().T
        return hermitize(out)

    def hf_hamiltonian(self, P: np.ndarray) -> np.ndarray:
        Q = self.as_block_density(P)
        return hermitize(self.h0 + self.hartree_hamiltonian(Q) + self.fock_hamiltonian(Q))

    def energy(self, P: np.ndarray) -> EnergyComponents:
        Q = self.as_block_density(P)
        one_body = block_trace_product(self.h0, Q)
        Hh = self.hartree_hamiltonian(Q)
        Hf = self.fock_hamiltonian(Q)
        hartree = 0.5 * block_trace_product(Hh, Q)
        fock = 0.5 * block_trace_product(Hf, Q)
        return EnergyComponents(
            total=float(one_body + hartree + fock),
            one_body=float(one_body),
            hartree=float(hartree),
            fock=float(fock),
        )

    def update_density(
        self,
        H: np.ndarray,
        n_occ_per_k: int,
        constraint=None,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        if constraint is not None and hasattr(constraint, "update_density"):
            return constraint.update_density(H, n_occ_per_k)
        return _fixed_per_k_aufbau(H, n_occ_per_k)


def _commutator_norm(H: np.ndarray, P: np.ndarray) -> float:
    comm = H @ P - P @ H
    return float(np.linalg.norm(comm))


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
) -> ContinuumHFDiagnostics:
    """Compute scalar diagnostics for one density."""

    H = backend.hf_hamiltonian(P)
    H_projected = constraint.project_operator(H) if constraint is not None else H
    P_aufbau, _evals, direct, indirect = backend.update_density(
        H_projected,
        params.n_occ_per_k,
        constraint,
    )
    energy = backend.energy(P).total
    idem_fro, idem_max = projector_idempotency_errors(P)
    trace = np.trace(P, axis1=-2, axis2=-1)
    expected_trace = backend.n_blocks * params.n_occ_per_k
    residual = float(np.linalg.norm(P - P_aufbau))
    constraint_error = (
        float(constraint.symmetry_error(P)) if constraint is not None else 0.0
    )
    return ContinuumHFDiagnostics(
        energy=float(energy),
        delta_energy=float("nan") if energy_prev is None else float(energy - energy_prev),
        delta_P=float("nan")
        if P_prev is None
        else float(np.linalg.norm(P - backend.as_block_density(P_prev))),
        idempotency_error_fro=idem_fro,
        idempotency_error_max=idem_max,
        constraint_error=constraint_error,
        aufbau_residual_norm=residual,
        commutator_norm=_commutator_norm(H_projected, P),
        trace_error=float(abs(np.real(np.sum(trace)) - expected_trace)),
        direct_gap_min=float(direct),
        indirect_gap=float(indirect),
        iteration=int(iteration),
        constraint_name=getattr(constraint, "name", None) if constraint is not None else None,
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
) -> ContinuumHFResult:
    """Run fixed-per-k zero-temperature HF with linear density mixing."""

    controls = params or ContinuumHFParams()
    P = backend.as_block_density(P_init)
    if constraint is not None:
        P = constraint.project_density(P)
    history: list[ContinuumHFDiagnostics] = []
    converged = False
    energy_prev: float | None = None
    diagnostics = compute_hf_diagnostics(
        backend,
        P,
        controls,
        constraint=constraint,
        iteration=0,
    )
    n_iter = 0
    for iteration in range(1, controls.max_iter + 1):
        n_iter = iteration
        H = backend.hf_hamiltonian(P)
        H_projected = constraint.project_operator(H) if constraint is not None else H
        P_next, _evals, _direct, _indirect = backend.update_density(
            H_projected,
            controls.n_occ_per_k,
            constraint,
        )
        P_prev = P
        energy_before = backend.energy(P_prev).total
        mix = float(controls.mixing)
        P = hermitize((1.0 - mix) * P_prev + mix * P_next)
        if constraint is not None:
            P = constraint.project_density(P)
        diagnostics = compute_hf_diagnostics(
            backend,
            P,
            controls,
            constraint=constraint,
            P_prev=P_prev,
            energy_prev=energy_prev,
            iteration=iteration,
        )
        history.append(diagnostics)
        delta_e = abs(diagnostics.energy - energy_before)
        energy_prev = diagnostics.energy
        if (
            iteration >= controls.min_iter
            and diagnostics.aufbau_residual_norm < controls.tolerance
            and diagnostics.constraint_error < controls.tolerance
            and diagnostics.trace_error < controls.tolerance
            and delta_e < controls.energy_tolerance
        ):
            converged = True
            break

    H_mixed = backend.hf_hamiltonian(P)
    H_projected = constraint.project_operator(H_mixed) if constraint is not None else H_mixed
    P_final, _evals, _direct, _indirect = backend.update_density(
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
