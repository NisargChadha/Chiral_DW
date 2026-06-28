"""Symmetry-constrained HF references and convex variational paths."""

from __future__ import annotations

import numpy as np

from chiral_dw.config import ContinuumHFParams
from chiral_dw.continuum.builder import build_continuum_bundle
from chiral_dw.continuum.hf import (
    ContinuumHFBackend,
    reference_hamiltonian_diagnostics,
    solve_hf,
)
from chiral_dw.continuum.models import (
    ConvexPathDiagnostics,
    ContinuumBundle,
    ContinuumHFResult,
    ReferenceHamiltonianDiagnostics,
    SymmetricHFReferences,
    hermitize,
    projector_idempotency_errors,
)
from chiral_dw.continuum.seeds import build_seed, mix_projector_seeds, random_projector_like_seed
from chiral_dw.continuum.symmetry import (
    TPrimeConstraint,
    ValleyU1Constraint,
    _fixed_per_k_aufbau,
    rotate_valley_u1,
)


def solve_reference_hf(
    bundle: ContinuumBundle,
    seed_name: str,
    params: ContinuumHFParams | None = None,
    *,
    constraint=None,
) -> ContinuumHFResult:
    """Solve one native continuum HF reference from a named seed."""

    controls = params or ContinuumHFParams()
    P0 = build_seed(
        seed_name,
        bundle.active,
        n_occ_per_k=controls.n_occ_per_k,
        ivc_angle=controls.ivc_angle,
        ivc_phase=controls.ivc_phase,
        random_seed_value=controls.random_seed,
    )
    if controls.seed_random_weight > 0.0:
        P_noise = random_projector_like_seed(P0, seed=controls.random_seed)
        if constraint is not None:
            P_noise = constraint.project_density(P_noise)
        P0 = mix_projector_seeds(
            P0,
            P_noise,
            ordered_weight=controls.seed_ordered_weight,
            random_weight=controls.seed_random_weight,
        )
        if constraint is not None:
            P0 = constraint.project_density(P0)
    return solve_hf(bundle.backend, P0, controls, constraint=constraint, seed=seed_name)


def build_symmetric_hf_references(
    bundle: ContinuumBundle | None = None,
    params: ContinuumHFParams | None = None,
) -> SymmetricHFReferences:
    """Solve VP+, VP-, and T-prime IVC native HF references."""

    hf_params = params or ContinuumHFParams()
    work_bundle = bundle or build_continuum_bundle()
    vp_plus_constraint = ValleyU1Constraint(work_bundle.active)
    vp_minus_constraint = ValleyU1Constraint(work_bundle.active)
    tprime_constraint = TPrimeConstraint(work_bundle.active)
    vp_plus = solve_reference_hf(
        work_bundle,
        "vp_plus",
        hf_params,
        constraint=vp_plus_constraint,
    )
    vp_minus = solve_reference_hf(
        work_bundle,
        "vp_minus",
        hf_params,
        constraint=vp_minus_constraint,
    )
    ivc_seed_name = "finite_q_ivc" if work_bundle.active.finite_q_enabled else "ivc"
    ivc = solve_reference_hf(
        work_bundle,
        ivc_seed_name,
        hf_params,
        constraint=tprime_constraint,
    )
    return SymmetricHFReferences(
        vp_plus=vp_plus,
        vp_minus=vp_minus,
        ivc=ivc,
        n_occ_per_k=hf_params.n_occ_per_k,
    )


def convex_weights(theta: float) -> tuple[float, float, float]:
    """Return (w_vp_plus, w_vp_minus, w_ivc) for the symmetric path."""

    c = float(np.cos(float(theta)))
    s = float(np.sin(float(theta)))
    return max(c, 0.0) ** 2, max(-c, 0.0) ** 2, s * s


def symmetric_convex_hamiltonian(
    refs: SymmetricHFReferences,
    theta: float,
    phi: float = 0.0,
) -> np.ndarray:
    """Build the full-HF convex variational Hamiltonian."""

    w_plus, w_minus, w_ivc = convex_weights(theta)
    rotated_ivc = rotate_valley_u1(refs.H_ivc, phi)
    H = w_plus * refs.H_vp_plus + w_minus * refs.H_vp_minus + w_ivc * rotated_ivc
    return hermitize(H)


def symmetric_convex_projector(
    refs: SymmetricHFReferences,
    theta: float,
    phi: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, ConvexPathDiagnostics]:
    """Diagonalize one symmetric convex Hamiltonian into a fixed-per-k projector."""

    H = symmetric_convex_hamiltonian(refs, theta, phi)
    P, _evals, direct, indirect = _fixed_per_k_aufbau(H, refs.n_occ_per_k)
    idem_fro, idem_max = projector_idempotency_errors(P)
    w_plus, w_minus, w_ivc = convex_weights(theta)
    diagnostics = ConvexPathDiagnostics(
        theta=float(theta),
        phi=float(phi),
        w_vp_plus=float(w_plus),
        w_vp_minus=float(w_minus),
        w_ivc=float(w_ivc),
        direct_gap_min=float(direct),
        indirect_gap=float(indirect),
        projector_idempotency_error_fro=idem_fro,
        projector_idempotency_error_max=idem_max,
    )
    return P, H, diagnostics


def symmetric_convex_path(
    refs: SymmetricHFReferences,
    theta_nodes: np.ndarray,
    phi: float = 0.0,
) -> tuple[np.ndarray, tuple[ConvexPathDiagnostics, ...]]:
    """Build a theta-indexed projector path from the three HF references."""

    theta = np.asarray(theta_nodes, dtype=float)
    projectors = np.zeros((theta.size, refs.n_blocks, refs.dim, refs.dim), dtype=complex)
    diagnostics: list[ConvexPathDiagnostics] = []
    for idx, value in enumerate(theta):
        P, _H, row = symmetric_convex_projector(refs, float(value), phi)
        projectors[idx] = P
        diagnostics.append(row)
    return projectors, tuple(diagnostics)


def reference_diagnostics(
    refs: SymmetricHFReferences,
) -> dict[str, ReferenceHamiltonianDiagnostics]:
    """Return channel diagnostics for all three raw reference Hamiltonians."""

    return {
        "vp_plus": reference_hamiltonian_diagnostics(refs.H_vp_plus),
        "vp_minus": reference_hamiltonian_diagnostics(refs.H_vp_minus),
        "ivc": reference_hamiltonian_diagnostics(refs.H_ivc),
    }


__all__ = [
    "ContinuumHFBackend",
    "build_symmetric_hf_references",
    "convex_weights",
    "reference_diagnostics",
    "solve_reference_hf",
    "symmetric_convex_hamiltonian",
    "symmetric_convex_path",
    "symmetric_convex_projector",
]
