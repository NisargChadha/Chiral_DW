"""Native continuum HF seed projectors."""

from __future__ import annotations

import numpy as np

from chiral_dw.continuum.models import ContinuumActiveSpace, VALLEY_K, VALLEY_KPRIME, hermitize
from chiral_dw.continuum.symmetry import _fixed_per_k_aufbau


def valley_polarized_seed(
    active: ContinuumActiveSpace,
    valley: str = VALLEY_K,
    *,
    n_occ_per_k: int = 1,
) -> np.ndarray:
    """Fill the lowest active states in one valley at every momentum."""

    iv = active.valley_index(valley)
    n = active.n_active
    n_occ = int(n_occ_per_k)
    if n_occ > n:
        raise ValueError("a fully valley-polarized seed needs n_occ_per_k <= n_active")
    P = np.zeros((active.n_k, active.dim, active.dim), dtype=complex)
    start = iv * n
    order = np.argsort(active.hole_energies[:, iv, :], axis=1)
    for ik in range(active.n_k):
        for local in order[ik, :n_occ]:
            a = start + int(local)
            P[ik, a, a] = 1.0
    return P


def ivc_seed(
    active: ContinuumActiveSpace,
    *,
    n_occ_per_k: int = 1,
    angle: float = 0.5 * np.pi,
    phase: float = 0.0,
) -> np.ndarray:
    """Build a Q=0 intervalley-coherent seed at every momentum."""

    n_occ = int(n_occ_per_k)
    if n_occ > active.n_active:
        raise ValueError("ivc_seed needs n_occ_per_k <= n_active")
    c = np.cos(0.5 * float(angle))
    s = np.sin(0.5 * float(angle))
    P = np.zeros((active.n_k, active.dim, active.dim), dtype=complex)
    energies = 0.5 * (active.hole_energies[:, 0, :] + active.hole_energies[:, 1, :])
    order = np.argsort(energies, axis=1)
    for ik in range(active.n_k):
        for band in order[ik, :n_occ]:
            b = int(band)
            v = np.zeros(active.dim, dtype=complex)
            v[b] = c
            v[active.n_active + b] = np.exp(1j * float(phase)) * s
            P[ik] += np.outer(v, v.conj())
    return hermitize(P)


def finite_q_ivc_seed(
    active: ContinuumActiveSpace,
    *,
    n_occ_per_k: int = 1,
    angle: float = 0.5 * np.pi,
    phase: float = 0.0,
) -> np.ndarray:
    """Build an active-frame IVC seed for a finite-Q active space."""

    if not active.finite_q_enabled:
        raise ValueError("finite_q_ivc seed requires an active space built with finite_q enabled")
    return ivc_seed(active, n_occ_per_k=n_occ_per_k, angle=angle, phase=phase)


def random_seed(
    active: ContinuumActiveSpace,
    *,
    n_occ_per_k: int = 1,
    seed: int = 1,
    scale: float = 1e-3,
) -> np.ndarray:
    """Build a deterministic random Slater seed from a perturbed h0."""

    rng = np.random.default_rng(int(seed))
    H = active.h0.copy()
    h_scale = max(1.0, float(np.std(np.real(np.diagonal(H, axis1=-2, axis2=-1)))))
    for ik in range(active.n_k):
        noise = rng.normal(size=(active.dim, active.dim)) + 1j * rng.normal(
            size=(active.dim, active.dim)
        )
        H[ik] += float(scale) * h_scale * hermitize(noise)
    P, _evals, _direct, _indirect = _fixed_per_k_aufbau(H, n_occ_per_k)
    return P


def _haar_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(z)
    phases = np.diag(r)
    phases = np.where(np.abs(phases) > 1e-15, phases / np.abs(phases), 1.0)
    return q * phases.conj()


def random_projector_like_seed(
    reference: np.ndarray,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Randomly rotate a projector while preserving each block spectrum."""

    generator = np.random.default_rng(seed) if rng is None else rng
    P_ref = np.asarray(reference, dtype=complex)
    if P_ref.ndim != 3 or P_ref.shape[-1] != P_ref.shape[-2]:
        raise ValueError("reference projector must have shape (n_k, dim, dim)")
    n_k, dim, _ = P_ref.shape
    out = np.empty_like(P_ref)
    for ik in range(n_k):
        U = _haar_unitary(dim, generator)
        out[ik] = U @ P_ref[ik] @ U.conj().T
    return hermitize(out)


def mix_projector_seeds(
    ordered: np.ndarray,
    random: np.ndarray,
    *,
    ordered_weight: float = 1.0,
    random_weight: float = 0.0,
    atol: float = 1e-12,
) -> np.ndarray:
    """Convexly mix two seed density matrices and Hermitian-symmetrize."""

    P_ordered = np.asarray(ordered, dtype=complex)
    P_random = np.asarray(random, dtype=complex)
    if P_ordered.shape != P_random.shape:
        raise ValueError("seed projectors must have the same shape")
    w_ordered = float(ordered_weight)
    w_random = float(random_weight)
    if not (np.isfinite(w_ordered) and np.isfinite(w_random)):
        raise ValueError("seed weights must be finite")
    if w_ordered < -atol or w_random < -atol:
        raise ValueError("seed weights must be nonnegative")
    if not np.isclose(w_ordered + w_random, 1.0, atol=atol):
        raise ValueError("seed weights must sum to one")
    return hermitize(w_ordered * P_ordered + w_random * P_random)


def build_seed(
    name: str,
    active: ContinuumActiveSpace,
    *,
    n_occ_per_k: int = 1,
    ivc_angle: float = 0.5 * np.pi,
    ivc_phase: float = 0.0,
    random_seed_value: int = 1,
) -> np.ndarray:
    """Build a named native continuum HF seed."""

    key = str(name).strip().lower()
    if key in {"vp_plus", "vp_k", "k"}:
        return valley_polarized_seed(active, VALLEY_K, n_occ_per_k=n_occ_per_k)
    if key in {"vp_minus", "vp_kprime", "kprime"}:
        return valley_polarized_seed(active, VALLEY_KPRIME, n_occ_per_k=n_occ_per_k)
    if key in {"ivc", "ivc_q0"}:
        return ivc_seed(
            active,
            n_occ_per_k=n_occ_per_k,
            angle=ivc_angle,
            phase=ivc_phase,
        )
    if key in {"finite_q_ivc", "finite_q_ivc_minus"}:
        return finite_q_ivc_seed(
            active,
            n_occ_per_k=n_occ_per_k,
            angle=ivc_angle,
            phase=ivc_phase,
        )
    if key == "random":
        return random_seed(active, n_occ_per_k=n_occ_per_k, seed=random_seed_value)
    raise ValueError(f"unknown continuum seed {name!r}")
