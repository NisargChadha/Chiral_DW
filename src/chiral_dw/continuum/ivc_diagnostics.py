"""Local diagnostics for constrained Taige IVC Hartree-Fock branches."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, computed_field
import numpy as np


class TaigeIvcDiagnosticPoint(BaseModel):
    """One local phase-space point for IVC branch diagnostics."""

    model_config = ConfigDict(frozen=True)

    u_index: int = Field(ge=0)
    theta_index: int = Field(ge=0)
    u_D: float
    theta_deg: float
    group: str = "scan"

    @computed_field
    @property
    def label(self) -> str:
        u_text = _float_label(self.u_D)
        theta_text = _float_label(self.theta_deg)
        return f"{self.group}_u_{self.u_index:03d}_{u_text}_theta_{self.theta_index:03d}_{theta_text}"


class TaigeIvcSeedSpec(BaseModel):
    """One ordered or mixed-random IVC seed specification."""

    model_config = ConfigDict(frozen=True)

    label: str
    ordered_weight: float = Field(ge=0.0)
    random_weight: float = Field(ge=0.0)
    random_seed: int | None = None


class ProjectorOverlapDiagnostics(BaseModel):
    """Gauge-invariant overlap diagnostics between two rank-fixed projector paths."""

    model_config = ConfigDict(frozen=True)

    n_blocks: int
    rank_per_block: float
    mean_overlap: float
    min_block_overlap: float
    max_block_overlap: float
    std_block_overlap: float
    one_minus_mean_overlap: float
    frobenius_distance: float
    max_block_frobenius_distance: float


def _float_label(value: float) -> str:
    text = f"{float(value):.6g}".replace("-", "m").replace(".", "p")
    return text


def _validate_projector_pair(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(P, dtype=complex)
    right = np.asarray(Q, dtype=complex)
    if left.shape != right.shape:
        raise ValueError(f"projectors must have matching shapes, got {left.shape} and {right.shape}")
    if left.ndim != 3 or left.shape[-1] != left.shape[-2]:
        raise ValueError("projectors must have shape (n_blocks, dim, dim)")
    return left, right


def projector_overlap_diagnostics(
    P: np.ndarray,
    Q: np.ndarray,
    *,
    n_occ_per_k: int = 1,
) -> ProjectorOverlapDiagnostics:
    """Return active-basis projector overlap diagnostics."""

    left, right = _validate_projector_pair(P, Q)
    per_block = np.einsum("kab,kba->k", left, right, optimize=True).real / float(n_occ_per_k)
    return _overlap_from_per_block(left, right, per_block, n_occ_per_k=n_occ_per_k)


def projector_overlap_diagnostics_with_frames(
    P: np.ndarray,
    Q: np.ndarray,
    frames_P: np.ndarray,
    frames_Q: np.ndarray,
    *,
    n_occ_per_k: int = 1,
) -> ProjectorOverlapDiagnostics:
    r"""Return physical projector overlap diagnostics using active Bloch frames.

    For neighboring parameter points the active bases differ.  This evaluates
    Tr(P U_P^\dagger U_Q Q U_Q^\dagger U_P) block by block, avoiding a dense
    materialization of the full embedded projectors.
    """

    left, right = _validate_projector_pair(P, Q)
    f_left = np.asarray(frames_P, dtype=complex)
    f_right = np.asarray(frames_Q, dtype=complex)
    if f_left.ndim != 3 or f_right.ndim != 3:
        raise ValueError("frames must have shape (n_blocks, full_dim, active_dim)")
    if f_left.shape[0] != left.shape[0] or f_right.shape[0] != right.shape[0]:
        raise ValueError("frames and projectors must have matching block counts")
    if f_left.shape[2] != left.shape[-1] or f_right.shape[2] != right.shape[-1]:
        raise ValueError("frames and projectors must have matching active dimensions")
    if f_left.shape[1] != f_right.shape[1]:
        raise ValueError("frames must embed into the same full basis dimension")

    overlaps = np.empty(left.shape[0], dtype=float)
    for ik in range(left.shape[0]):
        s = f_left[ik].conj().T @ f_right[ik]
        overlaps[ik] = float(
            np.real(np.trace(left[ik] @ s @ right[ik] @ s.conj().T)) / float(n_occ_per_k)
        )
    base = _overlap_from_per_block(left, right, overlaps, n_occ_per_k=n_occ_per_k)
    # In different active frames, the active-basis Frobenius norm of P-Q is not
    # meaningful.  For rank-fixed idempotent projectors, the embedded full-basis
    # distance follows from Tr(PQ).
    clipped = np.clip(overlaps, -1.0, 1.0)
    return base.model_copy(
        update={
            "frobenius_distance": float(np.sqrt(max(0.0, 2.0 - 2.0 * base.mean_overlap))),
            "max_block_frobenius_distance": float(
                np.sqrt(max(0.0, 2.0 - 2.0 * float(np.min(clipped))))
            ),
        }
    )


def _overlap_from_per_block(
    P: np.ndarray,
    Q: np.ndarray,
    per_block: np.ndarray,
    *,
    n_occ_per_k: int,
) -> ProjectorOverlapDiagnostics:
    rank = float(n_occ_per_k)
    n_blocks = int(P.shape[0])
    block_fro = np.linalg.norm(P - Q, axis=(-2, -1)) / np.sqrt(max(rank, 1e-30))
    fro = float(np.linalg.norm(P - Q) / np.sqrt(max(float(n_blocks) * rank, 1e-30)))
    mean = float(np.mean(per_block))
    return ProjectorOverlapDiagnostics(
        n_blocks=n_blocks,
        rank_per_block=rank,
        mean_overlap=mean,
        min_block_overlap=float(np.min(per_block)),
        max_block_overlap=float(np.max(per_block)),
        std_block_overlap=float(np.std(per_block)),
        one_minus_mean_overlap=float(1.0 - mean),
        frobenius_distance=fro,
        max_block_frobenius_distance=float(np.max(block_fro)),
    )


__all__ = [
    "ProjectorOverlapDiagnostics",
    "TaigeIvcDiagnosticPoint",
    "TaigeIvcSeedSpec",
    "projector_overlap_diagnostics",
    "projector_overlap_diagnostics_with_frames",
]
