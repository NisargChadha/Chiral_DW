"""Local diagnostics for constrained Taige IVC Hartree-Fock branches."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, computed_field
import numpy as np

from chiral_dw.continuum.models import hermitize


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


class ProjectorTransportDiagnostics(BaseModel):
    """Diagnostics for transporting a projector between two active frames."""

    model_config = ConfigDict(frozen=True)

    n_blocks: int
    rank_per_block: float
    mean_projected_trace: float
    min_projected_trace: float
    max_projected_trace: float
    mean_retained_weight: float
    min_retained_weight: float
    max_retained_weight: float
    mean_discarded_weight: float
    max_discarded_weight: float
    transported_trace_error: float
    transported_idempotency_error_fro: float
    transported_idempotency_error_max: float


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


def transport_projector_between_frames(
    P_source: np.ndarray,
    frames_source: np.ndarray,
    frames_target: np.ndarray,
    *,
    n_occ_per_k: int = 1,
) -> tuple[np.ndarray, ProjectorTransportDiagnostics]:
    r"""Transport a projector from one active Bloch frame to another.

    The source full-space projector is represented in the target active frame as
    ``U_target^\dagger U_source P_source U_source^\dagger U_target``.  Since the
    two active subspaces need not be identical, this projected density is then
    converted back to a rank-fixed idempotent seed by retaining the largest
    ``n_occ_per_k`` eigenvectors in each momentum block.
    """

    source = np.asarray(P_source, dtype=complex)
    if source.ndim != 3 or source.shape[-1] != source.shape[-2]:
        raise ValueError("P_source must have shape (n_blocks, dim, dim)")
    f_source = np.asarray(frames_source, dtype=complex)
    f_target = np.asarray(frames_target, dtype=complex)
    if f_source.ndim != 3 or f_target.ndim != 3:
        raise ValueError("frames must have shape (n_blocks, full_dim, active_dim)")
    if f_source.shape[0] != source.shape[0] or f_target.shape[0] != source.shape[0]:
        raise ValueError("frames and projector must have matching block counts")
    if f_source.shape[2] != source.shape[-1] or f_target.shape[2] != source.shape[-1]:
        raise ValueError("frames and projector must have matching active dimensions")
    if f_source.shape[1] != f_target.shape[1]:
        raise ValueError("frames must embed into the same full basis dimension")
    dim = int(source.shape[-1])
    n_occ = int(n_occ_per_k)
    if n_occ < 1 or n_occ > dim:
        raise ValueError("n_occ_per_k must be between one and the active dimension")

    out = np.zeros_like(source, dtype=complex)
    projected_traces = np.empty(source.shape[0], dtype=float)
    retained_weights = np.empty(source.shape[0], dtype=float)
    discarded_weights = np.empty(source.shape[0], dtype=float)
    for ik in range(source.shape[0]):
        s = f_target[ik].conj().T @ f_source[ik]
        projected = hermitize(s @ source[ik] @ s.conj().T)
        vals, vecs = np.linalg.eigh(projected)
        order = np.argsort(vals)[::-1]
        occ = order[:n_occ]
        retained_weights[ik] = float(np.sum(vals[occ]).real / float(n_occ))
        projected_traces[ik] = float(np.trace(projected).real / float(n_occ))
        discarded_weights[ik] = float(
            max(0.0, projected_traces[ik] - retained_weights[ik])
        )
        occupied = vecs[:, occ]
        out[ik] = occupied @ occupied.conj().T
    out = hermitize(out)
    traces = np.trace(out, axis1=-2, axis2=-1).real
    idem = out @ out - out
    diagnostics = ProjectorTransportDiagnostics(
        n_blocks=int(source.shape[0]),
        rank_per_block=float(n_occ),
        mean_projected_trace=float(np.mean(projected_traces)),
        min_projected_trace=float(np.min(projected_traces)),
        max_projected_trace=float(np.max(projected_traces)),
        mean_retained_weight=float(np.mean(retained_weights)),
        min_retained_weight=float(np.min(retained_weights)),
        max_retained_weight=float(np.max(retained_weights)),
        mean_discarded_weight=float(np.mean(discarded_weights)),
        max_discarded_weight=float(np.max(discarded_weights)),
        transported_trace_error=float(np.max(np.abs(traces - float(n_occ)))),
        transported_idempotency_error_fro=float(np.linalg.norm(idem)),
        transported_idempotency_error_max=float(np.max(np.abs(idem))),
    )
    return out, diagnostics


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
    "ProjectorTransportDiagnostics",
    "TaigeIvcDiagnosticPoint",
    "TaigeIvcSeedSpec",
    "projector_overlap_diagnostics",
    "projector_overlap_diagnostics_with_frames",
    "transport_projector_between_frames",
]
