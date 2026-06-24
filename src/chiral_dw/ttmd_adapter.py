"""Adapters for TMD_HF-backed VP/IVC source interpolation."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import ModuleType
from typing import Callable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from chiral_dw.config import SourceInterpolationParams, TMDHFReferenceParams
from chiral_dw.response import hermitian_part

TMD_HF_PATH_HINT = "/Users/nisargchadha/Documents/TMD_HF"


class TMDHFUnavailableError(ImportError):
    """Raised when the optional local TMD_HF package is unavailable."""


class TTMDReferenceDiagnostics(BaseModel):
    """Scalar diagnostics for raw TMD_HF source fields and reference projectors."""

    model_config = ConfigDict(frozen=True)

    h0_hermiticity_error: float
    vp_projector_hermiticity_error: float
    ivc_projector_hermiticity_error: float
    vp_projector_idempotency_error: float
    ivc_projector_idempotency_error: float
    vp_trace_mean: float
    ivc_trace_mean: float
    delta_vp_hermiticity_error: float
    delta_ivc_hermiticity_error: float
    delta_vp_norm: float
    delta_ivc_norm: float
    delta_vp_scalar_norm: float
    delta_ivc_scalar_norm: float
    delta_vp_traceless_norm: float
    delta_ivc_traceless_norm: float
    delta_vp_offdiag_norm: float
    delta_ivc_offdiag_norm: float


class TTMDInterpolationDiagnostics(BaseModel):
    """Scalar diagnostics for one source-interpolated projector."""

    model_config = ConfigDict(frozen=True)

    theta: float
    phi: float
    direct_gap_min: float
    indirect_gap: float
    projector_hermiticity_error: float
    projector_idempotency_error: float
    trace_mean: float


class TTMDEndpointDiagnostics(BaseModel):
    """Distances between interpolation endpoints and supplied VP/IVC references."""

    model_config = ConfigDict(frozen=True)

    theta0_vs_vp_relative_frobenius: float
    theta0_vs_vp_max_per_k_frobenius: float
    theta_pi_over_2_vs_ivc_relative_frobenius: float
    theta_pi_over_2_vs_ivc_max_per_k_frobenius: float


@dataclass(frozen=True)
class TTMDReferenceProjectors:
    """TMD_HF reference projectors and raw contracted source fields."""

    h0: np.ndarray
    P_vp: np.ndarray
    P_ivc: np.ndarray
    delta_vp: np.ndarray
    delta_ivc: np.ndarray
    n_occ_per_block: int = 1
    metadata: dict | None = None

    def __post_init__(self) -> None:
        h0 = _as_hermitian_blocks(self.h0, name="h0")
        P_vp = _as_block_stack(self.P_vp, name="P_vp", shape=h0.shape)
        P_ivc = _as_block_stack(self.P_ivc, name="P_ivc", shape=h0.shape)
        delta_vp = _as_hermitian_blocks(self.delta_vp, name="delta_vp", shape=h0.shape)
        delta_ivc = _as_hermitian_blocks(self.delta_ivc, name="delta_ivc", shape=h0.shape)
        n_occ = int(self.n_occ_per_block)
        if n_occ < 1 or n_occ > h0.shape[-1]:
            raise ValueError("n_occ_per_block must lie between 1 and the block dimension")
        object.__setattr__(self, "h0", h0)
        object.__setattr__(self, "P_vp", hermitian_part(P_vp))
        object.__setattr__(self, "P_ivc", hermitian_part(P_ivc))
        object.__setattr__(self, "delta_vp", delta_vp)
        object.__setattr__(self, "delta_ivc", delta_ivc)
        object.__setattr__(self, "n_occ_per_block", n_occ)

    @property
    def n_blocks(self) -> int:
        return int(self.h0.shape[0])

    @property
    def dim(self) -> int:
        return int(self.h0.shape[-1])

    @property
    def n_active_per_valley(self) -> int:
        if self.dim % 2:
            raise ValueError("TMD_HF active-space dimension must be even")
        return self.dim // 2


@dataclass(frozen=True)
class TTMDInterpolationResult:
    """Variational Hamiltonian, projector, and scalar diagnostics."""

    H: np.ndarray
    P: np.ndarray
    eigenvalues: np.ndarray
    diagnostics: TTMDInterpolationDiagnostics


def require_tmd_hf(
    importer: Callable[[str], ModuleType] | None = None,
) -> dict[str, ModuleType]:
    """Import optional TMD_HF modules with a local editable-install hint."""

    load = importlib.import_module if importer is None else importer
    modules: dict[str, ModuleType] = {}
    for name in ("ttmd.problem", "ttmd.seeds", "hartree_fock"):
        try:
            modules[name] = load(name)
        except ImportError as exc:
            raise TMDHFUnavailableError(
                "TMD_HF is required for continuum VP/IVC source interpolation. "
                f"Install it with `python -m pip install -e {TMD_HF_PATH_HINT}` "
                f"or set PYTHONPATH to `{TMD_HF_PATH_HINT}/src`."
            ) from exc
    return modules


def flavor_u1_rotation(dim: int, phi: float) -> np.ndarray:
    """Return U_phi=diag(exp(-i phi/2) I_N, exp(+i phi/2) I_N)."""

    n_dim = int(dim)
    if n_dim <= 0 or n_dim % 2:
        raise ValueError("active-space dimension must be a positive even integer")
    n = n_dim // 2
    half = 0.5 * float(phi)
    phases = np.concatenate(
        [np.exp(-1j * half) * np.ones(n), np.exp(1j * half) * np.ones(n)]
    )
    return np.diag(phases.astype(complex))


def rotate_flavor_blocks(blocks: np.ndarray, phi: float) -> np.ndarray:
    """Rotate block operators or projectors by the TMD_HF valley U(1)."""

    arr = np.asarray(blocks, dtype=complex)
    if arr.ndim < 2 or arr.shape[-1] != arr.shape[-2]:
        raise ValueError("blocks must end in square matrix axes")
    U = flavor_u1_rotation(arr.shape[-1], phi)
    rotated = np.einsum("ab,...bc,dc->...ad", U, arr, U.conj(), optimize=True)
    return hermitian_part(rotated)


def source_interpolation_hamiltonian(
    refs: TTMDReferenceProjectors,
    theta: float,
    phi: float = 0.0,
    params: SourceInterpolationParams | None = None,
) -> np.ndarray:
    """Build H0 + source_scale[cos(theta) Delta_VP + sin(theta) U Delta_IVC U^dagger]."""

    controls = params or SourceInterpolationParams(n_occ_per_block=refs.n_occ_per_block)
    if controls.field_policy != "raw_hermitian":
        raise ValueError("only raw_hermitian source fields are supported")
    rotated_ivc = rotate_flavor_blocks(refs.delta_ivc, phi)
    H = refs.h0 + float(controls.source_scale) * (
        np.cos(float(theta)) * refs.delta_vp + np.sin(float(theta)) * rotated_ivc
    )
    return _as_hermitian_blocks(H, name="H_var", shape=refs.h0.shape)


def source_interpolation_projector(
    refs: TTMDReferenceProjectors,
    theta: float,
    phi: float = 0.0,
    params: SourceInterpolationParams | None = None,
) -> TTMDInterpolationResult:
    """Diagonalize the variational Hamiltonian and fill `n_occ_per_block` bands."""

    controls = params or SourceInterpolationParams(n_occ_per_block=refs.n_occ_per_block)
    H = source_interpolation_hamiltonian(refs, theta, phi, controls)
    P, evals, direct_gap, indirect_gap = fill_block_projector(
        H,
        controls.n_occ_per_block,
        occupy=controls.occupy,
    )
    diagnostics = TTMDInterpolationDiagnostics(
        theta=float(theta),
        phi=float(phi),
        direct_gap_min=float(direct_gap),
        indirect_gap=float(indirect_gap),
        projector_hermiticity_error=_hermiticity_error(P),
        projector_idempotency_error=_idempotency_error(P),
        trace_mean=_trace_mean(P),
    )
    return TTMDInterpolationResult(H=H, P=P, eigenvalues=evals, diagnostics=diagnostics)


def source_interpolation_path(
    refs: TTMDReferenceProjectors,
    theta_nodes: np.ndarray,
    phi: float = 0.0,
    params: SourceInterpolationParams | None = None,
) -> tuple[np.ndarray, tuple[TTMDInterpolationDiagnostics, ...]]:
    """Build a theta-indexed projector path from raw VP/IVC source fields."""

    theta = np.asarray(theta_nodes, dtype=float)
    projectors = np.zeros((theta.size, refs.n_blocks, refs.dim, refs.dim), dtype=complex)
    diagnostics: list[TTMDInterpolationDiagnostics] = []
    for idx, value in enumerate(theta):
        result = source_interpolation_projector(refs, float(value), phi, params)
        projectors[idx] = result.P
        diagnostics.append(result.diagnostics)
    return projectors, tuple(diagnostics)


def contracted_source_fields_from_backend(
    backend,
    P_vp: np.ndarray,
    P_ivc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return h0 and raw Delta fields as `hf_hamiltonian(P_ref) - h0`."""

    if not hasattr(backend, "h0") or not hasattr(backend, "hf_hamiltonian"):
        raise TypeError("backend must expose h0 and hf_hamiltonian(P)")
    h0 = _as_hermitian_blocks(np.asarray(backend.h0, dtype=complex), name="backend.h0")
    P_vp_block = (
        backend.as_block_density(P_vp) if hasattr(backend, "as_block_density") else P_vp
    )
    P_ivc_block = (
        backend.as_block_density(P_ivc) if hasattr(backend, "as_block_density") else P_ivc
    )
    delta_vp = np.asarray(backend.hf_hamiltonian(P_vp_block), dtype=complex) - h0
    delta_ivc = np.asarray(backend.hf_hamiltonian(P_ivc_block), dtype=complex) - h0
    return (
        h0,
        _as_hermitian_blocks(delta_vp, name="delta_vp", shape=h0.shape),
        _as_hermitian_blocks(delta_ivc, name="delta_ivc", shape=h0.shape),
    )


def references_from_tmd_hf_bundle(
    bundle,
    P_vp: np.ndarray | None = None,
    P_ivc: np.ndarray | None = None,
    params: TMDHFReferenceParams | None = None,
) -> TTMDReferenceProjectors:
    """Create VP/IVC source fields from a TMD_HF bundle.

    If projectors are not supplied, the TMD_HF seed helpers provide simple VP
    and Q=0 IVC references. Production workflows can instead pass converged HF
    projectors obtained from TMD_HF.
    """

    controls = params or TMDHFReferenceParams()
    if not hasattr(bundle, "backend") or not hasattr(bundle, "active"):
        raise TypeError("bundle must expose TMD_HF active and backend attributes")
    if P_vp is None or P_ivc is None:
        modules = require_tmd_hf()
        seeds = modules["ttmd.seeds"]
        n_particles = bundle.active.n_k * int(controls.n_occ_per_block)
        if P_vp is None:
            P_vp = seeds.valley_polarized_seed(bundle.active, n_particles, controls.vp_valley)
        if P_ivc is None:
            P_ivc = seeds.ivc_seed(
                bundle.active,
                n_particles,
                angle=float(controls.ivc_angle),
                phase=float(controls.ivc_phase),
            )
    h0, delta_vp, delta_ivc = contracted_source_fields_from_backend(
        bundle.backend,
        P_vp,
        P_ivc,
    )
    return TTMDReferenceProjectors(
        h0=h0,
        P_vp=P_vp,
        P_ivc=P_ivc,
        delta_vp=delta_vp,
        delta_ivc=delta_ivc,
        n_occ_per_block=controls.n_occ_per_block,
        metadata={
            "source": "TMD_HF",
            "source_convention": controls.source_convention,
            "field_policy": "raw_hermitian",
        },
    )


def diagnose_reference_projectors(refs: TTMDReferenceProjectors) -> TTMDReferenceDiagnostics:
    """Return shape, projector, and raw-field diagnostics for reference data."""

    return TTMDReferenceDiagnostics(
        h0_hermiticity_error=_hermiticity_error(refs.h0),
        vp_projector_hermiticity_error=_hermiticity_error(refs.P_vp),
        ivc_projector_hermiticity_error=_hermiticity_error(refs.P_ivc),
        vp_projector_idempotency_error=_idempotency_error(refs.P_vp),
        ivc_projector_idempotency_error=_idempotency_error(refs.P_ivc),
        vp_trace_mean=_trace_mean(refs.P_vp),
        ivc_trace_mean=_trace_mean(refs.P_ivc),
        delta_vp_hermiticity_error=_hermiticity_error(refs.delta_vp),
        delta_ivc_hermiticity_error=_hermiticity_error(refs.delta_ivc),
        delta_vp_norm=_rms_frobenius_norm(refs.delta_vp),
        delta_ivc_norm=_rms_frobenius_norm(refs.delta_ivc),
        delta_vp_scalar_norm=_rms_frobenius_norm(scalar_channel(refs.delta_vp)),
        delta_ivc_scalar_norm=_rms_frobenius_norm(scalar_channel(refs.delta_ivc)),
        delta_vp_traceless_norm=_rms_frobenius_norm(traceless_channel(refs.delta_vp)),
        delta_ivc_traceless_norm=_rms_frobenius_norm(traceless_channel(refs.delta_ivc)),
        delta_vp_offdiag_norm=_rms_frobenius_norm(offdiag_valley_channel(refs.delta_vp)),
        delta_ivc_offdiag_norm=_rms_frobenius_norm(offdiag_valley_channel(refs.delta_ivc)),
    )


def endpoint_diagnostics(
    refs: TTMDReferenceProjectors,
    params: SourceInterpolationParams | None = None,
) -> TTMDEndpointDiagnostics:
    """Compare theta=0 and theta=pi/2 projectors to the reference projectors."""

    controls = params or SourceInterpolationParams(n_occ_per_block=refs.n_occ_per_block)
    vp_projector = source_interpolation_projector(refs, 0.0, 0.0, controls).P
    ivc_projector = source_interpolation_projector(refs, 0.5 * np.pi, 0.0, controls).P
    vp_dist = projector_distance(vp_projector, refs.P_vp)
    ivc_dist = projector_distance(ivc_projector, refs.P_ivc)
    return TTMDEndpointDiagnostics(
        theta0_vs_vp_relative_frobenius=vp_dist["relative_frobenius"],
        theta0_vs_vp_max_per_k_frobenius=vp_dist["max_per_k_frobenius"],
        theta_pi_over_2_vs_ivc_relative_frobenius=ivc_dist["relative_frobenius"],
        theta_pi_over_2_vs_ivc_max_per_k_frobenius=ivc_dist["max_per_k_frobenius"],
    )


def fill_block_projector(
    H: np.ndarray,
    n_occ_per_block: int,
    occupy: Literal["lowest", "highest"] = "lowest",
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fill a fixed number of bands independently in each momentum block."""

    if occupy not in {"lowest", "highest"}:
        raise ValueError("occupy must be 'lowest' or 'highest'")
    blocks = _as_hermitian_blocks(H, name="H")
    n_occ = int(n_occ_per_block)
    dim = blocks.shape[-1]
    if n_occ < 1 or n_occ > dim:
        raise ValueError("n_occ_per_block must lie between 1 and dim")

    evals, evecs = np.linalg.eigh(blocks)
    if occupy == "lowest":
        filled = evecs[..., :, :n_occ]
        if n_occ < dim:
            direct = evals[:, n_occ] - evals[:, n_occ - 1]
            indirect = float(np.min(evals[:, n_occ]) - np.max(evals[:, n_occ - 1]))
        else:
            direct = np.full(blocks.shape[0], np.inf)
            indirect = float("inf")
    else:
        filled = evecs[..., :, dim - n_occ :]
        if n_occ < dim:
            split = dim - n_occ
            direct = evals[:, split] - evals[:, split - 1]
            indirect = float(np.min(evals[:, split]) - np.max(evals[:, split - 1]))
        else:
            direct = np.full(blocks.shape[0], np.inf)
            indirect = float("inf")
    P = np.einsum("kai,kbi->kab", filled, filled.conj(), optimize=True)
    return hermitian_part(P), evals, float(np.min(direct)), indirect


def scalar_channel(blocks: np.ndarray) -> np.ndarray:
    """Return the blockwise identity/scalar channel."""

    arr = _as_block_stack(blocks, name="blocks")
    dim = arr.shape[-1]
    tr = np.trace(arr, axis1=-2, axis2=-1) / float(dim)
    eye = np.eye(dim, dtype=complex)
    return tr[:, None, None] * eye[None, :, :]


def traceless_channel(blocks: np.ndarray) -> np.ndarray:
    """Return the raw matrix with its blockwise scalar channel removed."""

    arr = _as_block_stack(blocks, name="blocks")
    return arr - scalar_channel(arr)


def offdiag_valley_channel(blocks: np.ndarray) -> np.ndarray:
    """Return only intervalley blocks in the [K bands, Kprime bands] basis."""

    arr = _as_block_stack(blocks, name="blocks")
    dim = arr.shape[-1]
    if dim % 2:
        raise ValueError("active-space dimension must be even")
    n = dim // 2
    out = np.zeros_like(arr)
    out[..., :n, n:] = arr[..., :n, n:]
    out[..., n:, :n] = arr[..., n:, :n]
    return hermitian_part(out)


def projector_distance(P: np.ndarray, Q: np.ndarray) -> dict[str, float]:
    """Return relative and max-per-block Frobenius distances."""

    arr = _as_block_stack(P, name="P")
    other = _as_block_stack(Q, name="Q", shape=arr.shape)
    diff = arr - other
    denom = max(float(np.linalg.norm(other)), 1e-15)
    per_block = np.linalg.norm(diff, axis=(-2, -1))
    return {
        "relative_frobenius": float(np.linalg.norm(diff) / denom),
        "max_per_k_frobenius": float(np.max(per_block)),
    }


def _as_block_stack(
    value: np.ndarray,
    *,
    name: str,
    shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    arr = np.asarray(value, dtype=complex)
    if arr.ndim != 3 or arr.shape[-1] != arr.shape[-2]:
        raise ValueError(f"{name} must have shape (n_blocks, dim, dim)")
    if shape is not None and arr.shape != shape:
        raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
    return arr


def _as_hermitian_blocks(
    value: np.ndarray,
    *,
    name: str,
    shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    return hermitian_part(_as_block_stack(value, name=name, shape=shape))


def _hermiticity_error(blocks: np.ndarray) -> float:
    arr = _as_block_stack(blocks, name="blocks")
    return float(np.max(np.abs(arr - np.swapaxes(arr.conj(), -1, -2))))


def _idempotency_error(projectors: np.ndarray) -> float:
    arr = _as_block_stack(projectors, name="projectors")
    return float(np.max(np.abs(arr @ arr - arr)))


def _trace_mean(projectors: np.ndarray) -> float:
    arr = _as_block_stack(projectors, name="projectors")
    trace = np.trace(arr, axis1=-2, axis2=-1)
    return float(np.real_if_close(np.mean(trace), tol=1000).real)


def _rms_frobenius_norm(blocks: np.ndarray) -> float:
    arr = _as_block_stack(blocks, name="blocks")
    return float(np.sqrt(np.mean(np.sum(np.abs(arr) ** 2, axis=(-2, -1)))))


__all__ = [
    "TMDHFUnavailableError",
    "TTMDEndpointDiagnostics",
    "TTMDInterpolationDiagnostics",
    "TTMDInterpolationResult",
    "TTMDReferenceDiagnostics",
    "TTMDReferenceProjectors",
    "contracted_source_fields_from_backend",
    "diagnose_reference_projectors",
    "endpoint_diagnostics",
    "fill_block_projector",
    "flavor_u1_rotation",
    "offdiag_valley_channel",
    "projector_distance",
    "references_from_tmd_hf_bundle",
    "require_tmd_hf",
    "rotate_flavor_blocks",
    "scalar_channel",
    "source_interpolation_hamiltonian",
    "source_interpolation_path",
    "source_interpolation_projector",
    "traceless_channel",
]
