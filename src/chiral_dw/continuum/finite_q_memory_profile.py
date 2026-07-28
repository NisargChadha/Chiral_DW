"""Fresh-process memory profiling for production Taige finite-Q backend builds."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Literal, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from chiral_dw.config import (
    ContinuumFiniteQParams,
    ContinuumGridParams,
    ContinuumInteractionParams,
)
from chiral_dw.continuum.builder import build_continuum_bundle
from chiral_dw.continuum.hf import (
    ContinuumHFBackend,
    _MAX_EXCHANGE_Q_POINTS_PER_SLAB,
)
from chiral_dw.continuum.models import MomentumGrid
from chiral_dw.continuum.taige import (
    build_taige_active_space,
    build_taige_density_vertices,
    compute_taige_bandstructure,
    reciprocal_box,
    taige_ivc_minus_half_shift_coord,
    taige_ivc_minus_q_coord,
    taige_model_params,
)

FiniteQBuildVariant = Literal[
    "direct_finite_q_workers4",
    "rolled_finite_q_workers1",
    "rolled_finite_q_workers4",
]
FINITE_Q_BUILD_VARIANTS: tuple[FiniteQBuildVariant, ...] = (
    "direct_finite_q_workers4",
    "rolled_finite_q_workers1",
    "rolled_finite_q_workers4",
)


class FiniteQBuildProfileParams(BaseModel):
    """Controls for one build-only production-layout memory measurement."""

    model_config = ConfigDict(frozen=True)

    n_k: int = Field(ge=6)
    theta_deg: float = Field(default=3.5, gt=0.0)
    u_D: float = 0.0
    plane_wave_shell: int = Field(default=5, ge=0)
    n_bands: int = Field(default=2, ge=1)
    n_active_bands_per_valley: int = Field(default=2, ge=1)
    local_field_cutoff: int = Field(default=4, ge=0)
    epsilon: float = Field(default=50.0, gt=0.0)
    gate_distance_nm: float = Field(default=30.0, gt=0.0)
    smear_length_nm: float = Field(default=0.347, ge=0.0)
    vertex_workers: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _exact_shift_is_available(self) -> "FiniteQBuildProfileParams":
        if self.n_k % 6:
            raise ValueError("exact finite-Q profiling requires n_k divisible by 6")
        if self.n_active_bands_per_valley > self.n_bands:
            raise ValueError("n_active_bands_per_valley cannot exceed n_bands")
        return self

    def model_params(self):
        return taige_model_params(
            theta_deg=self.theta_deg,
            u_D=self.u_D,
            plane_wave_shell=self.plane_wave_shell,
            n_bands=self.n_bands,
            n_active_bands_per_valley=self.n_active_bands_per_valley,
        )

    def finite_q_params(self) -> ContinuumFiniteQParams:
        return ContinuumFiniteQParams(
            enabled=True,
            q_coord=taige_ivc_minus_q_coord(self.n_k),
            half_shift_coord=taige_ivc_minus_half_shift_coord(self.n_k),
        )

    def interaction_params(self, *, exchange_workers: int) -> ContinuumInteractionParams:
        return ContinuumInteractionParams(
            coulomb_kind="dual_gate",
            epsilon=self.epsilon,
            gate_distance_nm=self.gate_distance_nm,
            smear_length_nm=self.smear_length_nm,
            q_mesh="full",
            q_shell=0,
            local_field_cutoff=self.local_field_cutoff,
            vertex_workers=self.vertex_workers,
            exchange_workers=int(exchange_workers),
            density_vertex_layout="valley_compact",
            exchange_representation="valley_sector",
            density_vertex_retention="hartree_only",
        )


class FiniteQBuildArrayEstimate(BaseModel):
    """Major complex128 arrays relevant to the finite-Q build peak."""

    model_config = ConfigDict(frozen=True)

    n_blocks: int
    n_q: int
    n_g: int
    compact_vertices_gib: float
    source_plus_rolled_vertices_gib: float
    valley_sector_exchange_gib: float
    max_worker_result_slab_gib: float
    target_minus_q_gib: float
    v_over_a_gib: float


class FiniteQBuildWorkerResult(BaseModel):
    """Build result emitted inside one fresh worker process."""

    model_config = ConfigDict(frozen=True)

    params: FiniteQBuildProfileParams
    variant: FiniteQBuildVariant
    worker_pid: int
    completed: bool
    build_wall_seconds: float | None = None
    final_rss_gib: float | None = None
    self_max_rss_gib: float | None = None
    exchange_workers: int
    n_blocks: int | None = None
    exchange_representation: str | None = None
    error: str | None = None
    estimates: FiniteQBuildArrayEstimate


class FiniteQBuildProfileResult(BaseModel):
    """Controller-side process-tree memory measurement for one build."""

    model_config = ConfigDict(frozen=True)

    worker: FiniteQBuildWorkerResult
    subprocess_wall_seconds: float
    parent_peak_rss_gib: float | None = None
    child_processes_peak_rss_gib: float | None = None
    process_tree_peak_rss_gib: float | None = None
    slurm_cgroup_max_rss_gib: float | None = None
    returncode: int
    stderr: str = ""


class FiniteQBuildProfileSummary(BaseModel):
    """Artifact summary for a build-only finite-Q memory profiling run."""

    model_config = ConfigDict(frozen=True)

    output_dir: str
    results: tuple[FiniteQBuildProfileResult, ...]


def _bytes_to_gib(value: int | float) -> float:
    return float(value) / float(1024**3)


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _current_rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        return _max_rss_bytes()


def estimate_finite_q_build_arrays(
    params: FiniteQBuildProfileParams,
) -> FiniteQBuildArrayEstimate:
    n_blocks = int(params.n_k**2)
    n_q = n_blocks
    n_g = len(reciprocal_box(params.local_field_cutoff))
    n_active = int(params.n_active_bands_per_valley)
    complex_bytes = np.dtype(np.complex128).itemsize
    compact = n_q * n_g * n_blocks * 2 * n_active**2 * complex_bytes
    sector_dim = n_blocks * n_active**2
    exchange = 4 * sector_dim**2 * complex_bytes
    slab_q = min(_MAX_EXCHANGE_Q_POINTS_PER_SLAB, n_q)
    worker_result = (
        slab_q
        * 4
        * 2
        * n_blocks
        * n_active**4
        * complex_bytes
    )
    return FiniteQBuildArrayEstimate(
        n_blocks=n_blocks,
        n_q=n_q,
        n_g=n_g,
        compact_vertices_gib=_bytes_to_gib(compact),
        source_plus_rolled_vertices_gib=_bytes_to_gib(2 * compact),
        valley_sector_exchange_gib=_bytes_to_gib(exchange),
        max_worker_result_slab_gib=_bytes_to_gib(worker_result),
        target_minus_q_gib=_bytes_to_gib(n_q * n_blocks * np.dtype(np.int64).itemsize),
        v_over_a_gib=_bytes_to_gib(n_q * n_g * np.dtype(float).itemsize),
    )


def _variant_exchange_workers(variant: FiniteQBuildVariant) -> int:
    return 1 if variant == "rolled_finite_q_workers1" else 4


def _build_direct_finite_q_backend(
    params: FiniteQBuildProfileParams,
    interaction: ContinuumInteractionParams,
) -> ContinuumHFBackend:
    grid = MomentumGrid(params.n_k)
    model = params.model_params()
    finite_q = params.finite_q_params()
    bands = compute_taige_bandstructure(model, grid)
    active, _ = build_taige_active_space(
        grid,
        model,
        finite_q,
        bands=bands,
    )
    vertices = build_taige_density_vertices(active, interaction)
    return ContinuumHFBackend(active.h0, vertices, interaction)


def run_finite_q_build_worker(
    params: FiniteQBuildProfileParams,
    variant: FiniteQBuildVariant,
) -> FiniteQBuildWorkerResult:
    """Construct one backend without starting any HF iteration."""

    estimates = estimate_finite_q_build_arrays(params)
    exchange_workers = _variant_exchange_workers(variant)
    interaction = params.interaction_params(exchange_workers=exchange_workers)
    start = time.perf_counter()
    try:
        if variant == "direct_finite_q_workers4":
            backend = _build_direct_finite_q_backend(params, interaction)
        else:
            backend = build_continuum_bundle(
                model=params.model_params(),
                grid=ContinuumGridParams(n_k=params.n_k),
                interaction=interaction,
                finite_q=params.finite_q_params(),
            ).backend
        elapsed = time.perf_counter() - start
        return FiniteQBuildWorkerResult(
            params=params,
            variant=variant,
            worker_pid=os.getpid(),
            completed=True,
            build_wall_seconds=elapsed,
            final_rss_gib=_bytes_to_gib(_current_rss_bytes()),
            self_max_rss_gib=_bytes_to_gib(_max_rss_bytes()),
            exchange_workers=exchange_workers,
            n_blocks=int(backend.n_blocks),
            exchange_representation=str(backend.exchange_representation),
            estimates=estimates,
        )
    except Exception as exc:
        return FiniteQBuildWorkerResult(
            params=params,
            variant=variant,
            worker_pid=os.getpid(),
            completed=False,
            build_wall_seconds=time.perf_counter() - start,
            final_rss_gib=_bytes_to_gib(_current_rss_bytes()),
            self_max_rss_gib=_bytes_to_gib(_max_rss_bytes()),
            exchange_workers=exchange_workers,
            error=f"{type(exc).__name__}: {exc}",
            estimates=estimates,
        )


def _slurm_cgroup_peak_bytes() -> int | None:
    if "SLURM_JOB_ID" not in os.environ:
        return None
    for path in (
        Path("/sys/fs/cgroup/memory.peak"),
        Path("/sys/fs/cgroup/memory/memory.max_usage_in_bytes"),
    ):
        try:
            value = int(path.read_text().strip())
        except (OSError, ValueError):
            continue
        if value >= 0:
            return value
    return None


def _sample_process_tree_rss(pid: int) -> tuple[int | None, int | None]:
    try:
        import psutil

        process = psutil.Process(int(pid))
        parent = int(process.memory_info().rss)
        children = 0
        for child in process.children(recursive=True):
            try:
                children += int(child.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return parent, children
    except Exception:
        return None, None


def run_finite_q_build_subprocess(
    *,
    script_path: Path,
    params: FiniteQBuildProfileParams,
    variant: FiniteQBuildVariant,
    sample_interval_seconds: float = 0.05,
) -> FiniteQBuildProfileResult:
    """Launch and monitor one fresh worker process and its joblib children."""

    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        "--params-json",
        params.model_dump_json(),
        "--variant",
        variant,
    ]
    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    parent_peak: int | None = None
    children_peak: int | None = None
    tree_peak: int | None = None
    while proc.poll() is None:
        parent, children = _sample_process_tree_rss(proc.pid)
        if parent is not None:
            parent_peak = max(parent_peak or 0, parent)
        if children is not None:
            children_peak = max(children_peak or 0, children)
        if parent is not None and children is not None:
            tree_peak = max(tree_peak or 0, parent + children)
        time.sleep(max(float(sample_interval_seconds), 0.01))
    stdout, stderr = proc.communicate()
    elapsed = time.perf_counter() - start
    try:
        worker = FiniteQBuildWorkerResult.model_validate_json(stdout.strip().splitlines()[-1])
    except Exception as exc:
        worker = FiniteQBuildWorkerResult(
            params=params,
            variant=variant,
            worker_pid=int(proc.pid),
            completed=False,
            exchange_workers=_variant_exchange_workers(variant),
            error=f"worker result unavailable: {type(exc).__name__}: {exc}",
            estimates=estimate_finite_q_build_arrays(params),
        )
    cgroup_peak = _slurm_cgroup_peak_bytes()
    return FiniteQBuildProfileResult(
        worker=worker,
        subprocess_wall_seconds=elapsed,
        parent_peak_rss_gib=None if parent_peak is None else _bytes_to_gib(parent_peak),
        child_processes_peak_rss_gib=(
            None if children_peak is None else _bytes_to_gib(children_peak)
        ),
        process_tree_peak_rss_gib=None if tree_peak is None else _bytes_to_gib(tree_peak),
        slurm_cgroup_max_rss_gib=(
            None if cgroup_peak is None else _bytes_to_gib(cgroup_peak)
        ),
        returncode=int(proc.returncode),
        stderr=stderr,
    )


def write_finite_q_build_profile(
    output_dir: Path,
    results: Sequence[FiniteQBuildProfileResult],
) -> FiniteQBuildProfileSummary:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = FiniteQBuildProfileSummary(
        output_dir=str(output_dir),
        results=tuple(results),
    )
    (output_dir / "finite_q_build_profile.json").write_text(
        json.dumps(summary.model_dump(mode="json"), indent=2)
    )
    rows: list[dict[str, object]] = []
    for result in results:
        worker = result.worker
        row: dict[str, object] = {
            "n_k": worker.params.n_k,
            "variant": worker.variant,
            "completed": worker.completed,
            "returncode": result.returncode,
            "worker_pid": worker.worker_pid,
            "exchange_workers": worker.exchange_workers,
            "build_wall_seconds": worker.build_wall_seconds,
            "subprocess_wall_seconds": result.subprocess_wall_seconds,
            "parent_peak_rss_gib": result.parent_peak_rss_gib,
            "child_processes_peak_rss_gib": result.child_processes_peak_rss_gib,
            "process_tree_peak_rss_gib": result.process_tree_peak_rss_gib,
            "slurm_cgroup_max_rss_gib": result.slurm_cgroup_max_rss_gib,
            "final_rss_gib": worker.final_rss_gib,
            "self_max_rss_gib": worker.self_max_rss_gib,
            "compact_vertices_gib": worker.estimates.compact_vertices_gib,
            "source_plus_rolled_vertices_gib": (
                worker.estimates.source_plus_rolled_vertices_gib
            ),
            "valley_sector_exchange_gib": (
                worker.estimates.valley_sector_exchange_gib
            ),
            "max_worker_result_slab_gib": (
                worker.estimates.max_worker_result_slab_gib
            ),
            "error": worker.error,
        }
        rows.append(row)
    with (output_dir / "finite_q_build_profile.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["n_k"])
        writer.writeheader()
        writer.writerows(rows)
    return summary


def run_finite_q_build_profile_suite(
    *,
    output_dir: Path,
    script_path: Path,
    params_by_n_k: Sequence[FiniteQBuildProfileParams],
    variants: Sequence[FiniteQBuildVariant] = FINITE_Q_BUILD_VARIANTS,
) -> FiniteQBuildProfileSummary:
    results = [
        run_finite_q_build_subprocess(
            script_path=script_path,
            params=params,
            variant=variant,
        )
        for params in params_by_n_k
        for variant in variants
    ]
    return write_finite_q_build_profile(output_dir, results)


__all__ = [
    "FINITE_Q_BUILD_VARIANTS",
    "FiniteQBuildArrayEstimate",
    "FiniteQBuildProfileParams",
    "FiniteQBuildProfileResult",
    "FiniteQBuildProfileSummary",
    "FiniteQBuildVariant",
    "FiniteQBuildWorkerResult",
    "estimate_finite_q_build_arrays",
    "run_finite_q_build_profile_suite",
    "run_finite_q_build_subprocess",
    "run_finite_q_build_worker",
    "write_finite_q_build_profile",
]
