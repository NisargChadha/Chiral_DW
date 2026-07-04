#!/usr/bin/env python3
"""Safely delete a per-mesh Taige hysteresis backend cache after merge."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_MERGED_OUTPUTS = (
    "hysteresis_sweep.csv",
    "hysteresis_comparison.csv",
    "hysteresis_all_branch_candidates.csv",
    "hysteresis_selected_trial_theta.csv",
    "hysteresis_vp_chern_numbers.csv",
)


class BackendCacheCleanupManifest(BaseModel):
    """Manifest for one backend-cache cleanup action."""

    model_config = ConfigDict(frozen=True)

    output_root: str
    cache_root: str
    status: str
    dry_run: bool
    disabled: bool = False
    deleted: bool = False
    file_count: int = Field(ge=0)
    byte_count: int = Field(ge=0)
    required_outputs: tuple[str, ...]
    missing_required_outputs: tuple[str, ...] = ()
    slurm_job_id: str | None = None
    started_at_utc: str
    finished_at_utc: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument(
        "--allowed-cache-base-root",
        action="append",
        default=None,
        help=(
            "Optional base directory under which an out-of-output backend cache "
            "may be deleted. Repeat to allow multiple scratch/project roots."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--disabled", action="store_true")
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _assert_safe_cache_root(
    output_root: Path,
    cache_root: Path,
    allowed_cache_base_roots: tuple[Path, ...] = (),
) -> None:
    if cache_root.name != "backend_cache":
        raise ValueError(f"refusing to delete cache root that does not end in backend_cache: {cache_root}")
    if cache_root == output_root:
        raise ValueError("refusing to delete OUTPUT_ROOT itself")
    try:
        cache_root.relative_to(output_root)
        return
    except ValueError as exc:
        output_root_error = exc
    for base_root in allowed_cache_base_roots:
        if base_root == Path(base_root.anchor):
            continue
        try:
            cache_root.relative_to(base_root)
        except ValueError:
            continue
        if cache_root == base_root:
            continue
        return
    allowed = ", ".join(str(path) for path in allowed_cache_base_roots) or "none"
    raise ValueError(
        "refusing to delete cache root outside output root and outside allowed cache bases: "
        f"{cache_root}; allowed_cache_base_roots={allowed}"
    ) from output_root_error


def _missing_required_outputs(output_root: Path) -> tuple[str, ...]:
    return tuple(name for name in REQUIRED_MERGED_OUTPUTS if not (output_root / name).exists())


def _cache_size(cache_root: Path) -> tuple[int, int]:
    file_count = 0
    byte_count = 0
    if not cache_root.exists():
        return 0, 0
    for dirpath, _dirnames, filenames in os.walk(cache_root):
        root = Path(dirpath)
        for filename in filenames:
            path = root / filename
            try:
                stat = path.stat()
            except OSError:
                continue
            file_count += 1
            byte_count += int(stat.st_size)
    return file_count, byte_count


def _write_manifest(output_root: Path, manifest: BackendCacheCleanupManifest) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "backend_cache_cleanup_manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True)
    )


def cleanup_backend_cache(args: argparse.Namespace) -> BackendCacheCleanupManifest:
    started = _utc_now()
    output_root = _resolve(args.output_root)
    cache_root = _resolve(args.cache_root or output_root / "backend_cache")
    allowed_cache_base_roots = tuple(_resolve(path) for path in (args.allowed_cache_base_root or ()))
    _assert_safe_cache_root(output_root, cache_root, allowed_cache_base_roots)
    missing = _missing_required_outputs(output_root)
    if args.disabled:
        manifest = BackendCacheCleanupManifest(
            output_root=str(output_root),
            cache_root=str(cache_root),
            status="disabled",
            dry_run=bool(args.dry_run),
            disabled=True,
            file_count=0,
            byte_count=0,
            required_outputs=REQUIRED_MERGED_OUTPUTS,
            missing_required_outputs=missing,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
            started_at_utc=started,
            finished_at_utc=_utc_now(),
        )
        _write_manifest(output_root, manifest)
        return manifest
    if missing:
        raise FileNotFoundError(
            "refusing backend-cache cleanup because merged outputs are missing: "
            + ", ".join(missing)
        )
    file_count, byte_count = _cache_size(cache_root)
    if not cache_root.exists():
        status = "cache_missing"
        deleted = False
    elif args.dry_run:
        status = "dry_run"
        deleted = False
    else:
        shutil.rmtree(cache_root)
        status = "deleted"
        deleted = True
    manifest = BackendCacheCleanupManifest(
        output_root=str(output_root),
        cache_root=str(cache_root),
        status=status,
        dry_run=bool(args.dry_run),
        deleted=deleted,
        file_count=file_count,
        byte_count=byte_count,
        required_outputs=REQUIRED_MERGED_OUTPUTS,
        missing_required_outputs=(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        started_at_utc=started,
        finished_at_utc=_utc_now(),
    )
    _write_manifest(output_root, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = cleanup_backend_cache(args)
    except Exception as exc:
        print(f"Backend-cache cleanup refused: {exc}", file=sys.stderr)
        return 2
    print(
        f"backend_cache_cleanup status={manifest.status} "
        f"files={manifest.file_count} bytes={manifest.byte_count} "
        f"cache_root={manifest.cache_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
