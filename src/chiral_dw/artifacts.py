"""Pydantic artifact models shared by command-line workflows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RunArtifact(BaseModel):
    """One generated artifact expected from a run."""

    model_config = ConfigDict(frozen=True)

    name: str
    path: str
    kind: Literal["array", "json", "plot", "table", "text", "other"]
    description: str
    required: bool = True
    exists: bool = False
    size_bytes: int | None = Field(default=None, ge=0)


class RunManifest(BaseModel):
    """Summary of artifacts and validation status for a run."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    result_dir: str
    generated_at: str
    artifacts: tuple[RunArtifact, ...]
    missing_required: tuple[str, ...]
    passed: bool

    @classmethod
    def from_artifacts(
        cls,
        *,
        run_id: str,
        result_dir: str,
        artifacts: tuple[RunArtifact, ...] | list[RunArtifact],
    ) -> "RunManifest":
        items = tuple(artifacts)
        missing = tuple(item.name for item in items if item.required and not item.exists)
        return cls(
            run_id=run_id,
            result_dir=result_dir,
            generated_at=datetime.now(timezone.utc).isoformat(),
            artifacts=items,
            missing_required=missing,
            passed=not missing,
        )
