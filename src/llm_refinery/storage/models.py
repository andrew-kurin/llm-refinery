from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from llm_refinery.core.runs import RUN_SCHEMA_VERSION, Artifact


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    benchmark_kind: str
    spec_hash: str
    suite: str
    trial_name: str
    status: str
    started_at: datetime
    ended_at: datetime
    duration_s: float
    command: str
    cwd: str
    config_json: dict[str, Any]
    parent_run_id: str | None = None
    schema_version: int = RUN_SCHEMA_VERSION
    metrics: dict[str, float] = field(default_factory=dict)
    system_json: dict[str, Any] = field(default_factory=dict)
    target_json: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[Artifact, ...] = ()
    llama_version: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class SampleRecord:
    run_id: str
    sample_id: str
    status: str
    payload_json: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_path: str | None = None
    error: str | None = None
