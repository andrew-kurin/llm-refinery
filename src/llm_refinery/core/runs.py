from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RUN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Artifact:
    role: str
    path: str
    media_type: str = "application/octet-stream"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunSpec:
    benchmark_kind: str
    suite: str
    label: str
    trial_name: str
    spec_hash: str
    command: str
    config_json: dict[str, Any]
    database: Path
    parent_run_id: str | None = None
    schema_version: int = RUN_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        benchmark_kind: str,
        suite: str,
        label: str,
        command: str,
        config_json: dict[str, Any],
        database: str | Path,
        parent_run_id: str | None = None,
    ) -> RunSpec:
        if not benchmark_kind.strip():
            raise ValueError("benchmark_kind cannot be empty")
        if not suite.strip():
            raise ValueError("suite cannot be empty")
        normalized_label = label.rstrip("/")
        if not normalized_label:
            raise ValueError("run label cannot be empty")
        normalized_config = json.loads(json.dumps(config_json, sort_keys=True, default=str))
        spec_hash = stable_hash(
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "benchmark_kind": benchmark_kind,
                "suite": suite,
                "command": command,
                "config": normalized_config,
            }
        )
        return cls(
            benchmark_kind=benchmark_kind,
            suite=suite,
            label=normalized_label,
            trial_name=f"{normalized_label}/{spec_hash}",
            spec_hash=spec_hash,
            command=command,
            config_json=normalized_config,
            database=Path(database),
            parent_run_id=parent_run_id,
        )


@dataclass(frozen=True)
class CompletedRun:
    run_id: str
    benchmark_kind: str
    spec_hash: str
    status: str
    duration_s: float
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None


def make_run_id(spec_hash: str) -> str:
    return f"{spec_hash}-{uuid.uuid4().hex[:8]}"


def artifact_dir_for_run(database: str | Path, run_id: str) -> Path:
    return Path(database).resolve().parent / "artifacts" / run_id


def prepare_artifact_dir(database: str | Path, run_id: str) -> Path:
    artifact_dir = artifact_dir_for_run(database, run_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
