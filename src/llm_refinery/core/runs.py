from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from llm_refinery.storage import ResultStore, RunRecord
from llm_refinery.utils.system import get_system_profile


def make_run_id(key: str) -> str:
    return f"{key}-{uuid.uuid4().hex[:8]}"


def artifact_dir_for_run(database: str | Path, run_id: str) -> Path:
    return Path(database).parent / "artifacts" / run_id


def prepare_artifact_dir(database: str | Path, run_id: str) -> Path:
    artifact_dir = artifact_dir_for_run(database, run_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def record_benchmark_run(
    store: ResultStore,
    *,
    run_id: str,
    suite: str,
    trial_name: str,
    status: str,
    started_at: Any,
    ended_at: Any,
    duration_s: float,
    command: str,
    config_json: dict[str, Any],
    metrics: dict[str, float],
    stdout_path: str | Path | None = None,
    stderr_path: str | Path | None = None,
    llama_version: str | None = None,
    error: str | None = None,
) -> None:
    store.record_run(
        RunRecord(
            run_id=run_id,
            suite=suite,
            trial_name=trial_name,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            duration_s=duration_s,
            command=command,
            cwd=str(Path.cwd()),
            config_json=config_json,
            metrics=metrics,
            system_json=get_system_profile(),
            stdout_path=str(stdout_path) if stdout_path else None,
            stderr_path=str(stderr_path) if stderr_path else None,
            llama_version=llama_version,
            error=error,
        )
    )
