from __future__ import annotations

import time
from pathlib import Path
from types import TracebackType
from typing import Any

from llm_refinery.core.runs import (
    Artifact,
    CompletedRun,
    RunSpec,
    make_run_id,
    prepare_artifact_dir,
)
from llm_refinery.storage.duckdb import ResultStore, utc_now
from llm_refinery.storage.models import RunRecord
from llm_refinery.utils.system import get_system_profile


class RunSession:
    """Own the durable lifecycle and artifacts for one benchmark run."""

    def __init__(
        self,
        store: ResultStore,
        spec: RunSpec,
        *,
        system_profile: dict[str, Any] | None = None,
    ) -> None:
        if store.database != spec.database.resolve():
            raise ValueError(
                f"run database {spec.database.resolve()} does not match store {store.database}"
            )
        self.store = store
        self.spec = spec
        self.run_id = make_run_id(spec.spec_hash)
        self.artifact_dir = prepare_artifact_dir(store.database, self.run_id)
        if system_profile is not None:
            self.system_profile = system_profile
        else:
            try:
                self.system_profile = get_system_profile()
            except Exception as exc:  # noqa: BLE001 - metadata must not prevent a benchmark
                self.system_profile = {"capture_error": f"{type(exc).__name__}: {exc}"}
        self.started_at = utc_now()
        self._started_monotonic = time.perf_counter()
        self._artifacts: dict[str, Artifact] = {}
        self._completed: CompletedRun | None = None
        self._entered = False

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._started_monotonic

    @property
    def completed(self) -> CompletedRun | None:
        return self._completed

    def __enter__(self) -> RunSession:
        if self._entered:
            raise RuntimeError(f"run {self.run_id} session has already been entered")
        self._entered = True
        self._record(status="running", metrics={}, error=None)
        return self

    def artifact(self, role: str, filename: str, media_type: str) -> Path:
        if not self._entered:
            raise RuntimeError("RunSession must be entered before creating artifacts")
        if role in self._artifacts:
            raise ValueError(f"artifact role already registered: {role}")
        path = (self.artifact_dir / filename).resolve()
        if not path.is_relative_to(self.artifact_dir):
            raise ValueError(f"artifact filename escapes run directory: {filename}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        self._artifacts[role] = Artifact(role=role, path=str(path), media_type=media_type)
        return path

    def complete(
        self,
        *,
        status: str = "ok",
        metrics: dict[str, float] | None = None,
        error: str | None = None,
        llama_version: str | None = None,
    ) -> CompletedRun:
        if not self._entered:
            raise RuntimeError("RunSession must be entered before completion")
        if self._completed is not None:
            raise RuntimeError(f"run {self.run_id} has already completed")
        metrics = metrics or {}
        duration_s = self.elapsed_s
        self._record(
            status=status,
            metrics=metrics,
            error=error,
            llama_version=llama_version,
            duration_s=duration_s,
        )
        self._completed = CompletedRun(
            run_id=self.run_id,
            benchmark_kind=self.spec.benchmark_kind,
            spec_hash=self.spec.spec_hash,
            status=status,
            duration_s=duration_s,
            metrics=metrics,
            error=error,
        )
        return self._completed

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, traceback
        if self._completed is not None:
            return
        if exc is None:
            self.complete()
            return
        self.complete(status="failed", error=f"{type(exc).__name__}: {exc}")

    def _record(
        self,
        *,
        status: str,
        metrics: dict[str, float],
        error: str | None,
        llama_version: str | None = None,
        duration_s: float = 0.0,
    ) -> None:
        ended_at = utc_now()
        artifacts = tuple(
            artifact for artifact in self._artifacts.values() if Path(artifact.path).exists()
        )
        self.store.record_run(
            RunRecord(
                run_id=self.run_id,
                benchmark_kind=self.spec.benchmark_kind,
                spec_hash=self.spec.spec_hash,
                parent_run_id=self.spec.parent_run_id,
                schema_version=self.spec.schema_version,
                suite=self.spec.suite,
                trial_name=self.spec.trial_name,
                status=status,
                started_at=self.started_at,
                ended_at=ended_at,
                duration_s=duration_s,
                command=self.spec.command,
                cwd=str(Path.cwd()),
                config_json=self.spec.config_json,
                metrics=metrics,
                system_json=self.system_profile,
                artifacts=artifacts,
                llama_version=llama_version,
                error=error,
            )
        )
