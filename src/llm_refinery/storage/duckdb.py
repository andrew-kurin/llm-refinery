from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from llm_refinery.storage.migrations import initialize_schema
from llm_refinery.storage.models import RunRecord, SampleRecord


class ResultStore:
    def __init__(self, database: str | Path):
        self.database = Path(database).resolve()
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(self.database))
        initialize_schema(self.connection, database=self.database)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ResultStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def record_run(self, record: RunRecord) -> None:
        self.connection.execute("BEGIN TRANSACTION")
        try:
            self.connection.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_id, benchmark_kind, spec_hash, parent_run_id, schema_version,
                    suite, trial_name, status, started_at, ended_at, duration_s,
                    command, cwd, llama_version, config_json, metrics_json,
                    system_json, target_json, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    record.run_id,
                    record.benchmark_kind,
                    record.spec_hash,
                    record.parent_run_id,
                    record.schema_version,
                    record.suite,
                    record.trial_name,
                    record.status,
                    record.started_at,
                    record.ended_at,
                    record.duration_s,
                    record.command,
                    record.cwd,
                    record.llama_version,
                    _json_dump(record.config_json),
                    _json_dump(record.metrics),
                    _json_dump(record.system_json),
                    _json_dump(record.target_json),
                    record.error,
                ],
            )
            self.connection.execute("DELETE FROM metrics WHERE run_id = ?", [record.run_id])
            if record.metrics:
                rows = [(record.run_id, key, float(value)) for key, value in record.metrics.items()]
                self.connection.executemany("INSERT INTO metrics VALUES (?, ?, ?)", rows)

            self.connection.execute("DELETE FROM artifacts WHERE run_id = ?", [record.run_id])
            if record.artifacts:
                artifact_rows = [
                    (
                        record.run_id,
                        artifact.role,
                        self._stored_artifact_path(artifact.path),
                        artifact.media_type,
                        _json_dump(artifact.metadata),
                    )
                    for artifact in record.artifacts
                ]
                self.connection.executemany(
                    "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?)", artifact_rows
                )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def record_sample(self, record: SampleRecord) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO samples (
                run_id, sample_id, status, payload_json, metrics_json, artifact_path, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.run_id,
                record.sample_id,
                record.status,
                _json_dump(record.payload_json),
                _json_dump(record.metrics),
                self._stored_artifact_path(record.artifact_path)
                if record.artifact_path
                else None,
                record.error,
            ],
        )

    def samples_for_run(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT sample_id, status, payload_json, metrics_json, artifact_path, error
            FROM samples
            WHERE run_id = ?
            ORDER BY sample_id
            """,
            [run_id],
        ).fetchall()
        return [
            {
                "sample_id": row[0],
                "status": row[1],
                "payload_json": json.loads(row[2] or "{}"),
                "metrics": json.loads(row[3] or "{}"),
                "artifact_path": self._resolved_artifact_path(row[4]) if row[4] else None,
                "error": row[5],
            }
            for row in rows
        ]

    def run_resume_state(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT benchmark_kind, spec_hash, parent_run_id, schema_version, trial_name,
                   status, started_at, duration_s, system_json, target_json
            FROM runs
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": run_id,
            "benchmark_kind": row[0],
            "spec_hash": row[1],
            "parent_run_id": row[2],
            "schema_version": row[3],
            "trial_name": row[4],
            "status": row[5],
            "started_at": row[6],
            "duration_s": float(row[7]),
            "system_json": json.loads(row[8] or "{}"),
            "target_json": json.loads(row[9] or "{}"),
            "artifacts": self.artifacts_for_runs([run_id]).get(run_id, {}),
        }

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT run_id, benchmark_kind, suite, trial_name, status, duration_s, command, ended_at
            FROM runs
            ORDER BY ended_at DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [
            {
                "run_id": row[0],
                "benchmark_kind": row[1],
                "suite": row[2],
                "trial_name": row[3],
                "status": row[4],
                "duration_s": row[5],
                "command": row[6],
                "ended_at": row[7],
            }
            for row in rows
        ]

    def top_by_metric(self, metric: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT r.run_id, r.benchmark_kind, r.suite, r.trial_name, r.status,
                   r.duration_s, m.value, r.command
            FROM runs r
            JOIN metrics m ON r.run_id = m.run_id
            WHERE m.name = ?
            ORDER BY m.value DESC
            LIMIT ?
            """,
            [metric, limit],
        ).fetchall()
        return [
            {
                "run_id": row[0],
                "benchmark_kind": row[1],
                "suite": row[2],
                "trial_name": row[3],
                "status": row[4],
                "duration_s": row[5],
                "metric": metric,
                "value": row[6],
                "command": row[7],
            }
            for row in rows
        ]

    def metric_names(self, limit: int = 50) -> list[tuple[str, int]]:
        return self.connection.execute(
            """
            SELECT name, COUNT(*) AS n
            FROM metrics
            GROUP BY name
            ORDER BY n DESC, name
            LIMIT ?
            """,
            [limit],
        ).fetchall()

    def comparison_runs(
        self, *, include_failed: bool = False, latest_per_trial: bool = True
    ) -> list[dict[str, Any]]:
        status_filter = "" if include_failed else "WHERE status = 'ok'"
        rows = self.connection.execute(
            f"""
            SELECT
                run_id, benchmark_kind, spec_hash, parent_run_id, schema_version,
                suite, trial_name, status, started_at, ended_at, duration_s,
                command, cwd, llama_version, config_json, system_json, target_json, error
            FROM runs
            {status_filter}
            ORDER BY ended_at DESC
            """
        ).fetchall()
        selected_rows = _latest_trial_rows(rows) if latest_per_trial else rows

        run_ids = [row[0] for row in selected_rows]
        metrics_by_run_id = self.metrics_for_runs(run_ids)
        artifacts_by_run_id = self.artifacts_for_runs(run_ids)
        return [
            {
                "run_id": row[0],
                "benchmark_kind": row[1],
                "spec_hash": row[2],
                "parent_run_id": row[3],
                "schema_version": row[4],
                "suite": row[5],
                "trial_name": row[6],
                "status": row[7],
                "started_at": row[8],
                "ended_at": row[9],
                "duration_s": row[10],
                "command": row[11],
                "cwd": row[12],
                "llama_version": row[13],
                "config_json": json.loads(row[14] or "{}"),
                "system_json": json.loads(row[15] or "{}"),
                "target_json": json.loads(row[16] or "{}"),
                "error": row[17],
                "metrics": metrics_by_run_id.get(row[0], {}),
                "artifacts": artifacts_by_run_id.get(row[0], {}),
            }
            for row in selected_rows
        ]

    def reparse_candidates(self, *, include_failed: bool = False) -> list[dict[str, Any]]:
        return [
            run
            for run in self.comparison_runs(
                include_failed=include_failed,
                latest_per_trial=False,
            )
            if run["artifacts"]
        ]

    def metrics_for_runs(self, run_ids: list[str]) -> dict[str, dict[str, float]]:
        if not run_ids:
            return {}
        placeholders = ", ".join("?" for _ in run_ids)
        rows = self.connection.execute(
            f"SELECT run_id, name, value FROM metrics WHERE run_id IN ({placeholders})",
            run_ids,
        ).fetchall()
        metrics_by_run_id: dict[str, dict[str, float]] = {run_id: {} for run_id in run_ids}
        for run_id, name, value in rows:
            metrics_by_run_id[run_id][name] = float(value)
        return metrics_by_run_id

    def artifacts_for_runs(self, run_ids: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
        if not run_ids:
            return {}
        placeholders = ", ".join("?" for _ in run_ids)
        rows = self.connection.execute(
            f"""
            SELECT run_id, role, path, media_type, metadata_json
            FROM artifacts
            WHERE run_id IN ({placeholders})
            ORDER BY role
            """,
            run_ids,
        ).fetchall()
        by_run_id: dict[str, dict[str, dict[str, Any]]] = {run_id: {} for run_id in run_ids}
        for run_id, role, path, media_type, metadata_json in rows:
            by_run_id[run_id][role] = {
                "path": self._resolved_artifact_path(path),
                "media_type": media_type,
                "metadata": json.loads(metadata_json or "{}"),
            }
        return by_run_id

    def _stored_artifact_path(self, path: str) -> str:
        resolved = Path(path).resolve()
        try:
            return str(resolved.relative_to(self.database.parent))
        except ValueError:
            return str(resolved)

    def _resolved_artifact_path(self, path: str) -> str:
        artifact_path = Path(path)
        if not artifact_path.is_absolute():
            artifact_path = self.database.parent / artifact_path
        return str(artifact_path.resolve())

    def update_run_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        self.connection.execute("BEGIN TRANSACTION")
        try:
            self.connection.execute(
                "UPDATE runs SET metrics_json = ? WHERE run_id = ?",
                [_json_dump(metrics), run_id],
            )
            self.connection.execute("DELETE FROM metrics WHERE run_id = ?", [run_id])
            if metrics:
                rows = [(run_id, key, float(value)) for key, value in metrics.items()]
                self.connection.executemany("INSERT INTO metrics VALUES (?, ?, ?)", rows)
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def system_json_backfill_count(self, *, overwrite: bool = False) -> int:
        where_clause = "" if overwrite else _missing_system_json_where_clause()
        count_row = self.connection.execute(
            f"SELECT COUNT(*) FROM runs {where_clause}"
        ).fetchone()
        assert count_row is not None
        return int(count_row[0])

    def backfill_system_json(self, system_json: dict[str, Any], *, overwrite: bool = False) -> int:
        where_clause = "" if overwrite else _missing_system_json_where_clause()
        count = self.system_json_backfill_count(overwrite=overwrite)
        if count:
            self.connection.execute(
                f"UPDATE runs SET system_json = ? {where_clause}",
                [_json_dump(system_json)],
            )
        return int(count)


def _latest_trial_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    selected = []
    seen_trials: set[str] = set()
    for row in rows:
        trial_name = row[6]
        if trial_name in seen_trials:
            continue
        seen_trials.add(trial_name)
        selected.append(row)
    return selected


def _json_dump(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _missing_system_json_where_clause() -> str:
    return "WHERE system_json IS NULL OR trim(system_json) = '' OR trim(system_json) = '{}'"


def utc_now() -> datetime:
    return datetime.now(UTC)
