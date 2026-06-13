from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    suite: str
    trial_name: str
    status: str
    started_at: datetime
    ended_at: datetime
    duration_s: float
    command: str
    cwd: str
    config_json: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    system_json: dict[str, Any] = field(default_factory=dict)
    stdout_path: str | None = None
    stderr_path: str | None = None
    llama_version: str | None = None
    error: str | None = None


class ResultStore:
    def __init__(self, database: str | Path):
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(str(self.database))
        self._init_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> ResultStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def record_run(self, record: RunRecord) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, suite, trial_name, status, started_at, ended_at, duration_s,
                command, cwd, llama_version, config_json, stdout_path, stderr_path,
                metrics_json, system_json, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                record.run_id,
                record.suite,
                record.trial_name,
                record.status,
                record.started_at,
                record.ended_at,
                record.duration_s,
                record.command,
                record.cwd,
                record.llama_version,
                json.dumps(record.config_json, sort_keys=True, default=str),
                record.stdout_path,
                record.stderr_path,
                json.dumps(record.metrics, sort_keys=True, default=str),
                json.dumps(record.system_json, sort_keys=True, default=str),
                record.error,
            ],
        )
        self.connection.execute("DELETE FROM metrics WHERE run_id = ?", [record.run_id])
        if record.metrics:
            rows = [(record.run_id, key, float(value)) for key, value in record.metrics.items()]
            self.connection.executemany("INSERT INTO metrics VALUES (?, ?, ?)", rows)

    def recent_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT run_id, suite, trial_name, status, duration_s, command, ended_at
            FROM runs
            ORDER BY ended_at DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        return [
            {
                "run_id": row[0],
                "suite": row[1],
                "trial_name": row[2],
                "status": row[3],
                "duration_s": row[4],
                "command": row[5],
                "ended_at": row[6],
            }
            for row in rows
        ]

    def top_by_metric(self, metric: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT r.run_id, r.suite, r.trial_name, r.status, r.duration_s, m.value, r.command
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
                "suite": row[1],
                "trial_name": row[2],
                "status": row[3],
                "duration_s": row[4],
                "metric": metric,
                "value": row[5],
                "command": row[6],
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
                run_id, suite, trial_name, status, started_at, ended_at, duration_s,
                command, cwd, llama_version, config_json, stdout_path, stderr_path,
                system_json, error
            FROM runs
            {status_filter}
            ORDER BY ended_at DESC
            """
        ).fetchall()

        selected_rows = []
        seen_trials: set[str] = set()
        for row in rows:
            trial_name = row[2]
            if latest_per_trial and trial_name in seen_trials:
                continue
            seen_trials.add(trial_name)
            selected_rows.append(row)

        metrics_by_run_id = self.metrics_for_runs([row[0] for row in selected_rows])
        return [
            {
                "run_id": row[0],
                "suite": row[1],
                "trial_name": row[2],
                "status": row[3],
                "started_at": row[4],
                "ended_at": row[5],
                "duration_s": row[6],
                "command": row[7],
                "cwd": row[8],
                "llama_version": row[9],
                "config_json": json.loads(row[10]),
                "stdout_path": row[11],
                "stderr_path": row[12],
                "system_json": json.loads(row[13] or "{}"),
                "error": row[14],
                "metrics": metrics_by_run_id.get(row[0], {}),
            }
            for row in selected_rows
        ]

    def runs_with_artifacts(self, *, include_failed: bool = False) -> list[dict[str, Any]]:
        status_filter = "" if include_failed else "WHERE status = 'ok'"
        rows = self.connection.execute(
            f"""
            SELECT run_id, stdout_path, stderr_path, status
            FROM runs
            {status_filter}
            ORDER BY ended_at DESC
            """
        ).fetchall()
        return [
            {"run_id": row[0], "stdout_path": row[1], "stderr_path": row[2], "status": row[3]}
            for row in rows
        ]

    def metrics_for_runs(self, run_ids: list[str]) -> dict[str, dict[str, float]]:
        if not run_ids:
            return {}

        rows = self.connection.execute("SELECT run_id, name, value FROM metrics").fetchall()
        wanted = set(run_ids)
        metrics_by_run_id: dict[str, dict[str, float]] = {run_id: {} for run_id in run_ids}
        for run_id, name, value in rows:
            if run_id in wanted:
                metrics_by_run_id[run_id][name] = float(value)
        return metrics_by_run_id

    def update_run_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        self.connection.execute(
            "UPDATE runs SET metrics_json = ? WHERE run_id = ?",
            [json.dumps(metrics, sort_keys=True, default=str), run_id],
        )
        self.connection.execute("DELETE FROM metrics WHERE run_id = ?", [run_id])
        if metrics:
            rows = [(run_id, key, float(value)) for key, value in metrics.items()]
            self.connection.executemany("INSERT INTO metrics VALUES (?, ?, ?)", rows)

    def backfill_system_json(self, system_json: dict[str, Any], *, overwrite: bool = False) -> int:
        where_clause = "" if overwrite else _missing_system_json_where_clause()
        count = self.connection.execute(f"SELECT COUNT(*) FROM runs {where_clause}").fetchone()[0]
        if count:
            self.connection.execute(
                f"UPDATE runs SET system_json = ? {where_clause}",
                [json.dumps(system_json, sort_keys=True, default=str)],
            )
        return int(count)

    def _init_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                suite TEXT NOT NULL,
                trial_name TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TIMESTAMP NOT NULL,
                ended_at TIMESTAMP NOT NULL,
                duration_s DOUBLE NOT NULL,
                command TEXT NOT NULL,
                cwd TEXT NOT NULL,
                llama_version TEXT,
                config_json TEXT NOT NULL,
                stdout_path TEXT,
                stderr_path TEXT,
                metrics_json TEXT NOT NULL,
                system_json TEXT,
                error TEXT
            )
            """
        )
        self._ensure_column("runs", "system_json", "TEXT")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                run_id TEXT NOT NULL,
                name TEXT NOT NULL,
                value DOUBLE NOT NULL,
                PRIMARY KEY (run_id, name)
            )
            """
        )

    def _ensure_column(self, table: str, column: str, column_type: str) -> None:
        rows = self.connection.execute(f"PRAGMA table_info('{table}')").fetchall()
        existing = {str(row[1]) for row in rows}
        if column not in existing:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def _missing_system_json_where_clause() -> str:
    return "WHERE system_json IS NULL OR trim(system_json) = '' OR trim(system_json) = '{}'"


def utc_now() -> datetime:
    return datetime.now(UTC)
