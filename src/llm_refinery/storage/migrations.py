from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from llm_refinery.core.runs import stable_hash

DATABASE_SCHEMA_VERSION = 3


def initialize_schema(
    connection: duckdb.DuckDBPyConnection, *, database: Path
) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            benchmark_kind TEXT,
            spec_hash TEXT,
            parent_run_id TEXT,
            schema_version INTEGER NOT NULL DEFAULT 1,
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
            metrics_json TEXT NOT NULL,
            system_json TEXT,
            target_json TEXT,
            error TEXT
        )
        """
    )
    _ensure_column(connection, "runs", "benchmark_kind", "TEXT")
    _ensure_column(connection, "runs", "spec_hash", "TEXT")
    _ensure_column(connection, "runs", "parent_run_id", "TEXT")
    _ensure_column(connection, "runs", "schema_version", "INTEGER DEFAULT 1")
    _ensure_column(connection, "runs", "system_json", "TEXT")
    _ensure_column(connection, "runs", "target_json", "TEXT")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            run_id TEXT NOT NULL,
            name TEXT NOT NULL,
            value DOUBLE NOT NULL,
            PRIMARY KEY (run_id, name)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            run_id TEXT NOT NULL,
            role TEXT NOT NULL,
            path TEXT NOT NULL,
            media_type TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            PRIMARY KEY (run_id, role)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS samples (
            run_id TEXT NOT NULL,
            sample_id TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            artifact_path TEXT,
            error TEXT,
            PRIMARY KEY (run_id, sample_id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TIMESTAMP NOT NULL
        )
        """
    )
    migrations = (
        (2, lambda: _migrate_legacy_runs(connection, database=database)),
        (3, lambda: _ensure_column(connection, "runs", "target_json", "TEXT")),
    )
    for version, migrate in migrations:
        applied = connection.execute(
            "SELECT 1 FROM schema_migrations WHERE version = ?",
            [version],
        ).fetchone()
        if applied is not None:
            continue
        migrate()
        connection.execute(
            "INSERT INTO schema_migrations VALUES (?, ?)",
            [version, datetime.now(UTC)],
        )


def _migrate_legacy_runs(
    connection: duckdb.DuckDBPyConnection, *, database: Path
) -> None:
    columns = _column_names(connection, "runs")
    legacy_artifacts = {"stdout_path", "stderr_path"}.issubset(columns)
    selected_columns = (
        "run_id, trial_name, command, config_json, benchmark_kind, spec_hash, cwd"
    )
    if legacy_artifacts:
        selected_columns += ", stdout_path, stderr_path"
    rows = connection.execute(f"SELECT {selected_columns} FROM runs").fetchall()

    for row in rows:
        run_id, trial_name, command, config_text, benchmark_kind, spec_hash_value, cwd = row[:7]
        config = _migrate_config(json.loads(config_text or "{}"))
        kind = benchmark_kind or _infer_benchmark_kind(config)
        fingerprint = spec_hash_value or stable_hash(
            {
                "benchmark_kind": kind,
                "trial_name": trial_name,
                "command": command,
                "config": config,
            }
        )
        normalized_trial_name = (
            trial_name if str(trial_name).endswith(fingerprint) else f"{trial_name}/{fingerprint}"
        )
        connection.execute(
            """
            UPDATE runs
            SET benchmark_kind = ?, spec_hash = ?, trial_name = ?, config_json = ?,
                schema_version = coalesce(schema_version, 1)
            WHERE run_id = ?
            """,
            [kind, fingerprint, normalized_trial_name, json.dumps(config), run_id],
        )

        if not legacy_artifacts:
            continue
        stdout_path, stderr_path = row[7], row[8]
        if stdout_path:
            role, media_type = _legacy_stdout_role(kind, stdout_path)
            _insert_legacy_artifact(
                connection,
                run_id,
                role,
                stdout_path,
                media_type,
                base_dir=Path(cwd),
                database=database,
            )
        if stderr_path:
            role = "errors" if kind in {"http_load", "agent_eval"} else "stderr"
            _insert_legacy_artifact(
                connection,
                run_id,
                role,
                stderr_path,
                "text/plain",
                base_dir=Path(cwd),
                database=database,
            )


def _insert_legacy_artifact(
    connection: duckdb.DuckDBPyConnection,
    run_id: str,
    role: str,
    path: str,
    media_type: str,
    *,
    base_dir: Path,
    database: Path,
) -> None:
    resolved = _resolve_legacy_path(path, base_dir=base_dir, database=database)
    connection.execute(
        "INSERT OR IGNORE INTO artifacts VALUES (?, ?, ?, ?, ?)",
        [run_id, role, _stored_artifact_path(resolved, database), media_type, "{}"],
    )


def _stored_artifact_path(path: Path, database: Path) -> str:
    try:
        return str(path.relative_to(database.parent))
    except ValueError:
        return str(path)


def _resolve_legacy_path(path: str, *, base_dir: Path, database: Path) -> Path:
    artifact_path = Path(path)
    if artifact_path.is_absolute():
        return artifact_path.resolve()
    if artifact_path.parts and artifact_path.parts[0] == database.parent.name:
        database_candidate = database.parent.parent / artifact_path
    else:
        database_candidate = database.parent / artifact_path
    candidates = [
        base_dir / artifact_path,
        database_candidate,
        Path.cwd() / artifact_path,
    ]
    return next(
        (candidate.resolve() for candidate in candidates if candidate.exists()),
        candidates[0].resolve(),
    )


def _ensure_column(
    connection: duckdb.DuckDBPyConnection,
    table: str,
    column: str,
    column_type: str,
) -> None:
    if column not in _column_names(connection, table):
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def _column_names(connection: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info('{table}')").fetchall()
    return {str(row[1]) for row in rows}


def _migrate_config(config: dict[str, Any]) -> dict[str, Any]:
    protocol_by_provider = {
        "openai": "openai_chat",
        "cerebras": "openai_chat",
        "ollama": "ollama_chat",
    }

    def migrate(value: Any) -> Any:
        if isinstance(value, dict):
            migrated = {key: migrate(child) for key, child in value.items()}
            provider = migrated.pop("provider", None)
            if "protocol" not in migrated and provider in protocol_by_provider:
                migrated["protocol"] = protocol_by_provider[provider]
            return migrated
        if isinstance(value, list):
            return [migrate(child) for child in value]
        return value

    migrated_config = migrate(config)
    assert isinstance(migrated_config, dict)
    if migrated_config.get("benchmark") == "lm-eval":
        migrated_config["benchmark"] = "lm_eval"
    return migrated_config


def _infer_benchmark_kind(config: dict[str, Any]) -> str:
    benchmark = config.get("benchmark")
    if benchmark in {"lm-eval", "lm_eval"}:
        return "lm_eval"
    if isinstance(benchmark, dict):
        return "agent_eval"
    params = config.get("params") or {}
    if isinstance(params, dict) and params.get("scenario") and (
        params.get("provider") or params.get("protocol")
    ):
        return "http_load"
    return "llama_bench"


def _legacy_stdout_role(kind: str, path: str) -> tuple[str, str]:
    suffix = Path(path).suffix.lower()
    if kind == "lm_eval":
        return "result", "application/json"
    if kind in {"http_load", "agent_eval"}:
        return "responses", "application/x-ndjson"
    media_type = "application/json" if suffix == ".json" else "text/plain"
    return "stdout", media_type
