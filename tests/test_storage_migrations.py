import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from llm_refinery.storage.duckdb import ResultStore
from llm_refinery.storage.migrations import DATABASE_SCHEMA_VERSION


def test_result_store_migrates_legacy_run_identity_and_artifacts(tmp_path: Path):
    project = tmp_path / "moved-project"
    database = project / "results" / "legacy.duckdb"
    database.parent.mkdir(parents=True)
    responses = project / "results" / "artifacts" / "legacy-run" / "responses.jsonl"
    responses.parent.mkdir(parents=True)
    responses.write_text('{"ok": true}\n', encoding="utf-8")
    connection = duckdb.connect(str(database))
    connection.execute(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY, suite TEXT NOT NULL, trial_name TEXT NOT NULL,
            status TEXT NOT NULL, started_at TIMESTAMP NOT NULL, ended_at TIMESTAMP NOT NULL,
            duration_s DOUBLE NOT NULL, command TEXT NOT NULL, cwd TEXT NOT NULL,
            llama_version TEXT, config_json TEXT NOT NULL, stdout_path TEXT,
            stderr_path TEXT, metrics_json TEXT NOT NULL, system_json TEXT, error TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE metrics (
            run_id TEXT NOT NULL, name TEXT NOT NULL, value DOUBLE NOT NULL,
            PRIMARY KEY (run_id, name)
        )
        """
    )
    now = datetime.now(UTC)
    config = {
        "params": {
            "provider": "openai",
            "scenario": "short",
            "concurrency": 1,
        }
    }
    connection.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            "legacy-run",
            "legacy-suite",
            "legacy-suite/local/short",
            "ok",
            now,
            now,
            1.0,
            "http-load",
            "/old/project/location",
            None,
            json.dumps(config),
            "results/artifacts/legacy-run/responses.jsonl",
            None,
            "{}",
            "{}",
            None,
        ],
    )
    connection.close()

    with ResultStore(database) as store:
        run = store.comparison_runs()[0]
        stored_path = store.connection.execute(
            "SELECT path FROM artifacts WHERE run_id = 'legacy-run'"
        ).fetchone()
        columns = {
            row[1] for row in store.connection.execute("PRAGMA table_info('runs')").fetchall()
        }
        versions = [
            row[0]
            for row in store.connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        ]

    assert stored_path == ("artifacts/legacy-run/responses.jsonl",)
    assert run["benchmark_kind"] == "http_load"
    assert run["config_json"]["params"]["protocol"] == "openai_chat"
    assert "provider" not in run["config_json"]["params"]
    assert run["spec_hash"]
    assert run["trial_name"].endswith(run["spec_hash"])
    assert run["artifacts"]["responses"]["path"] == str(responses.resolve())
    assert run["target_json"] == {}
    assert "target_json" in columns
    assert versions == [2, DATABASE_SCHEMA_VERSION]


def test_schema_v3_target_json_migration_is_ordered_and_idempotent(tmp_path: Path):
    database = tmp_path / "v2.duckdb"
    connection = duckdb.connect(str(database))
    connection.execute(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY, benchmark_kind TEXT, spec_hash TEXT,
            parent_run_id TEXT, schema_version INTEGER NOT NULL DEFAULT 1,
            suite TEXT NOT NULL, trial_name TEXT NOT NULL, status TEXT NOT NULL,
            started_at TIMESTAMP NOT NULL, ended_at TIMESTAMP NOT NULL,
            duration_s DOUBLE NOT NULL, command TEXT NOT NULL, cwd TEXT NOT NULL,
            llama_version TEXT, config_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
            system_json TEXT, error TEXT
        )
        """
    )
    connection.execute(
        """CREATE TABLE schema_migrations (
            version INTEGER PRIMARY KEY, applied_at TIMESTAMP NOT NULL
        )"""
    )
    connection.execute("INSERT INTO schema_migrations VALUES (2, ?)", [datetime.now(UTC)])
    connection.close()

    for _ in range(2):
        with ResultStore(database) as store:
            columns = {
                row[1] for row in store.connection.execute("PRAGMA table_info('runs')").fetchall()
            }
            versions = store.connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()

        assert "target_json" in columns
        assert versions == [(2,), (DATABASE_SCHEMA_VERSION,)]
