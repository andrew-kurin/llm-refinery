from pathlib import Path

import pytest

from llm_refinery.application.run_session import RunSession
from llm_refinery.core.runs import RunSpec, artifact_dir_for_run, make_run_id
from llm_refinery.storage.duckdb import ResultStore


def test_make_run_id_adds_random_suffix():
    run_id = make_run_id("abc123")

    assert run_id.startswith("abc123-")
    assert len(run_id) == len("abc123-") + 8


def test_artifact_dir_for_run_uses_database_parent(tmp_path: Path):
    assert artifact_dir_for_run(tmp_path / "db.duckdb", "run-1") == (
        tmp_path / "artifacts" / "run-1"
    )


def test_run_session_persists_unexpected_failures(tmp_path: Path):
    database = tmp_path / "runs.duckdb"
    spec = RunSpec.create(
        benchmark_kind="test",
        suite="suite",
        label="suite/trial",
        command="explode",
        config_json={},
        database=database,
    )

    with ResultStore(database) as store:
        with pytest.raises(RuntimeError, match="boom"), RunSession(
            store,
            spec,
            system_profile={"platform": {"python": "test"}},
        ):
            raise RuntimeError("boom")
        rows = store.comparison_runs(include_failed=True)

    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "RuntimeError: boom"
