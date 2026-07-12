from pathlib import Path

import pytest

from llm_refinery.application.run_session import RunSession
from llm_refinery.core.runs import RunSpec
from llm_refinery.storage.duckdb import ResultStore


def test_run_session_records_identity_metrics_and_typed_artifacts(tmp_path: Path):
    database = tmp_path / "runs.duckdb"
    spec = RunSpec.create(
        benchmark_kind="http_load",
        suite="load-suite",
        label="load-suite/local/short",
        command="http-load target=local scenario=short",
        config_json={"params": {"protocol": "openai_chat", "concurrency": 1}},
        database=database,
    )

    with ResultStore(database) as store:
        with RunSession(store, spec, system_profile={"hardware": {"model": "test"}}) as run:
            responses = run.artifact("responses", "responses.jsonl", "application/x-ndjson")
            responses.write_text('{"ok": true}\n', encoding="utf-8")
            completed = run.complete(metrics={"success_count": 1.0})

        rows = store.comparison_runs()
        stored_path = store.connection.execute(
            "SELECT path FROM artifacts WHERE run_id = ?",
            [completed.run_id],
        ).fetchone()

    assert stored_path == (f"artifacts/{completed.run_id}/responses.jsonl",)
    assert completed.status == "ok"
    assert len(rows) == 1
    assert rows[0]["benchmark_kind"] == "http_load"
    assert rows[0]["spec_hash"] == spec.spec_hash
    assert rows[0]["trial_name"].endswith(spec.spec_hash)
    assert rows[0]["metrics"] == {"success_count": 1.0}
    assert rows[0]["artifacts"]["responses"]["media_type"] == "application/x-ndjson"
    assert Path(rows[0]["artifacts"]["responses"]["path"]).read_text() == '{"ok": true}\n'


def test_run_session_refuses_to_resume_with_a_different_spec(tmp_path: Path):
    database = tmp_path / "runs.duckdb"
    original = RunSpec.create(
        benchmark_kind="dabstep",
        suite="dabstep",
        label="dabstep/local",
        command="python baseline/run.py --max-steps 10",
        config_json={"max_steps": 10},
        database=database,
    )
    changed = RunSpec.create(
        benchmark_kind="dabstep",
        suite="dabstep",
        label="dabstep/local",
        command="python baseline/run.py --max-steps 20",
        config_json={"max_steps": 20},
        database=database,
    )

    with ResultStore(database) as store:
        with RunSession(store, original, system_profile={}) as run:
            run.complete(status="failed", error="interrupted")
        with (
            pytest.raises(RuntimeError, match="does not match"),
            RunSession(
                store,
                changed,
                resume_run_id=run.run_id,
            ),
        ):
            pass
