from pathlib import Path

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
