from pathlib import Path

from llm_refinery.core.runs import artifact_dir_for_run, make_run_id, record_benchmark_run
from llm_refinery.storage import ResultStore, utc_now


def test_make_run_id_adds_random_suffix():
    run_id = make_run_id("abc123")

    assert run_id.startswith("abc123-")
    assert len(run_id) == len("abc123-") + 8


def test_artifact_dir_for_run_uses_database_parent(tmp_path: Path):
    assert artifact_dir_for_run(tmp_path / "db.duckdb", "run-1") == (
        tmp_path / "artifacts" / "run-1"
    )


def test_record_benchmark_run_fills_common_metadata(tmp_path: Path):
    database = tmp_path / "runs.duckdb"
    now = utc_now()
    stdout_path = tmp_path / "stdout.txt"
    stdout_path.write_text("ok", encoding="utf-8")

    with ResultStore(database) as store:
        record_benchmark_run(
            store,
            run_id="run-1",
            suite="suite",
            trial_name="suite/trial",
            status="ok",
            started_at=now,
            ended_at=now,
            duration_s=1.0,
            command="echo ok",
            config_json={"params": {"x": 1}},
            metrics={"score": 1.0},
            stdout_path=stdout_path,
        )
        rows = store.comparison_runs()

    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["stdout_path"] == str(stdout_path)
    assert rows[0]["metrics"] == {"score": 1.0}
    assert rows[0]["system_json"]["platform"]["python_version"]
