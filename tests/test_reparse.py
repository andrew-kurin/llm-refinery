import json
from pathlib import Path

from click.testing import CliRunner

from llm_refinery.application.run_session import RunSession
from llm_refinery.cli import main
from llm_refinery.core.runs import RunSpec
from llm_refinery.storage.duckdb import ResultStore


def test_reparse_dispatches_http_artifacts_without_erasing_metrics(tmp_path: Path):
    database = tmp_path / "runs.duckdb"
    spec = RunSpec.create(
        benchmark_kind="http_load",
        suite="http-suite",
        label="http-suite/local/short",
        command="http-load",
        config_json={"params": {"protocol": "openai_chat", "concurrency": 1, "max_tokens": 8}},
        database=database,
    )
    response = {
        "index": 0,
        "ok": True,
        "status_code": 200,
        "latency_s": 2.0,
        "ttft_s": 0.5,
        "prompt_tokens": 4,
        "completion_tokens": 8,
        "completion_chars": 16,
        "server_prompt_eval_duration_s": None,
        "server_eval_duration_s": None,
        "response_text": "complete",
        "check_passed": True,
        "error": None,
    }
    with ResultStore(database) as store, RunSession(store, spec, system_profile={}) as run:
        responses = run.artifact("responses", "responses.jsonl", "application/x-ndjson")
        responses.write_text(json.dumps(response) + "\n", encoding="utf-8")
        run.complete(metrics={"stale": 1.0})

    result = CliRunner().invoke(main, ["reparse", str(database)])

    assert result.exit_code == 0, result.output
    with ResultStore(database) as store:
        metrics = store.comparison_runs()[0]["metrics"]
    assert "stale" not in metrics
    assert metrics["success_count"] == 1.0
    assert metrics["latency_p95_s"] == 2.0
    assert metrics["completion_tokens_total"] == 8.0


def test_reparse_uses_dabstep_answers_and_task_manifest(tmp_path: Path):
    database = tmp_path / "runs.duckdb"
    spec = RunSpec.create(
        benchmark_kind="dabstep",
        suite="dabstep-dev",
        label="dabstep-dev/local",
        command="python baseline/run.py",
        config_json={"benchmark": "dabstep", "params": {"split": "dev"}},
        database=database,
    )
    task = {
        "task_id": "5",
        "question": "Which country?",
        "guidelines": "Country code only.",
        "level": "easy",
        "answer": "NL",
    }
    answer = {
        "task_id": "5",
        "agent_answer": "NL",
        "answer": "NL",
        "score": 1,
        "level": "easy",
    }
    with ResultStore(database) as store, RunSession(store, spec, system_profile={}) as run:
        tasks_path = run.artifact("tasks", "tasks.jsonl", "application/x-ndjson")
        answers_path = run.artifact("answers", "answers.jsonl", "application/x-ndjson")
        measurement = run.artifact("measurement", "measurement.json", "application/json")
        tasks_path.write_text(json.dumps(task) + "\n", encoding="utf-8")
        answers_path.write_text(json.dumps(answer) + "\n", encoding="utf-8")
        measurement.write_text(
            json.dumps({"wall_duration_s": 2.0, "attempts": []}),
            encoding="utf-8",
        )
        run.complete(metrics={"stale": 1.0})

    result = CliRunner().invoke(main, ["reparse", str(database)])

    assert result.exit_code == 0, result.output
    with ResultStore(database) as store:
        metrics = store.comparison_runs()[0]["metrics"]
    assert "stale" not in metrics
    assert metrics["answer_count"] == 1.0
    assert metrics["success_rate"] == 1.0
    assert metrics["wall_duration_s"] == 2.0


def test_reparse_uses_lm_eval_metric_normalization(tmp_path: Path):
    database = tmp_path / "runs.duckdb"
    spec = RunSpec.create(
        benchmark_kind="lm_eval",
        suite="quality",
        label="quality/local",
        command="lm_eval",
        config_json={"benchmark": "lm_eval", "tasks": "gsm8k"},
        database=database,
    )
    payload = {
        "results": {
            "gsm8k": {
                "exact_match,strict-match": 0.75,
                "exact_match_stderr,strict-match": 0.1,
            }
        }
    }
    with ResultStore(database) as store, RunSession(store, spec, system_profile={}) as run:
        result_path = run.artifact("result", "result.json", "application/json")
        result_path.write_text(json.dumps(payload), encoding="utf-8")
        run.complete(metrics={"stale": 1.0})

    result = CliRunner().invoke(main, ["reparse", str(database)])

    assert result.exit_code == 0, result.output
    with ResultStore(database) as store:
        metrics = store.comparison_runs()[0]["metrics"]
    assert metrics == {
        "gsm8k.strict-match.exact_match": 0.75,
        "gsm8k.strict-match.exact_match_stderr": 0.1,
        "gsm8k.strict-match.exact_match_ci95_low": 0.554,
        "gsm8k.strict-match.exact_match_ci95_high": 0.946,
    }
