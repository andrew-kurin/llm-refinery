from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_refinery.benchmarks.dabstep import tasks as dabstep_tasks
from llm_refinery.benchmarks.dabstep.config import DabstepSettings
from llm_refinery.benchmarks.dabstep.parser import parse_dabstep_metrics
from llm_refinery.benchmarks.dabstep.tasks import (
    DabstepTask,
    validate_dabstep_task_source,
)
from llm_refinery.cli import main
from llm_refinery.core.config import ConfigError
from llm_refinery.storage.duckdb import ResultStore


def _write_tasks(path: Path) -> None:
    tasks = [
        {
            "task_id": "5",
            "question": "Which country has the most transactions?",
            "guidelines": "Answer with a country code.",
            "level": "easy",
            "answer": "NL",
        },
        {
            "task_id": "49",
            "question": "Which country has the most fraud?",
            "guidelines": "Answer with one option.",
            "level": "easy",
            "answer": "B. BE",
        },
    ]
    path.write_text(
        "".join(json.dumps(task) + "\n" for task in tasks),
        encoding="utf-8",
    )


def test_dabstep_dry_run_builds_the_official_baseline_command(tmp_path: Path) -> None:
    tasks_path = tmp_path / "dev.jsonl"
    _write_tasks(tasks_path)
    workspace = tmp_path / "upstream"
    workspace.mkdir()
    config = tmp_path / "dabstep.yaml"
    config.write_text(
        f"""
name: dabstep-smoke
database: {tmp_path / "runs.duckdb"}
endpoint:
  name: local
  protocol: openai_chat
  base_url: http://127.0.0.1:8080/v1
  model: local-model
dabstep:
  workspace: {workspace}
  command: [python3, baseline/run.py]
  tasks_file: {tasks_path}
  tasks_file_arg: --tasks-file
  split: dev
  task_ids: [5, 49]
  concurrency: 2
  max_steps: 12
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["dabstep", str(config), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "python3 baseline/run.py" in result.output
    assert "--model-id openai/local-model" in result.output
    assert "--api-base http://127.0.0.1:8080/v1" in result.output
    assert "--split dev" in result.output
    assert "--concurrency 2" in result.output
    assert "--max-steps 12" in result.output
    assert "--tasks-ids 5 49" in result.output
    assert f"--tasks-file {tasks_path}" in result.output
    assert "tasks=2" in result.output


def test_dabstep_default_submission_reports_completion_without_local_score(
    tmp_path: Path,
) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        json.dumps(
            {
                "task_id": "1712",
                "question": "What are the total fees?",
                "guidelines": "Answer with a number.",
                "level": "hard",
                "answer": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    answers_path = tmp_path / "answers.jsonl"
    answers_path.write_text(
        json.dumps({"task_id": "1712", "agent_answer": "42.00"}) + "\n",
        encoding="utf-8",
    )

    metrics = parse_dabstep_metrics(answers_path, tasks_path)

    assert metrics["completion_rate"] == 1.0
    assert metrics["scored_count"] == 0.0
    assert "success_rate" not in metrics
    assert "accuracy" not in metrics


def test_dabstep_config_rejects_unknown_settings(tmp_path: Path) -> None:
    config = tmp_path / "dabstep.yaml"
    config.write_text(
        """
name: bad-dabstep
endpoint:
  name: local
  protocol: openai_chat
  base_url: http://127.0.0.1:8080/v1
  model: local-model
dabstep:
  workspace: upstream
  unsupported_generation_knob: true
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["dabstep", str(config), "--dry-run"])

    assert result.exit_code != 0
    assert "unknown field(s): unsupported_generation_knob" in result.output


def test_dabstep_custom_manifest_requires_a_compatible_wrapper(tmp_path: Path) -> None:
    tasks_path = tmp_path / "tasks.jsonl"
    _write_tasks(tasks_path)

    with pytest.raises(ConfigError, match="requires tasks_file_arg"):
        DabstepSettings(workspace=tmp_path, tasks_file=tasks_path)


def test_dabstep_official_source_requires_a_pinned_revision(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="pinned 40-character commit"):
        DabstepSettings(workspace=tmp_path, dataset_revision="main")


def test_dabstep_official_contract_rejects_manifest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = DabstepSettings(workspace=tmp_path, task_ids=(5,), limit=None)
    selected = [
        DabstepTask(
            task_id=5,
            question="Pinned question",
            guidelines="Pinned guidelines",
            level="easy",
            answer="NL",
        )
    ]
    live_manifest = json.dumps(
        {
            "task_id": "5",
            "question": "Changed live question",
            "guidelines": "Pinned guidelines",
            "level": "easy",
            "answer": "NL",
        }
    )
    monkeypatch.setattr(dabstep_tasks, "_download_task_text", lambda *_args: live_manifest)

    with pytest.raises(ConfigError, match="does not match"):
        validate_dabstep_task_source(settings, selected)


def test_dabstep_official_contract_records_matching_manifest_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = DabstepSettings(workspace=tmp_path, task_ids=(5,), limit=None)
    selected = [
        DabstepTask(
            task_id=5,
            question="Pinned question",
            guidelines="Pinned guidelines",
            level="easy",
            answer="NL",
        )
    ]
    live_manifest = json.dumps(selected[0].as_jsonable())
    monkeypatch.setattr(dabstep_tasks, "_download_task_text", lambda *_args: live_manifest)

    contract = validate_dabstep_task_source(settings, selected)

    assert contract.mode == "official_verified"
    assert contract.selected_manifest_sha256 == contract.official_main_manifest_sha256


def test_dabstep_run_records_official_answers_metrics_and_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks_path = tmp_path / "dev.jsonl"
    _write_tasks(tasks_path)
    workspace = tmp_path / "upstream"
    workspace.mkdir()
    fake_baseline = workspace / "fake_baseline.py"
    fake_baseline.write_text(
        """
import argparse
import json
import os
from pathlib import Path

assert os.environ["OPENAI_API_KEY"] == "top-secret"
assert "DABSTEP_UNRELATED_SECRET" not in os.environ
parser = argparse.ArgumentParser()
parser.add_argument("--model-id", required=True)
parser.add_argument("--api-base", required=True)
parser.add_argument("--split", required=True)
parser.add_argument("--concurrency", required=True)
parser.add_argument("--max-steps", required=True)
parser.add_argument("--timestamp", required=True)
parser.add_argument("--tasks-ids", nargs="+", type=int, required=True)
parser.add_argument("--tasks-file", required=True)
args = parser.parse_args()
task_ids = [
    json.loads(line)["task_id"]
    for line in Path(args.tasks_file).read_text().splitlines()
]
assert task_ids == ["5", "49"]
model_dir = args.model_id.replace("/", "_").replace(".", "_")
output = Path("runs") / model_dir / args.split / args.timestamp
output.mkdir(parents=True)
answers = {
    5: {"agent_answer": "NL", "answer": "NL", "score": 1, "level": "easy"},
    49: {"agent_answer": "A. NL", "answer": "B. BE", "score": 0, "level": "easy"},
}
with (output / "answers.jsonl").open("w") as handle:
    for task_id in args.tasks_ids:
        handle.write(json.dumps({"task_id": str(task_id), **answers[task_id]}) + "\\n")
(output / "logs.txt").write_text("official agent trace\\n")
(output / "config.yaml").write_text("split: dev\\n")
print(f"completed {len(args.tasks_ids)} tasks")
""",
        encoding="utf-8",
    )
    database = tmp_path / "runs.duckdb"
    config = tmp_path / "dabstep.yaml"
    config.write_text(
        f"""
name: dabstep-smoke
database: {database}
endpoint:
  name: local
  protocol: openai_chat
  base_url: http://127.0.0.1:8080/v1
  model: local-model
  api_key_env: DABSTEP_TEST_KEY
dabstep:
  workspace: {workspace}
  command: {json.dumps([sys.executable, str(fake_baseline)])}
  tasks_file: {tasks_path}
  tasks_file_arg: --tasks-file
  split: dev
  limit: all
""",
        encoding="utf-8",
    )

    monkeypatch.setenv("DABSTEP_TEST_KEY", "top-secret")
    monkeypatch.setenv("DABSTEP_UNRELATED_SECRET", "must-not-leak")
    result = CliRunner().invoke(main, ["dabstep", str(config)])

    assert result.exit_code == 0, result.output
    with ResultStore(database) as store:
        runs = store.comparison_runs()
        assert len(runs) == 1
        run = runs[0]
        samples = store.samples_for_run(run["run_id"])
    assert run["benchmark_kind"] == "dabstep"
    assert run["status"] == "ok"
    assert run["config_json"]["target"]["api_key_env"] == "DABSTEP_TEST_KEY"
    assert run["config_json"]["task_source_contract"]["mode"] == "wrapper_manifest"
    assert "top-secret" not in json.dumps(run["config_json"])
    assert "top-secret" not in run["command"]
    assert run["metrics"]["task_count"] == 2.0
    assert run["metrics"]["answer_count"] == 2.0
    assert run["metrics"]["success_rate"] == 0.5
    assert run["metrics"]["correct_count"] == 1.0
    assert len(samples) == 2
    assert all(sample["status"] == "ok" for sample in samples)
    scores = {sample["sample_id"]: sample["metrics"]["score"] for sample in samples}
    assert scores == {"5": 1.0, "49": 0.0}
    assert set(run["artifacts"]) >= {
        "answers",
        "tasks",
        "stdout",
        "stderr",
        "measurement",
        "upstream_logs",
        "upstream_configs",
    }
    assert "completed 2 tasks" in Path(run["artifacts"]["stdout"]["path"]).read_text()
    assert "official agent trace" in Path(run["artifacts"]["upstream_logs"]["path"]).read_text()
    assert "split: dev" in Path(run["artifacts"]["upstream_configs"]["path"]).read_text()


def test_dabstep_resume_keeps_the_run_id_and_only_runs_missing_tasks(tmp_path: Path) -> None:
    tasks_path = tmp_path / "dev.jsonl"
    _write_tasks(tasks_path)
    workspace = tmp_path / "upstream"
    workspace.mkdir()
    fake_baseline = workspace / "resumable_baseline.py"
    fake_baseline.write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", required=True)
parser.add_argument("--api-base", required=True)
parser.add_argument("--split", required=True)
parser.add_argument("--concurrency", required=True)
parser.add_argument("--max-steps", required=True)
parser.add_argument("--timestamp", required=True)
parser.add_argument("--tasks-ids", nargs="+", type=int, required=True)
parser.add_argument("--tasks-file", required=True)
args = parser.parse_args()
with Path("calls.jsonl").open("a") as calls:
    calls.write(json.dumps(args.tasks_ids) + "\\n")
model_dir = args.model_id.replace("/", "_").replace(".", "_")
output = Path("runs") / model_dir / args.split / args.timestamp
output.mkdir(parents=True)
marker = Path("first-attempt-failed")
selected = args.tasks_ids if marker.exists() else args.tasks_ids[:1]
answers = {
    5: {"agent_answer": "NL", "answer": "NL", "score": 1, "level": "easy"},
    49: {"agent_answer": "A. NL", "answer": "B. BE", "score": 0, "level": "easy"},
}
with (output / "answers.jsonl").open("w") as handle:
    for task_id in selected:
        handle.write(json.dumps({"task_id": str(task_id), **answers[task_id]}) + "\\n")
if not marker.exists():
    marker.touch()
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    database = tmp_path / "runs.duckdb"
    config = tmp_path / "dabstep.yaml"
    config.write_text(
        f"""
name: dabstep-resume
database: {database}
endpoint:
  name: local
  protocol: openai_chat
  base_url: http://127.0.0.1:8080/v1
  model: local-model
dabstep:
  workspace: {workspace}
  command: {json.dumps([sys.executable, str(fake_baseline)])}
  tasks_file: {tasks_path}
  tasks_file_arg: --tasks-file
  split: dev
  limit: all
  keep_going: true
""",
        encoding="utf-8",
    )

    first = CliRunner().invoke(main, ["dabstep", str(config)])
    assert first.exit_code == 0, first.output
    with ResultStore(database) as store:
        failed_runs = store.comparison_runs(include_failed=True)
    assert len(failed_runs) == 1
    assert failed_runs[0]["status"] == "failed"
    run_id = failed_runs[0]["run_id"]

    resumed = CliRunner().invoke(main, ["dabstep", str(config), "--resume", run_id])

    assert resumed.exit_code == 0, resumed.output
    with ResultStore(database) as store:
        runs = store.comparison_runs(include_failed=True, latest_per_trial=False)
        samples = store.samples_for_run(run_id)
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_id
    assert runs[0]["status"] == "ok"
    assert runs[0]["metrics"]["answer_count"] == 2.0
    assert runs[0]["metrics"]["success_rate"] == 0.5
    assert all(sample["status"] == "ok" for sample in samples)
    calls = [json.loads(line) for line in (workspace / "calls.jsonl").read_text().splitlines()]
    assert calls == [[5, 49], [49]]


@pytest.mark.skipif(os.name == "nt", reason="the interruption fixture uses POSIX signals")
def test_dabstep_interruption_checkpoints_answers_for_resume(tmp_path: Path) -> None:
    tasks_path = tmp_path / "dev.jsonl"
    _write_tasks(tasks_path)
    workspace = tmp_path / "upstream"
    workspace.mkdir()
    fake_baseline = workspace / "interruptible_baseline.py"
    fake_baseline.write_text(
        """
import argparse
import json
import os
import signal
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", required=True)
parser.add_argument("--api-base", required=True)
parser.add_argument("--split", required=True)
parser.add_argument("--concurrency", required=True)
parser.add_argument("--max-steps", required=True)
parser.add_argument("--timestamp", required=True)
parser.add_argument("--tasks-ids", nargs="+", type=int, required=True)
parser.add_argument("--tasks-file", required=True)
args = parser.parse_args()
with Path("calls.jsonl").open("a") as calls:
    calls.write(json.dumps(args.tasks_ids) + "\\n")
model_dir = args.model_id.replace("/", "_").replace(".", "_")
output = Path("runs") / model_dir / args.split / args.timestamp
output.mkdir(parents=True)
marker = Path("interruption-marker")
selected = args.tasks_ids if marker.exists() else args.tasks_ids[:1]
answers = {
    5: {"agent_answer": "NL", "answer": "NL", "score": 1, "level": "easy"},
    49: {"agent_answer": "A. NL", "answer": "B. BE", "score": 0, "level": "easy"},
}
with (output / "answers.jsonl").open("w") as handle:
    for task_id in selected:
        handle.write(json.dumps({"task_id": str(task_id), **answers[task_id]}) + "\\n")
if not marker.exists():
    marker.touch()
    os.kill(os.getppid(), signal.SIGINT)
    time.sleep(10)
""",
        encoding="utf-8",
    )
    database = tmp_path / "runs.duckdb"
    config = tmp_path / "dabstep.yaml"
    config.write_text(
        f"""
name: dabstep-interruption
database: {database}
endpoint:
  name: local
  protocol: openai_chat
  base_url: http://127.0.0.1:8080/v1
  model: local-model
dabstep:
  workspace: {workspace}
  command: {json.dumps([sys.executable, str(fake_baseline)])}
  tasks_file: {tasks_path}
  tasks_file_arg: --tasks-file
  split: dev
  limit: all
""",
        encoding="utf-8",
    )

    interrupted = CliRunner().invoke(main, ["dabstep", str(config)])

    assert interrupted.exit_code != 0
    with ResultStore(database) as store:
        failed_run = store.comparison_runs(include_failed=True)[0]
        first_samples = store.samples_for_run(failed_run["run_id"])
    assert failed_run["status"] == "failed"
    assert failed_run["run_id"] in interrupted.output
    assert failed_run["metrics"]["answer_count"] == 1.0
    assert failed_run["metrics"]["success_rate"] == 0.5
    assert failed_run["metrics"]["interruption_count"] == 1.0
    assert {sample["sample_id"]: sample["status"] for sample in first_samples} == {
        "49": "failed",
        "5": "ok",
    }

    resumed = CliRunner().invoke(
        main,
        ["dabstep", str(config), "--resume", failed_run["run_id"]],
    )

    assert resumed.exit_code == 0, resumed.output
    with ResultStore(database) as store:
        run = store.comparison_runs()[0]
    assert run["run_id"] == failed_run["run_id"]
    assert run["metrics"]["answer_count"] == 2.0
    calls = [json.loads(line) for line in (workspace / "calls.jsonl").read_text().splitlines()]
    assert calls == [[5, 49], [49]]


def test_dabstep_retries_only_tasks_without_answers(tmp_path: Path) -> None:
    tasks_path = tmp_path / "dev.jsonl"
    _write_tasks(tasks_path)
    workspace = tmp_path / "upstream"
    workspace.mkdir()
    fake_baseline = workspace / "retrying_baseline.py"
    fake_baseline.write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", required=True)
parser.add_argument("--api-base", required=True)
parser.add_argument("--split", required=True)
parser.add_argument("--concurrency", required=True)
parser.add_argument("--max-steps", required=True)
parser.add_argument("--timestamp", required=True)
parser.add_argument("--tasks-ids", nargs="+", type=int, required=True)
parser.add_argument("--tasks-file", required=True)
args = parser.parse_args()
with Path("calls.jsonl").open("a") as calls:
    calls.write(json.dumps(args.tasks_ids) + "\\n")
model_dir = args.model_id.replace("/", "_").replace(".", "_")
output = Path("runs") / model_dir / args.split / args.timestamp
output.mkdir(parents=True)
marker = Path("retry-marker")
selected = args.tasks_ids if marker.exists() else args.tasks_ids[:1]
answers = {
    5: {"agent_answer": "NL", "answer": "NL", "score": 1, "level": "easy"},
    49: {"agent_answer": "A. NL", "answer": "B. BE", "score": 0, "level": "easy"},
}
with (output / "answers.jsonl").open("w") as handle:
    for task_id in selected:
        handle.write(json.dumps({"task_id": str(task_id), **answers[task_id]}) + "\\n")
if not marker.exists():
    marker.touch()
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    database = tmp_path / "runs.duckdb"
    config = tmp_path / "dabstep.yaml"
    config.write_text(
        f"""
name: dabstep-retry
database: {database}
endpoint:
  name: local
  protocol: openai_chat
  base_url: http://127.0.0.1:8080/v1
  model: local-model
dabstep:
  workspace: {workspace}
  command: {json.dumps([sys.executable, str(fake_baseline)])}
  tasks_file: {tasks_path}
  tasks_file_arg: --tasks-file
  split: dev
  limit: all
  retries: 1
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["dabstep", str(config)])

    assert result.exit_code == 0, result.output
    with ResultStore(database) as store:
        run = store.comparison_runs()[0]
        samples = {sample["sample_id"]: sample for sample in store.samples_for_run(run["run_id"])}
    assert run["status"] == "ok"
    assert run["metrics"]["answer_count"] == 2.0
    assert run["metrics"]["process_attempt_count"] == 2.0
    assert run["metrics"]["process_retry_count"] == 1.0
    assert samples["5"]["metrics"]["retry_count"] == 0.0
    assert samples["49"]["metrics"]["retry_count"] == 1.0
    calls = [json.loads(line) for line in (workspace / "calls.jsonl").read_text().splitlines()]
    assert calls == [[5, 49], [49]]
