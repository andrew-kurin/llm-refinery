from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from llm_refinery.application.run_context import RunContext
from llm_refinery.application.run_session import RunSession
from llm_refinery.benchmarks.dabstep.command import (
    build_dabstep_command,
    effective_model_id,
    upstream_output_dir,
)
from llm_refinery.benchmarks.dabstep.config import DabstepConfig
from llm_refinery.benchmarks.dabstep.parser import (
    DabstepOutputError,
    parse_dabstep_metrics,
    read_dabstep_answers,
    write_dabstep_answers,
)
from llm_refinery.benchmarks.dabstep.tasks import (
    DabstepTask,
    DabstepTaskSourceContract,
    load_dabstep_tasks,
    validate_dabstep_task_source,
    write_task_manifest,
)
from llm_refinery.core.config import ConfigError
from llm_refinery.core.runs import CompletedRun, RunSpec, stable_hash
from llm_refinery.storage.duckdb import ResultStore
from llm_refinery.storage.models import SampleRecord

_SAFE_SUBPROCESS_ENV = frozenset(
    {
        "HOME",
        "PATH",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "XDG_CACHE_HOME",
        "NO_PROXY",
        "no_proxy",
        "OTEL_SDK_DISABLED",
        "SYSTEMROOT",
        "WINDIR",
    }
)


class DabstepFailed(RuntimeError):
    pass


def run_dabstep(
    config: DabstepConfig,
    *,
    dry_run: bool = False,
    resume_run_id: str | None = None,
    allow_unverified_executor: bool = False,
    store: ResultStore | None = None,
    parent_run_id: str | None = None,
    run_context: RunContext | None = None,
) -> CompletedRun | None:
    tasks = load_dabstep_tasks(config.dabstep)
    task_source_contract = validate_dabstep_task_source(config.dabstep, tasks)
    task_ids = [task.task_id for task in tasks]
    planned_command = build_dabstep_command(config, task_ids, run_token="<run-token>")
    command_text = shlex.join(planned_command)
    print(
        f"DABStep target={config.endpoint.name} model={effective_model_id(config)} "
        f"split={config.dabstep.split} tasks={len(tasks)}"
    )
    print(f"workspace={config.dabstep.workspace}")
    print(command_text)
    if dry_run:
        return None
    if not config.dabstep.workspace.is_dir():
        raise ConfigError(f"DABStep workspace does not exist: {config.dabstep.workspace}")
    if store is not None and store.database != config.database.resolve():
        raise ValueError(
            f"DABStep database {config.database.resolve()} does not match shared store"
        )

    store_context = nullcontext(store) if store is not None else ResultStore(config.database)
    with store_context as active_store:
        assert active_store is not None
        spec = _run_spec(
            config,
            command_text=command_text,
            tasks=tasks,
            task_source_contract=task_source_contract,
            database=active_store.database,
            parent_run_id=parent_run_id,
            run_context=run_context,
        )
        with RunSession(
            active_store,
            spec,
            resume_run_id=resume_run_id,
            run_context=run_context,
            allow_unverified_executor=allow_unverified_executor,
        ) as run:
            answers_path = run.artifact("answers", "answers.jsonl", "application/x-ndjson")
            tasks_path = run.artifact("tasks", "tasks.jsonl", "application/x-ndjson")
            stdout_path = run.artifact("stdout", "stdout.txt", "text/plain")
            stderr_path = run.artifact("stderr", "stderr.txt", "text/plain")
            measurement_path = run.artifact("measurement", "measurement.json", "application/json")
            upstream_logs_path = run.artifact("upstream_logs", "upstream-logs.txt", "text/plain")
            upstream_configs_path = run.artifact(
                "upstream_configs", "upstream-configs.yaml", "application/yaml"
            )
            write_task_manifest(tasks, tasks_path)
            answers = read_dabstep_answers(answers_path)
            completed_samples = {
                int(sample["sample_id"])
                for sample in active_store.samples_for_run(run.run_id)
                if sample["status"] == "ok"
            }
            completed_ids = completed_samples & set(answers)
            remaining_ids = [task_id for task_id in task_ids if task_id not in completed_ids]
            measurement = _read_measurement(measurement_path)
            returncode = 0
            process_ran = False
            interrupted: KeyboardInterrupt | None = None

            invocation_attempt = 0
            while remaining_ids and invocation_attempt <= config.dabstep.retries:
                artifact_attempt = len(measurement["attempts"]) + 1
                run_token = _run_token(run.run_id, artifact_attempt)
                source_output = upstream_output_dir(config, run_token=run_token)
                source_answers = source_output / "answers.jsonl"
                if source_answers.exists():
                    _merge_source_answers(
                        answers,
                        source_answers,
                        selected_task_ids=set(task_ids),
                    )
                    write_dabstep_answers(answers, answers_path, task_order=task_ids)
                    remaining_ids = [task_id for task_id in task_ids if task_id not in answers]
                    if not remaining_ids:
                        _append_upstream_artifact(
                            upstream_logs_path,
                            source_output / "logs.txt",
                            attempt=artifact_attempt,
                        )
                        _append_upstream_artifact(
                            upstream_configs_path,
                            source_output / "config.yaml",
                            attempt=artifact_attempt,
                        )
                        break
                attempted_ids = list(remaining_ids)
                command = build_dabstep_command(
                    config,
                    attempted_ids,
                    run_token=run_token,
                    tasks_manifest_path=tasks_path,
                )
                env = _subprocess_environment(config)
                process_ran = True
                started = time.perf_counter()
                timed_out = False
                try:
                    completed = subprocess.run(
                        command,
                        cwd=config.dabstep.workspace,
                        env=env,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=config.dabstep.timeout_s,
                    )
                    returncode = completed.returncode
                    stdout = completed.stdout or ""
                    stderr = completed.stderr or ""
                except subprocess.TimeoutExpired as exc:
                    timed_out = True
                    returncode = 124
                    stdout = _output_text(exc.stdout)
                    stderr = _output_text(exc.stderr)
                    stderr += f"\nDABStep process timed out after {config.dabstep.timeout_s}s\n"
                except KeyboardInterrupt as exc:
                    interrupted = exc
                    returncode = 130
                    stdout = ""
                    stderr = "DABStep process interrupted by user\n"
                except OSError as exc:
                    returncode = 127
                    stdout = ""
                    stderr = f"{type(exc).__name__}: {exc}\n"
                process_duration_s = time.perf_counter() - started
                _append_process_output(stdout_path, stdout, attempt=artifact_attempt)
                _append_process_output(stderr_path, stderr, attempt=artifact_attempt)

                if source_answers.exists():
                    _merge_source_answers(
                        answers,
                        source_answers,
                        selected_task_ids=set(task_ids),
                    )
                    write_dabstep_answers(answers, answers_path, task_order=task_ids)
                _append_upstream_artifact(
                    upstream_logs_path,
                    source_output / "logs.txt",
                    attempt=artifact_attempt,
                )
                _append_upstream_artifact(
                    upstream_configs_path,
                    source_output / "config.yaml",
                    attempt=artifact_attempt,
                )
                measurement["wall_duration_s"] += process_duration_s
                measurement["attempts"].append(
                    {
                        "task_ids": attempted_ids,
                        "returncode": returncode,
                        "timed_out": timed_out,
                        "interrupted": interrupted is not None,
                        "retry": invocation_attempt > 0,
                        "duration_s": process_duration_s,
                    }
                )
                _write_measurement(measurement_path, measurement)
                _record_samples(
                    active_store,
                    run_id=run.run_id,
                    tasks=tasks,
                    answers=answers,
                    answers_path=answers_path,
                    process_returncode=returncode,
                    attempts=measurement["attempts"],
                )
                remaining_ids = [task_id for task_id in task_ids if task_id not in answers]
                invocation_attempt += 1
                if interrupted is not None:
                    break
            _write_measurement(measurement_path, measurement)
            answers = read_dabstep_answers(answers_path)
            _record_samples(
                active_store,
                run_id=run.run_id,
                tasks=tasks,
                answers=answers,
                answers_path=answers_path,
                process_returncode=returncode,
                attempts=measurement["attempts"],
            )
            metrics = parse_dabstep_metrics(answers_path, tasks_path, measurement_path)
            missing = int(metrics["missing_count"])
            success = missing == 0 and interrupted is None and (not process_ran or returncode == 0)
            if interrupted is not None:
                error = "DABStep run interrupted by user"
            elif missing:
                error = f"DABStep baseline produced no answer for {missing} task(s)"
                if returncode != 0:
                    error += f"; last exit code was {returncode}"
            elif returncode != 0:
                error = f"DABStep baseline exited with code {returncode}"
            else:
                error = None
            outcome = run.complete(
                status="ok" if success else "failed",
                metrics=metrics,
                error=error,
            )
            if interrupted is not None:
                print(
                    f"checkpointed interrupted DABStep run {outcome.run_id}; "
                    f"resume with --resume {outcome.run_id}",
                    flush=True,
                )
                raise interrupted

    print(
        f"stored {outcome.status}: {outcome.run_id} "
        f"(answers={int(metrics['answer_count'])}/{int(metrics['task_count'])})"
    )
    if outcome.status != "ok" and not config.dabstep.keep_going:
        raise DabstepFailed(f"DABStep run failed: {outcome.error}")
    return outcome


def _run_spec(
    config: DabstepConfig,
    *,
    command_text: str,
    tasks: list[DabstepTask],
    task_source_contract: DabstepTaskSourceContract,
    database: Path,
    parent_run_id: str | None,
    run_context: RunContext | None,
) -> RunSpec:
    selected_ids = [task.task_id for task in tasks]
    config_json = {
        "benchmark": "dabstep",
        "target": config.endpoint.safe_json(),
        "dabstep": config.dabstep.safe_json(),
        "selected_task_ids": selected_ids,
        "selected_tasks_hash": stable_hash([task.as_jsonable() for task in tasks]),
        "task_source_contract": task_source_contract.as_jsonable(),
        "upstream": {"workspace_git_head": _workspace_git_head(config.dabstep.workspace)},
        "model": {"name": config.endpoint.model},
        "params": {
            "target": config.endpoint.name,
            "model": config.endpoint.model,
            "model_id": effective_model_id(config),
            "split": config.dabstep.split,
            "task_count": len(tasks),
            "concurrency": config.dabstep.concurrency,
            "max_steps": config.dabstep.max_steps,
        },
    }
    if run_context is not None and run_context.target_json:
        config_json["execution_target"] = run_context.target_identity_json()
    return RunSpec.create(
        benchmark_kind="dabstep",
        suite=config.name,
        label=f"{config.name}/{config.endpoint.name}/{config.dabstep.split}",
        command=command_text,
        config_json=config_json,
        database=database,
        parent_run_id=parent_run_id,
    )


def _workspace_git_head(workspace: Path) -> str | None:
    if not (workspace / ".git").exists():
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _subprocess_environment(config: DabstepConfig) -> dict[str, str]:
    env = {name: os.environ[name] for name in _SAFE_SUBPROCESS_ENV if name in os.environ}
    env.setdefault("PATH", os.defpath)
    for name in config.dabstep.pass_env:
        if name not in os.environ:
            raise ConfigError(f"DABStep pass_env variable is not set: {name}")
        env[name] = os.environ[name]
    api_key_env = config.endpoint.api_key_env
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ConfigError(f"DABStep endpoint API key variable is not set: {api_key_env}")
        env[api_key_env] = api_key
        env["OPENAI_API_KEY"] = api_key
    else:
        env["OPENAI_API_KEY"] = "local"
    env.setdefault("OTEL_SDK_DISABLED", "true")
    return env


def _record_samples(
    store: ResultStore,
    *,
    run_id: str,
    tasks: list[DabstepTask],
    answers: dict[int, dict[str, Any]],
    answers_path: Path,
    process_returncode: int,
    attempts: list[dict[str, Any]],
) -> None:
    for task in tasks:
        answer = answers.get(task.task_id)
        payload: dict[str, Any]
        task_attempts = [
            attempt for attempt in attempts if task.task_id in attempt.get("task_ids", [])
        ]
        attempt_count = len(task_attempts)
        metrics: dict[str, float] = {
            "process_attempt_count": float(attempt_count),
            "retry_count": float(max(0, attempt_count - 1)),
            "timeout_count": float(sum(1 for attempt in task_attempts if attempt.get("timed_out"))),
            "interruption_count": float(
                sum(1 for attempt in task_attempts if attempt.get("interrupted"))
            ),
        }
        if answer is None:
            error = f"DABStep produced no answer; last process exit code was {process_returncode}"
            status = "failed"
            payload = {
                "task_id": str(task.task_id),
                "level": task.level,
                "question_hash": stable_hash(task.question),
            }
        else:
            error = None
            status = "ok"
            if answer.get("score") is not None:
                metrics["score"] = float(answer["score"])
            payload = {
                "task_id": str(task.task_id),
                "level": str(answer.get("level") or task.level),
                "question_hash": stable_hash(task.question),
                "agent_answer_chars": len(str(answer["agent_answer"])),
            }
        store.record_sample(
            SampleRecord(
                run_id=run_id,
                sample_id=str(task.task_id),
                status=status,
                payload_json=payload,
                metrics=metrics,
                artifact_path=str(answers_path),
                error=error,
            )
        )


def _merge_source_answers(
    answers: dict[int, dict[str, Any]],
    source_path: Path,
    *,
    selected_task_ids: set[int],
) -> None:
    source_answers = read_dabstep_answers(source_path)
    unexpected = set(source_answers) - selected_task_ids
    if unexpected:
        values = ", ".join(str(task_id) for task_id in sorted(unexpected))
        raise DabstepOutputError(f"DABStep output contains unselected task ID(s): {values}")
    answers.update(source_answers)


def _run_token(run_id: str, attempt: int) -> str:
    return str(int(stable_hash({"run_id": run_id, "attempt": attempt}), 16))


def _read_measurement(path: Path) -> dict[str, Any]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return {"wall_duration_s": 0.0, "attempts": []}
    value = json.loads(path.read_text(encoding="utf-8"))
    return {
        "wall_duration_s": float(value.get("wall_duration_s") or 0.0),
        "attempts": list(value.get("attempts") or []),
    }


def _write_measurement(path: Path, measurement: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(measurement, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_process_output(path: Path, value: str, *, attempt: int) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== DABStep process attempt {attempt} =====\n")
        handle.write(value)
        if value and not value.endswith("\n"):
            handle.write("\n")


def _append_upstream_artifact(destination: Path, source: Path, *, attempt: int) -> None:
    if not source.exists():
        return
    last_character = ""
    with (
        destination.open("a", encoding="utf-8") as output,
        source.open("r", encoding="utf-8", errors="replace") as upstream,
    ):
        output.write(f"\n===== DABStep process attempt {attempt} =====\n")
        while chunk := upstream.read(1024 * 1024):
            output.write(chunk)
            last_character = chunk[-1]
        if last_character and last_character != "\n":
            output.write("\n")


def _output_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
