from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class DabstepOutputError(RuntimeError):
    pass


def read_dabstep_answers(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    answers: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DabstepOutputError(
                f"invalid DABStep answer JSON on line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise DabstepOutputError(f"DABStep answer on line {line_number} must be an object")
        if "task_id" not in value or "agent_answer" not in value:
            raise DabstepOutputError(
                f"DABStep answer on line {line_number} requires task_id and agent_answer"
            )
        try:
            task_id = int(value["task_id"])
        except (TypeError, ValueError) as exc:
            raise DabstepOutputError(
                f"DABStep answer on line {line_number} has an invalid task_id"
            ) from exc
        normalized = dict(value)
        normalized["task_id"] = str(task_id)
        normalized["agent_answer"] = str(value["agent_answer"])
        if value.get("score") is not None:
            try:
                score = float(value["score"])
            except (TypeError, ValueError) as exc:
                raise DabstepOutputError(
                    f"DABStep answer on line {line_number} has an invalid score"
                ) from exc
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise DabstepOutputError(
                    f"DABStep answer on line {line_number} has a score outside [0, 1]"
                )
            normalized["score"] = score
        answers[task_id] = normalized
    return answers


def write_dabstep_answers(
    answers: dict[int, dict[str, Any]],
    path: Path,
    *,
    task_order: list[int],
) -> None:
    ordered_ids = [task_id for task_id in task_order if task_id in answers]
    _atomic_write_text(
        path,
        "".join(json.dumps(answers[task_id], sort_keys=True) + "\n" for task_id in ordered_ids),
    )


def parse_dabstep_metrics(
    answers_path: Path,
    tasks_path: Path,
    measurement_path: Path | None = None,
) -> dict[str, float]:
    answers = read_dabstep_answers(answers_path)
    tasks = _read_jsonl(tasks_path, description="task")
    task_ids = {int(task["task_id"]) for task in tasks}
    selected_answers = [answer for task_id, answer in answers.items() if task_id in task_ids]
    scored = [answer for answer in selected_answers if answer.get("score") is not None]
    task_count = len(tasks)
    answer_count = len(selected_answers)
    metrics: dict[str, float] = {
        "task_count": float(task_count),
        "answer_count": float(answer_count),
        "missing_count": float(task_count - answer_count),
        "error_count": float(task_count - answer_count),
        "completion_rate": answer_count / task_count if task_count else 0.0,
        "scored_count": float(len(scored)),
    }
    if scored:
        score_sum = sum(float(answer["score"]) for answer in scored)
        correct_count = sum(float(answer["score"]) >= 1.0 for answer in scored)
        metrics.update(
            {
                "score_total": score_sum,
                "average_score": score_sum / task_count,
                "correct_count": float(correct_count),
                "success_rate": score_sum / task_count,
                "accuracy": correct_count / task_count,
            }
        )
        task_levels = {int(task["task_id"]): str(task.get("level") or "unknown") for task in tasks}
        for level in sorted(set(task_levels.values())):
            level_task_ids = {
                task_id for task_id, task_level in task_levels.items() if task_level == level
            }
            level_answers = [
                answer
                for task_id, answer in answers.items()
                if task_id in level_task_ids and answer.get("score") is not None
            ]
            level_score = sum(float(answer["score"]) for answer in level_answers)
            metrics[f"{level}.task_count"] = float(len(level_task_ids))
            metrics[f"{level}.scored_count"] = float(len(level_answers))
            metrics[f"{level}.success_rate"] = level_score / len(level_task_ids)

    if measurement_path is not None and measurement_path.exists():
        measurement = json.loads(measurement_path.read_text(encoding="utf-8"))
        metrics["wall_duration_s"] = float(measurement.get("wall_duration_s") or 0.0)
        attempts = measurement.get("attempts") or []
        metrics["process_attempt_count"] = float(len(attempts))
        metrics["process_error_count"] = float(
            sum(1 for attempt in attempts if int(attempt.get("returncode", 0)) != 0)
        )
        metrics["timeout_count"] = float(
            sum(1 for attempt in attempts if bool(attempt.get("timed_out")))
        )
        metrics["interruption_count"] = float(
            sum(1 for attempt in attempts if bool(attempt.get("interrupted")))
        )
        metrics["process_retry_count"] = float(
            sum(1 for attempt in attempts if bool(attempt.get("retry")))
        )
    return metrics


def reparse_dabstep_run(run: dict[str, Any]) -> dict[str, float]:
    artifacts = run.get("artifacts") or {}
    answers = artifacts.get("answers")
    tasks = artifacts.get("tasks")
    if not answers or not tasks:
        raise FileNotFoundError("DABStep run requires answers and tasks artifacts")
    measurement = artifacts.get("measurement")
    return parse_dabstep_metrics(
        Path(answers["path"]),
        Path(tasks["path"]),
        Path(measurement["path"]) if measurement else None,
    )


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _read_jsonl(path: Path, *, description: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DabstepOutputError(
                f"invalid DABStep {description} JSON on line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise DabstepOutputError(
                f"DABStep {description} on line {line_number} must be an object"
            )
        values.append(value)
    return values
