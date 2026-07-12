from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from llm_refinery.benchmarks.dabstep.config import DabstepSettings
from llm_refinery.core.config import ConfigError


@dataclass(frozen=True)
class DabstepTask:
    task_id: int
    question: str
    guidelines: str
    level: str
    answer: str | None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> DabstepTask:
        try:
            task_id = int(raw["task_id"])
            question = str(raw["question"])
            guidelines = str(raw["guidelines"])
            level = str(raw["level"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"invalid DABStep task: {exc}") from exc
        answer_raw = raw.get("answer")
        return cls(
            task_id=task_id,
            question=question,
            guidelines=guidelines,
            level=level,
            answer=str(answer_raw) if answer_raw is not None else None,
        )

    def as_jsonable(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "task_id": str(self.task_id),
            "question": self.question,
            "guidelines": self.guidelines,
            "level": self.level,
        }
        if self.answer is not None:
            value["answer"] = self.answer
        return value


@dataclass(frozen=True)
class DabstepTaskSourceContract:
    mode: str
    selected_manifest_sha256: str
    official_main_manifest_sha256: str | None = None

    def as_jsonable(self) -> dict[str, str | None]:
        return {
            "mode": self.mode,
            "selected_manifest_sha256": self.selected_manifest_sha256,
            "official_main_manifest_sha256": self.official_main_manifest_sha256,
        }


def load_dabstep_tasks(settings: DabstepSettings) -> list[DabstepTask]:
    if settings.tasks_file is not None:
        try:
            text = settings.tasks_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"could not read DABStep tasks file: {settings.tasks_file}") from exc
    else:
        text = _download_task_text(
            settings.dataset_repo,
            settings.dataset_revision,
            settings.split,
        )
    return _parse_and_select_tasks(text, settings)


def validate_dabstep_task_source(
    settings: DabstepSettings,
    tasks: list[DabstepTask],
) -> DabstepTaskSourceContract:
    selected_hash = task_manifest_sha256(tasks)
    if settings.tasks_file is not None:
        return DabstepTaskSourceContract(
            mode="wrapper_manifest",
            selected_manifest_sha256=selected_hash,
        )

    official_main_text = _download_task_text(settings.dataset_repo, "main", settings.split)
    official_main_tasks = _parse_and_select_tasks(official_main_text, settings)
    official_main_hash = task_manifest_sha256(official_main_tasks)
    if official_main_hash != selected_hash:
        raise ConfigError(
            "the pinned DABStep task manifest does not match the official baseline's current "
            "main manifest; refusing to label the run as the selected pinned revision"
        )
    return DabstepTaskSourceContract(
        mode="official_verified",
        selected_manifest_sha256=selected_hash,
        official_main_manifest_sha256=official_main_hash,
    )


def task_manifest_sha256(tasks: list[DabstepTask]) -> str:
    payload = json.dumps(
        [task.as_jsonable() for task in tasks],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_and_select_tasks(text: str, settings: DabstepSettings) -> list[DabstepTask]:
    tasks: list[DabstepTask] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"invalid DABStep task JSON on line {line_number}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"DABStep task on line {line_number} must be an object")
        tasks.append(DabstepTask.from_mapping(raw))

    ids = [task.task_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ConfigError("DABStep task source contains duplicate task IDs")
    if settings.task_ids:
        wanted = set(settings.task_ids)
        tasks = [task for task in tasks if task.task_id in wanted]
        missing = wanted - {task.task_id for task in tasks}
        if missing:
            values = ", ".join(str(task_id) for task_id in sorted(missing))
            raise ConfigError(f"unknown DABStep task ID(s): {values}")
        order = {task_id: index for index, task_id in enumerate(settings.task_ids)}
        tasks.sort(key=lambda task: order[task.task_id])
    if settings.limit is not None:
        tasks = tasks[: settings.limit]
    if not tasks:
        raise ConfigError("DABStep task selection is empty")
    return tasks


def write_task_manifest(tasks: list[DabstepTask], path: Path) -> None:
    content = "".join(json.dumps(task.as_jsonable(), sort_keys=True) + "\n" for task in tasks)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _dataset_url_for(dataset_repo: str, dataset_revision: str, split: str) -> str:
    filename = "dev.jsonl" if split == "dev" else "all.jsonl"
    return (
        f"https://huggingface.co/datasets/{dataset_repo}/resolve/"
        f"{dataset_revision}/data/tasks/{filename}"
    )


def _download_task_text(dataset_repo: str, dataset_revision: str, split: str) -> str:
    source_url = _dataset_url_for(dataset_repo, dataset_revision, split)
    request = Request(source_url, headers={"User-Agent": "llm-refinery/0.1"})
    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310 - configured HF source
            return response.read().decode("utf-8")
    except OSError as exc:
        raise ConfigError(f"could not download DABStep tasks: {source_url}") from exc
