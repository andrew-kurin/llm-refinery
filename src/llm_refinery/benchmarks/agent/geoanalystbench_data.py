from __future__ import annotations

import csv
import io
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from llm_refinery.core.config import ConfigError
from llm_refinery.core.runs import stable_hash

DEFAULT_GEOANALYSTBENCH_DATASET = (
    "https://raw.githubusercontent.com/GeoDS/GeoAnalystBench/master/dataset/GeoAnalystBench.csv"
)


@dataclass(frozen=True)
class GeoAnalystTask:
    task_id: int
    open_source: bool
    task: str
    instruction: str
    domain_knowledge: str
    dataset_description: str
    human_workflow: str
    task_length: int
    code: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> GeoAnalystTask:
        return cls(
            task_id=int(row["id"]),
            open_source=str(row.get("Open Source") or "").strip().upper() == "T",
            task=str(row.get("Task") or ""),
            instruction=str(row.get("Instruction") or ""),
            domain_knowledge=str(row.get("Domain Knowledge") or ""),
            dataset_description=str(row.get("Dataset Description") or ""),
            human_workflow=str(row.get("Human Designed Workflow") or ""),
            task_length=int(row.get("Task Length") or 0),
            code=str(row.get("CodeString") or ""),
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "open_source": self.open_source,
            "task": self.task,
            "task_length": self.task_length,
            "reference_workflow_hash": stable_hash(self.human_workflow),
            "reference_code_hash": stable_hash(self.code),
        }


def load_geoanalyst_tasks(dataset: str) -> list[GeoAnalystTask]:
    if is_url(dataset):
        with urllib.request.urlopen(dataset, timeout=60) as response:  # noqa: S310 user config
            text = response.read().decode("utf-8-sig", errors="replace")
    else:
        text = Path(dataset).read_text(encoding="utf-8-sig")

    rows = csv.DictReader(io.StringIO(text))
    return [GeoAnalystTask.from_row(row) for row in rows]


class GeoAnalystSelection(Protocol):
    @property
    def open_source_only(self) -> bool: ...

    @property
    def task_ids(self) -> tuple[int, ...]: ...

    @property
    def limit(self) -> int | None: ...


def select_geoanalyst_tasks(
    tasks: list[GeoAnalystTask], benchmark: GeoAnalystSelection
) -> list[GeoAnalystTask]:
    selected = tasks
    if benchmark.open_source_only:
        selected = [task for task in selected if task.open_source]
    if benchmark.task_ids:
        wanted = set(benchmark.task_ids)
        selected = [task for task in selected if task.task_id in wanted]
        missing = wanted - {task.task_id for task in selected}
        if missing:
            missing_text = ", ".join(map(str, sorted(missing)))
            raise ConfigError(f"GeoAnalystBench task id(s) not found: {missing_text}")
    if benchmark.limit is not None:
        selected = selected[: benchmark.limit]
    if not selected:
        raise ConfigError("GeoAnalystBench task selection is empty")
    return selected


def is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")
