from __future__ import annotations

import ast
import csv
import io
import re
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from llm_refinery.benchmarks.agent.base import (
    UNSET,
    AgentEvalRequest,
    AgentEvalRequestConfig,
    AgentEvalResult,
    AgentTask,
    LimitOverride,
)
from llm_refinery.config import ConfigError, coerce_list
from llm_refinery.core.metrics import add_distribution_metrics

DEFAULT_GEOANALYSTBENCH_DATASET = (
    "https://raw.githubusercontent.com/GeoDS/GeoAnalystBench/"
    "master/dataset/GeoAnalystBench.csv"
)
PROMPT_VARIANTS = {"original", "domain", "dataset", "domain_and_dataset"}
RESPONSE_TYPES = {"workflow", "code"}


@dataclass(frozen=True)
class GeoAnalystBenchSpec:
    kind: str = "geoanalystbench"
    dataset: str = DEFAULT_GEOANALYSTBENCH_DATASET
    task_ids: tuple[int, ...] = ()
    limit: int | None = 5
    open_source_only: bool = True
    prompt_variants: tuple[str, ...] = ("domain_and_dataset",)
    response_types: tuple[str, ...] = ("workflow", "code")

    @classmethod
    def from_mapping(
        cls, raw: dict[str, Any], *, source_path: Path | None = None
    ) -> GeoAnalystBenchSpec:
        dataset = str(
            raw.get("dataset") or raw.get("dataset_url") or DEFAULT_GEOANALYSTBENCH_DATASET
        )
        if source_path and not _is_url(dataset):
            dataset_path = Path(dataset)
            if not dataset_path.is_absolute():
                dataset = str(source_path.parent / dataset_path)

        task_ids = tuple(int(value) for value in coerce_list(raw.get("task_ids") or []))
        limit_raw = raw.get("limit", 5)
        limit = None if limit_raw is None or str(limit_raw).lower() == "all" else int(limit_raw)
        if limit is not None and limit <= 0:
            raise ConfigError("benchmark.limit must be a positive integer or 'all'")

        prompt_variants = tuple(
            str(v) for v in coerce_list(raw.get("prompt_variants") or ["domain_and_dataset"])
        )
        unknown_variants = sorted(set(prompt_variants) - PROMPT_VARIANTS)
        if unknown_variants:
            raise ConfigError(
                f"unknown GeoAnalystBench prompt variant(s): {', '.join(unknown_variants)}"
            )

        response_types = tuple(
            str(v) for v in coerce_list(raw.get("response_types") or ["workflow", "code"])
        )
        unknown_response_types = sorted(set(response_types) - RESPONSE_TYPES)
        if unknown_response_types:
            raise ConfigError(
                f"unknown GeoAnalystBench response type(s): {', '.join(unknown_response_types)}"
            )

        return cls(
            dataset=dataset,
            task_ids=task_ids,
            limit=limit,
            open_source_only=bool(raw.get("open_source_only", True)),
            prompt_variants=prompt_variants,
            response_types=response_types,
        )

    def with_overrides(
        self, *, limit: LimitOverride = UNSET, task_ids: tuple[int, ...] | None = None
    ) -> GeoAnalystBenchSpec:
        updates: dict[str, Any] = {}
        if limit is not UNSET:
            updates["limit"] = limit
        if task_ids is not None:
            updates["task_ids"] = task_ids
        return replace(self, **updates)

    def safe_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dataset": self.dataset,
            "task_ids": list(self.task_ids),
            "limit": self.limit,
            "open_source_only": self.open_source_only,
            "prompt_variants": list(self.prompt_variants),
            "response_types": list(self.response_types),
        }

    def load_tasks(self) -> list[AgentTask]:
        return list(select_geoanalyst_tasks(load_geoanalyst_tasks(self.dataset), self))

    def expand_requests(
        self, tasks: list[AgentTask], request_config: AgentEvalRequestConfig
    ) -> list[AgentEvalRequest]:
        geo_tasks = [task for task in tasks if isinstance(task, GeoAnalystTask)]
        if len(geo_tasks) != len(tasks):
            raise ConfigError("GeoAnalystBench received non-GeoAnalyst task objects")
        return expand_geoanalyst_requests(geo_tasks, self, request_config)

    def score_result(self, result: AgentEvalResult) -> AgentEvalResult:
        return score_geoanalyst_result(result)

    def summarize_results(
        self, results: list[AgentEvalResult], wall_duration_s: float
    ) -> dict[str, float]:
        return summarize_geoanalyst_results(results, wall_duration_s)


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
        }


def load_geoanalyst_tasks(dataset: str) -> list[GeoAnalystTask]:
    if _is_url(dataset):
        with urllib.request.urlopen(dataset, timeout=60) as response:  # noqa: S310 user config
            text = response.read().decode("utf-8-sig", errors="replace")
    else:
        text = Path(dataset).read_text(encoding="utf-8-sig")

    rows = csv.DictReader(io.StringIO(text))
    return [GeoAnalystTask.from_row(row) for row in rows]


def select_geoanalyst_tasks(
    tasks: list[GeoAnalystTask], benchmark: GeoAnalystBenchSpec
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


def expand_geoanalyst_requests(
    tasks: list[GeoAnalystTask],
    benchmark: GeoAnalystBenchSpec,
    request_config: AgentEvalRequestConfig,
) -> list[AgentEvalRequest]:
    requests: list[AgentEvalRequest] = []
    for task in tasks:
        for prompt_variant in benchmark.prompt_variants:
            for response_type in benchmark.response_types:
                system, prompt = build_geoanalyst_prompt(task, prompt_variant, response_type)
                requests.append(
                    AgentEvalRequest(
                        task=task,
                        prompt_variant=prompt_variant,
                        response_type=response_type,
                        system=system,
                        prompt=prompt,
                        config=request_config,
                    )
                )
    return requests


def build_geoanalyst_prompt(
    task: GeoAnalystTask, prompt_variant: str, response_type: str
) -> tuple[str, str]:
    system = (
        "You are a careful geospatial Python analyst. Prefer open-source Python GIS "
        "libraries such as GeoPandas, Shapely, Rasterio, PyProj, Xarray, NumPy, and "
        "Matplotlib unless the task explicitly requires ArcPy."
    )
    sections = [
        f"Task ID: {task.task_id}",
        f"Task: {task.task}",
        "Instruction:",
        task.instruction,
    ]
    if prompt_variant in {"domain", "domain_and_dataset"} and task.domain_knowledge.strip():
        sections.extend(["Domain knowledge:", task.domain_knowledge])
    if prompt_variant in {"dataset", "domain_and_dataset"} and task.dataset_description.strip():
        sections.extend(["Dataset description:", task.dataset_description])

    if response_type == "workflow":
        sections.extend(
            [
                "Output requirements:",
                "Return only a numbered spatial-analysis workflow. Keep each step concise. "
                "Do not include code.",
            ]
        )
    elif response_type == "code":
        sections.extend(
            [
                "Output requirements:",
                "Return only Python code. Put all logic in a function named model(). "
                "Do not include Markdown fences or explanatory prose.",
            ]
        )
    else:
        raise ConfigError(f"unsupported response type: {response_type}")

    return system, "\n\n".join(sections)


def score_geoanalyst_result(result: AgentEvalResult) -> AgentEvalResult:
    if not result.ok:
        return result
    if result.request.response_type == "workflow":
        step_count = extract_workflow_step_count(result.response_text)
        task_length = int(getattr(result.request.task, "task_length", 0))
        return replace(
            result,
            workflow_step_count=step_count,
            workflow_step_abs_error=abs(step_count - task_length),
        )
    if result.request.response_type == "code":
        code = extract_python_code(result.response_text)
        try:
            ast.parse(code)
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False
        return replace(result, code_syntax_ok=syntax_ok)
    return result


def summarize_geoanalyst_results(
    results: list[AgentEvalResult], wall_duration_s: float
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "request_count": float(len(results)),
        "success_count": float(sum(1 for result in results if result.ok)),
        "error_count": float(sum(1 for result in results if not result.ok)),
        "wall_duration_s": float(wall_duration_s),
    }
    if results:
        metrics["success_rate"] = metrics["success_count"] / len(results)
    successes = [result for result in results if result.ok]
    add_distribution_metrics(metrics, "latency", [result.latency_s for result in successes])
    add_distribution_metrics(
        metrics,
        "response_chars",
        [float(len(result.response_text)) for result in successes],
        unit_suffix="",
    )

    known_completion_tokens = [
        result.completion_tokens
        for result in successes
        if result.completion_tokens is not None
    ]
    if known_completion_tokens:
        metrics["completion_tokens_total"] = float(sum(known_completion_tokens))
        metrics["completion_tokens_avg"] = sum(known_completion_tokens) / len(
            known_completion_tokens
        )
        metrics["completion_tokens_per_second"] = (
            sum(known_completion_tokens) / wall_duration_s if wall_duration_s else 0.0
        )

    workflow_results = [
        result
        for result in successes
        if result.request.response_type == "workflow" and result.workflow_step_count is not None
    ]
    if workflow_results:
        metrics["workflow_count"] = float(len(workflow_results))
        metrics["workflow_step_count_avg"] = sum(
            result.workflow_step_count or 0 for result in workflow_results
        ) / len(workflow_results)
        metrics["workflow_step_abs_error_avg"] = sum(
            result.workflow_step_abs_error or 0 for result in workflow_results
        ) / len(workflow_results)

    code_results = [
        result
        for result in successes
        if result.request.response_type == "code" and result.code_syntax_ok is not None
    ]
    if code_results:
        syntax_pass = sum(1 for result in code_results if result.code_syntax_ok)
        metrics["code_count"] = float(len(code_results))
        metrics["code_syntax_pass_count"] = float(syntax_pass)
        metrics["code_syntax_pass_rate"] = syntax_pass / len(code_results)

    return metrics


def extract_workflow_step_count(text: str) -> int:
    numbers: list[int] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d{1,2})[\.)]\s+", line)
        if match:
            numbers.append(int(match.group(1)))
    if numbers:
        return max(numbers)
    bullet_lines = [line for line in text.splitlines() if re.match(r"^\s*[-*]\s+", line)]
    return len(bullet_lines)


def extract_python_code(text: str) -> str:
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")
