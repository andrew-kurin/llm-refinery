from __future__ import annotations

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
from llm_refinery.benchmarks.agent.geoanalystbench_data import (
    DEFAULT_GEOANALYSTBENCH_DATASET,
    GeoAnalystTask,
    is_url,
    load_geoanalyst_tasks,
    select_geoanalyst_tasks,
)
from llm_refinery.benchmarks.agent.geoanalystbench_prompts import (
    PROMPT_VARIANTS,
    RESPONSE_TYPES,
    build_geoanalyst_prompt,
    expand_geoanalyst_requests,
)
from llm_refinery.benchmarks.agent.geoanalystbench_scoring import (
    extract_python_code,
    extract_workflow_step_count,
    score_geoanalyst_result,
    summarize_geoanalyst_results,
)
from llm_refinery.config import ConfigError, coerce_list


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
        if source_path and not is_url(dataset):
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


__all__ = [
    "DEFAULT_GEOANALYSTBENCH_DATASET",
    "PROMPT_VARIANTS",
    "RESPONSE_TYPES",
    "GeoAnalystBenchSpec",
    "GeoAnalystTask",
    "build_geoanalyst_prompt",
    "expand_geoanalyst_requests",
    "extract_python_code",
    "extract_workflow_step_count",
    "load_geoanalyst_tasks",
    "score_geoanalyst_result",
    "select_geoanalyst_tasks",
    "summarize_geoanalyst_results",
]
