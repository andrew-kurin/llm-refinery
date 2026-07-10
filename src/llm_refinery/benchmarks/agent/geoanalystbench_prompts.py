from __future__ import annotations

from typing import Protocol

from llm_refinery.benchmarks.agent.base import AgentEvalRequest, AgentEvalRequestConfig
from llm_refinery.benchmarks.agent.geoanalystbench_data import GeoAnalystTask
from llm_refinery.core.config import ConfigError

PROMPT_VARIANTS = {"original", "domain", "dataset", "domain_and_dataset"}
RESPONSE_TYPES = {"workflow", "code"}


class GeoAnalystPromptSpec(Protocol):
    @property
    def prompt_variants(self) -> tuple[str, ...]: ...

    @property
    def response_types(self) -> tuple[str, ...]: ...


def expand_geoanalyst_requests(
    tasks: list[GeoAnalystTask],
    benchmark: GeoAnalystPromptSpec,
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
