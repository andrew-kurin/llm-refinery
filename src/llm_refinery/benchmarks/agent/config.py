from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_refinery.benchmarks.agent.base import AgentBenchmarkSpec, AgentEvalRequestConfig
from llm_refinery.benchmarks.agent.registry import load_agent_benchmark_spec
from llm_refinery.core.config import ConfigError, load_yaml_mapping, reject_unknown_keys
from llm_refinery.core.endpoints import OPENAI_CHAT, Endpoint


@dataclass(frozen=True)
class AgentEvalConfig:
    name: str
    database: Path
    benchmark: AgentBenchmarkSpec
    targets: list[Endpoint]
    request: AgentEvalRequestConfig = field(default_factory=AgentEvalRequestConfig)
    source_path: Path | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], source_path: Path | None = None) -> AgentEvalConfig:
        reject_unknown_keys(
            raw,
            {"name", "database", "benchmark", "targets", "request"},
            context="agent-eval configuration",
        )
        name = str(raw.get("name") or (source_path.stem if source_path else "agent-eval"))
        targets_raw = raw.get("targets") or []
        if not targets_raw:
            raise ConfigError("agent-eval config requires at least one target in 'targets'")

        benchmark_value = raw.get("benchmark") or {}
        if not isinstance(benchmark_value, dict):
            raise ConfigError("agent-eval benchmark must be a mapping")
        benchmark = load_agent_benchmark_spec(benchmark_value, source_path=source_path)
        if any(not isinstance(item, dict) for item in targets_raw):
            raise ConfigError("each agent-eval target must be a mapping")
        targets = [
            Endpoint.from_mapping(
                item,
                context="agent-eval target",
                allowed_protocols=frozenset({OPENAI_CHAT}),
            )
            for item in targets_raw
        ]
        if len({target.name for target in targets}) != len(targets):
            raise ConfigError("agent-eval target names must be unique")

        return cls(
            name=name,
            database=Path(str(raw.get("database") or "results/llm_refinery.duckdb")),
            benchmark=benchmark,
            targets=targets,
            request=AgentEvalRequestConfig.from_mapping(raw.get("request")),
            source_path=source_path,
        )


def load_agent_eval_config(path: str | Path) -> AgentEvalConfig:
    config_path, raw = load_yaml_mapping(path)
    return AgentEvalConfig.from_mapping(raw, source_path=config_path)
