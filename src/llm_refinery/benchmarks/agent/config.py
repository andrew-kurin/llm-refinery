from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llm_refinery.benchmarks.agent import load_agent_benchmark_spec
from llm_refinery.benchmarks.agent.base import AgentBenchmarkSpec, AgentEvalRequestConfig
from llm_refinery.config import ConfigError


@dataclass(frozen=True)
class AgentEvalTarget:
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> AgentEvalTarget:
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ConfigError("each agent-eval target requires a non-empty 'name'")

        provider = str(raw.get("provider") or "openai").strip().lower()
        if provider != "openai":
            raise ConfigError(f"target {name!r} provider must be 'openai', got {provider!r}")

        base_url = str(raw.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            raise ConfigError(f"target {name!r} requires 'base_url'")

        model = str(raw.get("model") or "").strip()
        if not model:
            raise ConfigError(f"target {name!r} requires 'model'")

        return cls(
            name=name,
            provider=provider,
            base_url=base_url,
            model=model,
            api_key_env=str(raw["api_key_env"]) if raw.get("api_key_env") else None,
            headers={str(k): str(v) for k, v in dict(raw.get("headers") or {}).items()},
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "header_names": sorted(self.headers),
        }


@dataclass(frozen=True)
class AgentEvalConfig:
    name: str
    database: Path
    benchmark: AgentBenchmarkSpec
    targets: list[AgentEvalTarget]
    request: AgentEvalRequestConfig = field(default_factory=AgentEvalRequestConfig)
    source_path: Path | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], source_path: Path | None = None) -> AgentEvalConfig:
        name = str(raw.get("name") or (source_path.stem if source_path else "agent-eval"))
        targets_raw = raw.get("targets") or []
        if not targets_raw:
            raise ConfigError("agent-eval config requires at least one target in 'targets'")

        benchmark_raw = dict(raw.get("benchmark") or {})
        benchmark = load_agent_benchmark_spec(benchmark_raw, source_path=source_path)

        return cls(
            name=name,
            database=Path(str(raw.get("database") or "results/llm_refinery.duckdb")),
            benchmark=benchmark,
            targets=[AgentEvalTarget.from_mapping(dict(item)) for item in targets_raw],
            request=AgentEvalRequestConfig.from_mapping(raw.get("request")),
            source_path=source_path,
        )


def load_agent_eval_config(path: str | Path) -> AgentEvalConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")
    return AgentEvalConfig.from_mapping(raw, source_path=config_path)
