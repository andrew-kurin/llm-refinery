from __future__ import annotations

from pathlib import Path
from typing import Any

from llm_refinery.benchmarks.agent.base import AgentBenchmarkSpec
from llm_refinery.benchmarks.agent.geoanalystbench import GeoAnalystBenchSpec
from llm_refinery.config import ConfigError


def load_agent_benchmark_spec(
    raw: dict[str, Any], *, source_path: Path | None = None
) -> AgentBenchmarkSpec:
    kind = str(raw.get("kind") or "geoanalystbench").strip().lower()
    if kind == "geoanalystbench":
        return GeoAnalystBenchSpec.from_mapping(raw, source_path=source_path)
    raise ConfigError(f"unsupported agent-eval benchmark kind: {kind!r}")


__all__ = ["AgentBenchmarkSpec", "GeoAnalystBenchSpec", "load_agent_benchmark_spec"]
