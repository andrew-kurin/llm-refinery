from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llm_refinery.config import ConfigError, coerce_list, stable_hash

PROVIDERS = {"openai", "ollama", "cerebras"}


@dataclass(frozen=True)
class HttpTarget:
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> HttpTarget:
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ConfigError("each HTTP target requires a non-empty 'name'")

        provider = str(raw.get("provider") or "openai").strip().lower()
        if provider not in PROVIDERS:
            raise ConfigError(
                f"target {name!r} provider must be one of {sorted(PROVIDERS)}, got {provider!r}"
            )

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
            headers={str(key): str(value) for key, value in dict(raw.get("headers") or {}).items()},
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
class HttpScenario:
    name: str
    prompt: str
    system: str | None = None
    max_tokens: list[int] = field(default_factory=lambda: [128])
    concurrency: list[int] = field(default_factory=lambda: [1])
    requests: int = 8
    warmup_requests: int = 0
    temperature: float = 0.0
    seed: int | None = None
    stream: bool = True
    timeout_s: float = 300.0
    prompt_repeat: int = 1
    expected_contains: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, base_dir: Path) -> HttpScenario:
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ConfigError("each HTTP scenario requires a non-empty 'name'")

        prompt = _scenario_prompt(raw, base_dir=base_dir)
        max_tokens = [int(value) for value in coerce_list(raw.get("max_tokens", 128))]
        concurrency = [int(value) for value in coerce_list(raw.get("concurrency", 1))]
        requests = int(raw.get("requests", 8))
        warmup_requests = int(raw.get("warmup_requests", 0))
        prompt_repeat = int(raw.get("prompt_repeat", 1))
        timeout_s = float(raw.get("timeout_s", 300.0))

        if any(value <= 0 for value in max_tokens):
            raise ConfigError(f"scenario {name!r} max_tokens values must be positive")
        if any(value <= 0 for value in concurrency):
            raise ConfigError(f"scenario {name!r} concurrency values must be positive")
        if requests <= 0:
            raise ConfigError(f"scenario {name!r} requests must be positive")
        if warmup_requests < 0:
            raise ConfigError(f"scenario {name!r} warmup_requests cannot be negative")
        if prompt_repeat <= 0:
            raise ConfigError(f"scenario {name!r} prompt_repeat must be positive")
        if timeout_s <= 0:
            raise ConfigError(f"scenario {name!r} timeout_s must be positive")

        return cls(
            name=name,
            prompt=prompt,
            system=str(raw["system"]) if raw.get("system") else None,
            max_tokens=max_tokens,
            concurrency=concurrency,
            requests=requests,
            warmup_requests=warmup_requests,
            temperature=float(raw.get("temperature", 0.0)),
            seed=int(raw["seed"]) if raw.get("seed") is not None else None,
            stream=bool(raw.get("stream", True)),
            timeout_s=timeout_s,
            prompt_repeat=prompt_repeat,
            expected_contains=[str(value) for value in coerce_list(raw.get("expected_contains"))],
        )

    @property
    def rendered_prompt(self) -> str:
        return "\n\n".join([self.prompt] * self.prompt_repeat)

    def safe_json(self) -> dict[str, Any]:
        rendered = self.rendered_prompt
        return {
            "name": self.name,
            "system": self.system,
            "prompt_preview": rendered[:240],
            "prompt_chars": len(rendered),
            "prompt_hash": stable_hash(rendered),
            "max_tokens": self.max_tokens,
            "concurrency": self.concurrency,
            "requests": self.requests,
            "warmup_requests": self.warmup_requests,
            "temperature": self.temperature,
            "seed": self.seed,
            "stream": self.stream,
            "timeout_s": self.timeout_s,
            "prompt_repeat": self.prompt_repeat,
            "expected_contains": self.expected_contains,
        }


@dataclass(frozen=True)
class HttpLoadConfig:
    name: str
    database: Path
    targets: list[HttpTarget]
    scenarios: list[HttpScenario]
    source_path: Path | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], source_path: Path | None = None) -> HttpLoadConfig:
        name = str(raw.get("name") or (source_path.stem if source_path else "http-load"))
        targets_raw = raw.get("targets") or []
        scenarios_raw = raw.get("scenarios") or []
        if not targets_raw:
            raise ConfigError("HTTP load config requires at least one target in 'targets'")
        if not scenarios_raw:
            raise ConfigError("HTTP load config requires at least one scenario in 'scenarios'")

        base_dir = source_path.parent if source_path else Path.cwd()
        return cls(
            name=name,
            database=Path(str(raw.get("database") or "results/llm_refinery.duckdb")),
            targets=[HttpTarget.from_mapping(dict(item)) for item in targets_raw],
            scenarios=[
                HttpScenario.from_mapping(dict(item), base_dir=base_dir) for item in scenarios_raw
            ],
            source_path=source_path,
        )


@dataclass(frozen=True)
class HttpLoadTrial:
    suite: str
    name: str
    key: str
    target: HttpTarget
    scenario: HttpScenario
    concurrency: int
    max_tokens: int

    @property
    def command_text(self) -> str:
        return (
            f"http-load provider={self.target.provider} base_url={self.target.base_url} "
            f"model={self.target.model} scenario={self.scenario.name} "
            f"concurrency={self.concurrency} requests={self.scenario.requests} "
            f"max_tokens={self.max_tokens} stream={self.scenario.stream}"
        )

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "name": self.name,
            "key": self.key,
            "model": {"name": self.target.model},
            "target": self.target.safe_json(),
            "scenario": self.scenario.safe_json(),
            "prompt_tokens": None,
            "gen_tokens": self.max_tokens,
            "params": {
                "target": self.target.name,
                "provider": self.target.provider,
                "scenario": self.scenario.name,
                "model": self.target.model,
                "concurrency": self.concurrency,
                "requests": self.scenario.requests,
                "max_tokens": self.max_tokens,
                "stream": self.scenario.stream,
                "temperature": self.scenario.temperature,
            },
        }


def load_http_load_config(path: str | Path) -> HttpLoadConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")
    return HttpLoadConfig.from_mapping(raw, source_path=config_path)


def expand_http_load_trials(
    config: HttpLoadConfig,
    *,
    target_names: tuple[str, ...] = (),
    scenario_names: tuple[str, ...] = (),
) -> list[HttpLoadTrial]:
    wanted_targets = set(target_names)
    wanted_scenarios = set(scenario_names)
    targets = [
        target for target in config.targets if not wanted_targets or target.name in wanted_targets
    ]
    scenarios = [
        scenario
        for scenario in config.scenarios
        if not wanted_scenarios or scenario.name in wanted_scenarios
    ]

    missing_targets = wanted_targets - {target.name for target in targets}
    if missing_targets:
        raise ConfigError(f"unknown HTTP load target(s): {', '.join(sorted(missing_targets))}")
    missing_scenarios = wanted_scenarios - {scenario.name for scenario in scenarios}
    if missing_scenarios:
        raise ConfigError(f"unknown HTTP load scenario(s): {', '.join(sorted(missing_scenarios))}")

    trials: list[HttpLoadTrial] = []
    for target in targets:
        for scenario in scenarios:
            for concurrency in scenario.concurrency:
                for max_tokens in scenario.max_tokens:
                    key_material = {
                        "suite": config.name,
                        "target": target.safe_json(),
                        "scenario": scenario.safe_json(),
                        "concurrency": concurrency,
                        "max_tokens": max_tokens,
                    }
                    key = stable_hash(key_material)
                    name = "/".join(
                        [
                            config.name,
                            target.name,
                            scenario.name,
                            f"c{concurrency}",
                            f"n{max_tokens}",
                            key,
                        ]
                    )
                    trials.append(
                        HttpLoadTrial(
                            suite=config.name,
                            name=name,
                            key=key,
                            target=target,
                            scenario=scenario,
                            concurrency=concurrency,
                            max_tokens=max_tokens,
                        )
                    )
    return trials


def print_http_load_plan(
    config: HttpLoadConfig,
    *,
    target_names: tuple[str, ...] = (),
    scenario_names: tuple[str, ...] = (),
    limit: int | None = None,
) -> None:
    all_trials = expand_http_load_trials(
        config,
        target_names=target_names,
        scenario_names=scenario_names,
    )
    trials = all_trials[:limit] if limit is not None else all_trials
    for index, trial in enumerate(trials):
        print(f"# [{index}] {trial.name}")
        print(trial.command_text)
        print()
    print(f"planned {len(trials)} of {len(all_trials)} HTTP load trial(s)")


def _scenario_prompt(raw: dict[str, Any], *, base_dir: Path) -> str:
    has_prompt = raw.get("prompt") is not None
    has_prompt_file = raw.get("prompt_file") is not None
    if has_prompt == has_prompt_file:
        raise ConfigError("each HTTP scenario must set exactly one of 'prompt' or 'prompt_file'")
    if has_prompt:
        return str(raw["prompt"])

    prompt_path = Path(str(raw["prompt_file"]))
    if not prompt_path.is_absolute():
        prompt_path = base_dir / prompt_path
    return prompt_path.read_text(encoding="utf-8")
