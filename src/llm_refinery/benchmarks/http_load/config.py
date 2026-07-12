from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_refinery.core.config import (
    ConfigError,
    coerce_list,
    load_yaml_mapping,
    reject_unknown_keys,
)
from llm_refinery.core.endpoints import CHAT_PROTOCOLS, Endpoint
from llm_refinery.core.runs import stable_hash

CACHE_MODES = {"shared", "unique"}
RECOMMENDED_MEASURED_REQUESTS = 100


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
    prompt_pool: tuple[str, ...] = ()
    cache_mode: str = "shared"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, base_dir: Path) -> HttpScenario:
        reject_unknown_keys(
            raw,
            {
                "name",
                "prompt",
                "prompt_file",
                "prompts",
                "prompt_files",
                "system",
                "max_tokens",
                "concurrency",
                "requests",
                "warmup_requests",
                "temperature",
                "seed",
                "stream",
                "timeout_s",
                "prompt_repeat",
                "cache_mode",
                "expected_contains",
            },
            context="HTTP scenario",
        )
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ConfigError("each HTTP scenario requires a non-empty 'name'")

        prompts = _scenario_prompts(raw, base_dir=base_dir)
        max_tokens = [int(value) for value in coerce_list(raw.get("max_tokens", 128))]
        concurrency = [int(value) for value in coerce_list(raw.get("concurrency", 1))]
        requests = int(raw.get("requests", 8))
        warmup_requests = int(raw.get("warmup_requests", 0))
        prompt_repeat = int(raw.get("prompt_repeat", 1))
        timeout_s = float(raw.get("timeout_s", 300.0))
        cache_mode = str(raw.get("cache_mode", "shared")).strip().lower()

        if any(value <= 0 for value in max_tokens):
            raise ConfigError(f"scenario {name!r} max_tokens values must be positive")
        if any(value <= 0 for value in concurrency):
            raise ConfigError(f"scenario {name!r} concurrency values must be positive")
        if requests <= 0:
            raise ConfigError(f"scenario {name!r} requests must be positive")
        if any(value > requests for value in concurrency):
            raise ConfigError(f"scenario {name!r} concurrency cannot exceed requests ({requests})")
        if warmup_requests < 0:
            raise ConfigError(f"scenario {name!r} warmup_requests cannot be negative")
        if prompt_repeat <= 0:
            raise ConfigError(f"scenario {name!r} prompt_repeat must be positive")
        if timeout_s <= 0:
            raise ConfigError(f"scenario {name!r} timeout_s must be positive")
        if cache_mode not in CACHE_MODES:
            choices = ", ".join(sorted(CACHE_MODES))
            raise ConfigError(f"scenario {name!r} cache_mode must be one of: {choices}")

        return cls(
            name=name,
            prompt=prompts[0],
            prompt_pool=tuple(prompts),
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
            cache_mode=cache_mode,
            expected_contains=[str(value) for value in coerce_list(raw.get("expected_contains"))],
        )

    @property
    def rendered_prompt(self) -> str:
        return self.rendered_prompt_for(0)

    @property
    def prompts(self) -> tuple[str, ...]:
        """Return the configured pool while retaining ``prompt`` compatibility."""
        return self.prompt_pool or (self.prompt,)

    def rendered_prompt_for(self, index: int, *, request_nonce: str | None = None) -> str:
        """Select a prompt deterministically and apply the configured cache policy.

        ``shared`` sends pool entries unchanged (a one-item pool is the legacy repeated-prompt
        behavior). ``unique`` adds a nonce before the prompt, preventing a long shared user-prefix
        from dominating measurements even when the pool wraps.
        """
        prompt = self.prompts[index % len(self.prompts)]
        rendered = "\n\n".join([prompt] * self.prompt_repeat)
        if self.cache_mode == "unique":
            nonce = request_nonce or "direct"
            return f"[llm-refinery cache-bust {nonce}:{index}]\n\n{rendered}"
        return rendered

    def safe_json(self) -> dict[str, Any]:
        rendered_prompts = ["\n\n".join([prompt] * self.prompt_repeat) for prompt in self.prompts]
        rendered = rendered_prompts[0]
        return {
            "name": self.name,
            "system": self.system,
            "prompt_preview": rendered[:240],
            "prompt_chars": len(rendered),
            "prompt_hash": stable_hash(rendered),
            "prompt_hashes": [stable_hash(prompt) for prompt in rendered_prompts],
            "prompt_pool_size": len(rendered_prompts),
            "max_tokens": self.max_tokens,
            "concurrency": self.concurrency,
            "requests": self.requests,
            "warmup_requests": self.warmup_requests,
            "temperature": self.temperature,
            "seed": self.seed,
            "stream": self.stream,
            "timeout_s": self.timeout_s,
            "prompt_repeat": self.prompt_repeat,
            "cache_mode": self.cache_mode,
            "expected_contains": self.expected_contains,
        }


@dataclass(frozen=True)
class HttpTransportConfig:
    """HTTP client environment and TLS settings shared by every trial."""

    trust_env: bool = True
    ca_bundle: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any] | None,
        *,
        base_dir: Path,
    ) -> HttpTransportConfig:
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ConfigError("HTTP-load transport must be a mapping")
        reject_unknown_keys(raw, {"trust_env", "ca_bundle"}, context="HTTP-load transport")
        trust_env = raw.get("trust_env", True)
        if not isinstance(trust_env, bool):
            raise ConfigError("HTTP-load transport.trust_env must be a boolean")
        ca_bundle_raw = raw.get("ca_bundle")
        ca_bundle = Path(str(ca_bundle_raw)) if ca_bundle_raw else None
        if ca_bundle is not None and not ca_bundle.is_absolute():
            ca_bundle = base_dir / ca_bundle
        if ca_bundle is not None and not ca_bundle.is_file():
            raise ConfigError(
                f"HTTP-load transport.ca_bundle is not a file: {ca_bundle}"
            )
        return cls(trust_env=trust_env, ca_bundle=ca_bundle)

    def safe_json(self) -> dict[str, Any]:
        return {
            "trust_env": self.trust_env,
            "ca_bundle": str(self.ca_bundle) if self.ca_bundle else None,
        }


@dataclass(frozen=True)
class HttpLoadConfig:
    name: str
    database: Path
    targets: list[Endpoint]
    scenarios: list[HttpScenario]
    transport: HttpTransportConfig = HttpTransportConfig()
    source_path: Path | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], source_path: Path | None = None) -> HttpLoadConfig:
        reject_unknown_keys(
            raw,
            {"name", "database", "targets", "scenarios", "transport"},
            context="HTTP-load configuration",
        )
        name = str(raw.get("name") or (source_path.stem if source_path else "http-load"))
        targets_raw = raw.get("targets") or []
        scenarios_raw = raw.get("scenarios") or []
        if not targets_raw:
            raise ConfigError("HTTP load config requires at least one target in 'targets'")
        if not scenarios_raw:
            raise ConfigError("HTTP load config requires at least one scenario in 'scenarios'")

        if any(not isinstance(item, dict) for item in targets_raw):
            raise ConfigError("each HTTP target must be a mapping")
        if any(not isinstance(item, dict) for item in scenarios_raw):
            raise ConfigError("each HTTP scenario must be a mapping")
        base_dir = source_path.parent if source_path else Path.cwd()
        targets = [
            Endpoint.from_mapping(
                item,
                context="HTTP target",
                allowed_protocols=CHAT_PROTOCOLS,
            )
            for item in targets_raw
        ]
        scenarios = [HttpScenario.from_mapping(item, base_dir=base_dir) for item in scenarios_raw]
        transport = HttpTransportConfig.from_mapping(raw.get("transport"), base_dir=base_dir)
        _require_unique_names([target.name for target in targets], context="HTTP target")
        _require_unique_names([scenario.name for scenario in scenarios], context="HTTP scenario")
        return cls(
            name=name,
            database=Path(str(raw.get("database") or "results/llm_refinery.duckdb")),
            targets=targets,
            scenarios=scenarios,
            transport=transport,
            source_path=source_path,
        )


@dataclass(frozen=True)
class HttpLoadTrial:
    suite: str
    name: str
    key: str
    target: Endpoint
    scenario: HttpScenario
    transport: HttpTransportConfig
    concurrency: int
    max_tokens: int

    @property
    def command_text(self) -> str:
        return (
            f"http-load protocol={self.target.protocol} base_url={self.target.base_url} "
            f"model={self.target.model} scenario={self.scenario.name} "
            f"concurrency={self.concurrency} requests={self.scenario.requests} "
            f"max_tokens={self.max_tokens} stream={self.scenario.stream} "
            f"cache_mode={self.scenario.cache_mode}"
        )

    @property
    def effective_warmup_requests(self) -> int:
        """Warm every client slot, even when old manifests requested fewer warmups."""
        return max(self.scenario.warmup_requests, self.concurrency)

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "name": self.name,
            "key": self.key,
            "model": {"name": self.target.model},
            "target": self.target.safe_json(),
            "scenario": self.scenario.safe_json(),
            "transport": self.transport.safe_json(),
            "prompt_tokens": None,
            "gen_tokens": self.max_tokens,
            "params": {
                "target": self.target.name,
                "protocol": self.target.protocol,
                "scenario": self.scenario.name,
                "model": self.target.model,
                "concurrency": self.concurrency,
                "requests": self.scenario.requests,
                "max_tokens": self.max_tokens,
                "stream": self.scenario.stream,
                "temperature": self.scenario.temperature,
                "cache_mode": self.scenario.cache_mode,
                "prompt_pool_size": len(self.scenario.prompts),
                "warmup_requests": self.effective_warmup_requests,
                "recommended_measured_requests": RECOMMENDED_MEASURED_REQUESTS,
            },
        }


def load_http_load_config(path: str | Path) -> HttpLoadConfig:
    config_path, raw = load_yaml_mapping(path)
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
                        "transport": config.transport.safe_json(),
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
                            transport=config.transport,
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


def _require_unique_names(names: list[str], *, context: str) -> None:
    if len(names) != len(set(names)):
        raise ConfigError(f"{context} names must be unique")


def _scenario_prompts(raw: dict[str, Any], *, base_dir: Path) -> list[str]:
    sources = [
        key
        for key in ("prompt", "prompt_file", "prompts", "prompt_files")
        if raw.get(key) is not None
    ]
    if len(sources) != 1:
        raise ConfigError(
            "each HTTP scenario must set exactly one of 'prompt', 'prompt_file', "
            "'prompts', or 'prompt_files'"
        )

    source = sources[0]
    if source == "prompt":
        prompts = [str(raw[source])]
    elif source == "prompts":
        prompts = [str(value) for value in coerce_list(raw[source])]
    else:
        paths = coerce_list(raw[source])
        prompts = [_read_prompt_file(value, base_dir=base_dir) for value in paths]

    if not prompts or any(not prompt.strip() for prompt in prompts):
        raise ConfigError("HTTP scenario prompt pools must contain non-empty prompts")
    return prompts


def _read_prompt_file(value: Any, *, base_dir: Path) -> str:
    prompt_path = Path(str(value))
    if not prompt_path.is_absolute():
        prompt_path = base_dir / prompt_path
    return prompt_path.read_text(encoding="utf-8")
