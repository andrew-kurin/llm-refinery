from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a sweep config is invalid."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf: str | None = None
    path: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ModelSpec:
        name = raw.get("name")
        if not name:
            raise ConfigError("each model requires a non-empty 'name'")

        hf = raw.get("hf")
        path = raw.get("path") or raw.get("model")
        if bool(hf) == bool(path):
            raise ConfigError(f"model {name!r} must set exactly one of 'hf' or 'path'")

        return cls(
            name=str(name),
            hf=str(hf) if hf else None,
            path=str(path) if path else None,
            params=dict(raw.get("params") or {}),
            extra_args=coerce_arg_list(raw.get("extra_args") or []),
        )


@dataclass(frozen=True)
class BenchSpec:
    prompt_tokens: list[int] = field(default_factory=lambda: [512])
    gen_tokens: list[int] = field(default_factory=lambda: [128])
    repetitions: int = 3
    output: str | None = "json"
    params: dict[str, Any] = field(default_factory=dict)
    omit_params: set[str] = field(default_factory=set)
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> BenchSpec:
        raw = raw or {}
        output = raw.get("output", "json")
        return cls(
            prompt_tokens=[int(v) for v in coerce_list(raw.get("prompt_tokens", [512]))],
            gen_tokens=[int(v) for v in coerce_list(raw.get("gen_tokens", [128]))],
            repetitions=int(raw.get("repetitions", 3)),
            output=None if output is None else str(output),
            params=dict(raw.get("params") or {}),
            omit_params={str(v) for v in coerce_list(raw.get("omit_params") or [])},
            extra_args=coerce_arg_list(raw.get("extra_args") or []),
        )


@dataclass(frozen=True)
class ServerSpec:
    params: dict[str, Any] = field(default_factory=dict)
    omit_params: set[str] = field(default_factory=set)
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    model_flag: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> ServerSpec:
        raw = raw or {}
        return cls(
            params=dict(raw.get("params") or {}),
            omit_params={str(v) for v in coerce_list(raw.get("omit_params") or [])},
            extra_args=coerce_arg_list(raw.get("extra_args") or []),
            env={str(k): str(v) for k, v in dict(raw.get("env") or {}).items()},
            model_flag=str(raw["model_flag"]) if raw.get("model_flag") else None,
        )


@dataclass(frozen=True)
class EvalSpec:
    tasks: str = "ifeval,gsm8k"
    limit: int | None = 50
    max_length: int = 8192
    eos_string: str = "<turn|>"
    gen_kwargs: str | None = None
    api_model: str = "local-model"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> EvalSpec:
        raw = raw or {}
        limit_raw = raw.get("limit", 50)
        if limit_raw is None or (isinstance(limit_raw, str) and limit_raw.lower() == "all"):
            limit = None
        else:
            limit = int(limit_raw)
            if limit <= 0:
                raise ConfigError("eval.limit must be a positive integer or 'all'")

        return cls(
            tasks=str(raw.get("tasks") or "ifeval,gsm8k"),
            limit=limit,
            max_length=int(raw.get("max_length", 8192)),
            eos_string=str(raw.get("eos_string") or "<turn|>"),
            gen_kwargs=str(raw["gen_kwargs"]) if raw.get("gen_kwargs") else None,
            api_model=str(raw.get("api_model") or "local-model"),
        )


@dataclass(frozen=True)
class TuneConfig:
    name: str
    database: Path
    commands: dict[str, list[str]]
    models: list[ModelSpec]
    defaults: dict[str, Any] = field(default_factory=dict)
    sweep: dict[str, list[Any]] = field(default_factory=dict)
    bench: BenchSpec = field(default_factory=BenchSpec)
    server: ServerSpec = field(default_factory=ServerSpec)
    eval: EvalSpec = field(default_factory=EvalSpec)
    source_path: Path | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], source_path: Path | None = None) -> TuneConfig:
        name = str(raw.get("name") or (source_path.stem if source_path else "llm-refinery"))
        commands = raw.get("commands") or {}
        models_raw = raw.get("models") or []
        if not models_raw:
            raise ConfigError("config requires at least one model in 'models'")

        return cls(
            name=name,
            database=Path(str(raw.get("database") or "results/llm_refinery.duckdb")),
            commands={
                "bench": coerce_command(commands.get("bench") or ["llama", "bench"]),
                "server": coerce_command(commands.get("server") or ["llama", "server"]),
            },
            models=[ModelSpec.from_mapping(dict(item)) for item in models_raw],
            defaults=dict(raw.get("defaults") or {}),
            sweep={str(k): coerce_list(v) for k, v in dict(raw.get("sweep") or {}).items()},
            bench=BenchSpec.from_mapping(raw.get("bench")),
            server=ServerSpec.from_mapping(raw.get("server")),
            eval=EvalSpec.from_mapping(raw.get("eval") or raw.get("lm_eval")),
            source_path=source_path,
        )


@dataclass(frozen=True)
class Trial:
    suite: str
    name: str
    key: str
    model: ModelSpec
    params: dict[str, Any]
    prompt_tokens: int | None = None
    gen_tokens: int | None = None

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "name": self.name,
            "key": self.key,
            "model": {
                "name": self.model.name,
                "hf": self.model.hf,
                "path": self.model.path,
                "params": self.model.params,
                "extra_args": self.model.extra_args,
            },
            "params": self.params,
            "prompt_tokens": self.prompt_tokens,
            "gen_tokens": self.gen_tokens,
        }


def load_config(path: str | Path) -> TuneConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")
    return TuneConfig.from_mapping(raw, source_path=config_path)


def expand_trials(config: TuneConfig, *, include_bench_dimensions: bool = True) -> list[Trial]:
    sweep_items = list(config.sweep.items())
    sweep_keys = [key for key, _ in sweep_items]
    sweep_values = [values for _, values in sweep_items]
    sweep_products = list(product(*sweep_values)) if sweep_values else [()]

    if include_bench_dimensions:
        prompt_values: list[int | None] = config.bench.prompt_tokens
        gen_values: list[int | None] = config.bench.gen_tokens
    else:
        prompt_values = [None]
        gen_values = [None]

    trials: list[Trial] = []
    for model in config.models:
        for sweep_tuple in sweep_products:
            params = dict(config.defaults)
            params.update(model.params)
            params.update(dict(zip(sweep_keys, sweep_tuple, strict=True)))

            for prompt_tokens in prompt_values:
                for gen_tokens in gen_values:
                    key_material = {
                        "suite": config.name,
                        "model": model.name,
                        "hf": model.hf,
                        "path": model.path,
                        "params": params,
                        "prompt_tokens": prompt_tokens,
                        "gen_tokens": gen_tokens,
                    }
                    key = stable_hash(key_material)
                    name_bits = [config.name, model.name, key]
                    trials.append(
                        Trial(
                            suite=config.name,
                            name="/".join(name_bits),
                            key=key,
                            model=model,
                            params=params,
                            prompt_tokens=prompt_tokens,
                            gen_tokens=gen_tokens,
                        )
                    )
    return trials


def coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def coerce_command(value: Any) -> list[str]:
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ConfigError(f"command must be a string or list, got {type(value).__name__}")


def coerce_arg_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ConfigError(f"extra args must be a string or list, got {type(value).__name__}")


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
