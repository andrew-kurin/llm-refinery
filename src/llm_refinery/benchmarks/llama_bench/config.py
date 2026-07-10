from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

from llm_refinery.core.config import (
    ConfigError,
    coerce_arg_list,
    coerce_command,
    coerce_list,
    load_yaml_mapping,
    reject_unknown_keys,
)
from llm_refinery.core.runs import stable_hash


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hf: str | None = None
    path: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ModelSpec:
        reject_unknown_keys(
            raw,
            {"name", "hf", "path", "model", "params", "extra_args"},
            context="model",
        )
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
        reject_unknown_keys(
            raw,
            {
                "prompt_tokens",
                "gen_tokens",
                "repetitions",
                "output",
                "params",
                "omit_params",
                "extra_args",
            },
            context="bench configuration",
        )
        output = raw.get("output", "json")
        prompt_tokens = [int(v) for v in coerce_list(raw.get("prompt_tokens", [512]))]
        gen_tokens = [int(v) for v in coerce_list(raw.get("gen_tokens", [128]))]
        repetitions = int(raw.get("repetitions", 3))
        if not prompt_tokens or any(value <= 0 for value in prompt_tokens):
            raise ConfigError("bench.prompt_tokens must contain positive integers")
        if not gen_tokens or any(value <= 0 for value in gen_tokens):
            raise ConfigError("bench.gen_tokens must contain positive integers")
        if repetitions <= 0:
            raise ConfigError("bench.repetitions must be positive")
        return cls(
            prompt_tokens=prompt_tokens,
            gen_tokens=gen_tokens,
            repetitions=repetitions,
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
        reject_unknown_keys(
            raw,
            {"params", "omit_params", "extra_args", "env", "model_flag"},
            context="server configuration",
        )
        return cls(
            params=dict(raw.get("params") or {}),
            omit_params={str(v) for v in coerce_list(raw.get("omit_params") or [])},
            extra_args=coerce_arg_list(raw.get("extra_args") or []),
            env={str(k): str(v) for k, v in dict(raw.get("env") or {}).items()},
            model_flag=str(raw["model_flag"]) if raw.get("model_flag") else None,
        )


@dataclass(frozen=True)
class LlamaSweepConfig:
    name: str
    database: Path
    commands: dict[str, list[str]]
    models: list[ModelSpec]
    defaults: dict[str, Any] = field(default_factory=dict)
    sweep: dict[str, list[Any]] = field(default_factory=dict)
    bench: BenchSpec = field(default_factory=BenchSpec)
    server: ServerSpec = field(default_factory=ServerSpec)
    source_path: Path | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], source_path: Path | None = None) -> LlamaSweepConfig:
        reject_unknown_keys(
            raw,
            {
                "name",
                "database",
                "commands",
                "models",
                "defaults",
                "sweep",
                "bench",
                "server",
            },
            context="llama configuration",
        )
        name = str(raw.get("name") or (source_path.stem if source_path else "llm-refinery"))
        commands = raw.get("commands") or {}
        if not isinstance(commands, dict):
            raise ConfigError("commands must be a mapping")
        models_raw = raw.get("models") or []
        if not isinstance(models_raw, list) or not models_raw:
            raise ConfigError("config requires at least one model in 'models'")
        if any(not isinstance(item, dict) for item in models_raw):
            raise ConfigError("each models entry must be a mapping")
        models = [ModelSpec.from_mapping(item) for item in models_raw]
        model_names = [model.name for model in models]
        if len(model_names) != len(set(model_names)):
            raise ConfigError("model names must be unique")
        sweep = {str(k): coerce_list(v) for k, v in dict(raw.get("sweep") or {}).items()}
        empty_sweeps = sorted(key for key, values in sweep.items() if not values)
        if empty_sweeps:
            raise ConfigError(f"sweep values cannot be empty: {', '.join(empty_sweeps)}")

        return cls(
            name=name,
            database=Path(str(raw.get("database") or "results/llm_refinery.duckdb")),
            commands={
                "bench": coerce_command(commands.get("bench") or ["llama", "bench"]),
                "server": coerce_command(commands.get("server") or ["llama", "server"]),
            },
            models=models,
            defaults=dict(raw.get("defaults") or {}),
            sweep=sweep,
            bench=BenchSpec.from_mapping(raw.get("bench")),
            server=ServerSpec.from_mapping(raw.get("server")),
            source_path=source_path,
        )


@dataclass(frozen=True)
class LlamaTrial:
    benchmark_kind: str
    suite: str
    name: str
    key: str
    model: ModelSpec
    params: dict[str, Any]
    execution: dict[str, Any]
    prompt_tokens: int | None = None
    gen_tokens: int | None = None

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark_kind,
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
            "execution": self.execution,
            "prompt_tokens": self.prompt_tokens,
            "gen_tokens": self.gen_tokens,
        }


def load_llama_config(path: str | Path) -> LlamaSweepConfig:
    config_path, raw = load_yaml_mapping(path)
    return LlamaSweepConfig.from_mapping(raw, source_path=config_path)


def expand_trials(config: LlamaSweepConfig, *, kind: str = "bench") -> list[LlamaTrial]:
    if kind not in {"bench", "server"}:
        raise ValueError(f"unknown llama trial kind: {kind}")
    sweep_items = list(config.sweep.items())
    sweep_keys = [key for key, _ in sweep_items]
    sweep_values = [values for _, values in sweep_items]
    sweep_products = list(product(*sweep_values)) if sweep_values else [()]

    if kind == "bench":
        prompt_values: list[int | None] = []
        prompt_values.extend(config.bench.prompt_tokens)
        gen_values: list[int | None] = []
        gen_values.extend(config.bench.gen_tokens)
        command_params = config.bench.params
        omit_params = config.bench.omit_params
        execution = {
            "command": config.commands["bench"],
            "repetitions": config.bench.repetitions,
            "output": config.bench.output,
            "extra_args": config.bench.extra_args,
        }
        benchmark_kind = "llama_bench"
    else:
        prompt_values = [None]
        gen_values = [None]
        command_params = config.server.params
        omit_params = config.server.omit_params
        execution = {
            "command": config.commands["server"],
            "model_flag": config.server.model_flag,
            "extra_args": config.server.extra_args,
            "env_names": sorted(config.server.env),
            "env_hash": stable_hash(config.server.env) if config.server.env else None,
        }
        benchmark_kind = "llama_server"

    trials: list[LlamaTrial] = []
    for model in config.models:
        for sweep_tuple in sweep_products:
            params = dict(config.defaults)
            params.update(model.params)
            params.update(dict(zip(sweep_keys, sweep_tuple, strict=True)))
            params.update(command_params)
            for omitted in omit_params:
                params.pop(omitted, None)

            for prompt_tokens in prompt_values:
                for gen_tokens in gen_values:
                    key_material = {
                        "benchmark": benchmark_kind,
                        "suite": config.name,
                        "model": {
                            "name": model.name,
                            "hf": model.hf,
                            "path": model.path,
                            "extra_args": model.extra_args,
                        },
                        "params": params,
                        "execution": execution,
                        "prompt_tokens": prompt_tokens,
                        "gen_tokens": gen_tokens,
                    }
                    key = stable_hash(key_material)
                    name_bits = [config.name, model.name, key]
                    trials.append(
                        LlamaTrial(
                            benchmark_kind=benchmark_kind,
                            suite=config.name,
                            name="/".join(name_bits),
                            key=key,
                            model=model,
                            params=params,
                            execution=dict(execution),
                            prompt_tokens=prompt_tokens,
                            gen_tokens=gen_tokens,
                        )
                    )
    return trials

