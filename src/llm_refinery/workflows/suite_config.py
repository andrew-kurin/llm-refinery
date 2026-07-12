from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_refinery.core.config import (
    ConfigError,
    coerce_list,
    load_yaml_mapping,
    reject_unknown_keys,
)
from llm_refinery.core.endpoints import OPENAI_CHAT, Endpoint
from llm_refinery.core.targets import TargetSpec, load_target_spec


@dataclass(frozen=True)
class QualityStep:
    enabled: bool = True
    tasks: str = "ifeval,gsm8k"
    limit: int | None = 50
    num_fewshot: int | None = None
    max_length: int = 8192
    eos_string: str | None = None
    tokenizer: str | None = None
    metadata: str | None = None
    gen_kwargs: str | None = None
    include_path: Path | None = None
    output_root: Path = Path("results/lm_eval")
    package_spec: str = "lm-eval[api]==0.4.12"
    extra_packages: tuple[str, ...] = ()
    offline: bool = True

    def __post_init__(self) -> None:
        if self.limit is not None and self.limit <= 0:
            raise ConfigError("suite quality.limit must be positive or None")
        if self.max_length <= 0:
            raise ConfigError("suite quality.max_length must be positive")
        if not self.package_spec.strip():
            raise ConfigError("suite quality.package_spec cannot be empty")
        if any(not package.strip() for package in self.extra_packages):
            raise ConfigError("suite quality.extra_packages cannot contain empty values")
        if self.metadata is not None:
            try:
                metadata = json.loads(self.metadata)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"suite quality.metadata must be valid JSON: {exc}") from exc
            if not isinstance(metadata, dict):
                raise ConfigError("suite quality.metadata must be a JSON object")

    @classmethod
    def from_mapping(
        cls, raw: dict[str, Any] | None, *, source_path: Path | None = None
    ) -> QualityStep:
        raw = raw or {}
        reject_unknown_keys(
            raw,
            {
                "enabled",
                "tasks",
                "limit",
                "num_fewshot",
                "max_length",
                "eos_string",
                "tokenizer",
                "metadata",
                "gen_kwargs",
                "include_path",
                "output_root",
                "package_spec",
                "extra_packages",
                "offline",
            },
            context="suite quality step",
        )
        limit_raw = raw.get("limit", 50)
        limit = None if limit_raw is None or str(limit_raw).lower() == "all" else int(limit_raw)
        if limit is not None and limit <= 0:
            raise ConfigError("suite quality.limit must be a positive integer or 'all'")
        max_length = int(raw.get("max_length", 8192))
        if max_length <= 0:
            raise ConfigError("suite quality.max_length must be positive")
        include_path = Path(str(raw["include_path"])) if raw.get("include_path") else None
        if include_path is not None and source_path is not None and not include_path.is_absolute():
            include_path = source_path.parent / include_path
        metadata_raw = raw.get("metadata")
        metadata = (
            json.dumps(metadata_raw, sort_keys=True, separators=(",", ":"))
            if isinstance(metadata_raw, dict)
            else str(metadata_raw)
            if metadata_raw is not None
            else None
        )
        return cls(
            enabled=bool(raw.get("enabled", True)),
            tasks=str(raw.get("tasks") or "ifeval,gsm8k"),
            limit=limit,
            num_fewshot=int(raw["num_fewshot"]) if raw.get("num_fewshot") is not None else None,
            max_length=max_length,
            eos_string=str(raw["eos_string"]) if raw.get("eos_string") else None,
            tokenizer=str(raw["tokenizer"]) if raw.get("tokenizer") else None,
            metadata=metadata,
            gen_kwargs=str(raw["gen_kwargs"]) if raw.get("gen_kwargs") else None,
            include_path=include_path,
            output_root=Path(str(raw.get("output_root") or "results/lm_eval")),
            package_spec=str(raw.get("package_spec") or "lm-eval[api]==0.4.12"),
            extra_packages=tuple(
                str(package) for package in coerce_list(raw.get("extra_packages"))
            ),
            offline=bool(raw.get("offline", True)),
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "tasks": self.tasks,
            "limit": self.limit,
            "num_fewshot": self.num_fewshot,
            "max_length": self.max_length,
            "eos_string": self.eos_string,
            "tokenizer": self.tokenizer,
            "metadata": self.metadata,
            "gen_kwargs": self.gen_kwargs,
            "include_path": str(self.include_path) if self.include_path else None,
            "output_root": str(self.output_root),
            "package_spec": self.package_spec,
            "extra_packages": list(self.extra_packages),
            "offline": self.offline,
        }


@dataclass(frozen=True)
class HttpLoadStep:
    enabled: bool = False
    config: Path | None = None
    targets: tuple[str, ...] = ()
    scenarios: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None, *, source_path: Path | None) -> HttpLoadStep:
        raw = raw or {}
        reject_unknown_keys(
            raw,
            {"enabled", "config", "targets", "scenarios"},
            context="suite HTTP-load step",
        )
        config_path = Path(str(raw["config"])) if raw.get("config") else None
        if config_path is not None and source_path is not None and not config_path.is_absolute():
            config_path = source_path.parent / config_path
        enabled = bool(raw.get("enabled", config_path is not None))
        if enabled and config_path is None:
            raise ConfigError("suite http_load.config is required when HTTP load is enabled")
        return cls(
            enabled=enabled,
            config=config_path,
            targets=tuple(str(value) for value in coerce_list(raw.get("targets"))),
            scenarios=tuple(str(value) for value in coerce_list(raw.get("scenarios"))),
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "config": str(self.config) if self.config else None,
            "targets": list(self.targets),
            "scenarios": list(self.scenarios),
        }


@dataclass(frozen=True)
class PreflightStep:
    enabled: bool = True
    require_clean: bool = True
    forbidden_ports: tuple[int, ...] = (8081, 8082, 8083)
    sanity_check: bool = True
    expected_response_model: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> PreflightStep:
        raw = raw or {}
        reject_unknown_keys(
            raw,
            {
                "enabled",
                "require_clean",
                "forbidden_ports",
                "sanity_check",
                "expected_response_model",
            },
            context="suite preflight step",
        )
        forbidden_ports = tuple(
            int(value) for value in coerce_list(raw.get("forbidden_ports", [8081, 8082, 8083]))
        )
        if any(port <= 0 or port > 65535 for port in forbidden_ports):
            raise ConfigError("suite preflight.forbidden_ports must be valid TCP ports")
        return cls(
            enabled=bool(raw.get("enabled", True)),
            require_clean=bool(raw.get("require_clean", True)),
            forbidden_ports=forbidden_ports,
            sanity_check=bool(raw.get("sanity_check", True)),
            expected_response_model=(
                str(raw["expected_response_model"]) if raw.get("expected_response_model") else None
            ),
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "require_clean": self.require_clean,
            "forbidden_ports": list(self.forbidden_ports),
            "sanity_check": self.sanity_check,
            "expected_response_model": self.expected_response_model,
        }


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    database: Path
    endpoint: Endpoint | None
    target: TargetSpec | None = None
    schema_version: int = 1
    quality: QualityStep = QualityStep()
    http_load: HttpLoadStep = HttpLoadStep()
    preflight: PreflightStep = PreflightStep()
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2}:
            raise ConfigError(
                f"unsupported suite schema_version {self.schema_version}; expected 1 or 2"
            )
        if (self.endpoint is None) == (self.target is None):
            raise ConfigError("suite requires exactly one of 'endpoint' or 'target'")
        if self.target is not None and self.schema_version < 2:
            raise ConfigError("suite target discovery requires schema_version: 2")

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], source_path: Path | None = None) -> SuiteConfig:
        reject_unknown_keys(
            raw,
            {
                "schema_version",
                "name",
                "database",
                "endpoint",
                "target",
                "quality",
                "http_load",
                "preflight",
            },
            context="suite configuration",
        )
        schema_version = int(raw.get("schema_version", 1))
        target_value = raw.get("target")
        endpoint_value = raw.get("endpoint")
        if target_value is not None and endpoint_value is not None:
            raise ConfigError("suite cannot define both 'endpoint' and 'target'")
        if target_value is not None:
            if isinstance(target_value, dict):
                target = TargetSpec.from_mapping(target_value)
            elif isinstance(target_value, str):
                target_path = Path(target_value)
                if source_path is not None and not target_path.is_absolute():
                    target_path = source_path.parent / target_path
                _resolved_path, target = load_target_spec(target_path)
            else:
                raise ConfigError("suite target must be a mapping or target YAML path")
            endpoint = None
        else:
            endpoint_value = endpoint_value or {
                "name": "local",
                "protocol": OPENAI_CHAT,
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "local-model",
            }
            if not isinstance(endpoint_value, dict):
                raise ConfigError("suite endpoint must be a mapping")
            endpoint = Endpoint.from_mapping(
                endpoint_value,
                context="suite endpoint",
                allowed_protocols=frozenset({OPENAI_CHAT}),
            )
            target = None
        for section in ("quality", "http_load", "preflight"):
            value = raw.get(section)
            if value is not None and not isinstance(value, dict):
                raise ConfigError(f"suite {section} must be a mapping")
        return cls(
            name=str(raw.get("name") or (source_path.stem if source_path else "suite")),
            database=Path(str(raw.get("database") or "results/llm_refinery.duckdb")),
            endpoint=endpoint,
            target=target,
            schema_version=schema_version,
            quality=QualityStep.from_mapping(raw.get("quality"), source_path=source_path),
            http_load=HttpLoadStep.from_mapping(raw.get("http_load"), source_path=source_path),
            preflight=PreflightStep.from_mapping(raw.get("preflight")),
            source_path=source_path,
        )

    def safe_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "database": str(self.database),
            "quality": self.quality.safe_json(),
            "http_load": self.http_load.safe_json(),
            "preflight": self.preflight.safe_json(),
        }
        if self.endpoint is not None:
            payload["endpoint"] = self.endpoint.safe_json()
        if self.target is not None:
            payload["target"] = self.target.safe_json()
        return payload


def load_suite_config(path: str | Path) -> SuiteConfig:
    config_path, raw = load_yaml_mapping(path)
    return SuiteConfig.from_mapping(raw, source_path=config_path)
