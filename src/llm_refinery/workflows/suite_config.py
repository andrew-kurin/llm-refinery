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


def _strict_bool(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{context} must be a boolean")
    return value


def _schema_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in {1, 2}:
        raise ConfigError(f"suite schema_version must be the integer 1 or 2, got {value!r}")
    return value


def _strict_integer(value: Any, *, context: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ConfigError(f"{context} must be a {qualifier} integer")
    if value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ConfigError(f"{context} must be a {qualifier} integer")
    return value


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
    # ``None`` inherits the schema-v2 target transport; legacy endpoint suites
    # use a deterministic direct quality path.
    trust_env: bool | None = None
    ca_bundle: Path | None = None

    def __post_init__(self) -> None:
        if self.limit is not None:
            _strict_integer(self.limit, context="suite quality.limit", minimum=1)
        if self.num_fewshot is not None:
            _strict_integer(
                self.num_fewshot,
                context="suite quality.num_fewshot",
                minimum=0,
            )
        _strict_integer(self.max_length, context="suite quality.max_length", minimum=1)
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
                "trust_env",
                "ca_bundle",
            },
            context="suite quality step",
        )
        limit_raw = raw.get("limit", 50)
        if limit_raw is None or (isinstance(limit_raw, str) and limit_raw.lower() == "all"):
            limit = None
        else:
            limit = _strict_integer(
                limit_raw,
                context="suite quality.limit",
                minimum=1,
            )
        max_length = _strict_integer(
            raw.get("max_length", 8192),
            context="suite quality.max_length",
            minimum=1,
        )
        num_fewshot_raw = raw.get("num_fewshot")
        num_fewshot = (
            _strict_integer(
                num_fewshot_raw,
                context="suite quality.num_fewshot",
                minimum=0,
            )
            if num_fewshot_raw is not None
            else None
        )
        include_path = Path(str(raw["include_path"])) if raw.get("include_path") else None
        if include_path is not None and source_path is not None and not include_path.is_absolute():
            include_path = source_path.parent / include_path
        ca_bundle_raw = raw.get("ca_bundle")
        if ca_bundle_raw is not None and (
            not isinstance(ca_bundle_raw, str) or not ca_bundle_raw.strip()
        ):
            raise ConfigError("suite quality.ca_bundle must be a non-empty path string")
        ca_bundle = Path(ca_bundle_raw).expanduser() if ca_bundle_raw is not None else None
        if ca_bundle is not None and source_path is not None and not ca_bundle.is_absolute():
            ca_bundle = source_path.parent / ca_bundle
        if ca_bundle is not None:
            ca_bundle = ca_bundle.resolve()
        if ca_bundle is not None and not ca_bundle.is_file():
            raise ConfigError(f"suite quality.ca_bundle is not a file: {ca_bundle}")
        metadata_raw = raw.get("metadata")
        metadata = (
            json.dumps(metadata_raw, sort_keys=True, separators=(",", ":"))
            if isinstance(metadata_raw, dict)
            else str(metadata_raw)
            if metadata_raw is not None
            else None
        )
        return cls(
            enabled=_strict_bool(raw.get("enabled", True), context="suite quality.enabled"),
            tasks=str(raw.get("tasks") or "ifeval,gsm8k"),
            limit=limit,
            num_fewshot=num_fewshot,
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
            offline=_strict_bool(raw.get("offline", True), context="suite quality.offline"),
            trust_env=(
                _strict_bool(raw["trust_env"], context="suite quality.trust_env")
                if "trust_env" in raw
                else None
            ),
            ca_bundle=ca_bundle,
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
            "trust_env": self.trust_env,
            "ca_bundle": str(self.ca_bundle) if self.ca_bundle else None,
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
        enabled = _strict_bool(
            raw.get("enabled", config_path is not None),
            context="suite http_load.enabled",
        )
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
            _strict_integer(
                value,
                context="suite preflight.forbidden_ports entries",
                minimum=1,
            )
            for value in coerce_list(raw.get("forbidden_ports", [8081, 8082, 8083]))
        )
        if any(port <= 0 or port > 65535 for port in forbidden_ports):
            raise ConfigError("suite preflight.forbidden_ports must be valid TCP ports")
        return cls(
            enabled=_strict_bool(raw.get("enabled", True), context="suite preflight.enabled"),
            require_clean=_strict_bool(
                raw.get("require_clean", True),
                context="suite preflight.require_clean",
            ),
            forbidden_ports=forbidden_ports,
            sanity_check=_strict_bool(
                raw.get("sanity_check", True),
                context="suite preflight.sanity_check",
            ),
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
        schema_version = _schema_version(raw.get("schema_version", 1))
        target_value = raw.get("target")
        endpoint_value = raw.get("endpoint")
        if target_value is not None and endpoint_value is not None:
            raise ConfigError("suite cannot define both 'endpoint' and 'target'")
        if target_value is not None:
            if isinstance(target_value, dict):
                target = TargetSpec.from_mapping(
                    target_value,
                    base_dir=source_path.parent if source_path else Path.cwd(),
                )
            elif isinstance(target_value, str):
                target_path = Path(target_value).expanduser()
                if source_path is not None and not target_path.is_absolute():
                    target_path = source_path.parent / target_path
                target_path = target_path.resolve()
                _resolved_path, target = load_target_spec(target_path)
            else:
                raise ConfigError("suite target must be a mapping or target YAML path")
            endpoint = None
        else:
            if endpoint_value is None:
                if schema_version >= 2:
                    raise ConfigError(
                        "suite schema_version 2 requires exactly one of 'endpoint' or 'target'"
                    )
                endpoint_value = {
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
