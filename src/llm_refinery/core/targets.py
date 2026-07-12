from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from llm_refinery.core.config import ConfigError, load_yaml_mapping, reject_unknown_keys
from llm_refinery.core.endpoints import CHAT_PROTOCOLS, OPENAI_CHAT, Endpoint
from llm_refinery.core.runs import stable_hash

HOST_ACCESS_LOCAL = "local"
HOST_ACCESS_SSH = "ssh"
HOST_ACCESS_MODES = frozenset({HOST_ACCESS_LOCAL, HOST_ACCESS_SSH})
MODEL_SELECTION_EXPLICIT = "explicit"
MODEL_SELECTION_SINGLE = "single"
MODEL_SELECTION_MODES = frozenset({MODEL_SELECTION_EXPLICIT, MODEL_SELECTION_SINGLE})
SERVER_INFO_OFF = "off"
SERVER_INFO_OPTIONAL = "optional"
SERVER_INFO_REQUIRED = "required"
SERVER_INFO_MODES = frozenset(
    {SERVER_INFO_OFF, SERVER_INFO_OPTIONAL, SERVER_INFO_REQUIRED}
)


def _positive_float(value: Any, *, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context} must be a positive number") from exc
    if parsed <= 0:
        raise ConfigError(f"{context} must be a positive number")
    return parsed


@dataclass(frozen=True)
class EndpointSpec:
    """An endpoint that may not have a concrete served model yet."""

    name: str
    protocol: str
    base_url: str
    model: str | None = None
    api_key_env: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name.strip()
        protocol = self.protocol.strip().lower()
        base_url = self.base_url.strip().rstrip("/")
        model = self.model.strip() if self.model else None
        if not name:
            raise ConfigError("endpoint spec name cannot be empty")
        if protocol not in CHAT_PROTOCOLS:
            raise ConfigError(
                f"endpoint spec protocol must be one of {sorted(CHAT_PROTOCOLS)}, "
                f"got {protocol!r}"
            )
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigError("endpoint spec base_url must be an HTTP(S) URL")
        if (
            protocol == OPENAI_CHAT
            and not base_url.endswith("/v1")
            and not base_url.endswith("/v1/chat/completions")
        ):
            base_url = f"{base_url}/v1"
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "headers", dict(self.headers))

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        default_name: str | None = None,
        context: str = "target.endpoint",
    ) -> EndpointSpec:
        reject_unknown_keys(
            raw,
            {"name", "protocol", "base_url", "model", "api_key_env", "headers"},
            context=context,
        )
        name = str(raw.get("name") or default_name or "").strip()
        if not name:
            raise ConfigError(f"{context} requires a non-empty 'name'")
        headers_raw = raw.get("headers") or {}
        if not isinstance(headers_raw, dict):
            raise ConfigError(f"{context} headers must be a mapping")
        return cls(
            name=name,
            protocol=str(raw.get("protocol") or "").strip(),
            base_url=str(raw.get("base_url") or "").strip(),
            model=str(raw["model"]).strip() if raw.get("model") else None,
            api_key_env=str(raw["api_key_env"]).strip() if raw.get("api_key_env") else None,
            headers={str(key): str(value) for key, value in headers_raw.items()},
        )

    @property
    def api_root_url(self) -> str:
        suffix = "/chat/completions"
        if self.base_url.endswith(suffix):
            return self.base_url[: -len(suffix)]
        return self.base_url

    @property
    def server_root_url(self) -> str:
        api_root = self.api_root_url
        if api_root.endswith("/v1"):
            return api_root[:-3]
        return api_root

    def resolve(self, model: str) -> Endpoint:
        return Endpoint(
            name=self.name,
            protocol=self.protocol,
            base_url=self.api_root_url,
            model=model,
            api_key_env=self.api_key_env,
            headers=self.headers,
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "protocol": self.protocol,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "header_names": sorted(self.headers),
            "headers_hash": stable_hash(self.headers) if self.headers else None,
        }


@dataclass(frozen=True)
class HostAccess:
    access: str = HOST_ACCESS_LOCAL
    destination: str | None = None
    connect_timeout_s: float = 5.0
    command_timeout_s: float = 20.0
    required: bool = True

    def __post_init__(self) -> None:
        access = self.access.strip().lower()
        destination = self.destination.strip() if self.destination else None
        if access not in HOST_ACCESS_MODES:
            raise ConfigError(
                f"host access must be one of {sorted(HOST_ACCESS_MODES)}, got {access!r}"
            )
        if access == HOST_ACCESS_SSH and not destination:
            raise ConfigError("SSH host access requires a non-empty destination")
        if access == HOST_ACCESS_LOCAL and destination:
            raise ConfigError("local host access does not accept a destination")
        if destination and (
            destination.startswith("-")
            or any(
                character.isspace() or ord(character) < 32 or ord(character) == 127
                for character in destination
            )
        ):
            raise ConfigError(
                "SSH destination cannot start with '-' or contain whitespace/control characters"
            )
        object.__setattr__(self, "access", access)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(
            self,
            "connect_timeout_s",
            _positive_float(self.connect_timeout_s, context="host.connect_timeout_s"),
        )
        object.__setattr__(
            self,
            "command_timeout_s",
            _positive_float(self.command_timeout_s, context="host.command_timeout_s"),
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "access": self.access,
            "destination": self.destination,
            "connect_timeout_s": self.connect_timeout_s,
            "command_timeout_s": self.command_timeout_s,
            "required": self.required,
        }

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, context: str = "target.host") -> HostAccess:
        reject_unknown_keys(
            raw,
            {"access", "destination", "connect_timeout_s", "command_timeout_s", "required"},
            context=context,
        )
        return cls(
            access=str(raw.get("access") or HOST_ACCESS_LOCAL),
            destination=str(raw["destination"]) if raw.get("destination") else None,
            connect_timeout_s=raw.get("connect_timeout_s", 5.0),
            command_timeout_s=raw.get("command_timeout_s", 20.0),
            required=bool(raw.get("required", True)),
        )


@dataclass(frozen=True)
class ModelSelection:
    selection: str = MODEL_SELECTION_SINGLE
    model_id: str | None = None
    tokenizer: str | None = None

    def __post_init__(self) -> None:
        selection = self.selection.strip().lower()
        model_id = self.model_id.strip() if self.model_id else None
        tokenizer = self.tokenizer.strip() if self.tokenizer else None
        if selection not in MODEL_SELECTION_MODES:
            raise ConfigError(
                f"model selection must be one of {sorted(MODEL_SELECTION_MODES)}, "
                f"got {selection!r}"
            )
        if selection == MODEL_SELECTION_EXPLICIT and not model_id:
            raise ConfigError("explicit model selection requires a non-empty id")
        if selection == MODEL_SELECTION_SINGLE and model_id:
            raise ConfigError("single model selection does not accept an id")
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "tokenizer", tokenizer)

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        endpoint_model: str | None = None,
        context: str = "target.model",
    ) -> ModelSelection:
        reject_unknown_keys(raw, {"selection", "id", "tokenizer"}, context=context)
        configured_id = str(raw["id"]).strip() if raw.get("id") else None
        if configured_id and endpoint_model and configured_id != endpoint_model:
            raise ConfigError(
                f"{context}.id conflicts with target.endpoint.model: "
                f"{configured_id!r} != {endpoint_model!r}"
            )
        model_id = configured_id or endpoint_model
        selection_default = MODEL_SELECTION_EXPLICIT if model_id else MODEL_SELECTION_SINGLE
        return cls(
            selection=str(raw.get("selection") or selection_default),
            model_id=model_id,
            tokenizer=str(raw["tokenizer"]).strip() if raw.get("tokenizer") else None,
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "selection": self.selection,
            "id": self.model_id,
            "tokenizer": self.tokenizer,
        }


@dataclass(frozen=True)
class DiscoveryPolicy:
    service_required: bool = True
    server_info: str = SERVER_INFO_OPTIONAL
    metrics: bool = True

    def __post_init__(self) -> None:
        server_info = self.server_info.strip().lower()
        if server_info not in SERVER_INFO_MODES:
            raise ConfigError(
                f"discovery.server_info must be one of {sorted(SERVER_INFO_MODES)}, "
                f"got {server_info!r}"
            )
        object.__setattr__(self, "server_info", server_info)

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        context: str = "target.discovery",
    ) -> DiscoveryPolicy:
        reject_unknown_keys(
            raw,
            {"service_required", "server_info", "metrics"},
            context=context,
        )
        server_info_raw = raw.get("server_info", SERVER_INFO_OPTIONAL)
        if isinstance(server_info_raw, bool):
            server_info_raw = SERVER_INFO_OPTIONAL if server_info_raw else SERVER_INFO_OFF
        return cls(
            service_required=bool(raw.get("service_required", True)),
            server_info=str(server_info_raw),
            metrics=bool(raw.get("metrics", True)),
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "service_required": self.service_required,
            "server_info": self.server_info,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class ModelDescriptor:
    id: str
    root: str | None = None
    max_model_len: int | None = None
    owned_by: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ConfigError("discovered model id cannot be empty")

    def safe_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "root": self.root,
            "max_model_len": self.max_model_len,
            "owned_by": self.owned_by,
        }


@dataclass(frozen=True)
class HostDiscovery:
    transport: str
    destination: str | None
    profile: dict[str, Any]

    def safe_json(self) -> dict[str, Any]:
        return {
            "transport": self.transport,
            "destination": self.destination,
            "profile": self.profile,
        }


@dataclass(frozen=True)
class ServiceDiscovery:
    implementation: str
    base_url: str
    health: str
    version: str | None
    models: tuple[ModelDescriptor, ...]
    server_info: dict[str, Any] | None = None
    errors: tuple[str, ...] = ()

    def safe_json(self, *, include_models: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "implementation": self.implementation,
            "base_url": self.base_url,
            "health": self.health,
            "version": self.version,
            "errors": list(self.errors),
        }
        if include_models:
            result["models"] = [model.safe_json() for model in self.models]
        if self.server_info is not None:
            result["server_info"] = self.server_info
        return result


@dataclass(frozen=True)
class TargetSpec:
    name: str
    host: HostAccess
    endpoint: EndpointSpec
    model: ModelSelection
    discovery: DiscoveryPolicy = field(default_factory=DiscoveryPolicy)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigError("target name cannot be empty")
        endpoint_host = urlparse(self.endpoint.base_url).hostname
        if self.host.access == HOST_ACCESS_SSH and _is_loopback_host(endpoint_host):
            raise ConfigError(
                "SSH target endpoint cannot use a loopback URL; configure the "
                "client-visible DGX address"
            )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, context: str = "target") -> TargetSpec:
        reject_unknown_keys(
            raw,
            {"name", "host", "endpoint", "model", "discovery"},
            context=context,
        )
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ConfigError(f"{context} requires a non-empty 'name'")
        for key in ("host", "endpoint", "model", "discovery"):
            value = raw.get(key) or {}
            if not isinstance(value, dict):
                raise ConfigError(f"{context}.{key} must be a mapping")
        endpoint = EndpointSpec.from_mapping(raw.get("endpoint") or {}, default_name=name)
        model = ModelSelection.from_mapping(
            raw.get("model") or {}, endpoint_model=endpoint.model
        )
        return cls(
            name=name,
            host=HostAccess.from_mapping(raw.get("host") or {}),
            endpoint=endpoint,
            model=model,
            discovery=DiscoveryPolicy.from_mapping(raw.get("discovery") or {}),
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "host": self.host.safe_json(),
            "endpoint": self.endpoint.safe_json(),
            "model": self.model.safe_json(),
            "discovery": self.discovery.safe_json(),
        }

@dataclass(frozen=True)
class ResolvedTarget:
    spec_name: str
    endpoint: Endpoint
    host: HostDiscovery
    service: ServiceDiscovery
    model: ModelDescriptor
    selection: str
    tokenizer: str | None = None

    @property
    def topology(self) -> dict[str, str]:
        return {
            "measurement_scope": _measurement_scope(
                self.host.transport,
                self.endpoint.base_url,
            )
        }

    def safe_json(self) -> dict[str, Any]:
        model_json = self.model.safe_json()
        model_json.update(
            {
                "requested_id": self.endpoint.model,
                "selection": self.selection,
                "tokenizer": self.tokenizer,
            }
        )
        return {
            "schema_version": 1,
            "name": self.spec_name,
            "host": self.host.safe_json(),
            "service": self.service.safe_json(include_models=False),
            "model": model_json,
            "topology": self.topology,
        }


@dataclass(frozen=True)
class TargetInspection:
    spec: TargetSpec
    host: HostDiscovery | None
    service: ServiceDiscovery | None
    resolved: ResolvedTarget | None
    errors: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.resolved is not None

    def safe_json(self) -> dict[str, Any]:
        if self.resolved is not None:
            result = self.resolved.safe_json()
            if self.service is not None:
                result["service"] = self.service.safe_json(include_models=True)
            result["status"] = "available"
            result["errors"] = list(self.errors)
            return result
        return {
            "schema_version": 1,
            "name": self.spec.name,
            "status": "unavailable",
            "host": self.host.safe_json() if self.host else None,
            "service": self.service.safe_json() if self.service else None,
            "model": None,
            "topology": {
                "measurement_scope": _measurement_scope(
                    self.spec.host.access,
                    self.spec.endpoint.base_url,
                )
            },
            "errors": list(self.errors),
        }


def _is_loopback_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _measurement_scope(host_access: str, base_url: str) -> str:
    if host_access == HOST_ACCESS_SSH:
        return "remote_client_to_server"
    if _is_loopback_host(urlparse(base_url).hostname):
        return "local_loopback"
    return "local_client_to_network_endpoint"


def load_target_spec(path: str | Path) -> tuple[Path, TargetSpec]:
    config_path, raw = load_yaml_mapping(path)
    if "target" in raw:
        reject_unknown_keys(raw, {"schema_version", "target"}, context=str(config_path))
        schema_version = raw.get("schema_version", 1)
        if schema_version != 1:
            raise ConfigError(f"{config_path} schema_version must be 1, got {schema_version!r}")
        target_raw = raw["target"]
        if not isinstance(target_raw, dict):
            raise ConfigError(f"{config_path} target must be a mapping")
    else:
        target_raw = raw
    return config_path, TargetSpec.from_mapping(target_raw)


__all__ = [
    "DiscoveryPolicy",
    "EndpointSpec",
    "HostAccess",
    "HostDiscovery",
    "ModelDescriptor",
    "ModelSelection",
    "ResolvedTarget",
    "ServiceDiscovery",
    "TargetInspection",
    "TargetSpec",
    "load_target_spec",
]
