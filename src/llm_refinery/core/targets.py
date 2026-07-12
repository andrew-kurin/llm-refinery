from __future__ import annotations

import ipaddress
import math
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from llm_refinery.core.config import ConfigError, load_yaml_mapping, reject_unknown_keys
from llm_refinery.core.endpoints import CHAT_PROTOCOLS, OPENAI_CHAT, Endpoint
from llm_refinery.core.http_safety import PinnedHttpRoute
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
SERVER_INFO_MODES = frozenset({SERVER_INFO_OFF, SERVER_INFO_OPTIONAL, SERVER_INFO_REQUIRED})


def _positive_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{context} must be a positive finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{context} must be a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ConfigError(f"{context} must be a positive finite number")
    return parsed


def _strict_bool(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{context} must be a boolean")
    return value


def _mapping_section(
    raw: dict[str, Any],
    key: str,
    *,
    context: str,
) -> dict[str, Any]:
    if key not in raw:
        return {}
    value = raw[key]
    if not isinstance(value, dict):
        raise ConfigError(f"{context}.{key} must be a mapping")
    return value


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
                f"endpoint spec protocol must be one of {sorted(CHAT_PROTOCOLS)}, got {protocol!r}"
            )
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ConfigError("endpoint spec base_url must be an HTTP(S) URL")
        try:
            hostname = parsed.hostname
            _port = parsed.port
        except ValueError as exc:
            raise ConfigError("endpoint spec base_url must be a valid HTTP(S) URL") from exc
        if hostname is None:
            raise ConfigError("endpoint spec base_url must include a hostname")
        if parsed.username is not None or parsed.password is not None:
            raise ConfigError("endpoint spec base_url cannot include user information")
        if parsed.query or parsed.fragment or "?" in base_url or "#" in base_url:
            raise ConfigError("endpoint spec base_url cannot include a query or fragment")
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
        if "name" in raw:
            name_value = raw["name"]
            if not isinstance(name_value, str) or not name_value.strip():
                raise ConfigError(f"{context}.name must be a non-empty string")
            name = name_value.strip()
        else:
            name = (default_name or "").strip()
        if not name:
            raise ConfigError(f"{context} requires a non-empty 'name'")
        protocol = raw.get("protocol")
        if not isinstance(protocol, str) or not protocol.strip():
            raise ConfigError(f"{context}.protocol must be a non-empty string")
        base_url = raw.get("base_url")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ConfigError(f"{context}.base_url must be a non-empty string")
        model = None
        if "model" in raw:
            model_value = raw["model"]
            if not isinstance(model_value, str) or not model_value.strip():
                raise ConfigError(f"{context}.model must be a non-empty string")
            model = model_value.strip()
        api_key_env = None
        if "api_key_env" in raw:
            api_key_env_value = raw["api_key_env"]
            if not isinstance(api_key_env_value, str) or not api_key_env_value.strip():
                raise ConfigError(f"{context}.api_key_env must be a non-empty string")
            api_key_env = api_key_env_value.strip()
        headers_raw = _mapping_section(raw, "headers", context=context)
        headers: dict[str, str] = {}
        for key, value in headers_raw.items():
            if not isinstance(key, str) or not key.strip():
                raise ConfigError(f"{context}.headers keys must be non-empty strings")
            if not isinstance(value, str):
                raise ConfigError(f"{context}.headers[{key!r}] must be a string")
            if key.casefold() == "authorization" and not value.strip():
                raise ConfigError(f"{context}.headers Authorization cannot be empty")
            headers[key] = value
        return cls(
            name=name,
            protocol=protocol,
            base_url=base_url,
            model=model,
            api_key_env=api_key_env,
            headers=headers,
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
    expected_fingerprint: str | None = None

    def __post_init__(self) -> None:
        access = self.access.strip().lower()
        destination = self.destination.strip() if self.destination else None
        if self.expected_fingerprint is not None and not isinstance(self.expected_fingerprint, str):
            raise ConfigError("host.expected_fingerprint must be a non-empty string")
        expected_fingerprint = (
            self.expected_fingerprint.strip() if self.expected_fingerprint else None
        )
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
        if self.expected_fingerprint is not None and not expected_fingerprint:
            raise ConfigError("host.expected_fingerprint cannot be empty")
        required = _strict_bool(self.required, context="host.required")
        object.__setattr__(self, "access", access)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(self, "expected_fingerprint", expected_fingerprint)
        object.__setattr__(self, "required", required)
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
            "expected_fingerprint": self.expected_fingerprint,
        }

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, context: str = "target.host") -> HostAccess:
        reject_unknown_keys(
            raw,
            {
                "access",
                "destination",
                "connect_timeout_s",
                "command_timeout_s",
                "required",
                "expected_fingerprint",
            },
            context=context,
        )
        expected_fingerprint = None
        if "expected_fingerprint" in raw:
            value = raw["expected_fingerprint"]
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"{context}.expected_fingerprint must be a non-empty string")
            expected_fingerprint = value
        access_value = raw.get("access", HOST_ACCESS_LOCAL)
        if not isinstance(access_value, str) or not access_value.strip():
            raise ConfigError(f"{context}.access must be a non-empty string")
        destination = None
        if "destination" in raw:
            destination_value = raw["destination"]
            if not isinstance(destination_value, str) or not destination_value.strip():
                raise ConfigError(f"{context}.destination must be a non-empty string")
            destination = destination_value
        return cls(
            access=access_value,
            destination=destination,
            connect_timeout_s=raw.get("connect_timeout_s", 5.0),
            command_timeout_s=raw.get("command_timeout_s", 20.0),
            required=_strict_bool(raw.get("required", True), context=f"{context}.required"),
            expected_fingerprint=expected_fingerprint,
        )


@dataclass(frozen=True)
class ModelSelection:
    selection: str = MODEL_SELECTION_SINGLE
    model_id: str | None = None

    def __post_init__(self) -> None:
        selection = self.selection.strip().lower()
        model_id = self.model_id.strip() if self.model_id else None
        if selection not in MODEL_SELECTION_MODES:
            raise ConfigError(
                f"model selection must be one of {sorted(MODEL_SELECTION_MODES)}, got {selection!r}"
            )
        if selection == MODEL_SELECTION_EXPLICIT and not model_id:
            raise ConfigError("explicit model selection requires a non-empty id")
        if selection == MODEL_SELECTION_SINGLE and model_id:
            raise ConfigError("single model selection does not accept an id")
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "model_id", model_id)

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        endpoint_model: str | None = None,
        context: str = "target.model",
    ) -> ModelSelection:
        reject_unknown_keys(raw, {"selection", "id"}, context=context)
        configured_id = None
        if "id" in raw:
            id_value = raw["id"]
            if not isinstance(id_value, str) or not id_value.strip():
                raise ConfigError(f"{context}.id must be a non-empty string")
            configured_id = id_value.strip()
        if configured_id and endpoint_model and configured_id != endpoint_model:
            raise ConfigError(
                f"{context}.id conflicts with target.endpoint.model: "
                f"{configured_id!r} != {endpoint_model!r}"
            )
        model_id = configured_id or endpoint_model
        selection_default = MODEL_SELECTION_EXPLICIT if model_id else MODEL_SELECTION_SINGLE
        selection = raw.get("selection", selection_default)
        if not isinstance(selection, str) or not selection.strip():
            raise ConfigError(f"{context}.selection must be a non-empty string")
        return cls(
            selection=selection,
            model_id=model_id,
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "selection": self.selection,
            "id": self.model_id,
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
        object.__setattr__(
            self,
            "service_required",
            _strict_bool(self.service_required, context="discovery.service_required"),
        )
        object.__setattr__(
            self,
            "metrics",
            _strict_bool(self.metrics, context="discovery.metrics"),
        )

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
        elif not isinstance(server_info_raw, str) or not server_info_raw.strip():
            raise ConfigError(f"{context}.server_info must be a boolean or non-empty string")
        return cls(
            service_required=_strict_bool(
                raw.get("service_required", True),
                context=f"{context}.service_required",
            ),
            server_info=str(server_info_raw),
            metrics=_strict_bool(raw.get("metrics", True), context=f"{context}.metrics"),
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "service_required": self.service_required,
            "server_info": self.server_info,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class TargetTransport:
    """HTTP environment and TLS settings for target discovery requests."""

    trust_env: bool = True
    ca_bundle: Path | None = None

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        base_dir: Path,
        context: str = "target.transport",
    ) -> TargetTransport:
        reject_unknown_keys(raw, {"trust_env", "ca_bundle"}, context=context)
        trust_env = _strict_bool(
            raw.get("trust_env", True),
            context=f"{context}.trust_env",
        )
        ca_bundle: Path | None = None
        if "ca_bundle" in raw:
            value = raw["ca_bundle"]
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"{context}.ca_bundle must be a non-empty path string")
            ca_bundle = Path(value).expanduser()
            if not ca_bundle.is_absolute():
                ca_bundle = base_dir / ca_bundle
            ca_bundle = ca_bundle.resolve()
            if not ca_bundle.is_file():
                raise ConfigError(f"{context}.ca_bundle is not a file: {ca_bundle}")
        return cls(trust_env=trust_env, ca_bundle=ca_bundle)

    def safe_json(self) -> dict[str, Any]:
        return {
            "trust_env": self.trust_env,
            "ca_bundle": str(self.ca_bundle) if self.ca_bundle else None,
        }


@dataclass(frozen=True)
class ModelDescriptor:
    id: str
    root: str | None = None
    max_model_len: int | None = None
    owned_by: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ConfigError("discovered model id must be a non-empty string")
        model_id = self.id.strip()
        root = _optional_metadata_string(self.root, field_name="root", model_id=model_id)
        owned_by = _optional_metadata_string(
            self.owned_by,
            field_name="owned_by",
            model_id=model_id,
        )
        if self.max_model_len is not None and (
            isinstance(self.max_model_len, bool)
            or not isinstance(self.max_model_len, int)
            or self.max_model_len <= 0
        ):
            raise ConfigError(
                f"discovered model {model_id!r} max_model_len must be a positive integer"
            )
        object.__setattr__(self, "id", model_id)
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "owned_by", owned_by)

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
    transport: TargetTransport = field(default_factory=TargetTransport)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigError("target name cannot be empty")
        if self.endpoint.protocol != OPENAI_CHAT:
            raise ConfigError("target discovery currently requires endpoint.protocol 'openai_chat'")
        endpoint_host = urlparse(self.endpoint.base_url).hostname
        if self.host.access == HOST_ACCESS_SSH and _is_client_local_host(endpoint_host):
            raise ConfigError(
                "SSH target endpoint cannot use a loopback or wildcard URL; configure the "
                "client-visible DGX address"
            )

    @classmethod
    def from_mapping(
        cls,
        raw: dict[str, Any],
        *,
        context: str = "target",
        base_dir: Path | None = None,
    ) -> TargetSpec:
        reject_unknown_keys(
            raw,
            {"name", "host", "endpoint", "model", "discovery", "transport"},
            context=context,
        )
        name_value = raw.get("name")
        if not isinstance(name_value, str) or not name_value.strip():
            raise ConfigError(f"{context} requires a non-empty 'name'")
        name = name_value.strip()
        host_raw = _mapping_section(raw, "host", context=context)
        endpoint_raw = _mapping_section(raw, "endpoint", context=context)
        model_raw = _mapping_section(raw, "model", context=context)
        discovery_raw = _mapping_section(raw, "discovery", context=context)
        transport_raw = _mapping_section(raw, "transport", context=context)
        endpoint = EndpointSpec.from_mapping(
            endpoint_raw,
            default_name=name,
            context=f"{context}.endpoint",
        )
        model = ModelSelection.from_mapping(
            model_raw,
            endpoint_model=endpoint.model,
            context=f"{context}.model",
        )
        return cls(
            name=name,
            host=HostAccess.from_mapping(host_raw, context=f"{context}.host"),
            endpoint=endpoint,
            model=model,
            discovery=DiscoveryPolicy.from_mapping(
                discovery_raw,
                context=f"{context}.discovery",
            ),
            transport=TargetTransport.from_mapping(
                transport_raw,
                base_dir=base_dir or Path.cwd(),
                context=f"{context}.transport",
            ),
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "host": self.host.safe_json(),
            "endpoint": self.endpoint.safe_json(),
            "model": self.model.safe_json(),
            "discovery": self.discovery.safe_json(),
            "transport": self.transport.safe_json(),
        }


@dataclass(frozen=True)
class ResolvedTarget:
    spec_name: str
    endpoint: Endpoint
    host: HostDiscovery
    service: ServiceDiscovery
    model: ModelDescriptor
    selection: str
    route: PinnedHttpRoute | None = None
    expected_host_fingerprint: str | None = None

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
            }
        )
        result = {
            "schema_version": 1,
            "name": self.spec_name,
            "host": self.host.safe_json(),
            "service": self.service.safe_json(include_models=False),
            "model": model_json,
            "topology": self.topology,
            "route": self.route.safe_json() if self.route is not None else None,
        }
        binding = _host_identity_binding(
            self.expected_host_fingerprint,
            self.host,
        )
        if binding is not None:
            result["host_identity_binding"] = binding
        return result


@dataclass(frozen=True)
class TargetInspection:
    spec: TargetSpec
    host: HostDiscovery | None
    service: ServiceDiscovery | None
    resolved: ResolvedTarget | None
    route: PinnedHttpRoute | None = None
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
            result["route"] = self.route.safe_json() if self.route is not None else None
            return result
        result = {
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
            "route": self.route.safe_json() if self.route is not None else None,
        }
        binding = _host_identity_binding(
            self.spec.host.expected_fingerprint,
            self.host,
        )
        if binding is not None:
            result["host_identity_binding"] = binding
        return result


def _is_loopback_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    normalized = hostname.casefold().rstrip(".")
    if (
        normalized == "localhost"
        or normalized.startswith("localhost.")
        or normalized.endswith(".localhost")
        or normalized in {"ip6-localhost", "localhost6", "localhost6.localdomain6"}
    ):
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        try:
            packed = socket.inet_aton(normalized)
        except OSError:
            return False
        return ipaddress.ip_address(packed).is_loopback


def _is_client_local_host(hostname: str | None) -> bool:
    if hostname is None or _is_loopback_host(hostname):
        return hostname is not None
    normalized = hostname.casefold().rstrip(".")
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        try:
            address = ipaddress.ip_address(socket.inet_aton(normalized))
        except OSError:
            return False
    return address.is_unspecified


def _host_identity_binding(
    expected_fingerprint: str | None,
    host: HostDiscovery | None,
) -> dict[str, Any] | None:
    if expected_fingerprint is None:
        return None
    actual = host.profile.get("host_fingerprint") if host is not None else None
    return {
        "expected_fingerprint": expected_fingerprint,
        "actual_fingerprint": actual,
        "verified": actual == expected_fingerprint,
    }


def _measurement_scope(host_access: str, base_url: str) -> str:
    if host_access == HOST_ACCESS_SSH:
        return "remote_client_to_server"
    if _is_client_local_host(urlparse(base_url).hostname):
        return "local_loopback"
    return "local_client_to_network_endpoint"


def _optional_metadata_string(
    value: Any,
    *,
    field_name: str,
    model_id: str,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"discovered model {model_id!r} {field_name} must be a non-empty string")
    return value.strip()


def load_target_spec(path: str | Path) -> tuple[Path, TargetSpec]:
    config_path, raw = load_yaml_mapping(path)
    if "target" in raw:
        reject_unknown_keys(raw, {"schema_version", "target"}, context=str(config_path))
        schema_version = raw.get("schema_version", 1)
        if isinstance(schema_version, bool) or schema_version != 1:
            raise ConfigError(f"{config_path} schema_version must be 1, got {schema_version!r}")
        target_raw = raw["target"]
        if not isinstance(target_raw, dict):
            raise ConfigError(f"{config_path} target must be a mapping")
    else:
        target_raw = raw
    return config_path, TargetSpec.from_mapping(
        target_raw,
        base_dir=config_path.parent,
    )


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
    "TargetTransport",
    "load_target_spec",
]
