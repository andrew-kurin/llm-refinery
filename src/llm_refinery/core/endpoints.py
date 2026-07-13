from __future__ import annotations

import ipaddress
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

from llm_refinery.core.config import ConfigError, reject_unknown_keys
from llm_refinery.core.runs import stable_hash

OPENAI_CHAT = "openai_chat"
OLLAMA_CHAT = "ollama_chat"
CHAT_PROTOCOLS = frozenset({OPENAI_CHAT, OLLAMA_CHAT})


@dataclass(frozen=True)
class Endpoint:
    name: str
    protocol: str
    base_url: str
    model: str
    api_key_env: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = self.name.strip()
        protocol = self.protocol.strip().lower()
        base_url = normalize_base_url(self.base_url, context="endpoint base_url")
        model = self.model.strip()
        if not name or not protocol or not model:
            raise ConfigError("endpoint name, protocol, and model cannot be empty")
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
        context: str = "endpoint",
        allowed_protocols: frozenset[str] | None = None,
    ) -> Endpoint:
        reject_unknown_keys(
            raw,
            {"name", "protocol", "base_url", "model", "api_key_env", "headers"},
            context=context,
        )
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ConfigError(f"{context} requires a non-empty 'name'")

        protocol = str(raw.get("protocol") or "").strip().lower()
        if not protocol:
            raise ConfigError(f"{context} {name!r} requires 'protocol'")
        if allowed_protocols is not None and protocol not in allowed_protocols:
            raise ConfigError(
                f"{context} {name!r} protocol must be one of "
                f"{sorted(allowed_protocols)}, got {protocol!r}"
            )

        base_url = str(raw.get("base_url") or "").strip()
        if not base_url:
            raise ConfigError(f"{context} {name!r} requires 'base_url'")
        base_url = normalize_base_url(
            base_url,
            context=f"{context} {name!r} base_url",
        )
        model = str(raw.get("model") or "").strip()
        if not model:
            raise ConfigError(f"{context} {name!r} requires 'model'")
        headers_raw = raw.get("headers") or {}
        if not isinstance(headers_raw, dict):
            raise ConfigError(f"{context} {name!r} headers must be a mapping")

        return cls(
            name=name,
            protocol=protocol,
            base_url=base_url,
            model=model,
            api_key_env=str(raw["api_key_env"]) if raw.get("api_key_env") else None,
            headers={str(key): str(value) for key, value in headers_raw.items()},
        )

    @property
    def chat_completions_url(self) -> str:
        return _replace_or_append_path(
            self.base_url,
            existing_suffix="/chat/completions",
            replacement_suffix="/chat/completions",
        )

    @property
    def completions_url(self) -> str:
        return _replace_or_append_path(
            self.base_url,
            existing_suffix="/chat/completions",
            replacement_suffix="/completions",
            alternate_existing_suffix="/completions",
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


def normalize_base_url(value: str, *, context: str) -> str:
    """Validate and normalize a credential-free HTTP endpoint base URL."""
    base_url = value.strip().rstrip("/")
    if not base_url or any(character.isspace() or ord(character) < 32 for character in base_url):
        raise ConfigError(f"{context} must be an HTTP(S) URL without whitespace")
    if "\\" in base_url:
        raise ConfigError(f"{context} cannot include backslashes")

    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError(f"{context} must be an HTTP(S) URL")
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ConfigError(f"{context} must include a valid hostname and port") from exc
    if hostname is None:
        raise ConfigError(f"{context} must include a hostname")
    try:
        explicit_address = ipaddress.ip_address(hostname)
    except ValueError:
        explicit_address = None
    if explicit_address is not None and explicit_address.is_unspecified:
        raise ConfigError(f"{context} cannot use a wildcard address")
    if port == 0 or parsed.netloc.endswith(":"):
        raise ConfigError(f"{context} must include a valid hostname and port")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigError(f"{context} cannot include user information")
    # Check the delimiters as well as the parsed fields so an empty query or
    # fragment (for example, ``...?``) cannot silently survive normalization.
    if parsed.query or parsed.fragment or "?" in base_url or "#" in base_url:
        raise ConfigError(f"{context} cannot include a query or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _replace_or_append_path(
    base_url: str,
    *,
    existing_suffix: str,
    replacement_suffix: str,
    alternate_existing_suffix: str | None = None,
) -> str:
    """Build an API URL by operating on its path, never its authority."""
    parsed = urlsplit(base_url)
    path = parsed.path
    if path.endswith(existing_suffix):
        path = path[: -len(existing_suffix)] + replacement_suffix
    elif alternate_existing_suffix is None or not path.endswith(alternate_existing_suffix):
        path = f"{path.rstrip('/')}{replacement_suffix}"
    result = SplitResult(parsed.scheme, parsed.netloc, path, "", "")
    return urlunsplit(result)
