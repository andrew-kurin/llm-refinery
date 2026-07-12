from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

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
        base_url = self.base_url.strip().rstrip("/")
        model = self.model.strip()
        if not name or not protocol or not model:
            raise ConfigError("endpoint name, protocol, and model cannot be empty")
        parsed_url = urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ConfigError("endpoint base_url must be an HTTP(S) URL")
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

        base_url = str(raw.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            raise ConfigError(f"{context} {name!r} requires 'base_url'")
        parsed_url = urlparse(base_url)
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ConfigError(f"{context} {name!r} base_url must be an HTTP(S) URL")
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
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    @property
    def completions_url(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url[: -len("/chat/completions")] + "/completions"
        if self.base_url.endswith("/completions"):
            return self.base_url
        return f"{self.base_url}/completions"

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
