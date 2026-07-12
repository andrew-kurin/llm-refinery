from __future__ import annotations

import os
import re
from typing import Any

import httpx

from llm_refinery.core.endpoints import OPENAI_CHAT
from llm_refinery.core.targets import (
    SERVER_INFO_OFF,
    DiscoveryPolicy,
    EndpointSpec,
    ModelDescriptor,
    ServiceDiscovery,
)
from llm_refinery.providers.openai_chat import has_header, json_headers

MAX_RESPONSE_BYTES = 2_000_000
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "token",
        "access_token",
        "auth_token",
        "bearer_token",
        "client_secret",
        "client_secret_key",
        "credential",
        "credentials",
        "hf_token",
        "private_key",
    }
)
_SECRET_SUFFIXES = (
    "_api_key",
    "_password",
    "_secret",
    "_access_token",
    "_auth_token",
    "_credential",
    "_credentials",
    "_private_key",
)


class OpenAIDiscoveryClient:
    """Read-only discovery for a vLLM/OpenAI-compatible serving endpoint."""

    def __init__(
        self,
        *,
        client: httpx.Client | None = None,
        timeout_s: float = 5.0,
    ) -> None:
        self._client = client
        self._timeout_s = timeout_s

    def discover(
        self,
        endpoint: EndpointSpec,
        policy: DiscoveryPolicy,
    ) -> ServiceDiscovery:
        if endpoint.protocol != OPENAI_CHAT:
            raise RuntimeError(
                "target model discovery currently requires protocol 'openai_chat'"
            )
        errors: list[str] = []
        headers = _headers(endpoint)
        owns_client = self._client is None
        client = self._client or httpx.Client(
            timeout=self._timeout_s,
            follow_redirects=True,
            trust_env=False,
        )
        health = "unavailable"
        version: str | None = None
        models: tuple[ModelDescriptor, ...] = ()
        server_info: dict[str, Any] | None = None
        health_transport_failed = False
        try:
            try:
                response = client.get(f"{endpoint.server_root_url}/health", headers=headers)
                response.raise_for_status()
                health = "ok"
            except (httpx.HTTPError, ValueError) as exc:
                errors.append(f"health: {_error_text(exc)}")
                health_transport_failed = isinstance(exc, httpx.TransportError)

            if not health_transport_failed:
                try:
                    response = client.get(f"{endpoint.server_root_url}/version", headers=headers)
                    response.raise_for_status()
                    payload = _json_object(response)
                    version_value = payload.get("version") or payload.get("vllm_version")
                    version = str(version_value) if version_value is not None else None
                    if version is None:
                        errors.append("version: response did not include a version")
                except (httpx.HTTPError, ValueError) as exc:
                    errors.append(f"version: {_error_text(exc)}")

                try:
                    response = client.get(_models_url(endpoint), headers=headers)
                    response.raise_for_status()
                    models = _models_from_response(response)
                except (httpx.HTTPError, ValueError) as exc:
                    errors.append(f"models: {_error_text(exc)}")

                if policy.server_info != SERVER_INFO_OFF:
                    try:
                        response = client.get(
                            f"{endpoint.server_root_url}/server_info",
                            params={"config_format": "json"},
                            headers=headers,
                        )
                        response.raise_for_status()
                        server_info = _sanitize(_json_object(response))
                    except (httpx.HTTPError, ValueError) as exc:
                        errors.append(f"server_info: {_error_text(exc)}")
        finally:
            if owns_client:
                client.close()

        return ServiceDiscovery(
            implementation="vllm",
            base_url=endpoint.api_root_url,
            health=health,
            version=version,
            models=models,
            server_info=server_info,
            errors=tuple(errors),
        )

    def metrics(self, endpoint: EndpointSpec) -> str:
        """Read the Prometheus snapshot without changing server state."""
        headers = _headers(endpoint)
        owns_client = self._client is None
        client = self._client or httpx.Client(
            timeout=self._timeout_s,
            follow_redirects=True,
            trust_env=False,
        )
        try:
            response = client.get(f"{endpoint.server_root_url}/metrics", headers=headers)
            response.raise_for_status()
            if len(response.content) > MAX_RESPONSE_BYTES:
                raise RuntimeError("metrics response is too large")
            return response.text
        except httpx.HTTPError as exc:
            raise RuntimeError(f"could not read vLLM metrics: {_error_text(exc)}") from exc
        finally:
            if owns_client:
                client.close()


def _models_url(endpoint: EndpointSpec) -> str:
    api_root = endpoint.api_root_url
    return f"{api_root}/models" if api_root.endswith("/v1") else f"{api_root}/v1/models"


def _headers(endpoint: EndpointSpec) -> dict[str, str]:
    headers = json_headers(
        endpoint.headers,
        api_key_env=endpoint.api_key_env,
        accept=True,
    )
    if (
        endpoint.api_key_env
        and not has_header(headers, "Authorization")
        and not os.environ.get(endpoint.api_key_env)
    ):
        raise RuntimeError(
            f"endpoint API key environment variable is not set: {endpoint.api_key_env}"
        )
    return headers


def _models_from_response(response: httpx.Response) -> tuple[ModelDescriptor, ...]:
    payload = _json_object(response)
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("response field 'data' must be a list")
    models: list[ModelDescriptor] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"model entry {index} must be an object")
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            raise ValueError(f"model entry {index} has no id")
        max_model_len = item.get("max_model_len")
        try:
            parsed_max_len = int(max_model_len) if max_model_len is not None else None
        except (TypeError, ValueError) as exc:
            raise ValueError(f"model {model_id!r} has invalid max_model_len") from exc
        models.append(
            ModelDescriptor(
                id=model_id,
                root=str(item["root"]) if item.get("root") is not None else None,
                max_model_len=parsed_max_len,
                owned_by=(
                    str(item["owned_by"]) if item.get("owned_by") is not None else None
                ),
            )
        )
    return tuple(models)


def _json_object(response: httpx.Response) -> dict[str, Any]:
    content_length = response.headers.get("content-length")
    if content_length:
        try:
            parsed_content_length = int(content_length)
        except ValueError:
            parsed_content_length = None
        if parsed_content_length is not None and parsed_content_length > MAX_RESPONSE_BYTES:
            raise ValueError("response is too large")
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise ValueError("response is too large")
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("response JSON must be an object")
    return payload


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize(child)
            for key, child in value.items()
            if not _is_secret_key(str(key))
        }
    if isinstance(value, list):
        return [_sanitize(child) for child in value]
    return value


def _is_secret_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    return normalized in _SECRET_KEYS or normalized.endswith(_SECRET_SUFFIXES)


def _error_text(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return str(exc) or type(exc).__name__


__all__ = ["OpenAIDiscoveryClient"]
