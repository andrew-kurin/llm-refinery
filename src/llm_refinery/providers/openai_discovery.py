from __future__ import annotations

import os
import re
import time
from typing import Any

import httpx

from llm_refinery.core.config import ConfigError
from llm_refinery.core.endpoints import OPENAI_CHAT
from llm_refinery.core.http_safety import http_origin, validate_request_url
from llm_refinery.core.targets import (
    SERVER_INFO_OFF,
    DiscoveryPolicy,
    EndpointSpec,
    ModelDescriptor,
    ServiceDiscovery,
)
from llm_refinery.providers.openai_chat import has_header, json_headers

MAX_RESPONSE_BYTES = 2_000_000
_monotonic = time.monotonic
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
    "_token",
)
_MAX_REDIRECTS = 5


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
            raise RuntimeError("target model discovery currently requires protocol 'openai_chat'")
        errors: list[str] = []
        headers = _headers(endpoint)
        owns_client = self._client is None
        client = self._client or httpx.Client(
            timeout=self._timeout_s,
            follow_redirects=False,
            trust_env=False,
        )
        health = "unavailable"
        version: str | None = None
        models: tuple[ModelDescriptor, ...] = ()
        server_info: dict[str, Any] | None = None
        health_transport_failed = False
        try:
            try:
                _get_bounded_response(
                    client,
                    f"{endpoint.server_root_url}/health",
                    headers=headers,
                    timeout_s=self._timeout_s,
                )
                health = "ok"
            except (httpx.HTTPError, ValueError) as exc:
                _raise_critical_discovery_error(exc, operation="health")
                errors.append(f"health: {_error_text(exc)}")
                health_transport_failed = isinstance(exc, httpx.TransportError)

            if not health_transport_failed:
                try:
                    response = _get_bounded_response(
                        client,
                        f"{endpoint.server_root_url}/version",
                        headers=headers,
                        timeout_s=self._timeout_s,
                    )
                    payload = _json_object(response)
                    version_value = payload.get("version") or payload.get("vllm_version")
                    version = str(version_value) if version_value is not None else None
                    if version is None:
                        errors.append("version: response did not include a version")
                except (httpx.HTTPError, ValueError) as exc:
                    errors.append(f"version: {_error_text(exc)}")

                try:
                    response = _get_bounded_response(
                        client,
                        _models_url(endpoint),
                        headers=headers,
                        timeout_s=self._timeout_s,
                    )
                    models = _models_from_response(response)
                except (httpx.HTTPError, ValueError) as exc:
                    _raise_critical_discovery_error(exc, operation="model discovery")
                    errors.append(f"models: {_error_text(exc)}")

                if policy.server_info != SERVER_INFO_OFF:
                    try:
                        response = _get_bounded_response(
                            client,
                            f"{endpoint.server_root_url}/server_info",
                            params={"config_format": "json"},
                            headers=headers,
                            timeout_s=self._timeout_s,
                        )
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
            follow_redirects=False,
            trust_env=False,
        )
        try:
            response = _get_bounded_response(
                client,
                f"{endpoint.server_root_url}/metrics",
                headers=headers,
                timeout_s=self._timeout_s,
            )
            return response.text
        except (httpx.HTTPError, ValueError) as exc:
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
        raise ConfigError(
            f"endpoint API key environment variable is not set: {endpoint.api_key_env}"
        )
    return headers


def _get_bounded_response(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, str] | None = None,
    timeout_s: float,
) -> httpx.Response:
    """Stream a small discovery response and stop as soon as its limit is exceeded."""
    expected_origin = http_origin(url)
    deadline = _monotonic() + timeout_s
    # HTTPX read timeouts reset after each chunk. A shorter per-read slice plus the
    # monotonic deadline prevents an endless trickle from keeping discovery alive.
    read_timeout_s = min(timeout_s, max(0.1, timeout_s / 4))
    timeout = httpx.Timeout(timeout_s, read=read_timeout_s)
    current_url = url
    current_params = params
    for _redirect_count in range(_MAX_REDIRECTS + 1):
        validate_request_url(
            current_url,
            expected_origin=expected_origin,
            resolve_addresses=False,
        )
        with client.stream(
            "GET",
            current_url,
            headers=headers,
            params=current_params,
            timeout=timeout,
            follow_redirects=False,
        ) as response:
            _check_deadline(deadline, response.request)
            if response.has_redirect_location:
                redirect_url = str(response.url.join(response.headers["location"]))
                validate_request_url(
                    redirect_url,
                    expected_origin=expected_origin,
                    resolve_addresses=False,
                )
                current_url = redirect_url
                current_params = None
                continue
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = None
                if declared_size is not None and declared_size > MAX_RESPONSE_BYTES:
                    raise ValueError("response is too large")
            content = bytearray()
            for chunk in response.iter_bytes():
                _check_deadline(deadline, response.request)
                if len(content) + len(chunk) > MAX_RESPONSE_BYTES:
                    raise ValueError("response is too large")
                content.extend(chunk)
            # iter_bytes() has already decoded transfer/content encodings. Do not retain
            # headers that would make the in-memory response decode the bytes a second time.
            bounded_headers = [
                (key, value)
                for key, value in response.headers.multi_items()
                if key.casefold()
                not in {"content-encoding", "content-length", "transfer-encoding"}
            ]
            return httpx.Response(
                status_code=response.status_code,
                headers=bounded_headers,
                content=bytes(content),
                request=response.request,
            )
    raise ConfigError("vLLM discovery exceeded the maximum same-origin redirects")


def _check_deadline(deadline: float, request: httpx.Request) -> None:
    if _monotonic() > deadline:
        raise httpx.ReadTimeout(
            "discovery response exceeded the total timeout",
            request=request,
        )


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
                owned_by=(str(item["owned_by"]) if item.get("owned_by") is not None else None),
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
    if isinstance(exc, httpx.LocalProtocolError):
        return "invalid HTTP request or response"
    return str(exc) or type(exc).__name__


def _raise_critical_discovery_error(exc: Exception, *, operation: str) -> None:
    if isinstance(exc, ConfigError):
        raise exc
    if isinstance(exc, httpx.HTTPStatusError) and 300 <= exc.response.status_code < 500:
        if exc.response.status_code in {401, 403}:
            raise ConfigError(
                f"vLLM {operation} authorization failed with HTTP "
                f"{exc.response.status_code}"
            ) from exc
        raise ConfigError(
            f"vLLM {operation} failed with HTTP {exc.response.status_code}"
        ) from exc
    if isinstance(exc, ValueError) and not isinstance(exc, httpx.HTTPError):
        raise ConfigError(f"vLLM {operation} returned an invalid response: {exc}") from exc


__all__ = ["OpenAIDiscoveryClient"]
