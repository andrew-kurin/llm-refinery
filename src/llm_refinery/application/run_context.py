from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from llm_refinery.core.runs import stable_hash
from llm_refinery.utils.system import host_identity as system_host_identity


@dataclass(frozen=True, slots=True)
class RunContext:
    """Environment metadata shared by all runs in one benchmark execution.

    ``executor_system_json`` describes the machine running the harness, while
    ``target_json`` describes the system and service being measured.  Keeping the
    two documents separate prevents a remote DGX from being mistaken for the
    client that generated the requests.
    """

    target_json: Mapping[str, Any] = field(default_factory=dict)
    executor_system_json: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_json", deepcopy(dict(self.target_json)))
        if self.executor_system_json is not None:
            object.__setattr__(
                self,
                "executor_system_json",
                deepcopy(dict(self.executor_system_json)),
            )

    def to_target_json(self) -> dict[str, Any]:
        """Return an independent document suitable for durable storage."""
        return deepcopy(dict(self.target_json))

    def to_executor_system_json(self) -> dict[str, Any] | None:
        """Return an independent executor profile, when one was supplied."""
        if self.executor_system_json is None:
            return None
        return deepcopy(dict(self.executor_system_json))

    def with_target_json(self, target_json: Mapping[str, Any]) -> RunContext:
        """Return a context with updated target discovery and the same executor."""
        return RunContext(
            target_json=target_json,
            executor_system_json=self.executor_system_json,
        )

    def target_identity_json(self) -> dict[str, Any]:
        """Return stable serving identity suitable for a child RunSpec hash."""
        target = self.to_target_json()
        if not target:
            return {}
        host = target.get("host") or {}
        profile: dict[str, Any] = {}
        if isinstance(host, dict):
            for key in ("profile", "inventory", "system_json"):
                candidate = host.get(key)
                if isinstance(candidate, dict):
                    profile = candidate
                    break
            if not profile:
                # Historical target rows stored inventory fields directly under
                # host. Preserve their identity when comparing old and new runs.
                profile = host
        service = target.get("service") or {}
        service = service if isinstance(service, dict) else {}
        route = target.get("route") or {}
        route = route if isinstance(route, dict) else {}
        logical_origin = route.get("logical_origin") or {}
        logical_origin = logical_origin if isinstance(logical_origin, dict) else {}
        server_info = service.get("server_info")
        host_fingerprint = system_host_identity(profile)
        if host_fingerprint == "unknown-host" and isinstance(host, dict):
            explicit_fingerprint = host.get("host_fingerprint") or host.get("fingerprint")
            if explicit_fingerprint:
                host_fingerprint = str(explicit_fingerprint)
        host_identity = (
            {"fingerprint": host_fingerprint}
            if host_fingerprint != "unknown-host"
            else {
                "hostname": profile.get("hostname"),
                "destination": host.get("destination") if isinstance(host, dict) else None,
            }
        )
        return {
            "schema_version": target.get("schema_version"),
            "name": target.get("name"),
            "host": host_identity,
            "service": {
                "implementation": service.get("implementation"),
                "base_url": service.get("base_url"),
                "version": service.get("version"),
                "server_info_hash": stable_hash(server_info) if server_info else None,
            },
            "route": {
                "logical_origin": {
                    "scheme": logical_origin.get("scheme"),
                    "hostname": logical_origin.get("hostname"),
                    "port": logical_origin.get("port"),
                },
                "selected_address": route.get("selected_address"),
                "authority": route.get("authority"),
            }
            if route
            else None,
            "model": target.get("model"),
            "topology": target.get("topology"),
        }
