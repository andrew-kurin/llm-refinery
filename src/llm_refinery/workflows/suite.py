from __future__ import annotations

import ipaddress
import json
import os
import socket
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rich.console import Console
from rich.table import Table

from llm_refinery.application.run_context import RunContext
from llm_refinery.application.run_session import RunSession
from llm_refinery.application.target_discovery import TargetResolver
from llm_refinery.benchmarks.http_load.config import HttpTransportConfig, load_http_load_config
from llm_refinery.benchmarks.http_load.runner import run_http_load
from llm_refinery.benchmarks.lm_eval.config import LmEvalConfig
from llm_refinery.benchmarks.lm_eval.runner import run_lm_eval
from llm_refinery.compare import build_compare_rows, build_compare_table_rows
from llm_refinery.core.config import ConfigError
from llm_refinery.core.endpoints import Endpoint
from llm_refinery.core.http_safety import PinnedHttpRoute, pinned_route_trust_env
from llm_refinery.core.runs import CompletedRun, RunSpec
from llm_refinery.core.targets import ResolvedTarget, TargetInspection, TargetSpec
from llm_refinery.storage.duckdb import ResultStore
from llm_refinery.utils.sanity import run_api_sanity_check
from llm_refinery.utils.system import get_system_snapshot, is_port_listening
from llm_refinery.workflows.suite_config import SuiteConfig

LmEvalRunner = Callable[..., list[CompletedRun]]
HttpLoadRunner = Callable[..., list[CompletedRun]]


@dataclass(frozen=True)
class SuiteResult:
    run: CompletedRun
    children: tuple[CompletedRun, ...]


class BenchmarkSuiteWorkflow:
    def __init__(
        self,
        config: SuiteConfig,
        *,
        console: Console | None = None,
        lm_eval_runner: LmEvalRunner = run_lm_eval,
        http_load_runner: HttpLoadRunner = run_http_load,
        port_listener: Callable[[int], bool] = is_port_listening,
        sanity_checker: Callable[..., dict[str, Any]] | None = None,
        system_snapshot: Callable[[], str] = get_system_snapshot,
        target_resolver: TargetResolver | None = None,
    ) -> None:
        self.config = config
        self.console = console or Console()
        self.lm_eval_runner = lm_eval_runner
        self.http_load_runner = http_load_runner
        self.port_listener = port_listener
        if sanity_checker is not None:
            self.sanity_checker = sanity_checker
        else:
            target_transport = config.target.transport if config.target is not None else None
            self.sanity_checker = lambda endpoint: run_api_sanity_check(
                endpoint,
                trust_env=(target_transport.trust_env if target_transport is not None else True),
                ca_bundle=(target_transport.ca_bundle if target_transport is not None else None),
                route=self._target_route,
            )
        self.system_snapshot = system_snapshot
        self.target_resolver = target_resolver or TargetResolver()
        self._endpoint: Endpoint | None = config.endpoint
        self._resolved_target: ResolvedTarget | None = None
        self._target_route: PinnedHttpRoute | None = None
        self._run_context: RunContext | None = None
        self._validation_warnings: list[str] = []

    def execute(self) -> SuiteResult:
        command = f"llm-refinery suite {self.config.name}"
        spec = RunSpec.create(
            benchmark_kind="suite",
            suite=self.config.name,
            label=self.config.name,
            command=command,
            config_json=self.config.safe_json(),
            database=self.config.database,
        )
        children: list[CompletedRun] = []
        with ResultStore(self.config.database) as store, RunSession(store, spec) as run:
            metrics_before_path = None
            target_spec = self.config.target
            initial_host_snapshot_available = False
            child_failure: Exception | None = None
            try:
                before_snapshot = self.system_snapshot()
                before_path = run.artifact(
                    "system_before",
                    "system-before.txt",
                    "text/plain",
                )
                before_path.write_text(before_snapshot, encoding="utf-8")
                if target_spec is not None:
                    try:
                        inspection = self.target_resolver.inspect(
                            target_spec,
                            allow_service_unavailable=True,
                        )
                    except Exception as exc:  # noqa: BLE001 - persist discovery failure context
                        detail = _target_error_summary(exc, target_spec)
                        partial = getattr(exc, "target_inspection", None)
                        inspection = (
                            replace(
                                partial,
                                errors=tuple(dict.fromkeys([*partial.errors, detail])),
                            )
                            if isinstance(partial, TargetInspection)
                            else TargetInspection(
                                spec=target_spec,
                                host=None,
                                service=None,
                                resolved=None,
                                errors=(detail,),
                            )
                        )
                        initial_host_snapshot_available = inspection.host is not None
                        failure = inspection.safe_json()
                        failure["failure_stage"] = "target_discovery"
                        failure["requested_target"] = target_spec.safe_json()
                        discovery_path = run.artifact(
                            "target_discovery",
                            "target-discovery.json",
                            "application/json",
                        )
                        server_before_path = run.artifact(
                            "server_before",
                            "server-before.json",
                            "application/json",
                        )
                        _write_json(discovery_path, failure)
                        _write_json(
                            server_before_path,
                            inspection.host.profile
                            if inspection.host is not None
                            else {"capture_error": detail},
                        )
                        run.set_target_json(failure)
                        raise
                    initial_host_snapshot_available = inspection.host is not None
                    discovery_path = run.artifact(
                        "target_discovery",
                        "target-discovery.json",
                        "application/json",
                    )
                    server_before_path = run.artifact(
                        "server_before",
                        "server-before.json",
                        "application/json",
                    )
                    _write_json(discovery_path, inspection.safe_json())
                    _write_json(
                        server_before_path,
                        inspection.host.profile
                        if inspection.host is not None
                        else {"capture_error": "target host inventory unavailable"},
                    )
                    run.set_target_json(inspection.safe_json())
                    if inspection.resolved is None:
                        detail = "; ".join(inspection.errors) or "target is unavailable"
                        raise RuntimeError(
                            f"could not resolve target {target_spec.name!r}: {detail}"
                        )
                    self._resolved_target = inspection.resolved
                    self._target_route = inspection.route
                    self._endpoint = inspection.resolved.endpoint
                    self._validate_resolved_target(inspection.resolved)
                    if target_spec.discovery.metrics:
                        metrics_before_path = run.artifact(
                            "vllm_metrics_before",
                            "vllm-metrics-before.prom",
                            "text/plain",
                        )
                        _write_text_observation(
                            metrics_before_path,
                            lambda: self.target_resolver.metrics(target_spec),
                        )
                self._run_context = run.run_context
                preflight_result = self.preflight(before_snapshot)
                preflight_path = run.artifact("preflight", "preflight.json", "application/json")
                preflight_path.write_text(
                    json.dumps(preflight_result, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                try:
                    children.extend(self.run_quality(store, parent_run_id=run.run_id))
                    children.extend(self.run_load(store, parent_run_id=run.run_id))
                except Exception as exc:  # noqa: BLE001 - summarize persisted children first
                    child_failure = exc
            finally:
                if target_spec is not None and initial_host_snapshot_available:
                    try:
                        after_profile = self.target_resolver.snapshot_host(target_spec).profile
                    except Exception as exc:  # noqa: BLE001 - retain best-effort telemetry
                        after_profile = {"capture_error": f"{type(exc).__name__}: {exc}"}
                    server_after_path = run.artifact(
                        "server_after",
                        "server-after.json",
                        "application/json",
                    )
                    _write_json(server_after_path, after_profile)
                if (
                    target_spec is not None
                    and metrics_before_path is not None
                    and self._resolved_target is not None
                ):
                    metrics_after_path = run.artifact(
                        "vllm_metrics_after",
                        "vllm-metrics-after.prom",
                        "text/plain",
                    )
                    _write_text_observation(
                        metrics_after_path,
                        lambda: self.target_resolver.metrics(target_spec),
                    )
                try:
                    after_snapshot = self.system_snapshot()
                except Exception as exc:  # noqa: BLE001 - retain the primary suite outcome
                    after_snapshot = f"capture_error: {type(exc).__name__}: {exc}"
                after_path = run.artifact(
                    "system_after",
                    "system-after.txt",
                    "text/plain",
                )
                after_path.write_text(after_snapshot, encoding="utf-8")
                self._log("Memory/process snapshot after")
                self.console.print(after_snapshot)

            children = _merge_linked_children(store, run.run_id, children)
            failed_children = sum(child.status != "ok" for child in children)
            if child_failure is not None:
                run.complete(
                    status="failed",
                    metrics={
                        "child_run_count": float(len(children)),
                        "failed_child_count": float(failed_children),
                    },
                    error=f"{type(child_failure).__name__}: {child_failure}",
                )
                raise child_failure
            outcome = run.complete(
                status="ok" if failed_children == 0 else "failed",
                metrics={
                    "child_run_count": float(len(children)),
                    "failed_child_count": float(failed_children),
                },
                error=None if failed_children == 0 else f"{failed_children} child run(s) failed",
            )
            if failed_children:
                raise RuntimeError(f"{failed_children} suite child run(s) failed")
        self._log("Done.")
        return SuiteResult(run=outcome, children=tuple(children))

    def preflight(self, snapshot: str | None = None) -> dict[str, Any]:
        config = self.config.preflight
        if not config.enabled:
            return {"enabled": False, "warnings": list(self._validation_warnings)}
        self._log("Performing preflight checks...")
        endpoint = self._effective_endpoint()
        parsed = urlparse(endpoint.base_url)
        endpoint_port = _port_from_url(endpoint.base_url)
        endpoint_is_loopback = _is_loopback_hostname(parsed.hostname)
        if endpoint_is_loopback and not self.port_listener(endpoint_port):
            raise RuntimeError(
                f"no model server listening on :{endpoint_port}; start it first, then rerun"
            )

        if config.require_clean:
            if not endpoint_is_loopback:
                raise ConfigError(
                    "preflight.require_clean cannot verify ports on a remote endpoint; "
                    "set require_clean: false explicitly"
                )
            for port in config.forbidden_ports:
                if port != endpoint_port and self.port_listener(port):
                    raise RuntimeError(
                        f"port :{port} is listening; stop other model servers for clean timing"
                    )

        self._log("Memory/process snapshot before")
        self.console.print(snapshot if snapshot is not None else self.system_snapshot())
        if not config.sanity_check:
            return {
                "enabled": True,
                "sanity_check": False,
                "warnings": list(self._validation_warnings),
            }

        self._log("Sanity check: content present")
        sanity = self.sanity_checker(endpoint)
        if not sanity["success"]:
            raise RuntimeError(str(sanity["error"]))
        expected_response_model = config.expected_response_model
        if expected_response_model is None and self._resolved_target is not None:
            expected_response_model = endpoint.model
        if (
            expected_response_model is not None
            and sanity.get("response_model") != expected_response_model
        ):
            raise RuntimeError(
                "endpoint returned model "
                f"{sanity.get('response_model')!r}; expected "
                f"{expected_response_model!r}"
            )
        for key in (
            "elapsed_s",
            "content_len",
            "reasoning_len",
            "finish_reason",
            "requested_model",
            "response_model",
            "model_matches",
            "content_preview",
        ):
            self.console.print(f"    {key}={sanity.get(key)}")
        return {
            "enabled": True,
            "sanity_check": True,
            "sanity": sanity,
            "warnings": list(self._validation_warnings),
        }

    def run_quality(self, store: ResultStore, *, parent_run_id: str) -> list[CompletedRun]:
        quality = self.config.quality
        if not quality.enabled:
            return []
        endpoint = self._effective_endpoint()
        target_transport = self.config.target.transport if self.config.target is not None else None
        trust_env = (
            quality.trust_env
            if quality.trust_env is not None
            else target_transport.trust_env
            if target_transport is not None
            else False
        )
        ca_bundle = quality.ca_bundle or (
            target_transport.ca_bundle if target_transport is not None else None
        )
        if self._target_route is not None:
            pinned_route_trust_env(endpoint.base_url, trust_env=trust_env)
        self._log(f"Running lm-eval quality (tasks={quality.tasks}, limit={quality.limit})")
        lm_config = LmEvalConfig(
            target=endpoint.name,
            model_backend=quality.model_backend,
            limit=quality.limit,
            tasks=quality.tasks,
            max_length=quality.max_length,
            eos_string=quality.eos_string,
            tokenizer=quality.tokenizer,
            metadata=quality.metadata,
            num_fewshot=quality.num_fewshot,
            gen_kwargs=quality.gen_kwargs,
            include_path=quality.include_path,
            output_root=quality.output_root,
            package_spec=quality.package_spec,
            extra_packages=quality.extra_packages,
            offline=quality.offline,
            trust_env=trust_env,
            ca_bundle=ca_bundle,
            pinned_route=self._target_route,
            suite_name=self.config.name,
            database=self.config.database,
            log_samples=True,
            targets={endpoint.name: endpoint},
        )
        return self.lm_eval_runner(
            lm_config,
            parent_run_id=parent_run_id,
            store=store,
            run_context=self._run_context,
        )

    def run_load(self, store: ResultStore, *, parent_run_id: str) -> list[CompletedRun]:
        step = self.config.http_load
        if not step.enabled:
            return []
        assert step.config is not None
        http_config = load_http_load_config(step.config)
        if self.config.target is None:
            endpoint = self._effective_endpoint()
            targets = step.targets or (endpoint.name,)
            self._log(f"Running HTTP load (targets={','.join(targets)})")
            outcomes = self.http_load_runner(
                http_config,
                target_names=targets,
                scenario_names=step.scenarios,
                database_override=self.config.database,
                parent_run_id=parent_run_id,
                store=store,
                run_context=self._run_context,
            )
            self._print_http_comparison(store, http_config.name)
            return outcomes

        if len(step.targets) > 1:
            raise ConfigError("suite HTTP load supports one resolved target")
        endpoint = self._effective_endpoint()
        target_name = step.targets[0] if step.targets else endpoint.name
        effective_target = replace(endpoint, name=target_name)
        assert self.config.target is not None
        target_transport = self.config.target.transport
        http_config = replace(
            http_config,
            targets=[effective_target],
            transport=HttpTransportConfig(
                trust_env=target_transport.trust_env,
                ca_bundle=target_transport.ca_bundle,
                pinned_route=self._target_route,
            ),
        )
        targets = (target_name,)
        self._log(f"Running HTTP load (targets={','.join(targets)})")
        outcomes = self.http_load_runner(
            http_config,
            target_names=targets,
            scenario_names=step.scenarios,
            database_override=self.config.database,
            parent_run_id=parent_run_id,
            store=store,
            run_context=self._run_context,
        )
        self._print_http_comparison(store, http_config.name)
        return outcomes

    def _effective_endpoint(self) -> Endpoint:
        if self._endpoint is None:
            raise RuntimeError("suite target has not been resolved")
        return self._endpoint

    def _validate_resolved_target(self, target: ResolvedTarget) -> None:
        max_model_len = target.model.max_model_len
        if max_model_len is None:
            if self.config.quality.enabled or self.config.http_load.enabled:
                self._warn_validation(
                    "served model discovery did not report max_model_len; suite context "
                    "budgets cannot be validated"
                )
            return
        quality = self.config.quality
        if quality.enabled and quality.max_length > max_model_len:
            raise ConfigError(
                f"quality.max_length {quality.max_length} exceeds served model limit "
                f"{max_model_len}"
            )
        if not self.config.http_load.enabled or self.config.http_load.config is None:
            return
        http_config = load_http_load_config(self.config.http_load.config)
        wanted_scenarios = set(self.config.http_load.scenarios)
        scenarios = [
            scenario
            for scenario in http_config.scenarios
            if not wanted_scenarios or scenario.name in wanted_scenarios
        ]
        missing_scenarios = wanted_scenarios - {scenario.name for scenario in scenarios}
        if missing_scenarios:
            raise ConfigError(
                "unknown HTTP load scenario(s): " + ", ".join(sorted(missing_scenarios))
            )
        requested_max = max(
            token_count for scenario in scenarios for token_count in scenario.max_tokens
        )
        if requested_max >= max_model_len:
            raise ConfigError(
                f"HTTP load max_tokens {requested_max} leaves no context for the "
                f"non-empty prompt within served model limit {max_model_len}"
            )
        for scenario in scenarios:
            prompt_chars = max(
                len(scenario.rendered_prompt_for(index, request_nonce="context-check"))
                for index in range(len(scenario.prompts))
            )
            input_chars = prompt_chars + len(scenario.system or "")
            remaining_tokens = max_model_len - max(scenario.max_tokens)
            warning = (
                f"HTTP scenario {scenario.name!r} leaves {remaining_tokens} model tokens "
                f"for a rendered prompt/system of up to {input_chars} characters; exact "
                "context fit cannot be verified without the served tokenizer"
            )
            self._warn_validation(warning)

    def _warn_validation(self, warning: str) -> None:
        if warning in self._validation_warnings:
            return
        self._validation_warnings.append(warning)
        self.console.print(f"[yellow]warning:[/yellow] {warning}")

    def _print_http_comparison(self, store: ResultStore, suite_name: str) -> None:
        runs = [
            run
            for run in store.comparison_runs(latest_per_trial=False)
            if run["suite"] == suite_name
        ]
        rows = build_compare_rows(
            runs,
            metrics=(
                "observed_latency_p95_s",
                "visible_ttft_p95_s",
                "reasoning_ttft_p95_s",
                "tpot_p95_s",
                "completion_tokens_per_second",
                "check_pass_rate",
            ),
            params=("target", "protocol", "scenario", "concurrency"),
            sort_key="observed_latency_p95_s",
            ascending=True,
            limit=20,
        )
        table_rows = build_compare_table_rows(rows)
        if not table_rows:
            return
        self._log("HTTP comparison for this suite")
        table = Table(*[str(header) for header in table_rows[0]])
        for row in table_rows[1:]:
            table.add_row(*[str(cell) for cell in row])
        self.console.print(table)

    def _log(self, message: str) -> None:
        self.console.print(f"[bold blue]==>[/bold blue] {message}")


def _port_from_url(url: str) -> int:
    parsed = urlparse(url)
    if parsed.port is not None:
        return parsed.port
    if parsed.scheme == "https":
        return 443
    return 80


def _is_loopback_hostname(hostname: str | None) -> bool:
    if hostname is None:
        return False
    normalized = hostname.casefold().rstrip(".")
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        try:
            return ipaddress.ip_address(socket.inet_aton(normalized)).is_loopback
        except OSError:
            return False


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text_observation(path: Path, capture: Callable[[], str]) -> None:
    try:
        value = capture()
    except Exception as exc:  # noqa: BLE001 - telemetry must not fail a benchmark
        value = f"# capture_error: {type(exc).__name__}: {exc}\n"
    path.write_text(value, encoding="utf-8")


def _merge_linked_children(
    store: ResultStore,
    parent_run_id: str,
    returned: list[CompletedRun],
) -> list[CompletedRun]:
    """Return child outcomes with persisted rows as the source of truth."""
    by_run_id = {child.run_id: child for child in returned}
    for row in store.comparison_runs(include_failed=True, latest_per_trial=False):
        if row["parent_run_id"] != parent_run_id:
            continue
        by_run_id[row["run_id"]] = CompletedRun(
            run_id=row["run_id"],
            benchmark_kind=row["benchmark_kind"],
            spec_hash=row["spec_hash"],
            status=row["status"],
            duration_s=float(row["duration_s"]),
            metrics=row["metrics"],
            error=row["error"],
        )
    return list(by_run_id.values())


def _target_error_summary(exc: Exception, target: TargetSpec) -> str:
    detail = f"{type(exc).__name__}: {exc}"
    secrets = list(target.endpoint.headers.values())
    if target.endpoint.api_key_env:
        secrets.append(os.environ.get(target.endpoint.api_key_env, ""))
    for secret in secrets:
        if secret:
            detail = detail.replace(secret, "<redacted>")
    return detail
