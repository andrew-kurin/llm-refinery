from __future__ import annotations

import os
import shlex
import signal
import ssl
import subprocess
import threading
import time
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager, nullcontext, suppress
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from llm_refinery.application.run_context import RunContext
from llm_refinery.application.run_session import RunSession
from llm_refinery.benchmarks.lm_eval.command import build_lm_eval_command, lm_eval_api_key
from llm_refinery.benchmarks.lm_eval.config import LmEvalConfig, resolve_target_names
from llm_refinery.benchmarks.lm_eval.parser import (
    ParsedLmEvalSample,
    latest_lm_eval_result,
    lm_eval_sample_files,
    parse_lm_eval_metrics,
    parse_lm_eval_samples,
    summarize_lm_eval_samples,
)
from llm_refinery.benchmarks.lm_eval.presets import default_targets
from llm_refinery.core.config import ConfigError
from llm_refinery.core.endpoints import Endpoint
from llm_refinery.core.http_safety import (
    environment_proxy_applies,
    pinned_route_trust_env,
    resolve_request_route,
)
from llm_refinery.core.runs import CompletedRun, RunSpec, stable_hash
from llm_refinery.storage.duckdb import ResultStore
from llm_refinery.storage.models import SampleRecord
from llm_refinery.utils.terminal import sanitize_terminal_text


class LmEvalFailed(RuntimeError):
    pass


_MAX_RELAY_BODY_BYTES = 64_000_000
_PROCESS_TERMINATION_GRACE_S = 0.5
_RELAY_SHUTDOWN_GRACE_S = 2.0
_RELAY_RESPONSE_HEADERS = {
    "content-type": "Content-Type",
    "retry-after": "Retry-After",
    "retry-after-ms": "retry-after-ms",
}
_LOOPBACK_NO_PROXY = ("127.0.0.1", "localhost")


class _RelayState:
    """Own relay handler threads and their cancellable upstream clients."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._handler_threads: set[threading.Thread] = set()
        self._clients: set[httpx.Client] = set()
        self._closing = False

    def add_handler(self, thread: threading.Thread) -> None:
        with self._condition:
            self._handler_threads.add(thread)

    def remove_handler(self, thread: threading.Thread) -> None:
        with self._condition:
            self._handler_threads.discard(thread)
            self._condition.notify_all()

    def add_client(self, client: httpx.Client) -> bool:
        with self._condition:
            if self._closing:
                return False
            self._clients.add(client)
            return True

    def remove_client(self, client: httpx.Client) -> None:
        with self._condition:
            self._clients.discard(client)
            self._condition.notify_all()

    def begin_shutdown(self) -> None:
        with self._condition:
            self._closing = True
            clients = tuple(self._clients)
        for client in clients:
            with suppress(Exception):
                client.close()

    def join_handlers(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        current = threading.current_thread()
        while True:
            with self._condition:
                threads = tuple(
                    thread
                    for thread in self._handler_threads
                    if thread is not current and thread.is_alive()
                )
            if not threads:
                return
            for thread in threads:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                thread.join(timeout=remaining)


class _RelayServer(ThreadingHTTPServer):
    """Threaded relay whose request threads can be cancelled and joined."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        state: _RelayState,
    ) -> None:
        self.relay_state = state
        super().__init__(server_address, handler)

    def process_request(self, request: Any, client_address: Any) -> None:
        thread = threading.Thread(
            target=self._tracked_process_request,
            args=(request, client_address),
            name="llm-refinery-lm-eval-relay-request",
            daemon=self.daemon_threads,
        )
        self.relay_state.add_handler(thread)
        try:
            thread.start()
        except BaseException:
            self.relay_state.remove_handler(thread)
            self.shutdown_request(request)
            raise

    def _tracked_process_request(self, request: Any, client_address: Any) -> None:
        try:
            self.process_request_thread(request, client_address)
        finally:
            self.relay_state.remove_handler(threading.current_thread())


def run_lm_eval(
    config: LmEvalConfig,
    *,
    dry_run: bool = False,
    parent_run_id: str | None = None,
    store: ResultStore | None = None,
    run_context: RunContext | None = None,
) -> list[CompletedRun]:
    targets = {**default_targets(), **config.targets}
    selected = resolve_target_names(config.target, set(targets))
    config.output_root.mkdir(parents=True, exist_ok=True)

    env = _lm_eval_environment(os.environ)
    offline_value = "1" if config.offline else "0"
    env["HF_DATASETS_OFFLINE"] = offline_value
    env["HF_HUB_OFFLINE"] = offline_value
    if store is not None and store.database != config.database.resolve():
        raise ValueError(
            f"lm-eval database {config.database.resolve()} does not match shared store"
        )

    outcomes: list[CompletedRun] = []
    store_context = nullcontext(store) if store is not None else ResultStore(config.database)
    with store_context as active_store, ExitStack() as relay_stack:
        assert active_store is not None
        for target_name in selected:
            logical_target = targets[target_name]
            api_key = None if dry_run else lm_eval_api_key(logical_target, environ=os.environ)
            target = relay_stack.enter_context(
                _lm_eval_target(
                    logical_target,
                    config,
                    dry_run=dry_run,
                    api_key=api_key,
                )
            )
            limit_text = str(config.limit) if config.limit is not None else "all"
            command_output_path = config.output_root / target.name / "<run-id>"
            command_template = build_lm_eval_command(
                config,
                logical_target,
                output_path=command_output_path,
            )
            command_text = shlex.join(command_template)
            print(
                sanitize_terminal_text(
                    f"==> Running lm-eval target={target.name} "
                    f"tasks={config.tasks} limit={limit_text}"
                )
            )
            print(
                sanitize_terminal_text(
                    f"    model={target.model} base_url={logical_target.base_url}"
                )
            )
            if dry_run:
                print(sanitize_terminal_text(f"    output_path={command_output_path}"))
                print(sanitize_terminal_text(command_text))
                relay_stack.pop_all().close()
                continue

            spec = _run_spec(
                config,
                target_name=target.name,
                target_model=target.model,
                target_base_url=logical_target.base_url,
                target_api_key_env=logical_target.api_key_env,
                target_headers=logical_target.headers,
                command_text=command_text,
                database=active_store.database,
                parent_run_id=parent_run_id,
                run_context=run_context,
            )
            session = RunSession(active_store, spec, run_context=run_context)
            output_path = config.output_root / target.name / session.run_id
            cmd = build_lm_eval_command(config, target, output_path=output_path)
            print(sanitize_terminal_text(f"    output_path={output_path}"))
            with session as run:
                stdout_path = run.artifact("stdout", "stdout.txt", "text/plain")
                stderr_path = run.artifact("stderr", "stderr.txt", "text/plain")
                result_path = run.artifact("result", "result.json", "application/json")
                target_env = env.copy()
                # lm-eval's API adapter reads only OPENAI_API_KEY. Resolve the target's
                # credential into the child environment and never place it in command argv.
                target_env.pop("OPENAI_API_KEY", None)
                if api_key is not None:
                    target_env["OPENAI_API_KEY"] = api_key
                completed, process_timed_out = _run_lm_eval_process(
                    cmd,
                    env=target_env,
                    timeout_s=config.process_timeout_s,
                )
                stdout_path.write_text(
                    _redact_subprocess_output(completed.stdout or "", api_key),
                    encoding="utf-8",
                )
                stderr_path.write_text(
                    _redact_subprocess_output(completed.stderr or "", api_key),
                    encoding="utf-8",
                )
                source_result = latest_lm_eval_result(output_path)
                metrics: dict[str, float] = {}
                parsed_samples: list[ParsedLmEvalSample] = []
                sample_error: str | None = None
                source_samples: list[Path] = []
                if source_result is not None:
                    result_path.write_bytes(source_result.read_bytes())
                    metrics = parse_lm_eval_metrics(result_path)
                    source_samples = lm_eval_sample_files(source_result)
                    try:
                        for sample_index, source_sample in enumerate(source_samples):
                            sample_path = run.artifact(
                                f"samples.{sample_index}",
                                f"samples/{source_sample.name}",
                                "application/x-ndjson",
                            )
                            sample_path.write_bytes(source_sample.read_bytes())
                            file_samples = parse_lm_eval_samples(
                                sample_path,
                                result_path=source_result,
                            )
                            parsed_samples.extend(file_samples)
                            for sample in file_samples:
                                active_store.record_sample(
                                    SampleRecord(
                                        run_id=run.run_id,
                                        sample_id=sample.sample_id,
                                        status="ok",
                                        payload_json=sample.payload,
                                        metrics=sample.metrics,
                                        artifact_path=str(sample_path),
                                    )
                                )
                    except (OSError, ValueError) as exc:
                        sample_error = f"could not retain lm-eval samples: {exc}"
                    if source_samples:
                        metrics.update(summarize_lm_eval_samples(parsed_samples))

                missing_samples = config.log_samples and not parsed_samples
                success = (
                    completed.returncode == 0
                    and source_result is not None
                    and sample_error is None
                    and not missing_samples
                )
                status = "ok" if success else "failed"
                if process_timed_out:
                    error = f"lm-eval process timed out after {config.process_timeout_s:g}s"
                elif completed.returncode != 0:
                    error = f"exit code {completed.returncode}"
                elif source_result is None:
                    error = "lm-eval produced no result artifact"
                elif sample_error is not None:
                    error = sample_error
                elif missing_samples:
                    error = "lm-eval was asked to log samples but produced no sample records"
                else:
                    error = None
                outcome = run.complete(status=status, metrics=metrics, error=error)
                outcomes.append(outcome)

            if status != "ok":
                relay_stack.pop_all().close()
                raise LmEvalFailed(f"lm-eval failed for {target.name}: {error}")
            relay_stack.pop_all().close()
    return outcomes


def _lm_eval_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Preserve child network policy while forcing model traffic off proxies."""
    result = dict(environment)
    entries: list[str] = []
    seen: set[str] = set()
    for name in ("NO_PROXY", "no_proxy"):
        for raw_entry in environment.get(name, "").split(","):
            entry = raw_entry.strip()
            key = entry.casefold()
            if entry and key not in seen:
                entries.append(entry)
                seen.add(key)
    for entry in _LOOPBACK_NO_PROXY:
        if entry.casefold() not in seen:
            entries.append(entry)
            seen.add(entry.casefold())
    merged = ",".join(entries)
    result["NO_PROXY"] = merged
    result["no_proxy"] = merged
    return result


def _run_spec(
    config: LmEvalConfig,
    *,
    target_name: str,
    target_model: str,
    target_base_url: str,
    target_api_key_env: str | None,
    target_headers: dict[str, str],
    command_text: str,
    database: str | Path,
    parent_run_id: str | None,
    run_context: RunContext | None,
) -> RunSpec:
    config_json = {
        "benchmark": "lm_eval",
        "model_backend": config.model_backend,
        "package_spec": config.package_spec,
        "extra_packages": list(config.extra_packages),
        "apply_chat_template": config.apply_chat_template,
        "target": target_name,
        "model": target_model,
        "base_url": target_base_url,
        "api_key_env": target_api_key_env,
        "header_names": sorted(target_headers),
        "headers_hash": stable_hash(target_headers) if target_headers else None,
        "tasks": config.tasks,
        "limit": config.limit,
        "num_concurrent": config.num_concurrent,
        "max_retries": config.max_retries,
        "request_timeout_s": config.request_timeout_s,
        "process_timeout_s": config.process_timeout_s,
        "max_length": config.max_length,
        "eos_string": config.eos_string,
        "tokenizer": config.tokenizer,
        "metadata": config.metadata,
        "log_samples": config.log_samples,
        "num_fewshot": config.num_fewshot,
        "gen_kwargs": config.gen_kwargs,
        "offline": config.offline,
        "transport": {
            "trust_env": config.trust_env,
            "ca_bundle": str(config.ca_bundle) if config.ca_bundle else None,
            "pinned_route": (
                config.pinned_route.safe_json() if config.pinned_route is not None else None
            ),
        },
        "include_path": str(config.include_path) if config.include_path else None,
        "output_root": str(config.output_root),
        "params": {"target": target_name, "model": target_model},
    }
    if run_context is not None and run_context.target_json:
        config_json["execution_target"] = run_context.target_identity_json()
    return RunSpec.create(
        benchmark_kind="lm_eval",
        suite=config.suite_name,
        label=f"{config.suite_name}/{target_name}",
        command=command_text,
        config_json=config_json,
        database=database,
        parent_run_id=parent_run_id,
    )


def _redact_subprocess_output(value: str, api_key: str | None) -> str:
    if not api_key:
        return value
    return value.replace(f"Bearer {api_key}", "Bearer [REDACTED]").replace(
        api_key,
        "[REDACTED]",
    )


def _run_lm_eval_process(
    command: list[str],
    *,
    env: dict[str, str],
    timeout_s: float,
) -> tuple[subprocess.CompletedProcess[str], bool]:
    """Run lm-eval with an absolute deadline and no orphaned descendants."""
    process = subprocess.Popen(  # noqa: S603 - command is built from validated config
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_group(process)
        stdout, stderr = process.communicate()
        stderr = (stderr or "") + f"\nlm-eval process timed out after {timeout_s:g}s\n"
    except BaseException:
        _terminate_process_group(process)
        process.communicate()
        raise
    return (
        subprocess.CompletedProcess(
            args=command,
            returncode=124 if timed_out else process.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
        ),
        timed_out,
    )


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate the dedicated uvx/lm-eval process group, escalating if needed."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        if process.poll() is None:
            process.terminate()
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=_PROCESS_TERMINATION_GRACE_S)

    # The uvx leader may exit before lm-eval. Address the group again so a
    # descendant cannot keep issuing requests after the run is marked failed.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        if process.poll() is None:
            process.kill()
    if process.poll() is None:
        process.wait()


@contextmanager
def _lm_eval_target(
    target: Endpoint,
    config: LmEvalConfig,
    *,
    dry_run: bool,
    api_key: str | None = None,
):
    route = config.pinned_route
    if dry_run:
        yield target
        return
    if route is not None:
        route.request_url(target.base_url)

    # Resolve every permitted non-loopback target once within the request
    # budget. Synchronous proxy DNS cannot be cancelled safely: returning a
    # timeout while that lookup continues can let the worker send credentials
    # after the relay has already reported failure. Require model traffic to
    # bypass environment proxies; the lm-eval child may still use them for
    # package, dataset, and tokenizer downloads.
    if config.trust_env and environment_proxy_applies(target.base_url):
        raise ConfigError(
            "lm-eval target requests cannot use an environment proxy; add the target "
            "host to NO_PROXY, set quality.trust_env=false in a suite, or pass "
            "--no-trust-env"
        )
    if route is None:
        route = resolve_request_route(
            target.base_url,
            require_resolution=True,
            resolution_timeout_s=config.request_timeout_s,
        )

    client_trust_env = pinned_route_trust_env(
        target.base_url,
        trust_env=config.trust_env,
        route_is_pinned=route is not None,
    )
    try:
        verify = httpx.create_ssl_context(
            verify=str(config.ca_bundle) if config.ca_bundle is not None else True,
            trust_env=config.trust_env,
        )
    except (OSError, ssl.SSLError) as exc:
        raise ConfigError(f"could not load lm-eval CA bundle: {config.ca_bundle}") from exc
    logical = urlsplit(target.base_url)
    allowed_requests = {
        ("POST", urlsplit(target.chat_completions_url).path),
        ("POST", urlsplit(target.completions_url).path),
    }
    tokenizer_requests: set[tuple[str, str]] = set()
    if config.model_backend == "local-completions" and config.tokenizer is None:
        tokenizer_base = (
            target.completions_url.replace("/v1/completions", "")
            .replace("/v1/chat/completions", "")
            .rstrip("/")
        )
        tokenizer_path = urlsplit(tokenizer_base).path.rstrip("/")
        tokenizer_requests = {
            ("GET", f"{tokenizer_path}/tokenizer_info"),
            ("POST", f"{tokenizer_path}/tokenize"),
            ("POST", f"{tokenizer_path}/detokenize"),
        }
        allowed_requests.update(tokenizer_requests)
    relay_state = _RelayState()

    class RelayHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self._forward("GET")

        def do_POST(self) -> None:
            self._forward("POST")

        def _forward(self, method: str) -> None:
            request_path = urlsplit(self.path).path
            if (method, request_path) not in allowed_requests:
                self.send_error(404)
                return
            body: bytes | None = None
            if method == "POST":
                try:
                    content_length = int(self.headers.get("Content-Length", ""))
                except ValueError:
                    content_length = -1
                if content_length < 0 or content_length > _MAX_RELAY_BODY_BYTES:
                    self.send_error(413)
                    return
                body = self.rfile.read(content_length)
            forward_url = urlunsplit(
                (logical.scheme, logical.netloc, request_path, urlsplit(self.path).query, "")
            )
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.casefold()
                not in {
                    "connection",
                    "content-length",
                    "host",
                    "proxy-authorization",
                    "transfer-encoding",
                }
            }
            # lm-eval 0.4.12 does not attach its OPENAI_API_KEY to remote
            # tokenizer probes. Inject it only for those narrowly allowlisted
            # routes; model requests must authenticate to the loopback relay.
            if api_key is not None and (method, request_path) in tokenizer_requests:
                headers = {
                    key: value
                    for key, value in headers.items()
                    if key.casefold() != "authorization"
                }
                headers["Authorization"] = f"Bearer {api_key}"
            if route is not None:
                upstream_url = route.request_url(forward_url)
                upstream_headers = route.request_headers(headers)
                extensions = {"sni_hostname": route.sni_hostname}
            else:
                upstream_url = forward_url
                upstream_headers = headers
                extensions = {}
            client = httpx.Client(
                follow_redirects=False,
                trust_env=client_trust_env,
                verify=verify,
            )
            if not relay_state.add_client(client):
                with suppress(Exception):
                    client.close()
                self.send_error(503, "relay is shutting down")
                return
            deadline = time.monotonic() + config.request_timeout_s
            expired = threading.Event()

            def expire() -> None:
                expired.set()
                with suppress(Exception):
                    client.close()

            timer = threading.Timer(config.request_timeout_s, expire)
            timer.daemon = True
            timer.start()
            try:
                with client.stream(
                    method,
                    upstream_url,
                    content=body,
                    headers=upstream_headers,
                    extensions=extensions,
                    timeout=config.request_timeout_s,
                ) as response:
                    if 300 <= response.status_code < 400:
                        raise RuntimeError("upstream redirect refused")
                    response_body = bytearray()
                    for chunk in response.iter_bytes():
                        if expired.is_set() or time.monotonic() > deadline:
                            raise TimeoutError("upstream request exceeded its total timeout")
                        if len(response_body) + len(chunk) > _MAX_RELAY_BODY_BYTES:
                            raise RuntimeError("upstream response is too large")
                        response_body.extend(chunk)
                    self.send_response(response.status_code)
                    for source_name, output_name in _RELAY_RESPONSE_HEADERS.items():
                        header_value = response.headers.get(source_name)
                        if _safe_relay_response_header(header_value):
                            assert header_value is not None
                            self.send_header(output_name, header_value)
                    self.send_header("Content-Length", str(len(response_body)))
                    self.end_headers()
                    self.wfile.write(response_body)
            except Exception:  # noqa: BLE001 - return a fixed, credential-free relay error
                self.send_error(502, "upstream request failed")
            finally:
                timer.cancel()
                timer.join()
                with suppress(Exception):
                    client.close()
                relay_state.remove_client(client)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    server = _RelayServer(("127.0.0.1", 0), RelayHandler, relay_state)
    thread = threading.Thread(
        target=server.serve_forever,
        name="llm-refinery-lm-eval-relay",
        daemon=True,
    )
    thread.start()
    relay_netloc = f"127.0.0.1:{server.server_port}"
    relay_base_url = urlunsplit(("http", relay_netloc, logical.path, "", ""))
    try:
        yield replace(target, base_url=relay_base_url)
    finally:
        relay_state.begin_shutdown()
        server.shutdown()
        server.server_close()
        relay_state.join_handlers(_RELAY_SHUTDOWN_GRACE_S)
        thread.join(timeout=2)


def _safe_relay_response_header(value: str | None) -> bool:
    if value is None or not value or len(value) > 1024 or value != value.strip(" \t"):
        return False
    try:
        value.encode("ascii")
    except UnicodeEncodeError:
        return False
    return not any(ord(character) < 32 or ord(character) == 127 for character in value)
