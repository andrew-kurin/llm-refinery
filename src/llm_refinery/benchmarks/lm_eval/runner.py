from __future__ import annotations

import os
import shlex
import ssl
import subprocess
import threading
import time
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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
from llm_refinery.core.http_safety import pinned_route_trust_env
from llm_refinery.core.runs import CompletedRun, RunSpec, stable_hash
from llm_refinery.storage.duckdb import ResultStore
from llm_refinery.storage.models import SampleRecord


class LmEvalFailed(RuntimeError):
    pass


_MAX_RELAY_BODY_BYTES = 64_000_000


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

    env = os.environ.copy()
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
            target = relay_stack.enter_context(
                _lm_eval_target(logical_target, config, dry_run=dry_run)
            )
            limit_text = str(config.limit) if config.limit is not None else "all"
            output_path = config.output_root / target.name
            cmd = build_lm_eval_command(config, target)
            command_text = shlex.join(cmd)
            print(
                f"==> Running lm-eval target={target.name} tasks={config.tasks} limit={limit_text}"
            )
            print(f"    model={target.model} base_url={target.base_url}")
            print(f"    output_path={output_path}")
            if dry_run:
                print(command_text)
                continue

            recorded_command_text = command_text.replace(
                target.base_url,
                logical_target.base_url,
            )
            spec = _run_spec(
                config,
                target_name=target.name,
                target_model=target.model,
                target_base_url=logical_target.base_url,
                target_api_key_env=logical_target.api_key_env,
                target_headers=logical_target.headers,
                command_text=recorded_command_text,
                database=active_store.database,
                parent_run_id=parent_run_id,
                run_context=run_context,
            )
            with RunSession(active_store, spec, run_context=run_context) as run:
                stdout_path = run.artifact("stdout", "stdout.txt", "text/plain")
                stderr_path = run.artifact("stderr", "stderr.txt", "text/plain")
                result_path = run.artifact("result", "result.json", "application/json")
                result_started_mtime = time.time()
                target_env = env.copy()
                # lm-eval's API adapter reads only OPENAI_API_KEY. Resolve the target's
                # credential into the child environment and never place it in command argv.
                target_env.pop("OPENAI_API_KEY", None)
                api_key = lm_eval_api_key(target, environ=os.environ)
                if api_key is not None:
                    target_env["OPENAI_API_KEY"] = api_key
                completed = subprocess.run(
                    cmd,
                    env=target_env,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                stdout_path.write_text(
                    _redact_subprocess_output(completed.stdout or "", api_key),
                    encoding="utf-8",
                )
                stderr_path.write_text(
                    _redact_subprocess_output(completed.stderr or "", api_key),
                    encoding="utf-8",
                )
                source_result = latest_lm_eval_result(
                    output_path,
                    newer_than=result_started_mtime,
                )
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
                if completed.returncode != 0:
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
                raise LmEvalFailed(f"lm-eval failed for {target.name}: {error}")
    return outcomes


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


@contextmanager
def _lm_eval_target(
    target: Endpoint,
    config: LmEvalConfig,
    *,
    dry_run: bool,
):
    route = config.pinned_route
    if route is None or dry_run:
        yield target
        return
    pinned_route = route
    pinned_route.request_url(target.base_url)

    client_trust_env = pinned_route_trust_env(
        target.base_url,
        trust_env=config.trust_env,
    )
    try:
        verify = httpx.create_ssl_context(
            verify=str(config.ca_bundle) if config.ca_bundle is not None else True,
            trust_env=config.trust_env,
        )
    except (OSError, ssl.SSLError) as exc:
        raise ConfigError(f"could not load lm-eval CA bundle: {config.ca_bundle}") from exc
    client = httpx.Client(
        follow_redirects=False,
        trust_env=client_trust_env,
        verify=verify,
    )
    logical = urlsplit(target.base_url)
    allowed_paths = {
        urlsplit(target.chat_completions_url).path,
        urlsplit(target.completions_url).path,
    }

    class RelayHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request_path = urlsplit(self.path).path
            if request_path not in allowed_paths:
                self.send_error(404)
                return
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
            try:
                with client.stream(
                    "POST",
                    pinned_route.request_url(forward_url),
                    content=body,
                    headers=pinned_route.request_headers(headers),
                    extensions={"sni_hostname": pinned_route.sni_hostname},
                    timeout=300.0,
                ) as response:
                    if 300 <= response.status_code < 400:
                        raise RuntimeError("upstream redirect refused")
                    response_body = bytearray()
                    for chunk in response.iter_bytes():
                        if len(response_body) + len(chunk) > _MAX_RELAY_BODY_BYTES:
                            raise RuntimeError("upstream response is too large")
                        response_body.extend(chunk)
                    self.send_response(response.status_code)
                    content_type = response.headers.get("content-type")
                    if content_type:
                        self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(response_body)))
                    self.end_headers()
                    self.wfile.write(response_body)
            except Exception:  # noqa: BLE001 - return a fixed, credential-free relay error
                self.send_error(502, "upstream request failed")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), RelayHandler)
    server.daemon_threads = True
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
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        client.close()
