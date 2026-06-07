from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from llm_refinery.config import ConfigError, coerce_list, stable_hash
from llm_refinery.storage import ResultStore, RunRecord, utc_now

PROVIDERS = {"openai", "ollama"}
PERCENTILES = (50, 90, 95, 99)


@dataclass(frozen=True)
class HttpTarget:
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> HttpTarget:
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ConfigError("each HTTP target requires a non-empty 'name'")

        provider = str(raw.get("provider") or "openai").strip().lower()
        if provider not in PROVIDERS:
            raise ConfigError(
                f"target {name!r} provider must be one of {sorted(PROVIDERS)}, got {provider!r}"
            )

        base_url = str(raw.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            raise ConfigError(f"target {name!r} requires 'base_url'")

        model = str(raw.get("model") or "").strip()
        if not model:
            raise ConfigError(f"target {name!r} requires 'model'")

        return cls(
            name=name,
            provider=provider,
            base_url=base_url,
            model=model,
            api_key_env=str(raw["api_key_env"]) if raw.get("api_key_env") else None,
            headers={str(key): str(value) for key, value in dict(raw.get("headers") or {}).items()},
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "header_names": sorted(self.headers),
        }


@dataclass(frozen=True)
class HttpScenario:
    name: str
    prompt: str
    system: str | None = None
    max_tokens: list[int] = field(default_factory=lambda: [128])
    concurrency: list[int] = field(default_factory=lambda: [1])
    requests: int = 8
    warmup_requests: int = 0
    temperature: float = 0.0
    seed: int | None = None
    stream: bool = True
    timeout_s: float = 300.0
    prompt_repeat: int = 1
    expected_contains: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], *, base_dir: Path) -> HttpScenario:
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ConfigError("each HTTP scenario requires a non-empty 'name'")

        prompt = _scenario_prompt(raw, base_dir=base_dir)
        max_tokens = [int(value) for value in coerce_list(raw.get("max_tokens", 128))]
        concurrency = [int(value) for value in coerce_list(raw.get("concurrency", 1))]
        requests = int(raw.get("requests", 8))
        warmup_requests = int(raw.get("warmup_requests", 0))
        prompt_repeat = int(raw.get("prompt_repeat", 1))
        timeout_s = float(raw.get("timeout_s", 300.0))

        if any(value <= 0 for value in max_tokens):
            raise ConfigError(f"scenario {name!r} max_tokens values must be positive")
        if any(value <= 0 for value in concurrency):
            raise ConfigError(f"scenario {name!r} concurrency values must be positive")
        if requests <= 0:
            raise ConfigError(f"scenario {name!r} requests must be positive")
        if warmup_requests < 0:
            raise ConfigError(f"scenario {name!r} warmup_requests cannot be negative")
        if prompt_repeat <= 0:
            raise ConfigError(f"scenario {name!r} prompt_repeat must be positive")
        if timeout_s <= 0:
            raise ConfigError(f"scenario {name!r} timeout_s must be positive")

        return cls(
            name=name,
            prompt=prompt,
            system=str(raw["system"]) if raw.get("system") else None,
            max_tokens=max_tokens,
            concurrency=concurrency,
            requests=requests,
            warmup_requests=warmup_requests,
            temperature=float(raw.get("temperature", 0.0)),
            seed=int(raw["seed"]) if raw.get("seed") is not None else None,
            stream=bool(raw.get("stream", True)),
            timeout_s=timeout_s,
            prompt_repeat=prompt_repeat,
            expected_contains=[str(value) for value in coerce_list(raw.get("expected_contains"))],
        )

    @property
    def rendered_prompt(self) -> str:
        return "\n\n".join([self.prompt] * self.prompt_repeat)

    def safe_json(self) -> dict[str, Any]:
        rendered = self.rendered_prompt
        return {
            "name": self.name,
            "system": self.system,
            "prompt_preview": rendered[:240],
            "prompt_chars": len(rendered),
            "prompt_hash": stable_hash(rendered),
            "max_tokens": self.max_tokens,
            "concurrency": self.concurrency,
            "requests": self.requests,
            "warmup_requests": self.warmup_requests,
            "temperature": self.temperature,
            "seed": self.seed,
            "stream": self.stream,
            "timeout_s": self.timeout_s,
            "prompt_repeat": self.prompt_repeat,
            "expected_contains": self.expected_contains,
        }


@dataclass(frozen=True)
class HttpLoadConfig:
    name: str
    database: Path
    targets: list[HttpTarget]
    scenarios: list[HttpScenario]
    source_path: Path | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], source_path: Path | None = None) -> HttpLoadConfig:
        name = str(raw.get("name") or (source_path.stem if source_path else "http-load"))
        targets_raw = raw.get("targets") or []
        scenarios_raw = raw.get("scenarios") or []
        if not targets_raw:
            raise ConfigError("HTTP load config requires at least one target in 'targets'")
        if not scenarios_raw:
            raise ConfigError("HTTP load config requires at least one scenario in 'scenarios'")

        base_dir = source_path.parent if source_path else Path.cwd()
        return cls(
            name=name,
            database=Path(str(raw.get("database") or "results/llm_refinery.duckdb")),
            targets=[HttpTarget.from_mapping(dict(item)) for item in targets_raw],
            scenarios=[
                HttpScenario.from_mapping(dict(item), base_dir=base_dir) for item in scenarios_raw
            ],
            source_path=source_path,
        )


@dataclass(frozen=True)
class HttpLoadTrial:
    suite: str
    name: str
    key: str
    target: HttpTarget
    scenario: HttpScenario
    concurrency: int
    max_tokens: int

    @property
    def command_text(self) -> str:
        return (
            f"http-load provider={self.target.provider} base_url={self.target.base_url} "
            f"model={self.target.model} scenario={self.scenario.name} "
            f"concurrency={self.concurrency} requests={self.scenario.requests} "
            f"max_tokens={self.max_tokens} stream={self.scenario.stream}"
        )

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "name": self.name,
            "key": self.key,
            "model": {"name": self.target.model},
            "target": self.target.safe_json(),
            "scenario": self.scenario.safe_json(),
            "prompt_tokens": None,
            "gen_tokens": self.max_tokens,
            "params": {
                "target": self.target.name,
                "provider": self.target.provider,
                "scenario": self.scenario.name,
                "model": self.target.model,
                "concurrency": self.concurrency,
                "requests": self.scenario.requests,
                "max_tokens": self.max_tokens,
                "stream": self.scenario.stream,
                "temperature": self.scenario.temperature,
            },
        }


@dataclass(frozen=True)
class RequestResult:
    index: int
    ok: bool
    status_code: int | None
    latency_s: float
    ttft_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    completion_chars: int = 0
    server_prompt_eval_duration_s: float | None = None
    server_eval_duration_s: float | None = None
    response_text: str = ""
    check_passed: bool | None = None
    error: str | None = None

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "ok": self.ok,
            "status_code": self.status_code,
            "latency_s": self.latency_s,
            "ttft_s": self.ttft_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "completion_chars": self.completion_chars,
            "server_prompt_eval_duration_s": self.server_prompt_eval_duration_s,
            "server_eval_duration_s": self.server_eval_duration_s,
            "response_text": self.response_text,
            "check_passed": self.check_passed,
            "error": self.error,
        }


def load_http_load_config(path: str | Path) -> HttpLoadConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")
    return HttpLoadConfig.from_mapping(raw, source_path=config_path)


def expand_http_load_trials(
    config: HttpLoadConfig,
    *,
    target_names: tuple[str, ...] = (),
    scenario_names: tuple[str, ...] = (),
) -> list[HttpLoadTrial]:
    wanted_targets = set(target_names)
    wanted_scenarios = set(scenario_names)
    targets = [
        target for target in config.targets if not wanted_targets or target.name in wanted_targets
    ]
    scenarios = [
        scenario
        for scenario in config.scenarios
        if not wanted_scenarios or scenario.name in wanted_scenarios
    ]

    missing_targets = wanted_targets - {target.name for target in targets}
    if missing_targets:
        raise ConfigError(f"unknown HTTP load target(s): {', '.join(sorted(missing_targets))}")
    missing_scenarios = wanted_scenarios - {scenario.name for scenario in scenarios}
    if missing_scenarios:
        raise ConfigError(f"unknown HTTP load scenario(s): {', '.join(sorted(missing_scenarios))}")

    trials: list[HttpLoadTrial] = []
    for target in targets:
        for scenario in scenarios:
            for concurrency in scenario.concurrency:
                for max_tokens in scenario.max_tokens:
                    key_material = {
                        "suite": config.name,
                        "target": target.safe_json(),
                        "scenario": scenario.safe_json(),
                        "concurrency": concurrency,
                        "max_tokens": max_tokens,
                    }
                    key = stable_hash(key_material)
                    name = "/".join(
                        [
                            config.name,
                            target.name,
                            scenario.name,
                            f"c{concurrency}",
                            f"n{max_tokens}",
                            key,
                        ]
                    )
                    trials.append(
                        HttpLoadTrial(
                            suite=config.name,
                            name=name,
                            key=key,
                            target=target,
                            scenario=scenario,
                            concurrency=concurrency,
                            max_tokens=max_tokens,
                        )
                    )
    return trials


def print_http_load_plan(
    config: HttpLoadConfig,
    *,
    target_names: tuple[str, ...] = (),
    scenario_names: tuple[str, ...] = (),
    limit: int | None = None,
) -> None:
    all_trials = expand_http_load_trials(
        config,
        target_names=target_names,
        scenario_names=scenario_names,
    )
    trials = all_trials[:limit] if limit is not None else all_trials
    for index, trial in enumerate(trials):
        print(f"# [{index}] {trial.name}")
        print(trial.command_text)
        print()
    print(f"planned {len(trials)} of {len(all_trials)} HTTP load trial(s)")


def run_http_load(
    config: HttpLoadConfig,
    *,
    target_names: tuple[str, ...] = (),
    scenario_names: tuple[str, ...] = (),
    limit: int | None = None,
    dry_run: bool = False,
    keep_going: bool = False,
    database_override: str | Path | None = None,
) -> int:
    trials = expand_http_load_trials(
        config,
        target_names=target_names,
        scenario_names=scenario_names,
    )
    if limit is not None:
        trials = trials[:limit]

    if dry_run:
        print_http_load_plan(
            config,
            target_names=target_names,
            scenario_names=scenario_names,
            limit=limit,
        )
        return 0

    if not trials:
        print("no HTTP load trials to run")
        return 0

    database = Path(database_override) if database_override else config.database
    with ResultStore(database) as store:
        for index, trial in enumerate(trials, start=1):
            try:
                _run_one_http_load(config, trial, store, index=index, total=len(trials))
            except Exception as exc:  # noqa: BLE001 - keep-going needs to persist failures
                if keep_going:
                    print(f"failed: {trial.name}: {exc}")
                    continue
                raise
    return 0


def summarize_request_results(
    results: list[RequestResult], *, wall_duration_s: float, concurrency: int, max_tokens: int
) -> dict[str, float]:
    successes = [result for result in results if result.ok]
    request_count = len(results)
    error_count = request_count - len(successes)
    metrics: dict[str, float] = {
        "request_count": float(request_count),
        "success_count": float(len(successes)),
        "error_count": float(error_count),
        "error_rate": float(error_count / request_count) if request_count else 0.0,
        "concurrency": float(concurrency),
        "max_tokens": float(max_tokens),
        "wall_duration_s": wall_duration_s,
        "requests_per_second": float(len(successes) / wall_duration_s)
        if wall_duration_s > 0
        else 0.0,
    }

    _add_distribution_metrics(metrics, "latency", [result.latency_s for result in successes])
    _add_distribution_metrics(
        metrics,
        "ttft",
        [result.ttft_s for result in successes if result.ttft_s is not None],
    )

    completion_chars_total = sum(result.completion_chars for result in successes)
    metrics["completion_chars_total"] = float(completion_chars_total)
    metrics["completion_chars_per_second"] = (
        float(completion_chars_total / wall_duration_s) if wall_duration_s > 0 else 0.0
    )

    known_completion_tokens = [
        result.completion_tokens for result in successes if result.completion_tokens is not None
    ]
    if known_completion_tokens:
        completion_tokens_total = sum(known_completion_tokens)
        metrics["completion_tokens_total"] = float(completion_tokens_total)
        metrics["completion_tokens_per_second"] = (
            float(completion_tokens_total / wall_duration_s) if wall_duration_s > 0 else 0.0
        )

    known_prompt_tokens = [
        result.prompt_tokens for result in successes if result.prompt_tokens is not None
    ]
    if known_prompt_tokens:
        metrics["prompt_tokens_total"] = float(sum(known_prompt_tokens))

    checked_results = [result for result in successes if result.check_passed is not None]
    if checked_results:
        check_pass_count = sum(1 for result in checked_results if result.check_passed)
        metrics["check_pass_count"] = float(check_pass_count)
        metrics["check_fail_count"] = float(len(checked_results) - check_pass_count)
        metrics["check_pass_rate"] = check_pass_count / len(checked_results)

    eval_tps = [
        result.completion_tokens / result.server_eval_duration_s
        for result in successes
        if result.completion_tokens is not None
        and result.server_eval_duration_s is not None
        and result.server_eval_duration_s > 0
    ]
    _add_distribution_metrics(metrics, "server_eval_tps", eval_tps)

    prompt_eval_tps = [
        result.prompt_tokens / result.server_prompt_eval_duration_s
        for result in successes
        if result.prompt_tokens is not None
        and result.server_prompt_eval_duration_s is not None
        and result.server_prompt_eval_duration_s > 0
    ]
    _add_distribution_metrics(metrics, "server_prompt_eval_tps", prompt_eval_tps)
    return metrics


def _run_one_http_load(
    config: HttpLoadConfig,
    trial: HttpLoadTrial,
    store: ResultStore,
    *,
    index: int,
    total: int,
) -> None:
    run_id = f"{trial.key}-{uuid.uuid4().hex[:8]}"
    artifact_dir = config.database.parent / "artifacts" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = artifact_dir / "responses.jsonl"
    stderr_path = artifact_dir / "errors.txt"

    print(f"[{index}/{total}] {trial.name}")
    print(trial.command_text)

    if trial.scenario.warmup_requests:
        _run_requests(trial, count=trial.scenario.warmup_requests)

    started = utc_now()
    monotonic_start = time.perf_counter()
    results = _run_requests(trial, count=trial.scenario.requests)
    ended = utc_now()
    duration_s = time.perf_counter() - monotonic_start

    metrics = summarize_request_results(
        results,
        wall_duration_s=duration_s,
        concurrency=trial.concurrency,
        max_tokens=trial.max_tokens,
    )
    status = "ok" if metrics["error_count"] == 0 and metrics["success_count"] > 0 else "failed"
    error = _first_error(results) if status != "ok" else None

    stdout_path.write_text(
        "\n".join(json.dumps(result.as_jsonable(), sort_keys=True) for result in results) + "\n",
        encoding="utf-8",
    )
    stderr_path.write_text(
        "\n".join(result.error or "" for result in results if result.error),
        encoding="utf-8",
    )

    store.record_run(
        RunRecord(
            run_id=run_id,
            suite=trial.suite,
            trial_name=trial.name,
            status=status,
            started_at=started,
            ended_at=ended,
            duration_s=duration_s,
            command=trial.command_text,
            cwd=str(Path.cwd()),
            config_json=trial.as_jsonable(),
            metrics=metrics,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            error=error,
        )
    )

    summary = _http_metric_summary(metrics)
    print(f"stored {status}: {run_id} ({summary})")
    if status != "ok":
        raise RuntimeError(f"{trial.name} had HTTP load errors: {error}; artifacts: {stderr_path}")


def _run_requests(trial: HttpLoadTrial, *, count: int) -> list[RequestResult]:
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=trial.concurrency) as executor:
        futures = {
            executor.submit(_execute_http_request, trial, request_index): request_index
            for request_index in range(count)
        }
        for future in as_completed(futures):
            request_index = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - convert worker bugs into stored errors
                results.append(
                    RequestResult(
                        index=request_index,
                        ok=False,
                        status_code=None,
                        latency_s=0.0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
    return sorted(results, key=lambda result: result.index)


def _execute_http_request(trial: HttpLoadTrial, index: int) -> RequestResult:
    if trial.target.provider == "openai":
        return _execute_openai_request(trial, index)
    if trial.target.provider == "ollama":
        return _execute_ollama_request(trial, index)
    raise ValueError(f"unsupported provider: {trial.target.provider}")


def _execute_openai_request(trial: HttpLoadTrial, index: int) -> RequestResult:
    scenario = trial.scenario
    payload: dict[str, Any] = {
        "model": trial.target.model,
        "messages": _messages_for_scenario(scenario),
        "max_tokens": trial.max_tokens,
        "temperature": scenario.temperature,
        "stream": scenario.stream,
    }
    if scenario.seed is not None:
        payload["seed"] = scenario.seed
    if scenario.stream:
        payload["stream_options"] = {"include_usage": True}

    return _post_json(
        trial,
        index,
        url=f"{trial.target.base_url}/chat/completions",
        payload=payload,
        stream_reader=_read_openai_stream if scenario.stream else None,
        body_reader=_read_openai_body,
    )


def _execute_ollama_request(trial: HttpLoadTrial, index: int) -> RequestResult:
    scenario = trial.scenario
    options: dict[str, Any] = {
        "num_predict": trial.max_tokens,
        "temperature": scenario.temperature,
    }
    if scenario.seed is not None:
        options["seed"] = scenario.seed

    payload = {
        "model": trial.target.model,
        "messages": _messages_for_scenario(scenario),
        "stream": scenario.stream,
        "options": options,
    }
    return _post_json(
        trial,
        index,
        url=f"{trial.target.base_url}/api/chat",
        payload=payload,
        stream_reader=_read_ollama_stream if scenario.stream else None,
        body_reader=_read_ollama_body,
    )


def _post_json(
    trial: HttpLoadTrial,
    index: int,
    *,
    url: str,
    payload: dict[str, Any],
    stream_reader: Any,
    body_reader: Any,
) -> RequestResult:
    start = time.perf_counter()
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers_for_target(trial.target),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=trial.scenario.timeout_s) as response:  # noqa: S310 - user-configured local/server URL
            status_code = response.status
            if stream_reader is not None:
                result = stream_reader(index, response, start, status_code)
            else:
                body = response.read().decode("utf-8", errors="replace")
                result = body_reader(index, body, start, status_code)
            return _with_check_result(result, trial.scenario)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[-1000:]
        return RequestResult(
            index=index,
            ok=False,
            status_code=exc.code,
            latency_s=time.perf_counter() - start,
            error=f"HTTP {exc.code}: {body}",
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return RequestResult(
            index=index,
            ok=False,
            status_code=None,
            latency_s=time.perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
        )


def _read_openai_stream(
    index: int,
    response: Any,
    start: float,
    status_code: int,
) -> RequestResult:
    text_parts: list[str] = []
    ttft_s: float | None = None
    usage: dict[str, Any] | None = None
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            continue
        chunk = json.loads(data)
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            content = _openai_choice_text(choice)
            if content:
                if ttft_s is None:
                    ttft_s = time.perf_counter() - start
                text_parts.append(content)

    response_text = "".join(text_parts)
    return RequestResult(
        index=index,
        ok=200 <= status_code < 300,
        status_code=status_code,
        latency_s=time.perf_counter() - start,
        ttft_s=ttft_s,
        prompt_tokens=_int_from_mapping(usage, "prompt_tokens"),
        completion_tokens=_int_from_mapping(usage, "completion_tokens"),
        completion_chars=len(response_text),
        response_text=response_text,
    )


def _read_openai_body(index: int, body: str, start: float, status_code: int) -> RequestResult:
    payload = json.loads(body)
    text_parts: list[str] = []
    for choice in payload.get("choices") or []:
        content = _openai_choice_text(choice)
        if content:
            text_parts.append(content)
    response_text = "".join(text_parts)
    usage = payload.get("usage") or {}
    return RequestResult(
        index=index,
        ok=200 <= status_code < 300,
        status_code=status_code,
        latency_s=time.perf_counter() - start,
        prompt_tokens=_int_from_mapping(usage, "prompt_tokens"),
        completion_tokens=_int_from_mapping(usage, "completion_tokens"),
        completion_chars=len(response_text),
        response_text=response_text,
    )


def _read_ollama_stream(
    index: int,
    response: Any,
    start: float,
    status_code: int,
) -> RequestResult:
    text_parts: list[str] = []
    ttft_s: float | None = None
    final: dict[str, Any] = {}
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        chunk = json.loads(line)
        if chunk.get("error"):
            raise RuntimeError(str(chunk["error"]))
        message = chunk.get("message") or {}
        content = message.get("content") or message.get("thinking") or chunk.get("response") or ""
        if content:
            if ttft_s is None:
                ttft_s = time.perf_counter() - start
            text_parts.append(str(content))
        if chunk.get("done"):
            final = chunk

    response_text = "".join(text_parts)
    return RequestResult(
        index=index,
        ok=200 <= status_code < 300,
        status_code=status_code,
        latency_s=time.perf_counter() - start,
        ttft_s=ttft_s,
        prompt_tokens=_int_from_mapping(final, "prompt_eval_count"),
        completion_tokens=_int_from_mapping(final, "eval_count"),
        completion_chars=len(response_text),
        server_prompt_eval_duration_s=_ns_to_s(final.get("prompt_eval_duration")),
        server_eval_duration_s=_ns_to_s(final.get("eval_duration")),
        response_text=response_text,
    )


def _read_ollama_body(index: int, body: str, start: float, status_code: int) -> RequestResult:
    payload = json.loads(body)
    message = payload.get("message") or {}
    response_text = str(
        message.get("content") or message.get("thinking") or payload.get("response") or ""
    )
    return RequestResult(
        index=index,
        ok=200 <= status_code < 300,
        status_code=status_code,
        latency_s=time.perf_counter() - start,
        prompt_tokens=_int_from_mapping(payload, "prompt_eval_count"),
        completion_tokens=_int_from_mapping(payload, "eval_count"),
        completion_chars=len(response_text),
        server_prompt_eval_duration_s=_ns_to_s(payload.get("prompt_eval_duration")),
        server_eval_duration_s=_ns_to_s(payload.get("eval_duration")),
        response_text=response_text,
    )


def _openai_choice_text(choice: dict[str, Any]) -> str:
    parts: list[str] = []
    for mapping in (choice.get("delta"), choice.get("message"), choice):
        if not isinstance(mapping, dict):
            continue
        for key in ("content", "reasoning_content", "thinking", "text"):
            value = mapping.get(key)
            if value:
                parts.append(str(value))
    return "".join(parts)


def _with_check_result(result: RequestResult, scenario: HttpScenario) -> RequestResult:
    if not result.ok or not scenario.expected_contains:
        return result
    response_text = result.response_text.lower()
    check_passed = all(fragment.lower() in response_text for fragment in scenario.expected_contains)
    return replace(result, check_passed=check_passed)


def _messages_for_scenario(scenario: HttpScenario) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if scenario.system:
        messages.append({"role": "system", "content": scenario.system})
    messages.append({"role": "user", "content": scenario.rendered_prompt})
    return messages


def _headers_for_target(target: HttpTarget) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        **target.headers,
    }
    if target.api_key_env and "Authorization" not in headers:
        token = os.environ.get(target.api_key_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


def _scenario_prompt(raw: dict[str, Any], *, base_dir: Path) -> str:
    has_prompt = raw.get("prompt") is not None
    has_prompt_file = raw.get("prompt_file") is not None
    if has_prompt == has_prompt_file:
        raise ConfigError("each HTTP scenario must set exactly one of 'prompt' or 'prompt_file'")
    if has_prompt:
        return str(raw["prompt"])

    prompt_path = Path(str(raw["prompt_file"]))
    if not prompt_path.is_absolute():
        prompt_path = base_dir / prompt_path
    return prompt_path.read_text(encoding="utf-8")


def _add_distribution_metrics(
    metrics: dict[str, float], prefix: str, values: list[float | None]
) -> None:
    clean_values = sorted(float(value) for value in values if value is not None)
    if not clean_values:
        return

    unit_suffix = "" if prefix.endswith("tps") else "_s"
    metrics[f"{prefix}_avg{unit_suffix}"] = sum(clean_values) / len(clean_values)
    metrics[f"{prefix}_min{unit_suffix}"] = clean_values[0]
    metrics[f"{prefix}_max{unit_suffix}"] = clean_values[-1]
    for percentile in PERCENTILES:
        metrics[f"{prefix}_p{percentile}{unit_suffix}"] = _percentile(
            clean_values,
            percentile / 100,
        )


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _int_from_mapping(mapping: dict[str, Any] | None, key: str) -> int | None:
    if not mapping:
        return None
    value = mapping.get(key)
    if value is None:
        return None
    return int(value)


def _ns_to_s(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) / 1_000_000_000


def _first_error(results: list[RequestResult]) -> str | None:
    for result in results:
        if result.error:
            return result.error
    return None


def _http_metric_summary(metrics: dict[str, float]) -> str:
    keys = [
        "requests_per_second",
        "latency_p95_s",
        "ttft_p95_s",
        "completion_tokens_per_second",
        "check_pass_rate",
        "error_count",
    ]
    return ", ".join(f"{key}={metrics[key]:.3f}" for key in keys if key in metrics)
