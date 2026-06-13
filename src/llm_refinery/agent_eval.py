from __future__ import annotations

import ast
import csv
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import yaml

from llm_refinery.config import ConfigError, coerce_list, stable_hash
from llm_refinery.storage import ResultStore, RunRecord, utc_now
from llm_refinery.utils.system import get_system_profile

DEFAULT_GEOANALYSTBENCH_DATASET = (
    "https://raw.githubusercontent.com/GeoDS/GeoAnalystBench/"
    "master/dataset/GeoAnalystBench.csv"
)
PERCENTILES = (50, 90, 95, 99)
PROMPT_VARIANTS = {"original", "domain", "dataset", "domain_and_dataset"}
RESPONSE_TYPES = {"workflow", "code"}


class _Unset:
    pass


_UNSET = _Unset()


class ChatClient(Protocol):
    def complete(self, target: AgentEvalTarget, request: AgentEvalRequest) -> AgentEvalResult: ...


@dataclass(frozen=True)
class AgentEvalTarget:
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> AgentEvalTarget:
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ConfigError("each agent-eval target requires a non-empty 'name'")

        provider = str(raw.get("provider") or "openai").strip().lower()
        if provider != "openai":
            raise ConfigError(f"target {name!r} provider must be 'openai', got {provider!r}")

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
            headers={str(k): str(v) for k, v in dict(raw.get("headers") or {}).items()},
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
class AgentEvalRequestConfig:
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 600.0
    retries: int = 1
    seed: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> AgentEvalRequestConfig:
        raw = raw or {}
        return cls(
            temperature=float(raw.get("temperature", 0.0)),
            max_tokens=int(raw.get("max_tokens", 1024)),
            timeout_s=float(raw.get("timeout_s", 600.0)),
            retries=int(raw.get("retries", 1)),
            seed=int(raw["seed"]) if raw.get("seed") is not None else None,
            extra_body=dict(raw.get("extra_body") or {}),
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_s": self.timeout_s,
            "retries": self.retries,
            "seed": self.seed,
            "extra_body": self.extra_body,
        }


@dataclass(frozen=True)
class GeoAnalystBenchSpec:
    dataset: str = DEFAULT_GEOANALYSTBENCH_DATASET
    task_ids: tuple[int, ...] = ()
    limit: int | None = 5
    open_source_only: bool = True
    prompt_variants: tuple[str, ...] = ("domain_and_dataset",)
    response_types: tuple[str, ...] = ("workflow", "code")

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> GeoAnalystBenchSpec:
        dataset = str(
            raw.get("dataset") or raw.get("dataset_url") or DEFAULT_GEOANALYSTBENCH_DATASET
        )
        task_ids = tuple(int(value) for value in coerce_list(raw.get("task_ids") or []))
        limit_raw = raw.get("limit", 5)
        limit = None if limit_raw is None or str(limit_raw).lower() == "all" else int(limit_raw)
        if limit is not None and limit <= 0:
            raise ConfigError("benchmark.limit must be a positive integer or 'all'")

        prompt_variants = tuple(
            str(v) for v in coerce_list(raw.get("prompt_variants") or ["domain_and_dataset"])
        )
        unknown_variants = sorted(set(prompt_variants) - PROMPT_VARIANTS)
        if unknown_variants:
            raise ConfigError(
                f"unknown GeoAnalystBench prompt variant(s): {', '.join(unknown_variants)}"
            )

        response_types = tuple(
            str(v) for v in coerce_list(raw.get("response_types") or ["workflow", "code"])
        )
        unknown_response_types = sorted(set(response_types) - RESPONSE_TYPES)
        if unknown_response_types:
            raise ConfigError(
                f"unknown GeoAnalystBench response type(s): {', '.join(unknown_response_types)}"
            )

        return cls(
            dataset=dataset,
            task_ids=task_ids,
            limit=limit,
            open_source_only=bool(raw.get("open_source_only", True)),
            prompt_variants=prompt_variants,
            response_types=response_types,
        )

    def with_overrides(
        self, *, limit: int | None | object = _UNSET, task_ids: tuple[int, ...] | None = None
    ) -> GeoAnalystBenchSpec:
        updates: dict[str, Any] = {}
        if limit is not _UNSET:
            updates["limit"] = limit
        if task_ids is not None:
            updates["task_ids"] = task_ids
        return replace(self, **updates)

    def safe_json(self) -> dict[str, Any]:
        return {
            "kind": "geoanalystbench",
            "dataset": self.dataset,
            "task_ids": list(self.task_ids),
            "limit": self.limit,
            "open_source_only": self.open_source_only,
            "prompt_variants": list(self.prompt_variants),
            "response_types": list(self.response_types),
        }


@dataclass(frozen=True)
class AgentEvalConfig:
    name: str
    database: Path
    benchmark: GeoAnalystBenchSpec
    targets: list[AgentEvalTarget]
    request: AgentEvalRequestConfig = field(default_factory=AgentEvalRequestConfig)
    source_path: Path | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], source_path: Path | None = None) -> AgentEvalConfig:
        name = str(raw.get("name") or (source_path.stem if source_path else "agent-eval"))
        targets_raw = raw.get("targets") or []
        if not targets_raw:
            raise ConfigError("agent-eval config requires at least one target in 'targets'")

        benchmark_raw = dict(raw.get("benchmark") or {})
        kind = str(benchmark_raw.get("kind") or "geoanalystbench").strip().lower()
        if kind != "geoanalystbench":
            raise ConfigError(f"unsupported agent-eval benchmark kind: {kind!r}")

        benchmark = GeoAnalystBenchSpec.from_mapping(benchmark_raw)
        if source_path and not _is_url(benchmark.dataset):
            dataset_path = Path(benchmark.dataset)
            if not dataset_path.is_absolute():
                benchmark = replace(benchmark, dataset=str(source_path.parent / dataset_path))

        return cls(
            name=name,
            database=Path(str(raw.get("database") or "results/llm_refinery.duckdb")),
            benchmark=benchmark,
            targets=[AgentEvalTarget.from_mapping(dict(item)) for item in targets_raw],
            request=AgentEvalRequestConfig.from_mapping(raw.get("request")),
            source_path=source_path,
        )


def load_agent_eval_config(path: str | Path) -> AgentEvalConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")
    return AgentEvalConfig.from_mapping(raw, source_path=config_path)


@dataclass(frozen=True)
class GeoAnalystTask:
    task_id: int
    open_source: bool
    task: str
    instruction: str
    domain_knowledge: str
    dataset_description: str
    human_workflow: str
    task_length: int
    code: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> GeoAnalystTask:
        return cls(
            task_id=int(row["id"]),
            open_source=str(row.get("Open Source") or "").strip().upper() == "T",
            task=str(row.get("Task") or ""),
            instruction=str(row.get("Instruction") or ""),
            domain_knowledge=str(row.get("Domain Knowledge") or ""),
            dataset_description=str(row.get("Dataset Description") or ""),
            human_workflow=str(row.get("Human Designed Workflow") or ""),
            task_length=int(row.get("Task Length") or 0),
            code=str(row.get("CodeString") or ""),
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "open_source": self.open_source,
            "task": self.task,
            "task_length": self.task_length,
        }


@dataclass(frozen=True)
class AgentEvalRequest:
    task: GeoAnalystTask
    prompt_variant: str
    response_type: str
    system: str
    prompt: str
    config: AgentEvalRequestConfig

    @property
    def key(self) -> str:
        return stable_hash(
            {
                "task_id": self.task.task_id,
                "prompt_variant": self.prompt_variant,
                "response_type": self.response_type,
                "prompt": self.prompt,
                "request": self.config.safe_json(),
            }
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "task": self.task.safe_json(),
            "prompt_variant": self.prompt_variant,
            "response_type": self.response_type,
            "system": self.system,
            "prompt_preview": self.prompt[:500],
            "prompt_chars": len(self.prompt),
            "prompt_hash": stable_hash(self.prompt),
            "request": self.config.safe_json(),
        }


@dataclass(frozen=True)
class AgentEvalResult:
    request: AgentEvalRequest
    ok: bool
    latency_s: float
    response_text: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    workflow_step_count: int | None = None
    workflow_step_abs_error: int | None = None
    code_syntax_ok: bool | None = None
    error: str | None = None

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "request": self.request.safe_json(),
            "ok": self.ok,
            "latency_s": self.latency_s,
            "response_text": self.response_text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "workflow_step_count": self.workflow_step_count,
            "workflow_step_abs_error": self.workflow_step_abs_error,
            "code_syntax_ok": self.code_syntax_ok,
            "error": self.error,
        }


class OpenAIChatClient:
    def complete(self, target: AgentEvalTarget, request: AgentEvalRequest) -> AgentEvalResult:
        started = time.perf_counter()
        last_error: str | None = None
        for attempt in range(request.config.retries + 1):
            try:
                return self._complete_once(target, request, started_at=started)
            except Exception as exc:  # noqa: BLE001 - store benchmark request failures
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= request.config.retries:
                    break
        return AgentEvalResult(
            request=request,
            ok=False,
            latency_s=time.perf_counter() - started,
            error=last_error,
        )

    def _complete_once(
        self, target: AgentEvalTarget, request: AgentEvalRequest, *, started_at: float
    ) -> AgentEvalResult:
        payload: dict[str, Any] = {
            "model": target.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": request.config.temperature,
            "max_tokens": request.config.max_tokens,
            "stream": False,
        }
        if request.config.seed is not None:
            payload["seed"] = request.config.seed
        payload.update(request.config.extra_body)

        headers = {"Content-Type": "application/json", **target.headers}
        if target.api_key_env and os.environ.get(target.api_key_env):
            headers.setdefault("Authorization", f"Bearer {os.environ[target.api_key_env]}")
        url = _chat_completions_url(target.base_url)
        http_request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 - benchmark target is user-configured
                http_request,
                timeout=request.config.timeout_s,
            ) as response:
                body = response.read().decode(errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[-2000:]
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

        data = json.loads(body)
        choice = data["choices"][0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "")
        usage = data.get("usage") or {}
        return _score_result(
            AgentEvalResult(
                request=request,
                ok=bool(content.strip()),
                latency_s=time.perf_counter() - started_at,
                response_text=content,
                prompt_tokens=_int_or_none(usage.get("prompt_tokens")),
                completion_tokens=_int_or_none(usage.get("completion_tokens")),
                total_tokens=_int_or_none(usage.get("total_tokens")),
                error=None if content.strip() else "empty response content",
            )
        )


def run_agent_eval(
    config: AgentEvalConfig,
    *,
    target_names: tuple[str, ...] = (),
    limit: int | None | object = _UNSET,
    task_ids: tuple[int, ...] = (),
    dry_run: bool = False,
    client: ChatClient | None = None,
) -> int:
    benchmark = config.benchmark.with_overrides(
        limit=limit,
        task_ids=task_ids or None,
    )
    tasks = select_geoanalyst_tasks(load_geoanalyst_tasks(benchmark.dataset), benchmark)
    targets = _selected_targets(config.targets, target_names)
    requests = expand_geoanalyst_requests(tasks, benchmark, config.request)

    if dry_run:
        for target in targets:
            print(
                f"agent-eval benchmark=geoanalystbench target={target.name} "
                f"model={target.model} requests={len(requests)} tasks={len(tasks)}"
            )
        return 0

    chat_client = client or OpenAIChatClient()
    database = config.database
    artifact_root = database.parent / "artifacts"
    with ResultStore(database) as store:
        for target in targets:
            _run_target(config, benchmark, target, requests, chat_client, artifact_root, store)
    return 0


def load_geoanalyst_tasks(dataset: str) -> list[GeoAnalystTask]:
    if _is_url(dataset):
        with urllib.request.urlopen(dataset, timeout=60) as response:  # noqa: S310 user config
            text = response.read().decode("utf-8-sig", errors="replace")
    else:
        text = Path(dataset).read_text(encoding="utf-8-sig")

    rows = csv.DictReader(io.StringIO(text))
    return [GeoAnalystTask.from_row(row) for row in rows]


def select_geoanalyst_tasks(
    tasks: list[GeoAnalystTask], benchmark: GeoAnalystBenchSpec
) -> list[GeoAnalystTask]:
    selected = tasks
    if benchmark.open_source_only:
        selected = [task for task in selected if task.open_source]
    if benchmark.task_ids:
        wanted = set(benchmark.task_ids)
        selected = [task for task in selected if task.task_id in wanted]
        missing = wanted - {task.task_id for task in selected}
        if missing:
            missing_text = ", ".join(map(str, sorted(missing)))
            raise ConfigError(f"GeoAnalystBench task id(s) not found: {missing_text}")
    if benchmark.limit is not None:
        selected = selected[: benchmark.limit]
    if not selected:
        raise ConfigError("GeoAnalystBench task selection is empty")
    return selected


def expand_geoanalyst_requests(
    tasks: list[GeoAnalystTask],
    benchmark: GeoAnalystBenchSpec,
    request_config: AgentEvalRequestConfig,
) -> list[AgentEvalRequest]:
    requests: list[AgentEvalRequest] = []
    for task in tasks:
        for prompt_variant in benchmark.prompt_variants:
            for response_type in benchmark.response_types:
                system, prompt = build_geoanalyst_prompt(task, prompt_variant, response_type)
                requests.append(
                    AgentEvalRequest(
                        task=task,
                        prompt_variant=prompt_variant,
                        response_type=response_type,
                        system=system,
                        prompt=prompt,
                        config=request_config,
                    )
                )
    return requests


def build_geoanalyst_prompt(
    task: GeoAnalystTask, prompt_variant: str, response_type: str
) -> tuple[str, str]:
    system = (
        "You are a careful geospatial Python analyst. Prefer open-source Python GIS "
        "libraries such as GeoPandas, Shapely, Rasterio, PyProj, Xarray, NumPy, and "
        "Matplotlib unless the task explicitly requires ArcPy."
    )
    sections = [
        f"Task ID: {task.task_id}",
        f"Task: {task.task}",
        "Instruction:",
        task.instruction,
    ]
    if prompt_variant in {"domain", "domain_and_dataset"} and task.domain_knowledge.strip():
        sections.extend(["Domain knowledge:", task.domain_knowledge])
    if prompt_variant in {"dataset", "domain_and_dataset"} and task.dataset_description.strip():
        sections.extend(["Dataset description:", task.dataset_description])

    if response_type == "workflow":
        sections.extend(
            [
                "Output requirements:",
                "Return only a numbered spatial-analysis workflow. Keep each step concise. "
                "Do not include code.",
            ]
        )
    elif response_type == "code":
        sections.extend(
            [
                "Output requirements:",
                "Return only Python code. Put all logic in a function named model(). "
                "Do not include Markdown fences or explanatory prose.",
            ]
        )
    else:
        raise ConfigError(f"unsupported response type: {response_type}")

    return system, "\n\n".join(sections)


def summarize_agent_eval_results(
    results: list[AgentEvalResult], wall_duration_s: float
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "request_count": float(len(results)),
        "success_count": float(sum(1 for result in results if result.ok)),
        "error_count": float(sum(1 for result in results if not result.ok)),
        "wall_duration_s": float(wall_duration_s),
    }
    if results:
        metrics["success_rate"] = metrics["success_count"] / len(results)
    successes = [result for result in results if result.ok]
    _add_distribution_metrics(metrics, "latency", [result.latency_s for result in successes])
    _add_distribution_metrics(
        metrics,
        "response_chars",
        [float(len(result.response_text)) for result in successes],
        unit_suffix="",
    )

    known_completion_tokens = [
        result.completion_tokens
        for result in successes
        if result.completion_tokens is not None
    ]
    if known_completion_tokens:
        metrics["completion_tokens_total"] = float(sum(known_completion_tokens))
        metrics["completion_tokens_avg"] = sum(known_completion_tokens) / len(
            known_completion_tokens
        )
        metrics["completion_tokens_per_second"] = (
            sum(known_completion_tokens) / wall_duration_s if wall_duration_s else 0.0
        )

    workflow_results = [
        result
        for result in successes
        if result.request.response_type == "workflow" and result.workflow_step_count is not None
    ]
    if workflow_results:
        metrics["workflow_count"] = float(len(workflow_results))
        metrics["workflow_step_count_avg"] = sum(
            result.workflow_step_count or 0 for result in workflow_results
        ) / len(workflow_results)
        metrics["workflow_step_abs_error_avg"] = sum(
            result.workflow_step_abs_error or 0 for result in workflow_results
        ) / len(workflow_results)

    code_results = [
        result
        for result in successes
        if result.request.response_type == "code" and result.code_syntax_ok is not None
    ]
    if code_results:
        syntax_pass = sum(1 for result in code_results if result.code_syntax_ok)
        metrics["code_count"] = float(len(code_results))
        metrics["code_syntax_pass_count"] = float(syntax_pass)
        metrics["code_syntax_pass_rate"] = syntax_pass / len(code_results)

    return metrics


def _run_target(
    config: AgentEvalConfig,
    benchmark: GeoAnalystBenchSpec,
    target: AgentEvalTarget,
    requests: list[AgentEvalRequest],
    client: ChatClient,
    artifact_root: Path,
    store: ResultStore,
) -> None:
    key = stable_hash(
        {
            "suite": config.name,
            "benchmark": benchmark.safe_json(),
            "target": target.safe_json(),
            "request_count": len(requests),
            "request": config.request.safe_json(),
        }
    )
    run_id = f"{key}-{uuid.uuid4().hex[:8]}"
    artifact_dir = artifact_root / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    responses_path = artifact_dir / "geoanalystbench-responses.jsonl"
    errors_path = artifact_dir / "errors.txt"

    print(
        f"agent-eval benchmark=geoanalystbench target={target.name} "
        f"requests={len(requests)} model={target.model}"
    )
    started = utc_now()
    monotonic_start = time.perf_counter()
    results: list[AgentEvalResult] = []
    for index, request in enumerate(requests, start=1):
        print(
            f"[{index}/{len(requests)}] task={request.task.task_id} "
            f"type={request.response_type} variant={request.prompt_variant}",
            flush=True,
        )
        result = client.complete(target, request)
        results.append(result)
    ended = utc_now()
    duration_s = time.perf_counter() - monotonic_start

    metrics = summarize_agent_eval_results(results, duration_s)
    status = "ok" if metrics.get("error_count", 0.0) == 0 and results else "failed"
    error = _first_error(results) if status != "ok" else None

    responses_path.write_text(
        "\n".join(json.dumps(result.as_jsonable(), sort_keys=True) for result in results) + "\n",
        encoding="utf-8",
    )
    errors_path.write_text(
        "\n".join(result.error or "" for result in results if result.error),
        encoding="utf-8",
    )

    trial_name = f"{config.name}/{target.name}/geoanalystbench/{key}"
    store.record_run(
        RunRecord(
            run_id=run_id,
            suite=config.name,
            trial_name=trial_name,
            status=status,
            started_at=started,
            ended_at=ended,
            duration_s=duration_s,
            command=(
                f"agent-eval benchmark=geoanalystbench target={target.name} "
                f"model={target.model} requests={len(requests)}"
            ),
            cwd=str(Path.cwd()),
            config_json={
                "benchmark": benchmark.safe_json(),
                "target": target.safe_json(),
                "request": config.request.safe_json(),
                "params": {
                    "benchmark": "geoanalystbench",
                    "target": target.name,
                    "model": target.model,
                    "prompt_variants": ",".join(benchmark.prompt_variants),
                    "response_types": ",".join(benchmark.response_types),
                    "task_count": len({request.task.task_id for request in requests}),
                    "request_count": len(requests),
                },
                "model": {"name": target.model},
                "prompt_tokens": None,
                "gen_tokens": config.request.max_tokens,
            },
            metrics=metrics,
            system_json=get_system_profile(),
            stdout_path=str(responses_path),
            stderr_path=str(errors_path),
            error=error,
        )
    )
    print(f"stored {status}: {run_id} ({_metric_summary(metrics)})")


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _selected_targets(
    targets: list[AgentEvalTarget], target_names: tuple[str, ...]
) -> list[AgentEvalTarget]:
    wanted = set(target_names)
    selected = [target for target in targets if not wanted or target.name in wanted]
    missing = wanted - {target.name for target in selected}
    if missing:
        raise ConfigError(f"unknown agent-eval target(s): {', '.join(sorted(missing))}")
    return selected


def _score_result(result: AgentEvalResult) -> AgentEvalResult:
    if not result.ok:
        return result
    if result.request.response_type == "workflow":
        step_count = extract_workflow_step_count(result.response_text)
        return replace(
            result,
            workflow_step_count=step_count,
            workflow_step_abs_error=abs(step_count - result.request.task.task_length),
        )
    if result.request.response_type == "code":
        code = extract_python_code(result.response_text)
        try:
            ast.parse(code)
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False
        return replace(result, code_syntax_ok=syntax_ok)
    return result


def extract_workflow_step_count(text: str) -> int:
    numbers: list[int] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d{1,2})[\.)]\s+", line)
        if match:
            numbers.append(int(match.group(1)))
    if numbers:
        return max(numbers)
    bullet_lines = [line for line in text.splitlines() if re.match(r"^\s*[-*]\s+", line)]
    return len(bullet_lines)


def extract_python_code(text: str) -> str:
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _chat_completions_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    return f"{stripped}/chat/completions"


def _first_error(results: list[AgentEvalResult]) -> str | None:
    for result in results:
        if result.error:
            return result.error
    return None


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _add_distribution_metrics(
    metrics: dict[str, float],
    prefix: str,
    values: list[float | None],
    *,
    unit_suffix: str = "_s",
) -> None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return
    metrics[f"{prefix}_avg{unit_suffix}"] = sum(clean) / len(clean)
    metrics[f"{prefix}_min{unit_suffix}"] = clean[0]
    metrics[f"{prefix}_max{unit_suffix}"] = clean[-1]
    for percentile in PERCENTILES:
        metrics[f"{prefix}_p{percentile}{unit_suffix}"] = _percentile(clean, percentile)


def _percentile(values: list[float], percentile: int) -> float:
    if len(values) == 1:
        return values[0]
    rank = (percentile / 100) * (len(values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(values) - 1)
    weight = rank - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _metric_summary(metrics: dict[str, float]) -> str:
    keys = [
        "success_rate",
        "latency_p95_s",
        "workflow_step_abs_error_avg",
        "code_syntax_pass_rate",
        "error_count",
    ]
    return ", ".join(f"{key}={metrics[key]:.3f}" for key in keys if key in metrics)
