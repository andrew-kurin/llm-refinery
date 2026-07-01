from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from llm_refinery.core.runs import record_benchmark_run
from llm_refinery.storage import ResultStore, utc_now

TARGET_CHOICES = ("llama_cpp", "ollama", "mlx_e4b", "mlx_26b", "both", "all")
TARGET_ORDER = ("llama_cpp", "ollama", "mlx_e4b", "mlx_26b")


@dataclass(frozen=True)
class LmEvalTarget:
    name: str
    model: str
    base_url: str


@dataclass(frozen=True)
class LmEvalConfig:
    target: str = "llama_cpp"
    limit: int | None = 50
    tasks: str = "ifeval,gsm8k"
    num_concurrent: int = 1
    max_retries: int = 3
    max_length: int = 16384
    eos_string: str = "<turn|>"
    log_samples: bool = False
    gen_kwargs: str | None = None
    output_root: Path = Path("results/lm_eval")
    offline: bool = True
    model_backend: str = "local-chat-completions"
    apply_chat_template: bool = True
    include_path: Path | None = None
    suite_name: str = "lm-eval"
    database: Path = Path("results/llm_refinery.duckdb")
    targets: dict[str, LmEvalTarget] = field(default_factory=dict)


def default_targets(env: dict[str, str] | None = None) -> dict[str, LmEvalTarget]:
    env = os.environ if env is None else env
    return {
        "llama_cpp": LmEvalTarget(
            name="llama_cpp",
            model=env.get("LLAMA_CPP_MODEL", "local-model"),
            base_url=env.get(
                "LLAMA_CPP_BASE_URL", "http://127.0.0.1:8080/v1/chat/completions"
            ),
        ),
        "ollama": LmEvalTarget(
            name="ollama",
            model=env.get("OLLAMA_MODEL", "gemma4:26b"),
            base_url=env.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1/chat/completions"),
        ),
        "mlx_e4b": LmEvalTarget(
            name="mlx_e4b",
            model=env.get("MLX_E4B_MODEL", "mlx-community/gemma-4-e4b-it-OptiQ-4bit"),
            base_url=env.get("MLX_E4B_BASE_URL", "http://127.0.0.1:8081/v1/chat/completions"),
        ),
        "mlx_26b": LmEvalTarget(
            name="mlx_26b",
            model=env.get("MLX_26B_MODEL", "mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit"),
            base_url=env.get("MLX_26B_BASE_URL", "http://127.0.0.1:8082/v1/chat/completions"),
        ),
    }


def resolve_target_names(target: str) -> list[str]:
    if target == "both":
        return ["llama_cpp", "ollama"]
    if target == "all":
        return list(TARGET_ORDER)
    if target not in TARGET_ORDER:
        raise ValueError(f"unknown lm-eval target: {target}")
    return [target]


def build_lm_eval_command(config: LmEvalConfig, target: LmEvalTarget) -> list[str]:
    output_path = str(config.output_root / target.name)
    model_args = (
        f"model={target.model},"
        f"base_url={target.base_url},"
        f"num_concurrent={config.num_concurrent},"
        f"max_retries={config.max_retries},"
        f"eos_string={config.eos_string},"
        f"max_length={config.max_length}"
    )

    cmd = [
        "uvx",
        "--from",
        "lm-eval[api]",
        "--with",
        "langdetect",
        "--with",
        "immutabledict",
        "lm_eval",
        "--model",
        config.model_backend,
        "--model_args",
        model_args,
        "--tasks",
        config.tasks,
        "--batch_size",
        "1",
    ]

    if config.limit is not None:
        cmd.extend(["--limit", str(config.limit)])

    if config.apply_chat_template:
        cmd.append("--apply_chat_template")

    if config.include_path is not None:
        cmd.extend(["--include_path", str(config.include_path)])
    if config.log_samples:
        cmd.append("--log_samples")
    if config.gen_kwargs:
        cmd.extend(["--gen_kwargs", config.gen_kwargs])

    cmd.extend(["--output_path", output_path])
    return cmd


def run_lm_eval(config: LmEvalConfig, *, dry_run: bool = False) -> None:
    targets = {**default_targets(), **config.targets}
    selected = resolve_target_names(config.target)
    config.output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_DATASETS_OFFLINE"] = "1" if config.offline else "0"
    for target_name in selected:
        target = targets[target_name]
        limit_text = str(config.limit) if config.limit is not None else "all"
        model_args = (
            f"model={target.model},"
            f"base_url={target.base_url},"
            f"num_concurrent={config.num_concurrent},"
            f"max_retries={config.max_retries},"
            f"eos_string={config.eos_string},"
            f"max_length={config.max_length}"
        )
        output_path = config.output_root / target.name

        print(
            f"==> Running lm-eval target={target.name} "
            f"tasks={config.tasks} limit={limit_text}"
        )
        print(f"    model_args={model_args}")
        print(f"    output_path={output_path}")

        cmd = build_lm_eval_command(config, target)
        if dry_run:
            print(shlex.join(cmd))
            continue

        started_at = utc_now()
        result_started_mtime = time.time()
        completed = subprocess.run(cmd, env=env, check=False)
        ended_at = utc_now()
        result_json = latest_lm_eval_result(output_path, newer_than=result_started_mtime)
        metrics = parse_lm_eval_metrics(result_json) if result_json else {}
        with ResultStore(config.database) as store:
            record_benchmark_run(
                store,
                run_id=(
                    f"{config.suite_name}-{target.name}-"
                    f"{started_at.strftime('%Y%m%d%H%M%S%f')}"
                ),
                suite=config.suite_name,
                trial_name=f"{config.suite_name}/{target.name}",
                status="ok" if completed.returncode == 0 else "failed",
                started_at=started_at,
                ended_at=ended_at,
                duration_s=(ended_at - started_at).total_seconds(),
                command=shlex.join(cmd),
                config_json={
                    "benchmark": "lm-eval",
                    "model_backend": config.model_backend,
                    "apply_chat_template": config.apply_chat_template,
                    "target": target.name,
                    "model": target.model,
                    "base_url": target.base_url,
                    "tasks": config.tasks,
                    "limit": config.limit,
                    "max_length": config.max_length,
                    "eos_string": config.eos_string,
                    "gen_kwargs": config.gen_kwargs,
                    "include_path": str(config.include_path) if config.include_path else None,
                },
                metrics=metrics,
                stdout_path=result_json,
                error=None if completed.returncode == 0 else f"exit code {completed.returncode}",
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"lm-eval failed for {target.name}: exit code {completed.returncode}"
            )


def latest_lm_eval_result(output_path: Path, *, newer_than: float | None = None) -> Path | None:
    if not output_path.exists():
        return None
    result_candidates = [p for p in output_path.rglob("*.json") if "result" in p.name.lower()]
    candidates = result_candidates or list(output_path.rglob("*.json"))
    if newer_than is not None:
        candidates = [path for path in candidates if path.stat().st_mtime >= newer_than]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def parse_lm_eval_metrics(result_path: Path) -> dict[str, float]:
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    results = raw.get("results") or {}
    metrics: dict[str, float] = {}
    for task_name, task_results in results.items():
        if not isinstance(task_results, dict):
            continue
        for raw_name, value in task_results.items():
            if not isinstance(value, (int, float)):
                continue
            metric_name = normalize_lm_eval_metric_name(str(task_name), str(raw_name))
            metrics[metric_name] = float(value)
    return metrics


def normalize_lm_eval_metric_name(task_name: str, raw_name: str) -> str:
    if "," not in raw_name:
        return f"{task_name}.{raw_name}"
    metric, filter_name = raw_name.split(",", 1)
    return f"{task_name}.{filter_name}.{metric}"
