from __future__ import annotations

import os
import shlex
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

from llm_refinery.application.run_session import RunSession
from llm_refinery.benchmarks.lm_eval.command import build_lm_eval_command
from llm_refinery.benchmarks.lm_eval.config import LmEvalConfig, resolve_target_names
from llm_refinery.benchmarks.lm_eval.parser import latest_lm_eval_result, parse_lm_eval_metrics
from llm_refinery.benchmarks.lm_eval.presets import default_targets
from llm_refinery.core.runs import CompletedRun, RunSpec
from llm_refinery.storage.duckdb import ResultStore


class LmEvalFailed(RuntimeError):
    pass


def run_lm_eval(
    config: LmEvalConfig,
    *,
    dry_run: bool = False,
    parent_run_id: str | None = None,
    store: ResultStore | None = None,
) -> list[CompletedRun]:
    targets = {**default_targets(), **config.targets}
    selected = resolve_target_names(config.target, set(targets))
    config.output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_DATASETS_OFFLINE"] = "1" if config.offline else "0"
    if store is not None and store.database != config.database.resolve():
        raise ValueError(
            f"lm-eval database {config.database.resolve()} does not match shared store"
        )

    outcomes: list[CompletedRun] = []
    store_context = nullcontext(store) if store is not None else ResultStore(config.database)
    with store_context as active_store:
        assert active_store is not None
        for target_name in selected:
            target = targets[target_name]
            limit_text = str(config.limit) if config.limit is not None else "all"
            output_path = config.output_root / target.name
            cmd = build_lm_eval_command(config, target)
            command_text = shlex.join(cmd)
            print(
                f"==> Running lm-eval target={target.name} "
                f"tasks={config.tasks} limit={limit_text}"
            )
            print(f"    model={target.model} base_url={target.base_url}")
            print(f"    output_path={output_path}")
            if dry_run:
                print(command_text)
                continue

            spec = _run_spec(
                config,
                target_name=target.name,
                target_model=target.model,
                target_base_url=target.base_url,
                target_api_key_env=target.api_key_env,
                command_text=command_text,
                database=active_store.database,
                parent_run_id=parent_run_id,
            )
            with RunSession(active_store, spec) as run:
                stdout_path = run.artifact("stdout", "stdout.txt", "text/plain")
                stderr_path = run.artifact("stderr", "stderr.txt", "text/plain")
                result_path = run.artifact("result", "result.json", "application/json")
                result_started_mtime = time.time()
                target_env = env.copy()
                if target.api_key_env and os.environ.get(target.api_key_env):
                    target_env["OPENAI_API_KEY"] = os.environ[target.api_key_env]
                completed = subprocess.run(
                    cmd,
                    env=target_env,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                stdout_path.write_text(completed.stdout or "", encoding="utf-8")
                stderr_path.write_text(completed.stderr or "", encoding="utf-8")
                source_result = latest_lm_eval_result(
                    output_path,
                    newer_than=result_started_mtime,
                )
                metrics: dict[str, float] = {}
                if source_result is not None:
                    result_path.write_bytes(source_result.read_bytes())
                    metrics = parse_lm_eval_metrics(result_path)

                success = completed.returncode == 0 and source_result is not None
                status = "ok" if success else "failed"
                if completed.returncode != 0:
                    error = f"exit code {completed.returncode}"
                elif source_result is None:
                    error = "lm-eval produced no result artifact"
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
    command_text: str,
    database: str | Path,
    parent_run_id: str | None,
) -> RunSpec:
    config_json = {
        "benchmark": "lm_eval",
        "model_backend": config.model_backend,
        "package_spec": config.package_spec,
        "apply_chat_template": config.apply_chat_template,
        "target": target_name,
        "model": target_model,
        "base_url": target_base_url,
        "api_key_env": target_api_key_env,
        "tasks": config.tasks,
        "limit": config.limit,
        "num_concurrent": config.num_concurrent,
        "max_retries": config.max_retries,
        "max_length": config.max_length,
        "eos_string": config.eos_string,
        "log_samples": config.log_samples,
        "num_fewshot": config.num_fewshot,
        "gen_kwargs": config.gen_kwargs,
        "offline": config.offline,
        "include_path": str(config.include_path) if config.include_path else None,
        "output_root": str(config.output_root),
        "params": {"target": target_name, "model": target_model},
    }
    return RunSpec.create(
        benchmark_kind="lm_eval",
        suite=config.suite_name,
        label=f"{config.suite_name}/{target_name}",
        command=command_text,
        config_json=config_json,
        database=database,
        parent_run_id=parent_run_id,
    )
