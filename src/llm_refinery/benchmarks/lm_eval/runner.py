from __future__ import annotations

import os
import shlex
import subprocess
import time

from llm_refinery.benchmarks.lm_eval.command import build_lm_eval_command
from llm_refinery.benchmarks.lm_eval.config import (
    LmEvalConfig,
    default_targets,
    resolve_target_names,
)
from llm_refinery.benchmarks.lm_eval.parser import latest_lm_eval_result, parse_lm_eval_metrics
from llm_refinery.core.runs import record_benchmark_run
from llm_refinery.storage import ResultStore, utc_now


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
