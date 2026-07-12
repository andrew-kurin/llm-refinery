from __future__ import annotations

from pathlib import Path

from llm_refinery.benchmarks.dabstep.config import DabstepConfig


def build_dabstep_command(
    config: DabstepConfig,
    task_ids: list[int],
    *,
    run_token: str,
    tasks_manifest_path: Path | None = None,
) -> list[str]:
    settings = config.dabstep
    command = [
        *settings.command,
        "--model-id",
        effective_model_id(config),
        "--api-base",
        _litellm_api_base(config.endpoint.base_url),
        "--split",
        settings.split,
        "--concurrency",
        str(settings.concurrency),
        "--max-steps",
        str(settings.max_steps),
        "--timestamp",
        run_token,
        "--tasks-ids",
        *(str(task_id) for task_id in task_ids),
    ]
    if settings.tasks_file_arg:
        manifest_path = tasks_manifest_path or settings.tasks_file
        if manifest_path is None:
            raise ValueError("tasks_file_arg requires a task manifest path")
        command.extend([settings.tasks_file_arg, str(manifest_path)])
    return command


def effective_model_id(config: DabstepConfig) -> str:
    configured = config.dabstep.model_id
    if configured:
        return configured
    if config.endpoint.model.startswith("openai/"):
        return config.endpoint.model
    return f"openai/{config.endpoint.model}"


def upstream_output_dir(config: DabstepConfig, *, run_token: str) -> Path:
    model_dir = effective_model_id(config).replace("/", "_").replace(".", "_")
    return config.dabstep.workspace / "runs" / model_dir / config.dabstep.split / run_token


def _litellm_api_base(base_url: str) -> str:
    suffix = "/chat/completions"
    return base_url[: -len(suffix)] if base_url.endswith(suffix) else base_url
