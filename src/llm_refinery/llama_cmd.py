from __future__ import annotations

import shlex
from typing import Any

from llm_refinery.assets import resolve_mtp_head
from llm_refinery.config import Trial, TuneConfig

# Explicit aliases for common llama.cpp flags. Unknown keys fall back to --kebab-case.
FLAG_ALIASES = {
    "ctx_size": "--ctx-size",
    "hff": "-hff",
    "hf_file": "--hf-file",
    "n_gpu_layers": "--n-gpu-layers",
    "gpu_layers": "--n-gpu-layers",
    "flash_attn": "--flash-attn",
    "cache_type_k": "--cache-type-k",
    "cache_type_v": "--cache-type-v",
    "batch_size": "--batch-size",
    "ubatch_size": "--ubatch-size",
    "threads": "--threads",
    "threads_batch": "--threads-batch",
    "parallel": "--parallel",
    "main_gpu": "--main-gpu",
    "tensor_split": "--tensor-split",
    "mlock": "--mlock",
    "no_mmap": "--no-mmap",
    "perf": "--perf",
    "model_draft": "--model-draft",
}


def build_bench_command(config: TuneConfig, trial: Trial) -> list[str]:
    cmd = list(config.commands["bench"])
    cmd.extend(model_args(trial))
    params = effective_params(trial.params, config.bench.params, config.bench.omit_params)
    cmd.extend(params_args(params))

    if trial.prompt_tokens is not None:
        cmd.extend(["-p", str(trial.prompt_tokens)])
    if trial.gen_tokens is not None:
        cmd.extend(["-n", str(trial.gen_tokens)])
    if config.bench.repetitions:
        cmd.extend(["-r", str(config.bench.repetitions)])
    if config.bench.output:
        cmd.extend(["-o", config.bench.output])

    cmd.extend(trial.model.extra_args)
    cmd.extend(config.bench.extra_args)
    return cmd


def build_server_command(config: TuneConfig, trial: Trial) -> list[str]:
    cmd = list(config.commands["server"])
    cmd.extend(model_args(trial))
    params = effective_params(trial.params, config.server.params, config.server.omit_params)
    cmd.extend(params_args(params))
    cmd.extend(trial.model.extra_args)
    cmd.extend(config.server.extra_args)
    return cmd


def effective_params(
    base_params: dict[str, Any], overrides: dict[str, Any], omit_params: set[str]
) -> dict[str, Any]:
    params = dict(base_params)
    params.update(overrides)
    for key in omit_params:
        params.pop(key, None)
    return params


def model_args(trial: Trial) -> list[str]:
    if trial.model.hf:
        return ["-hf", trial.model.hf]
    if trial.model.path:
        return ["-m", trial.model.path]
    raise ValueError(f"trial {trial.name!r} has no model source")


def params_args(params: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in params.items():
        if key.startswith("_") or value is None:
            continue

        if key == "mtp_head":
            args.extend(["--model-draft", str(resolve_mtp_head(value).path)])
            continue

        flag = flag_for_key(key)
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue

        if isinstance(value, (list, tuple)):
            rendered = ",".join(str(item) for item in value)
        else:
            rendered = str(value)
        args.extend([flag, rendered])
    return args


def flag_for_key(key: str) -> str:
    if key.startswith("-"):
        return key
    return FLAG_ALIASES.get(key, f"--{key.replace('_', '-')}")


def shell_join(cmd: list[str]) -> str:
    return shlex.join(cmd)
