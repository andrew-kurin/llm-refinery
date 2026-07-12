from __future__ import annotations

from llm_refinery.benchmarks.lm_eval.config import LmEvalConfig
from llm_refinery.core.endpoints import Endpoint


def build_lm_eval_command(config: LmEvalConfig, target: Endpoint) -> list[str]:
    output_path = str(config.output_root / target.name)
    model_args_parts = [
        f"model={target.model}",
        f"base_url={target.chat_completions_url}",
        f"num_concurrent={config.num_concurrent}",
        f"max_retries={config.max_retries}",
        f"max_length={config.max_length}",
    ]
    if config.eos_string:
        model_args_parts.append(f"eos_string={config.eos_string}")
    if config.tokenizer:
        model_args_parts.append(f"tokenizer={config.tokenizer}")
    model_args = ",".join(model_args_parts)

    cmd = [
        "uvx",
        "--from",
        config.package_spec,
        "--with",
        "langdetect",
        "--with",
        "immutabledict",
    ]
    for package in config.extra_packages:
        cmd.extend(["--with", package])
    cmd.extend(
        [
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
    )

    if config.limit is not None:
        cmd.extend(["--limit", str(config.limit)])

    if config.num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(config.num_fewshot)])

    if config.apply_chat_template:
        cmd.append("--apply_chat_template")

    if config.include_path is not None:
        cmd.extend(["--include_path", str(config.include_path)])
    if config.log_samples:
        cmd.append("--log_samples")
    if config.gen_kwargs:
        cmd.extend(["--gen_kwargs", config.gen_kwargs])
    if config.metadata:
        cmd.extend(["--metadata", config.metadata])

    cmd.extend(["--output_path", output_path])
    return cmd
