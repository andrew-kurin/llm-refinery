from __future__ import annotations

from llm_refinery.benchmarks.lm_eval.config import LmEvalConfig, LmEvalTarget


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
