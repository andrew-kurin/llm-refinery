from __future__ import annotations

from pathlib import Path

import click

from llm_refinery.benchmarks.lm_eval.config import LmEvalConfig
from llm_refinery.benchmarks.lm_eval.presets import default_targets
from llm_refinery.benchmarks.lm_eval.runner import run_lm_eval
from llm_refinery.commands.common import parse_lm_eval_limit
from llm_refinery.core.endpoints import OPENAI_CHAT, Endpoint


@click.command("lm-eval", help="Run lm-eval against local OpenAI-compatible endpoints.")
@click.argument("target", required=False, default="llama_cpp")
@click.argument("limit_text", required=False)
@click.option("--tasks", default="ifeval,gsm8k", show_default=True, help="Comma-separated tasks.")
@click.option("--num-concurrent", type=int, default=1, show_default=True)
@click.option("--max-retries", type=int, default=3, show_default=True)
@click.option("--max-length", type=int, default=16384, show_default=True)
@click.option(
    "--eos-string",
    help="Optional model-specific EOS string; omit to use the evaluator/backend default.",
)
@click.option(
    "--tokenizer",
    help=(
        "Tokenizer id/path for a backend that supports client tokenization; "
        "local-chat-completions rejects this option because it ignores it."
    ),
)
@click.option(
    "--metadata",
    help='lm-eval metadata JSON, e.g. {"max_seq_lengths":[4096,8192]}.',
)
@click.option("--gen-kwargs", help="Extra lm-eval generation kwargs for API backends.")
@click.option(
    "--model-backend",
    default="local-chat-completions",
    show_default=True,
    type=click.Choice(["local-chat-completions", "local-completions"]),
    help="lm-eval API model backend.",
)
@click.option(
    "--package-spec",
    default="lm-eval[api]==0.4.12",
    show_default=True,
    help="uvx package spec; pin a version for reproducible runs.",
)
@click.option(
    "--with-package",
    "extra_packages",
    multiple=True,
    help="Additional pinned package for the evaluator environment; repeat as needed.",
)
@click.option(
    "--apply-chat-template/--no-apply-chat-template",
    default=True,
    show_default=True,
    help="Pass --apply_chat_template to lm-eval.",
)
@click.option(
    "--include-path",
    type=click.Path(file_okay=False, path_type=Path),
    help="Additional lm-eval task directory, e.g. evals/lm_eval_tasks.",
)
@click.option(
    "--suite-name",
    default="lm-eval",
    show_default=True,
    help="Suite name for DuckDB records.",
)
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("results/llm_refinery.duckdb"),
    show_default=True,
    help="DuckDB path for parsed lm-eval metrics.",
)
@click.option("--log-samples", is_flag=True, help="Pass --log_samples to lm-eval.")
@click.option("--num-fewshot", type=int, help="Override task few-shot count for lm-eval.")
@click.option("--model", help="Override model name for a single target.")
@click.option("--base-url", help="Override chat-completions URL for a single target.")
@click.option("--api-key-env", help="Environment variable containing the endpoint API key.")
@click.option(
    "--trust-env/--no-trust-env",
    default=False,
    show_default=True,
    help="Honor proxy environment variables (supported only with --num-concurrent 1).",
)
@click.option(
    "--ca-bundle",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="PEM CA bundle used consistently by lm-eval HTTP clients.",
)
@click.option(
    "--output-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results/lm_eval"),
    show_default=True,
)
@click.option(
    "--offline/--online",
    default=True,
    show_default=True,
    help="Set Hugging Face datasets and hub offline mode.",
)
@click.option("--dry-run", is_flag=True, help="Print uvx commands without running lm-eval.")
def lm_eval_command(
    target: str,
    limit_text: str | None,
    tasks: str,
    num_concurrent: int,
    max_retries: int,
    max_length: int,
    eos_string: str | None,
    tokenizer: str | None,
    metadata: str | None,
    gen_kwargs: str | None,
    model_backend: str,
    package_spec: str,
    extra_packages: tuple[str, ...],
    apply_chat_template: bool,
    include_path: Path | None,
    suite_name: str,
    db: Path,
    log_samples: bool,
    num_fewshot: int | None,
    model: str | None,
    base_url: str | None,
    api_key_env: str | None,
    trust_env: bool,
    ca_bundle: Path | None,
    output_root: Path,
    offline: bool,
    dry_run: bool,
) -> None:
    if (model or base_url or api_key_env) and target in {"both", "all"}:
        raise click.BadParameter(
            "--model/--base-url/--api-key-env can only override a single target"
        )

    presets = default_targets()
    targets = {}
    if target not in {*presets, "both", "all"}:
        if not model or not base_url:
            raise click.BadParameter(
                "custom targets require both --model and --base-url",
                param_hint="target",
            )
        targets[target] = Endpoint(
            name=target,
            protocol=OPENAI_CHAT,
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
        )
    elif target in presets and (model or base_url or api_key_env):
        target_defaults = presets[target]
        targets[target] = Endpoint(
            name=target,
            protocol=OPENAI_CHAT,
            model=model or target_defaults.model,
            base_url=base_url or target_defaults.base_url,
            api_key_env=api_key_env or target_defaults.api_key_env,
        )

    run_lm_eval(
        LmEvalConfig(
            target=target,
            limit=parse_lm_eval_limit(limit_text) if limit_text is not None else 50,
            tasks=tasks,
            num_concurrent=num_concurrent,
            max_retries=max_retries,
            max_length=max_length,
            eos_string=eos_string,
            tokenizer=tokenizer,
            metadata=metadata,
            log_samples=log_samples,
            num_fewshot=num_fewshot,
            gen_kwargs=gen_kwargs,
            output_root=output_root,
            offline=offline,
            model_backend=model_backend,
            package_spec=package_spec,
            extra_packages=extra_packages,
            apply_chat_template=apply_chat_template,
            include_path=include_path,
            trust_env=trust_env,
            ca_bundle=ca_bundle,
            suite_name=suite_name,
            database=db,
            targets=targets,
        ),
        dry_run=dry_run,
    )
