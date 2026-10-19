"""Run with the pinned MLX environment in diffusiongemma-26b-mlx-vlm-sweep.yaml.

Only the serve/recall entry points import MLX. The server hooks
observe cache construction and diffusion arguments without changing either.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
import threading
import uuid
from datetime import UTC, datetime
from functools import wraps
from importlib.metadata import version
from pathlib import Path
from types import MethodType


class RuntimeReport:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {
            "instance_id": uuid.uuid4().hex,
            "started_at": datetime.now(UTC).isoformat(),
            "pid": os.getpid(),
            "python": sys.version,
            "argv": sys.argv[1:],
            "packages": {
                name: version(name)
                for name in (
                    "mlx",
                    "mlx-lm",
                    "mlx-vlm",
                    "transformers",
                    "tokenizers",
                    "huggingface-hub",
                )
            },
            "models": {},
        }
        self.update()

    def update(self, **values):
        with self.lock:
            self.data.update(values)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            temporary.replace(self.path)

    def observe_cache(self, model):
        original = model.make_cache

        @wraps(original)
        def make_cache(_model, *args, **kwargs):
            caches = original(*args, **kwargs)
            self.update(
                observed_cache_factory=f"{type(model).__module__}.{type(model).__name__}",
                observed_cache_layout=[
                    {
                        "layer": i,
                        "type": f"{type(cache).__module__}.{type(cache).__name__}",
                        "keep": getattr(cache, "keep", None),
                        "max_size": getattr(cache, "max_size", None),
                    }
                    for i, cache in enumerate(caches)
                ],
            )
            return caches

        # Keep the observer out of MLX's parameter/module traversal.
        object.__setattr__(model, "make_cache", MethodType(make_cache, model))

    def observe_diffusion(self, original):
        signature = inspect.signature(original)

        @wraps(original)
        def generate(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            self.update(
                observed_diffusion_arguments={
                    name: bound.arguments.get(name)
                    for name in ("prefill_step_size", "max_kv_size", "kv_bits")
                },
                diffusion_supported_arguments=[
                    name
                    for name in ("prefill_step_size", "max_kv_size", "kv_bits")
                    if name in signature.parameters
                ],
            )
            return original(*args, **kwargs)

        return generate


def serve(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--revision")
    args, server_args = parser.parse_known_args(argv)
    if args.revision is not None and not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be an immutable 40-character commit")
    if "--reload" in server_args:
        parser.error("--reload would bypass the runtime observers")

    from mlx_vlm.server import generation, main
    from mlx_vlm.utils import get_model_path

    report = RuntimeReport(args.runtime_report)
    report.update(server_arguments=server_args)
    original_load = generation.load

    @wraps(original_load)
    def load(model_id, *load_args, **kwargs):
        path = get_model_path(model_id, revision=args.revision)
        model, processor = original_load(path, *load_args, **kwargs)
        resolved = Path(path).absolute()
        models = dict(report.data["models"])
        models[str(model_id)] = {
            "path": str(resolved),
            "revision": resolved.name if resolved.parent.name == "snapshots" else None,
            "requested_revision": args.revision,
            "model_type": getattr(model.config, "model_type", None),
        }
        report.update(
            models=models,
            server_environment={
                key: os.environ.get(key)
                for key in (
                    "PREFILL_STEP_SIZE",
                    "MAX_KV_SIZE",
                    "KV_BITS",
                    "KV_QUANT_SCHEME",
                    "KV_GROUP_SIZE",
                    "QUANTIZED_KV_START",
                    "MLX_VLM_ENABLE_THINKING",
                    "MLX_VLM_TOKEN_QUEUE_TIMEOUT",
                )
            },
        )
        report.observe_cache(model)
        return model, processor

    generation.load = load
    generation.stream_diffusion_generate = report.observe_diffusion(
        generation.stream_diffusion_generate
    )
    sys.argv = ["mlx_vlm.server", *server_args]
    main()


SYSTEM = "Answer using only the reference note in the user message."
ANSWER = "RF7391XZ"
FILLER = "The archive contains routine building inspection notes. "
NEEDLE = f"Reference note: the site trailer access code is {ANSWER}."
QUESTION = "What is the site trailer access code? Reply with only the code."


def recall_prompt(count_tokens, target: int, position: str = "near-end") -> str:
    """Fit a single needle to the exact chat-token budget using neutral padding."""

    def render(repeats, padding=0):
        filler = FILLER * repeats + " x" * padding
        if position == "start":
            return f"{NEEDLE}\n\n{filler}\n\n{QUESTION}"
        # Leave neutral text between the fact and question; the fact stays in-window.
        return f"{filler}\n\n{NEEDLE}\n\n{FILLER * 8}\n\n{QUESTION}"

    low, high = 0, target
    while low < high:
        middle = (low + high + 1) // 2
        if count_tokens(render(middle)) <= target:
            low = middle
        else:
            high = middle - 1
    for padding in range(256):
        prompt = render(low, padding)
        length = count_tokens(prompt)
        if length == target:
            return prompt
        if length > target:
            break
    raise ValueError(f"cannot fit recall prompt to exactly {target} tokens")


def prepare_recall(argv):
    parser = argparse.ArgumentParser(
        description="Generate exact-length single-needle HTTP fixtures"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True, help="Immutable model/tokenizer commit")
    parser.add_argument("--output", type=Path, default=Path("results/mlx-recall"))
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[4096, 8191, 8192, 8193, 12288, 16384]
    )
    parser.add_argument("--position", choices=["near-end", "start"], default="near-end")
    parser.add_argument(
        "--runtime-report", type=Path, default=Path("results/mlx-runtime-8082.json")
    )
    args = parser.parse_args(argv)
    if any(length <= 0 for length in args.tokens):
        parser.error("--tokens must be positive")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be an immutable 40-character commit")

    import yaml
    from huggingface_hub import snapshot_download
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config, load_processor, prepare_inputs

    path = snapshot_download(
        args.model,
        revision=args.revision,
        allow_patterns=["*.json", "*.jinja"],
    )
    processor = load_processor(Path(path))
    config = load_config(Path(path))
    if config["model_type"] != "diffusion_gemma":
        parser.error("recall preprocessing currently supports diffusion_gemma checkpoints")

    def count_tokens(prompt):
        text = apply_chat_template(
            processor,
            config,
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            add_generation_prompt=True,
            num_images=0,
            num_audios=0,
            tools=None,
            enable_thinking=False,
        )
        # In mlx-vlm 0.6.3, _cpu_preprocess uses True for diffusion_gemma.
        inputs = prepare_inputs(processor, prompts=text, add_special_tokens=True)
        return inputs["input_ids"].size

    args.output.mkdir(parents=True, exist_ok=True)
    scenarios = []
    for length in args.tokens:
        prompt = recall_prompt(count_tokens, length, args.position)
        name = f"recall-{args.position}-{length}"
        (args.output / f"{name}.txt").write_text(prompt, encoding="utf-8")
        scenarios.append(
            {
                "name": name,
                "system": SYSTEM,
                "prompt_file": f"{name}.txt",
                "expected_exact": ANSWER,
                "min_prompt_tokens": length,
                "max_prompt_tokens": length,
                "max_tokens": [32],
                "concurrency": [1],
                "requests": 3,
                "warmup_requests": 1,
                "temperature": 0,
                "seed": 42,
                "stream": True,
                "timeout_s": 300,
            }
        )
    manifest = {
        "name": f"mlx-single-needle-{args.position}",
        "database": "results/llm_refinery.duckdb",
        "runtime_report": os.path.relpath(args.runtime_report.resolve(), args.output.resolve()),
        "transport": {"trust_env": False},
        "targets": [
            {
                "name": "mlx-diffusiongemma-26b-4bit",
                "protocol": "openai_chat",
                "base_url": "http://127.0.0.1:8082/v1",
                "model": args.model,
            }
        ],
        "scenarios": scenarios,
    }
    (args.output / "http-load.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (args.output / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "revision": Path(path).name,
                "position": args.position,
                "prompt_tokens": args.tokens,
                "enable_thinking": False,
            },
            indent=2,
        )
    )
    print(f"Wrote {len(scenarios)} exact-length fixtures to {args.output / 'http-load.yaml'}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in {"serve", "recall"}:
        raise SystemExit("usage: mlx_runtime.py {serve,recall} [options]")
    if sys.argv[1] == "serve":
        serve(sys.argv[2:])
    else:
        prepare_recall(sys.argv[2:])
