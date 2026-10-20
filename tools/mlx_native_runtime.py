"""Observe the pinned MLX-VLM 0.7.9 server without changing generation arguments."""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from functools import wraps
from importlib.metadata import version
from pathlib import Path


def _token_count(value):
    if isinstance(value, (list, tuple, set)):
        return len(value)
    return None if value is None else 1


def observe_native_diffusion(report, original):
    signature = inspect.signature(original)

    @wraps(original)
    def generate(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        values = bound.arguments["kwargs"]
        tokenizer = bound.arguments["tokenizer"]
        model = bound.arguments["model"]
        report.update(
            observed_diffusion_arguments={
                key: values.get(key)
                for key in ("prefill_step_size", "max_kv_size", "kv_bits", "seed", "max_tokens")
            },
            requests_started=report.data.get("requests_started", 0) + 1,
        )
        try:
            return (yield from original(*args, **kwargs))
        finally:
            report.update(
                requests_completed=report.data.get("requests_completed", 0) + 1,
                stop_token_count=_token_count(tokenizer.stopping_criteria.eos_token_ids),
                config_stop_token_count=_token_count(model.config.eos_token_id),
            )

    return generate


def main(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args, server_args = parser.parse_known_args(argv)
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be an immutable 40-character commit")
    if "--reload" in server_args:
        parser.error("--reload would bypass the runtime observers")
    if version("mlx-vlm") != "0.7.9":
        parser.error("this native-runtime probe requires mlx-vlm==0.7.9")

    from mlx_runtime import RuntimeReport
    from mlx_vlm.server import generation
    from mlx_vlm.server import main as server_main
    from mlx_vlm.utils import get_model_path

    report = RuntimeReport(args.runtime_report)
    report.update(server_arguments=server_args, runtime_policy={"diffusion": "upstream_native"})
    original_load = generation.load

    @wraps(original_load)
    def load(model_id, *load_args, **kwargs):
        path = get_model_path(model_id, revision=args.revision)
        model, processor = original_load(path, *load_args, **kwargs)
        models = dict(report.data["models"])
        models[str(model_id)] = {
            "revision": Path(path).name,
            "requested_revision": args.revision,
            "model_type": model.config.model_type,
        }
        report.update(models=models)
        report.observe_cache(model)
        return model, processor

    generation.load = load
    generation.stream_diffusion_generate_from_kwargs = observe_native_diffusion(
        report, generation.stream_diffusion_generate_from_kwargs
    )
    sys.argv = ["mlx_vlm.server", *server_args]
    server_main()


if __name__ == "__main__":
    main(sys.argv[1:])
