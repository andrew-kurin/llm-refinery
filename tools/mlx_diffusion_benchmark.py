"""Measure native DiffusionGemma prefill with the pinned sweep environment.

This calls the generation API directly because mlx-vlm 0.6.3's HTTP diffusion
path does not forward prefill_step_size. It preserves the model's native caches.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import platform
import statistics
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from mlx_runtime import ANSWER, SYSTEM, isolate_stop_token_ids, recall_prompt

MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
REVISION = "0d2cee4aa112548df2fad35a7f017eeb1ac8f352"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/mlx-diffusion-benchmark"))
    parser.add_argument("--tokens", type=int, nargs="+", default=[4096, 8193])
    parser.add_argument("--prefill-steps", nargs="+", default=["none", "4096", "2048", "512"])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    args = parser.parse_args()
    if min(*args.tokens, args.repetitions, args.max_tokens) <= 0:
        parser.error("token counts and repetitions must be positive")
    steps = [None if value == "none" else int(value) for value in args.prefill_steps]
    if any(step is not None and step <= 0 for step in steps):
        parser.error("prefill steps must be positive or 'none'")

    import mlx.core as mx
    from mlx_vlm.generate.common import generation_stream, wired_limit
    from mlx_vlm.generate.diffusion import stream_diffusion_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import get_model_path, load, prepare_inputs

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "revision": REVISION,
        "python": platform.python_version(),
        "device": mx.device_info(),
        "packages": {
            name: version(name)
            for name in ("mlx", "mlx-lm", "mlx-vlm", "transformers", "tokenizers")
        },
        "method": {
            "entry_point": "mlx_vlm.generate.diffusion.stream_diffusion_generate",
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "seed": 42,
            "warmups_per_cell": 1,
            "measured_repetitions": args.repetitions,
            "cache": "native, fresh for each call; no KV quantization",
            "order": "all warmups first; measured cell order rotates each round",
            "memory": "MLX peak bytes reset per call; includes resident model weights",
            "stop_token_ids": "isolated per call to prevent upstream config-list mutation",
        },
        "fixtures": {},
        "runs": [],
    }

    def save():
        temporary = args.output / "results.tmp"
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output / "results.json")

    save()
    print("Loading pinned DiffusionGemma checkpoint", flush=True)
    model_path = get_model_path(MODEL, revision=REVISION)
    model, processor = load(model_path)
    tokenizer = getattr(processor, "tokenizer", processor)
    config = json.loads((Path(model_path) / "config.json").read_text())

    def preprocess(prompt):
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
        return prepare_inputs(processor, prompts=text, add_special_tokens=True)

    inputs_by_length = {}
    for length in args.tokens:
        prompt = recall_prompt(lambda text: preprocess(text)["input_ids"].size, length)
        inputs = preprocess(prompt)
        assert inputs["input_ids"].size == length
        mx.eval(inputs)
        inputs_by_length[length] = inputs
        (args.output / f"recall-near-end-{length}.txt").write_text(prompt, encoding="utf-8")
        report["fixtures"][str(length)] = {
            "prompt_tokens": length,
            "expected_exact": ANSWER,
            "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
    report["cache_layout"] = [
        {
            "layer": index,
            "type": type(cache).__name__,
            "keep": getattr(cache, "keep", None),
            "max_size": getattr(cache, "max_size", None),
        }
        for index, cache in enumerate(model.make_cache())
    ]
    save()

    # Observe actual encoder call lengths; preserve arguments, results and math.
    encoder_type = type(model.model.encoder)
    original_call = encoder_type.__call__
    encoder_chunks = []

    def observe_encoder(encoder, input_ids, *positional, **kwargs):
        if encoder is model.model.encoder:
            encoder_chunks.append(int(input_ids.shape[1]))
        return original_call(encoder, input_ids, *positional, **kwargs)

    encoder_type.__call__ = observe_encoder

    def run_cell(length, step, phase, repetition):
        gc.collect()
        mx.synchronize()
        mx.clear_cache()
        mx.random.seed(42)
        tokenizer.stopping_criteria.reset(model.config.eos_token_id)
        isolate_stop_token_ids(tokenizer)
        encoder_chunks.clear()
        mx.reset_peak_memory()
        baseline = mx.get_active_memory()
        inputs = inputs_by_length[length]
        started = time.perf_counter()
        first_text_s = None
        text = ""
        final = None
        with wired_limit(model, [generation_stream]):
            for result in stream_diffusion_generate(
                model,
                processor,
                tokenizer,
                inputs["input_ids"],
                inputs.get("pixel_values"),
                inputs.get("attention_mask"),
                max_tokens=args.max_tokens,
                temperature=0,
                skip_special_token_ids=set(tokenizer.all_special_ids or []),
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
                prefill_step_size=step,
            ):
                if result.text and first_text_s is None:
                    first_text_s = time.perf_counter() - started
                text += result.text
                final = result
        mx.synchronize()
        elapsed = time.perf_counter() - started
        assert final is not None
        record = {
            "phase": phase,
            "repetition": repetition,
            "prompt_tokens": length,
            "prefill_step_size": step,
            "encoder_chunk_lengths": list(encoder_chunks),
            "text": text,
            "exact_pass": text.strip() == ANSWER,
            "wall_s": elapsed,
            "first_text_s": first_text_s,
            "baseline_active_bytes": baseline,
            "peak_metal_bytes": mx.get_peak_memory(),
            "stop_token_count": len(tokenizer.stopping_criteria.eos_token_ids),
            "config_stop_token_count": len(model.config.eos_token_id),
            "generation_result": asdict(final),
        }
        report["runs"].append(record)
        save()
        print(
            json.dumps({key: value for key, value in record.items() if key != "generation_result"}),
            flush=True,
        )

    cells = list(itertools.product(args.tokens, steps))
    try:
        for length, step in cells:
            run_cell(length, step, "warmup", 0)
        for repetition in range(args.repetitions):
            order = cells[repetition:] + cells[:repetition]
            for length, step in order:
                run_cell(length, step, "measured", repetition + 1)
    finally:
        encoder_type.__call__ = original_call

    report["summary"] = []
    for length, step in cells:
        runs = [
            run
            for run in report["runs"]
            if run["phase"] == "measured"
            and run["prompt_tokens"] == length
            and run["prefill_step_size"] == step
        ]
        report["summary"].append(
            {
                "prompt_tokens": length,
                "prefill_step_size": step,
                "exact_passes": sum(run["exact_pass"] for run in runs),
                "requests": len(runs),
                "median_wall_s": statistics.median(run["wall_s"] for run in runs),
                "median_peak_metal_bytes": statistics.median(
                    run["peak_metal_bytes"] for run in runs
                ),
                "median_prompt_tps": statistics.median(
                    run["generation_result"]["prompt_tps"] for run in runs
                ),
            }
        )
    report["finished_at"] = datetime.now(UTC).isoformat()
    save()
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
