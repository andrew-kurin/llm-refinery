"""Optional Apple Silicon probes; run with the pinned MLX environment in the guide."""

import argparse
import gc
import hashlib
import json
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.models.base import create_causal_mask
from mlx_lm.models.cache import KVCache, RotatingKVCache

MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
REVISION = "8b403126fc14f14cfc99bb4cfa72ecbc129ea677"
OUTPUT = Path("results/mlx-cache-probe")
REPORT = OUTPUT / "results.json"


class SuppliedSinkCache(RotatingKVCache):
    """Robert Price's supplied implementation, retained as a test comparison."""

    def make_mask(self, N, window_size=None, return_array=False):
        if N == 1 or self.keep == 0:
            return super().make_mask(N, window_size=window_size, return_array=return_array)
        window_size = window_size or self.max_size
        offset = min(self.max_size - 1, self.offset)
        if offset + N <= window_size and not return_array:
            return "causal"
        mask = create_causal_mask(N, offset, window_size=window_size - self.keep)
        cols = mx.arange(offset + N)[None]
        rows = mx.arange(offset, offset + N)[:, None]
        return mask | ((cols < self.keep) & (rows >= cols))


class PRSinkCache(RotatingKVCache):
    """Formula and explicit-window guard from upstream PR #1978."""

    def make_mask(self, N, window_size=None, return_array=False):
        effective_window = window_size or self.max_size
        if N == 1 or self.keep == 0 or effective_window != self.max_size:
            return super().make_mask(N, window_size=window_size, return_array=return_array)
        offset = min(self.max_size - 1, self.offset)
        if offset + N <= effective_window and not return_array:
            return "causal"
        cols = mx.arange(offset + N)[None]
        rows = mx.arange(offset, offset + N)[:, None]
        return (rows >= cols) & ((cols < self.keep) | (rows < cols + effective_window - self.keep))


def mask_visibility_case(cls, history, n, keep, explicit_window=None):
    size = 8
    cache = cls(max_size=size, keep=keep)
    for token in range(history):
        kv = mx.array([token], dtype=mx.float32).reshape(1, 1, 1, 1)
        cache.update_and_fetch(kv, kv)
    mask = cache.make_mask(n, window_size=explicit_window, return_array=True)
    kv = mx.arange(history, history + n, dtype=mx.float32).reshape(1, 1, n, 1)
    keys, _ = cache.update_and_fetch(kv, kv)
    ids = keys[0, 0, :, 0].tolist()
    rows = mask.tolist() if mask is not None else None
    mismatches = []
    for row in range(n):
        end = history + row + 1
        actual = sorted(token for col, token in enumerate(ids) if rows is None or rows[row][col])
        if explicit_window is not None and explicit_window != size:
            expected = list(range(max(0, end - explicit_window), end))
        else:
            expected = list(range(min(keep, end))) + list(
                range(max(keep, end - (size - keep)), end)
            )
        if actual != expected:
            mismatches.append({"query": row, "expected": expected, "actual": actual})
    return mismatches


def mask_tests():
    results = {}
    for cls in [RotatingKVCache, SuppliedSinkCache, PRSinkCache]:
        failures = []
        count = 0
        for history in [0, 3, 8, 19]:
            for n in [1, 2, 4, 10]:
                for keep in [0, 2]:
                    count += 1
                    mismatch = mask_visibility_case(cls, history, n, keep)
                    if mismatch:
                        failures.append(
                            {
                                "history": history,
                                "n": n,
                                "keep": keep,
                                "first_mismatch": mismatch[0],
                            }
                        )
        explicit = mask_visibility_case(cls, history=12, n=4, keep=2, explicit_window=4)
        results[cls.__name__] = {
            "default_window_cases": count,
            "default_window_pass_count": count - len(failures),
            "default_window_failures": failures,
            "explicit_smaller_window_pass": not explicit,
            "explicit_smaller_window_mismatches": explicit,
        }
        print(
            f"MASK {cls.__name__}: {count - len(failures)}/{count} default-window cases; "
            f"smaller-window={'PASS' if not explicit else 'FAIL'}",
            flush=True,
        )
    return results


FILLER = (
    "The building permit was issued for a multifamily structure. Framing is "
    "exposed and the electrical rough-in has not yet started. The general "
    "contractor has not been reached. "
)
NEEDLE = "The access code for the site trailer is 7391."
INSTRUCTION = (
    "You are a site assistant. Read the notes below carefully. At the end you "
    "will be asked one question; answer with only the number requested."
)
QUESTION = "What is the access code for the site trailer? Answer with only the number."


def prompt_for(tokenizer, filler_tokens, position):
    text = ""
    while len(tokenizer.encode(text)) < filler_tokens:
        text += FILLER * 8
    text = tokenizer.decode(tokenizer.encode(text)[:filler_tokens])
    notes = NEEDLE + " " + text if position == "start" else text + " " + NEEDLE
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": INSTRUCTION},
            {"role": "user", "content": f"NOTES:\n{notes}\n\nQUESTION: {QUESTION}"},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    needle_end = len(tokenizer.encode(prompt.split(NEEDLE)[0] + NEEDLE))
    # Match stream_generate's string preprocessing explicitly.
    add_special = tokenizer.bos_token is None or not prompt.startswith(tokenizer.bos_token)
    tokens = tokenizer.encode(prompt, add_special_tokens=add_special)
    return prompt, tokens, needle_end


def persist(report):
    temporary = REPORT.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(REPORT)


def run_one(model, tokenizer, tokens, position, filler_tokens, window, step, kind, keep):
    cls = {"unbounded": KVCache, "stock": RotatingKVCache, "supplied_patch": SuppliedSinkCache}[
        kind
    ]
    caches = [
        KVCache() if kind == "unbounded" else cls(max_size=window, keep=keep) for _ in model.layers
    ]
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()
    progress = []
    start = time.perf_counter()
    responses = list(
        stream_generate(
            model,
            tokenizer,
            tokens,
            max_tokens=12,
            prompt_cache=caches,
            prefill_step_size=step,
            sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
            prompt_progress_callback=lambda done, total: progress.append([done, total]),
        )
    )
    answer = "".join(r.text for r in responses).strip()
    result = {
        "filler_tokens": filler_tokens,
        "actual_prompt_tokens": len(tokens),
        "position": position,
        "cache": kind,
        "window": window if kind != "unbounded" else None,
        "keep": keep if kind != "unbounded" else None,
        "prefill_step_size": step,
        "answer": answer,
        "exact_correct": answer == "7391",
        "contains_correct": "7391" in answer,
        "elapsed_s": time.perf_counter() - start,
        "peak_metal_gb": mx.get_peak_memory() / 1e9,
        "prefill_progress": progress,
        "generated_tokens": responses[-1].generation_tokens if responses else 0,
    }
    print(
        f"RECALL prompt={len(tokens)} position={position} step={step} "
        f"cache={kind} keep={result['keep']} exact={result['exact_correct']} answer={answer!r}",
        flush=True,
    )
    return result


def main():
    global OUTPUT, REPORT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--mask-only", action="store_true", help="Skip model loading and retrieval")
    args = parser.parse_args()
    OUTPUT = args.output
    REPORT = OUTPUT / "results.json"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "model": MODEL,
        "revision": REVISION,
        "packages": {name: version(name) for name in ["mlx", "mlx-lm", "mlx-vlm", "transformers"]},
        "hardware": mx.device_info(),
        "mask_tests": mask_tests(),
        "retrieval": [],
        "note": "Greedy correctness probes; no throughput or memory benchmarking controls.",
    }
    persist(report)
    if args.mask_only:
        return
    print(f"Loading pinned model {MODEL}@{REVISION}", flush=True)
    model, tokenizer = load(MODEL, revision=REVISION)
    for filler_tokens, window, positions, steps in [
        (1500, 750, ["start", "end"], [512, 4096]),
        (12000, 8192, ["end"], [4096]),
    ]:
        for position in positions:
            prompt, tokens, needle_end = prompt_for(tokenizer, filler_tokens, position)
            (OUTPUT / f"prompt-{filler_tokens}-{position}.txt").write_text(prompt)
            report.setdefault("prompts", {})[f"{filler_tokens}-{position}"] = {
                "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "actual_prompt_tokens": len(tokens),
                "needle_end": needle_end,
            }
            for step in steps:
                configs = [("unbounded", 0), ("stock", 4), ("supplied_patch", 4)]
                if position == "start":
                    configs.append(("supplied_patch", needle_end))
                for kind, keep in configs:
                    result = run_one(
                        model, tokenizer, tokens, position, filler_tokens, window, step, kind, keep
                    )
                    report["retrieval"].append(result)
                    persist(report)
    report["finished_at"] = datetime.now(UTC).isoformat()
    persist(report)
    print(f"Saved {len(report['retrieval'])} retrieval results to {REPORT}", flush=True)


if __name__ == "__main__":
    main()
