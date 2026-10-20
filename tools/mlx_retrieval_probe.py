"""Generate varied, exact-length DiffusionGemma HTTP retrieval fixtures."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import string
from pathlib import Path

MODEL = "mlx-community/diffusiongemma-26B-A4B-it-4bit"
REVISION = "0d2cee4aa112548df2fad35a7f017eeb1ac8f352"
SYSTEM = "Use the records to answer the question. Follow the requested output format exactly."


def retrieval_case(count_tokens, length, position, task, seed):
    rng = random.Random(seed)
    codes = [
        "".join(rng.choices(string.ascii_uppercase, k=3)) + str(rng.randrange(100, 999))
        for _ in range(3)
    ]
    ticket = f"J-{rng.randrange(10000, 99999)}"
    if task == "single":
        needle = f"Verified record for {ticket}: the lockbox key is {codes[0]}."
        question = f"What is the verified lockbox key for {ticket}? Reply with only the key."
        expected = codes[0]
        archived = ""
    elif task == "pair":
        needle = f"Verified record for {ticket}: gate key = {codes[0]}; cabinet key = {codes[1]}."
        question = (
            f"Give the gate key and cabinet key for {ticket}, in that order, "
            "separated by | with no spaces. Output only those two keys."
        )
        expected = f"{codes[0]}|{codes[1]}"
        archived = ""
    elif task == "revision":
        needle = f"APPROVED revision 7 for {ticket}: the current seal code is {codes[0]}."
        archived = (
            f"ARCHIVE ONLY: superseded revision 6 for {ticket} listed {codes[2]}. "
            "This obsolete entry is retained for audit and is not the current code."
        )
        question = (
            f"What is the APPROVED revision 7 seal code for {ticket}? Reply with only the code."
        )
        expected = codes[0]
    else:
        raise ValueError(f"unsupported task: {task}")

    trades = ["electrical", "roofing", "plumbing", "concrete", "landscaping", "survey"]
    findings = [
        "delivery checked",
        "drawing reviewed",
        "equipment inspected",
        "work area cleared",
        "measurements verified",
        "inventory reconciled",
    ]
    records = [
        f"Log {index:04d}: crew {rng.randrange(10, 99)} completed {rng.choice(trades)} work "
        f"in zone {rng.randrange(100, 999)}. Status: {rng.choice(findings)}. "
        f"The next inspection is on day {rng.randrange(1, 29)} of month {rng.randrange(1, 13)}."
        for index in range(2048)
    ]

    def render(count, padding=0):
        index = {"start": 0, "middle": count // 2, "near-end": max(0, count - 6)}[position]
        lines = [*records[:index], needle, *records[index:count]]
        lines.extend(["General note: routine checks continue according to the daily schedule."] * 2)
        if archived:
            lines.append(archived)
        return "\n".join(lines) + " idle" * padding + "\n\n" + question

    low, high = 0, len(records)
    while low < high:
        middle = (low + high + 1) // 2
        if count_tokens(render(middle)) <= length:
            low = middle
        else:
            high = middle - 1
    for padding in range(256):
        prompt = render(low, padding)
        tokens = count_tokens(prompt)
        if tokens == length:
            assert prompt.count(codes[0]) == 1
            if task == "pair":
                assert prompt.count(codes[1]) == 1
            return {"prompt": prompt, "expected_exact": expected, "seed": seed}
        if tokens > length:
            break
    raise ValueError(f"could not fit {task}/{position} to {length} tokens")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/mlx-varied-retrieval"))
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 16384])
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--database", type=Path, default=Path("results/llm_refinery.duckdb"))
    args = parser.parse_args()
    if any(length <= 0 for length in args.tokens):
        parser.error("token counts must be positive")

    import yaml
    from huggingface_hub import snapshot_download
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config, load_processor, prepare_inputs

    path = Path(snapshot_download(MODEL, revision=REVISION, allow_patterns=["*.json", "*.jinja"]))
    processor = load_processor(path)
    config = load_config(path)

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
        return prepare_inputs(processor, prompts=text, add_special_tokens=True)["input_ids"].size

    args.output.mkdir(parents=True, exist_ok=True)
    scenarios, cases = [], []
    for index, (length, position, task) in enumerate(
        itertools.product(
            args.tokens, ["start", "middle", "near-end"], ["single", "pair", "revision"]
        )
    ):
        case = retrieval_case(count_tokens, length, position, task, 600 + index)
        name = f"{task}-{position}-{length}"
        (args.output / f"{name}.txt").write_text(case["prompt"], encoding="utf-8")
        cases.append(
            {
                "name": name,
                "prompt_tokens": length,
                "position": position,
                "task": task,
                "expected_exact": case["expected_exact"],
                "fixture_seed": case["seed"],
                "sha256": hashlib.sha256(case["prompt"].encode()).hexdigest(),
            }
        )
        scenarios.append(
            {
                "name": name,
                "system": SYSTEM,
                "prompt_file": f"{name}.txt",
                "expected_exact": case["expected_exact"],
                "min_prompt_tokens": length,
                "max_prompt_tokens": length,
                "max_tokens": [32],
                "concurrency": [1],
                "requests": 1,
                "warmup_requests": 1,
                "temperature": 0,
                "seed": 42,
                "stream": False,
                "timeout_s": 300,
            }
        )
    manifest = {
        "name": "mlx-varied-retrieval",
        "database": str(args.database.resolve()),
        "runtime_report": os.path.relpath(args.runtime_report.resolve(), args.output.resolve()),
        "transport": {"trust_env": False},
        "targets": [
            {
                "name": "diffusiongemma-4bit",
                "protocol": "openai_chat",
                "base_url": "http://127.0.0.1:8082/v1",
                "model": MODEL,
            }
        ],
        "scenarios": scenarios,
    }
    (args.output / "http-load.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    (args.output / "fixtures.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "revision": REVISION,
                "system": SYSTEM,
                "cases": cases,
            },
            indent=2,
        )
    )
    print(f"Wrote {len(cases)} varied retrieval fixtures to {args.output}")


if __name__ == "__main__":
    main()
