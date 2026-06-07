# Local Serving Benchmark Summary

Last updated: 2026-06-07

This file summarizes local serving experiments so far across llama.cpp, Ollama, MLX, and MLX-VLM on Apple Silicon. It started as a Gemma 4 benchmark log and now also tracks Qwen triage for local coding-agent use.

## Executive summary

Current practical conclusions:

1. **Best math/limited-eval score so far:** llama.cpp Gemma 4 31B QAT `Q4_0` hits GSM8K 0.9600/0.9600, but it is slow and memory-tight. **Best instruction-following score so far:** Ollama `hf.co/ggml-org/gemma-4-12B-it-GGUF:Q8_0`.
2. **Best llama.cpp 26B daily default candidate:** Unsloth Gemma 4 26B `UD-Q4_K_XL` with `q8_0/q8_0` KV, reasoning disabled, 8k context. It beats the older ggml-org `Q4_K_M` baseline on GSM8K and IFEval, and HTTP latency is practical.
3. **Best MLX quality mode:** MLX Gemma 4 26B OptiQ with thinking disabled. It scores well and handles short/coding prompts well, but long-context TTFT is much worse than llama.cpp/Ollama because prompt cache reuse appears weak.
4. **Most important eval fix:** disable thinking/reasoning and use the right model-specific stop token.
5. **Qwen status:** Qwen3.6 35B-A3B `UD-IQ4_NL` is fast enough to be interesting (~15-17 tok/s coding), but limited IFEval was weak and visible empty `<think>` tags remained in `content`. Qwen3.6 27B `IQ4_NL` is dense and is rejected for daily use: limited quality did not improve, lm-eval took ~147m, and HTTP coding speed was only ~3.2 tok/s.
6. **Workflow status:** one-off shell scripts were retired. Quality evals now run through `uv run llama-tune lm-eval` or `uv run llama-tune suite`, with eval settings stored in YAML when possible.

Reasoning/thinking-disabled settings:

| Runtime | Required setting |
|---|---|
| llama.cpp | `--reasoning off` |
| Ollama OpenAI-compatible API | `--gen-kwargs 'reasoning_effort="none"'` for `llama-tune lm-eval`, or JSON field `"reasoning_effort": "none"` |
| MLX `mlx_lm.server` | `--chat-template-args '{"enable_thinking":false}'` |
| Qwen llama.cpp | `--reasoning off --reasoning-format deepseek` |
| Gemma lm-eval stop token | `<turn|>` |
| Qwen lm-eval stop token | `<|im_end|>` |

Without those fixes, the model often emits reasoning text and little/no final `content`, which makes the scores invalid. For Qwen3.6 specifically, even `--reasoning off --reasoning-format deepseek` has still produced a visible empty `<think></think>` prefix in llama.cpp OpenAI responses; treat Qwen IFEval as provisional until that prefix is removed or stripped.

## Contamination caveat

Some recent 12B timing numbers may be contaminated because an MLX-VLM server was left running in the background. That mainly affects **speed, wall time, memory pressure, and HTTP latency**. It should not normally change lm-eval **quality scores** unless it caused request errors, timeouts, OOM, or output truncation. Treat recent 12B runtime comparisons as provisional until rerun with other model servers stopped.

## What the quality metrics mean

| Metric | Meaning |
|---|---|
| GSM8K strict | Grade-school math word problems; exact final numeric answer match using strict extraction. |
| GSM8K flexible | Same task, more permissive answer extraction. |
| IFEval prompt strict | Whole-prompt instruction following. A prompt only passes if all constraints are satisfied. |
| IFEval inst strict | Per-instruction instruction following. Gives partial credit when some constraints pass. |

For coding-agent use, **IFEval prompt strict** is especially important because agent prompts often contain multiple simultaneous constraints.

## Current Gemma leaderboard

Superseded variants are excluded here. Rows are ranked primarily by IFEval prompt strict, then IFEval inst strict, with practical speed/memory notes. Qwen runs are tracked separately until corrected reruns are complete.

| Rank | Model | Role | IFEval prompt strict | IFEval inst strict | GSM8K strict | Practical speed |
|---:|---|---|---:|---:|---:|---:|
| 1 | Ollama `hf.co/ggml-org/gemma-4-12B-it-GGUF:Q8_0` | Best instruction-following quality | **0.9200** | **0.9474** | 0.8800 | ~7.2 tok/s coding |
| 2 | llama.cpp 31B Google QAT `Q4_0` | Best math / high-quality fallback | 0.9000 | 0.9342 | **0.9600** | ~3.6 tok/s coding |
| 3 | MLX `mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit` | Best MLX quality mode | 0.8800 | 0.9211 | 0.8000 | ~20.9 tok/s coding; poor long-ctx TTFT |
| 4 | llama.cpp 26B Unsloth `UD-Q4_K_XL` | Best practical llama.cpp daily default | 0.8800 | 0.9079 | 0.8200 | ~21.0 tok/s coding; excellent long-ctx TTFT |
| 5 | llama.cpp 12B Google QAT `Q4_0` | Compact llama.cpp fallback | 0.8600 | 0.8947 | 0.8600 | ~12.5 tok/s observed; HTTP TBD |

Removed from the active leaderboard:

| Model | Reason |
|---|---|
| Ollama `gemma4:12b` | Superseded by Ollama 12B Q8: same prompt strict, lower inst/GSM8K, slower in token check. |
| Ollama `hf.co/ggml-org/gemma-4-12B-it-GGUF:Q4_K_M` | Superseded by Ollama 12B Q8: lower quality with similar measured token speed. |
| llama.cpp 26B ggml-org `Q4_K_M` | Superseded by llama.cpp 26B Unsloth `UD-Q4_K_XL`: lower GSM8K and IFEval; only slightly faster raw generation. |

## Valid historical Gemma lm-eval quality results

All rows below are `limit=50`, so they are useful for local comparison but not publication-grade metrics. This table intentionally keeps superseded but valid Gemma runs for audit/history; use the leaderboard above for current Gemma decisions.

| Runtime / model | Notes | GSM8K strict | GSM8K flexible | IFEval prompt strict | IFEval inst strict | Eval time |
|---|---|---:|---:|---:|---:|---:|
| llama.cpp 31B Google QAT `Q4_0` | q8 KV, reasoning off, ctx 8192; memory-tight | **0.9600** | **0.9600** | 0.9000 | 0.9342 | 123.5m |
| Ollama `hf.co/ggml-org/gemma-4-12B-it-GGUF:Q8_0` | `reasoning_effort="none"`, Q8 12B | 0.8800 | 0.8800 | **0.9200** | **0.9474** | 74.4m |
| llama.cpp 12B Google QAT `Q4_0` | q8 KV, reasoning off, ctx 16384; memory-comfortable | 0.8600 | 0.8600 | 0.8600 | 0.8947 | 48.7m active |
| Ollama `gemma4:12b` | `reasoning_effort="none"`, Ollama library quant, ~7.6 GB | 0.8200 | 0.8200 | **0.9200** | 0.9342 | 90.3m progress; 123m shell wall |
| Ollama `hf.co/ggml-org/gemma-4-12B-it-GGUF:Q4_K_M` | `reasoning_effort="none"`, official Q4_K_M | 0.8000 | 0.8200 | 0.9000 | 0.9211 | 56.3m |
| llama.cpp 12B ggml-org `Q8_0` | q8 KV, reasoning off | 0.8200 | 0.8200 | 0.8600 | 0.9079 | 81.6m |
| MLX `mlx-community/gemma-4-12B-it-4bit` | `mlx-vlm`, `enable_thinking=False` | 0.8800 | **0.9200** | 0.8000 | 0.8684 | 73.8m |
| MLX `mlx-community/gemma-4-12B-it-8bit` | `mlx-vlm`, `enable_thinking=False` | 0.8200 | 0.8200 | 0.8600 | 0.9079 | 77.6m clean rerun |
| MLX `mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit` | thinking disabled | 0.8000 | 0.8200 | 0.8800 | 0.9211 | 21.8m |
| Ollama `gemma4:26b` | `reasoning_effort="none"`; valid rerun | 0.8200 | 0.8400 | 0.8600 | 0.8947 | 22.2m |
| MLX `mlx-community/gemma-4-26B-A4B-it-qat-mxfp4` | `enable_thinking=False`; memory-comfortable | 0.6800 | 0.7200 | 0.8600 | 0.8947 | 18.8m |
| llama.cpp 26B Unsloth `UD-Q4_K_XL` | q8 KV, reasoning off, ctx 8192; best llama.cpp 26B quality so far | 0.8200 | 0.8200 | 0.8800 | 0.9079 | 27.7m |
| llama.cpp 26B Google QAT `Q4_0` | q8 KV, reasoning off, ctx 8192; fast | 0.8000 | 0.8000 | 0.8400 | 0.8816 | 18.9m |
| Ollama 26B QAT `Q4_0` LM Studio mirror | `reasoning_effort="none"`; backend used 32k ctx + mmproj | 0.7200 | 0.7200 | 0.8400 | 0.8684 | 20.1m |
| llama.cpp 26B Unsloth `UD-Q5_K_M` | q8 KV, reasoning off | 0.8000 | 0.8000 | 0.8600 | 0.8947 | 29.3m |
| llama.cpp 26B ggml-org `Q4_K_M` | q8 KV, reasoning off | 0.8000 | 0.8000 | 0.8600 | 0.8816 | 20.5m |

Interpretation:

- The **31B QAT Q4_0 llama.cpp result is the strongest math result so far** and validates the new QAT weights, but its IFEval scores still trail Ollama 12B Q8 and runtime is much slower.
- The **12B Q8 Ollama result remains the strongest instruction-following result so far**, despite being a smaller model.
- Ollama `gemma4:12b` is a strong lower-disk/memory 12B option: it matches Q8 on IFEval prompt strict but trails Q8 on GSM8K and instruction strict.
- Ollama official `Q4_K_M` is the fastest 12B Ollama eval so far, but gives up quality versus both `gemma4:12b` and Q8.
- llama.cpp 12B QAT Q4_0 is fast and memory-comfortable. It improves GSM8K over llama.cpp 12B Q8, but does **not** improve IFEval prompt strict and trails Ollama 12B Q4/Q8 on instruction following.
- llama.cpp 26B QAT Q4_0 is fast, but quality is disappointing: GSM8K matches the old 26B Q4 baseline, IFEval prompt strict is worse, and it does not replace 26B `Q4_K_M` as the default.
- Ollama 26B QAT Q4_0 did not rescue the model: it matched llama.cpp on IFEval prompt strict but regressed badly on GSM8K and used more memory due 32k ctx + mmproj.
- MLX 26B QAT MXFP4 is memory-comfortable and fast, but quality trails MLX 26B OptiQ on GSM8K, IFEval prompt strict, and IFEval inst strict. It is not a keeper unless needed for additional MXFP4-specific testing.
- Unsloth 26B `UD-Q4_K_XL` is the best llama.cpp 26B quality result so far: it improves over ggml-org 26B `Q4_K_M` on GSM8K, IFEval prompt strict, and IFEval inst strict while remaining practical in HTTP load. It is the likely llama.cpp daily-default replacement.
- Ollama `gemma4:26b` is now valid after `reasoning_effort="none"`; it is fast and has better GSM8K than the old llama.cpp 26B `Q4_K_M` baseline, but instruction-following does not beat Ollama 12B Q8, MLX 26B OptiQ, or Unsloth 26B `UD-Q4_K_XL`.
- llama.cpp 12B Q8 is a strong result: it beats llama.cpp 26B Q4 on GSM8K and IFEval instruction strict, but does not match Ollama 12B Q8.
- MLX 12B 4bit is a mixed result: excellent GSM8K, but meaningfully worse IFEval prompt strict. For coding-agent use, the instruction-following drop matters more than the math gain.
- MLX 12B 8bit exactly matched llama.cpp 12B Q8 on this limited eval. A clean rerun took 77.6m, close to Ollama 12B Q8 and slightly faster than llama.cpp 12B Q8.
- The 12B Q8/8bit/library-quant runs were much slower in lm-eval than the 26B llama.cpp/MLX runs.

## Qwen exploratory lm-eval quality results

All rows are `limit=50`. Qwen runs use `eos_string: "<|im_end|>"`.

| Runtime / model | Notes | GSM8K strict | GSM8K flexible | IFEval prompt strict | IFEval inst strict | Eval time |
|---|---|---:|---:|---:|---:|---:|
| llama.cpp Qwen3.6 35B-A3B `UD-IQ4_NL` | q8 KV, ctx 8192; visible empty `<think>` tags remained in `content`; rerun only if template/output stripping is fixed | 0.8800 | 0.9000 | 0.6200 | 0.7237 | 27.2m |
| llama.cpp Qwen3.6 27B `IQ4_NL` | q8 KV, ctx 8192; dense model; visible empty `<think>` tags remained in `content`; too slow for daily use | 0.8600 | 0.9000 | 0.6200 | 0.7368 | 147.2m |

Interpretation:

- GSM8K was strong enough to keep Qwen interesting for math, but IFEval was too weak for a coding-agent default.
- Visible `<think>\n\n</think>` appeared in content during sanity checks for both 35B-A3B and 27B. That likely undercounts IFEval and means these Qwen quality scores should not be treated as final until the chat template/output stripping issue is fixed.
- Qwen3.6 27B is dense, not MoE. It was far slower than 35B-A3B and did not improve instruction following, so reject it for practical daily use on the 32 GB Mac.

## Invalid or superseded lm-eval results

These should not be used for model comparison:

| Run | Why invalid / superseded |
|---|---|
| Earlier Ollama 26B runs with `EOS_STRING=<end_of_turn>` | Wrong Gemma 4 stop token and reasoning not disabled; GSM8K was 0.0000. |
| Earlier llama.cpp Q4/Q5/q4-KV runs with `EOS_STRING=<end_of_turn>` | Same issue; measured thinking/output formatting failure more than quality. |
| llama.cpp Q5 first run | Thinking was enabled; model filled `reasoning_content`, produced empty final `content`, and hit length. |
| Ollama 12B Q8 first run without `reasoning_effort="none"` | Runtime was 3h23m; GSM8K 0.0200 and IFEval prompt 0.4400 because it mostly generated reasoning. |

## HTTP / coding-agent load results

HTTP load was run with `sweeps/gemma-http-load-ollama-compare.yaml`. The target name `llama-f16-kv` is stale; it means “whatever llama.cpp server was on port 8080 for that run.” These results are mainly useful for latency/TTFT behavior.

Additional Ollama 12B token-speed checks were run through the OpenAI-compatible API with `reasoning_effort="none"`, using the coding-assistant prompt, concurrency 1, `max_tokens=512`, 1 warmup + 3 measured requests:

| Target | mean completion tok/s | p50 completion tok/s | TTFT p95 | check pass |
|---|---:|---:|---:|---:|
| Ollama `hf.co/ggml-org/gemma-4-12B-it-GGUF:Q8_0` | 7.20 | 7.49 | 1.32s | 3/3 |
| Ollama `hf.co/ggml-org/gemma-4-12B-it-GGUF:Q4_K_M` | 7.04 | 7.03 | 1.86s | 3/3 |
| Ollama `gemma4:12b` | 6.60 | 6.46 | 1.01s | 3/3 |

### Interactive short prompt, concurrency 1

| Target | latency p95 | TTFT p95 | completion tok/s | errors |
|---|---:|---:|---:|---:|
| Ollama `gemma4:26b` | **4.371s** | 0.284s | **29.990** | 0 |
| llama.cpp target | 4.764s | 0.563s | 28.450 | 0 |
| MLX E4B OptiQ | 5.147s | n/a | 25.208 | 0 |
| MLX 26B OptiQ | 5.479s | 0.287s | 24.335 | 0 |

### Coding assistant prompt, concurrency 1

| Target | latency p95 | TTFT p95 | completion tok/s | check pass |
|---|---:|---:|---:|---:|
| MLX 26B OptiQ | **7.887s** | **0.313s** | 20.861 | 1.000 |
| llama.cpp target | 21.463s | 1.019s | **24.159** | 1.000 |
| Ollama `gemma4:26b` | 22.481s | 0.314s | 23.632 | 1.000 |
| MLX E4B OptiQ | 23.120s | 15.452s | 22.833 | 1.000 |

### Long-context recall, concurrency 1

| Target | latency p95 | TTFT p95 | completion tok/s | check pass |
|---|---:|---:|---:|---:|
| llama.cpp target | **5.915s** | **0.180s** | **22.359** | 1.000 |
| Ollama `gemma4:26b` | 6.935s | 0.448s | 19.663 | 1.000 |
| MLX 26B OptiQ, prefill 4096 | 22.299s | 20.763s | 1.774 | 1.000 |
| MLX 26B OptiQ, prefill 8192 | 26.885s | 25.325s | 1.491 | 1.000 |
| MLX E4B OptiQ | 22.860s | n/a | 5.641 | 0.000 |

### llama.cpp 26B Unsloth UD-Q4_K_XL, concurrency 1

HTTP load was run with `sweeps/gemma4-26b-qat-http-load.yaml` while the server was actually `unsloth/gemma-4-26B-A4B-it-GGUF` / `gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf`, q8 KV, reasoning off, ctx 8192. The stored target label is stale (`llama-gemma4-26b-a4b-qat-q4_0`).

| Scenario | latency p95 | TTFT p95 | completion tok/s | check pass | errors |
|---|---:|---:|---:|---:|---:|
| interactive-short | 7.016s | 0.732s | 19.694 | n/a | 0 |
| coding-assistant | 7.798s | 0.731s | 20.965 | 1.000 | 0 |
| long-context-recall | 2.222s | 0.065s | 19.483 | 1.000 | 0 |

### llama.cpp 26B QAT Q4_0, concurrency 1

Dedicated HTTP load was run with `sweeps/gemma4-26b-qat-http-load.yaml` against `google/gemma-4-26B-A4B-it-qat-q4_0-gguf`, q8 KV, reasoning off, ctx 8192.

| Scenario | latency p95 | TTFT p95 | completion tok/s | check pass | errors |
|---|---:|---:|---:|---:|---:|
| interactive-short | 5.903s | 1.343s | 23.288 | n/a | 0 |
| coding-assistant | 8.249s | 1.145s | 24.456 | 1.000 | 0 |
| long-context-recall | 1.853s | 0.059s | 23.001 | 1.000 | 0 |

### llama.cpp 31B QAT Q4_0, concurrency 1

Dedicated HTTP load was run with `sweeps/gemma4-31b-qat-http-load.yaml` against `google/gemma-4-31B-it-qat-q4_0-gguf`, q8 KV, reasoning off, ctx 8192.

| Scenario | latency p95 | TTFT p95 | completion tok/s | check pass | errors |
|---|---:|---:|---:|---:|---:|
| interactive-short | 37.000s | 4.824s | 3.502 | n/a | 0 |
| coding-assistant | 46.212s | 4.828s | 3.583 | 1.000 | 0 |
| long-context-recall | 11.320s | 0.315s | 3.496 | 1.000 | 0 |

Interpretation:

- Unsloth 26B `UD-Q4_K_XL` is slightly slower in completion tok/s than some 26B baselines, but coding latency is excellent because it produces concise successful answers; long-context TTFT is also excellent.
- 26B QAT Q4_0 has a good latency profile and excellent prompt-cache behavior, but its limited-eval instruction-following score is worse than the older 26B `Q4_K_M`.
- 31B QAT Q4_0 is quality-strong but **too slow for the daily coding-agent default** at ~3.5 completion tok/s and ~4.8s TTFT on short/coding prompts.
- It is useful as a high-quality local fallback when latency is less important.
- MLX 26B became viable after disabling thinking and passed coding/long-context checks.
- MLX 26B has a **large long-context TTFT penalty**: ~20–25s vs sub-second for llama.cpp/Ollama.
- Direct checks showed MLX was only reporting about `19 / 5915` prompt tokens cached on repeated long-context requests, so it appears to re-prefill almost the whole prompt. llama.cpp/Ollama appear to benefit from much stronger prompt/prefix cache reuse.

### llama.cpp Qwen3.6 35B-A3B `UD-IQ4_NL`, concurrency 1, preliminary

HTTP load was run with `sweeps/qwen36-http-load.yaml` against `unsloth/Qwen3.6-35B-A3B-GGUF` / `Qwen3.6-35B-A3B-UD-IQ4_NL.gguf`, q8 KV, ctx 8192. Visible empty `<think>` tags remained in content during sanity checks, so strict instruction quality needs a template/output-stripping fix before final judgment.

| Scenario | latency p95 | TTFT p95 | completion tok/s | check pass | errors |
|---|---:|---:|---:|---:|---:|
| interactive-short | 8.306s | 1.292s | 16.068 | n/a | 0 |
| coding-assistant | 10.352s | 0.824s | 16.817 | 1.000 | 0 |
| long-context-recall | 3.340s | 0.159s | 15.347 | 1.000 | 0 |

Memory pressure was high for this 32 GB machine after the run: swap climbed from ~3.3 GB before to ~6.6 GB after. That makes 35B-A3B plausible but not obviously comfortable as an always-on default on the current Mac.

### llama.cpp Qwen3.6 27B `IQ4_NL`, concurrency 1

HTTP load was run with `sweeps/qwen36-27b-http-load.yaml` against `unsloth/Qwen3.6-27B-GGUF` / `Qwen3.6-27B-IQ4_NL.gguf`, q8 KV, ctx 8192. This is a dense model and was much slower than the 35B-A3B MoE.

| Scenario | latency p95 | TTFT p95 | completion tok/s | check pass | errors |
|---|---:|---:|---:|---:|---:|
| interactive-short | 41.694s | 3.376s | 3.104 | n/a | 0 |
| coding-assistant | 56.477s | 3.508s | 3.170 | 1.000 | 0 |
| long-context-recall | 14.148s | 0.924s | 3.249 | 1.000 | 0 |

Verdict: reject for daily use on the 32 GB Mac. It is ~5x slower than Qwen3.6 35B-A3B in HTTP coding throughput and does not improve IFEval.

## Memory observations

Activity Monitor “App Memory” can understate Metal/unified-memory allocations; watch system memory pressure, wired memory, compressed memory, and swap.

| Runtime / model | Observed memory behavior |
|---|---|
| llama.cpp 26B `Q4_K_M` + q8 KV + 8k ctx | Practical baseline; acceptable pressure compared with Q5/f16 KV. |
| llama.cpp 26B Unsloth `UD-Q5_K_M` + q8 KV | Yellow pressure; ~30.5 / 32 GB used, ~24.2 GB wired, ~3.75 GB swap. Quality gain too small to justify memory. |
| MLX 26B OptiQ, prefill 4096 | Green pressure; ~26.6 / 32 GB used, ~19.2 GB wired, ~0.7 GB compressed, ~4.9 GB swap. |
| MLX 26B OptiQ, prefill 8192 | Green but tighter; ~29.7 / 32 GB used, ~23.0 GB wired, ~1.7 GB compressed, ~4.6 GB swap. Performance regressed vs 4096. |
| llama.cpp Gemma 4 12B Q8 | Green pressure; ~26.0 / 32 GB used, ~16.2 GB wired, ~0.9 GB compressed, ~4.2 GB swap. Looks comfortable. |
| llama.cpp 12B Google QAT `Q4_0` + q8 KV + 16k ctx | Very comfortable. During eval: green pressure, ~18.4 / 32 GB used, ~9.9 GB wired, ~0.7 GB compressed, ~3.1 GB swap. |
| llama.cpp 26B Google QAT `Q4_0` + q8 KV + 8k ctx | Comfortable enough. During eval: green pressure, ~24.6 / 32 GB used, ~16.8 GB wired, ~1.0 GB compressed, ~2.9 GB swap. After HTTP load, llama RSS grew to ~16.8 GB and compressed memory rose. |
| llama.cpp 26B Unsloth `UD-Q4_K_XL` + q8 KV + 8k ctx | Similar model/RSS footprint to Google 26B QAT. During eval/HTTP reruns, pressure stayed acceptable with llama RSS around ~14.3–17.2 GB. Swap remained high after a long benchmarking session, so restart/quit other apps for clean daily-use memory readings. |
| Ollama 26B QAT `Q4_0` LM Studio mirror | Yellow pressure near end of eval. Ollama backend used 32k ctx + mmproj; ~29.9 / 32 GB used, ~18.5 GB wired, ~7.3 GB compressed, ~5.7 GB swap. |
| MLX 26B QAT `mxfp4` | Green and comfortable during eval; ~21.6 / 32 GB used, ~15.7 GB wired, ~0.2 GB compressed, ~4.7 GB swap. |
| llama.cpp 31B Google QAT `Q4_0` + q8 KV + 8k ctx | Fits, but tight. During/after eval: green pressure, ~30 / 32 GB used, ~20.5 GB wired, ~3 GB compressed, ~3.3 GB swap. Not ideal as an always-on default. |
| llama.cpp Qwen3.6 35B-A3B `UD-IQ4_NL` + q8 KV + 8k ctx | Fits but swap-heavy on the 32 GB Mac. After eval + HTTP load, swap was ~6.6 GB and free swap was low. Rerun after reboot/cleanup for a cleaner daily-use reading. |
| llama.cpp Qwen3.6 27B `IQ4_NL` + q8 KV + 8k ctx | Fits but not comfortable after a long eval. Swap was ~3.6 GB before and ~5.9 GB after; generation speed was too slow to justify the footprint. |

## Runtime tuning findings

Best llama.cpp runtime defaults found from `llama bench` and sweeps:

```yaml
batch_size: 512
ubatch_size: 512
threads: 4
poll: 0
flash_attn: auto   # or 1 in llama bench
cache_type_k: q8_0
cache_type_v: q8_0
```

KV cache conclusions for 26B:

- `q8_0/q8_0` KV is the best quality/memory compromise.
- `f16/f16` KV did not improve quality enough and increased memory pressure.
- `q4_0/q4_0` KV reduced memory pressure but hurt IFEval.

MLX prefill tuning:

- `MLX_PREFILL_STEP_SIZE=2048`: long-context TTFT ~26.3s.
- `MLX_PREFILL_STEP_SIZE=4096`: best observed, TTFT ~20.8s.
- `MLX_PREFILL_STEP_SIZE=8192`: regressed to TTFT ~25.3s and used more memory.

## Workflow notes

One-off shell scripts have been retired. Use Python/YAML workflow commands instead:

```bash
uv run llama-tune lm-eval llama_cpp 50 \
  --eos-string '<turn|>' \
  --max-length 8192

uv run llama-tune suite sweeps/qwen36-35b-llama-sweep.yaml \
  --http-load-config sweeps/qwen36-http-load.yaml \
  --target llama-qwen36-35b-a3b-ud-iq4-nl
```

YAML configs can carry eval defaults:

```yaml
eval:
  tasks: ifeval,gsm8k
  limit: 50
  max_length: 8192
  eos_string: "<|im_end|>"
  gen_kwargs: reasoning_effort="none"
```

## Recommended commands

### llama.cpp Gemma 4 26B daily default

```bash
llama server \
  -hf unsloth/gemma-4-26B-A4B-it-GGUF \
  -hff gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  --no-mmproj \
  --reasoning off \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --batch-size 512 \
  --ubatch-size 512 \
  --flash-attn auto \
  --threads 4 \
  --poll 0 \
  --ctx-size 8192 \
  --n-gpu-layers all \
  --parallel 1 \
  --perf
```

### llama.cpp Gemma 4 12B Q8 text-only

Use `--no-mmproj` for text-only until the local llama.cpp build supports Gemma 4 `gemma4uv` projector loading.

```bash
llama server \
  -hf ggml-org/gemma-4-12B-it-GGUF \
  -hff gemma-4-12B-it-Q8_0.gguf \
  --no-mmproj \
  --reasoning off \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --batch-size 512 \
  --ubatch-size 512 \
  --flash-attn auto \
  --threads 4 \
  --poll 0 \
  --ctx-size 16384 \
  --n-gpu-layers all \
  --parallel 1 \
  --perf
```

### Qwen3.6 35B-A3B debugging candidate

Only trust IFEval after confirming the sanity preview no longer starts with an empty `<think></think>` block.

Terminal 1:

```bash
kill $(lsof -tiTCP:8080 -sTCP:LISTEN) 2>/dev/null || true
uv run llama-tune server sweeps/qwen36-35b-llama-sweep.yaml --index 0
```

Terminal 2:

```bash
uv run llama-tune suite sweeps/qwen36-35b-llama-sweep.yaml \
  --http-load-config sweeps/qwen36-http-load.yaml \
  --target llama-qwen36-35b-a3b-ud-iq4-nl
```

### Qwen3-Coder 30B-A3B Q4_K_M candidate

Terminal 1:

```bash
kill $(lsof -tiTCP:8080 -sTCP:LISTEN) 2>/dev/null || true
uv run llama-tune server sweeps/qwen3-coder-30b-llama-sweep.yaml --index 0
```

Terminal 2:

```bash
uv run llama-tune suite sweeps/qwen3-coder-30b-llama-sweep.yaml \
  --http-load-config sweeps/qwen3-coder-http-load.yaml \
  --target llama-qwen3-coder-30b-a3b-q4km
```

### Ollama Gemma 4 12B Q8 quality eval

```bash
uv run llama-tune lm-eval ollama 50 \
  --model 'hf.co/ggml-org/gemma-4-12B-it-GGUF:Q8_0' \
  --gen-kwargs 'reasoning_effort="none"' \
  --max-length 8192
```

### MLX 26B OptiQ quality mode

```bash
uvx --from mlx-lm mlx_lm.server \
  --model mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit \
  --host 127.0.0.1 \
  --port 8082 \
  --temp 0.0 \
  --max-tokens 1280 \
  --prompt-cache-size 1 \
  --prefill-step-size 4096 \
  --chat-template-args '{"enable_thinking":false}' \
  --decode-concurrency 1 \
  --prompt-concurrency 1
```

## Multimodal note

The `llama.app` build tested was:

```text
b9444-6f165c1c6
```

It failed to load Gemma 4 12B mmproj with:

```text
unknown projector type: gemma4uv
```

Upstream llama.cpp HEAD contains `gemma4uv` support. For multimodal Gemma 4 with llama.cpp, build upstream HEAD and run `llama-server` from that build with `--mmproj-auto`. For text-only, use `--no-mmproj`.

Fallback multimodal path:

```bash
uvx --from mlx-vlm mlx_vlm.generate \
  --model mlx-community/gemma-4-12B-it-8bit \
  --image image.png \
  --prompt "Describe this image briefly." \
  --max-tokens 256
```

## Next suggested experiments

1. Investigate Qwen3.6 empty `<think></think>` prefixes in llama.cpp OpenAI responses. Try a custom/generic ChatML template or response-stripping proxy before trusting IFEval; if fixed, rerun Qwen3.6 35B-A3B `UD-IQ4_NL`.
2. Run Qwen3-Coder 30B-A3B `Q4_K_M` with `sweeps/qwen3-coder-30b-llama-sweep.yaml` and compare against Gemma 4 26B `UD-Q4_K_XL`.
3. Add a dedicated HTTP target/sweep for Gemma 4 26B `UD-Q4_K_XL` so future results are not stored under stale QAT labels.
4. Add a lightweight deterministic agentic eval harness: patch applies, pytest repair, JSON/tool validity, multi-file edit, long-context repo task, retry behavior.
5. Run a real multimodal smoke test for Ollama 12B Q8 with an image.
6. Capture `prompt_tokens_details.cached_tokens` in `http-load` so prompt-cache behavior is visible in DB comparisons.
7. If a 128 GB Mac becomes available, try DS4 / DeepSeek V4 Flash q2-imatrix as the high-memory local frontier-ish candidate.
