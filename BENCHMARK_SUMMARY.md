# Local Serving Benchmark Summary

Last updated: 2026-06-09

This file summarizes local serving experiments so far across llama.cpp, Ollama, MLX, and MLX-VLM on Apple Silicon. It started as a Gemma 4 benchmark log and now also tracks Qwen triage for local coding-agent use.

## Executive summary

Current practical conclusions:

1. **Best math/limited-eval score so far:** llama.cpp Gemma 4 31B QAT `Q4_0` hits GSM8K 0.9600/0.9600, but it is slow and memory-tight. **Best instruction-following score so far:** Ollama `hf.co/ggml-org/gemma-4-12B-it-GGUF:Q8_0`. New llama.cpp Unsloth Gemma 4 12B `UD-Q4_K_XL` is quality-strong (IFEval 0.8800/0.9211) but too slow for daily coding-agent use on the 32 GB Mac; the Unsloth 12B QAT `UD-Q4_K_XL` variant is much faster and stronger on GSM8K but lower on IFEval. Gemma 4 MTP works in `b9553`, but current full-suite MTP runs are not keepers on the M4: 12B QAT MTP regressed badly with cache disabled, and 12B Q8 MTP with cache enabled is only roughly comparable to non-MTP 12B options while using more memory/swap.
2. **Best llama.cpp 26B daily default candidate:** Unsloth Gemma 4 26B `UD-Q4_K_XL` with `q8_0/q8_0` KV, reasoning disabled, 8k context. It beats the older ggml-org `Q4_K_M` baseline on GSM8K and IFEval, and HTTP latency is practical. SuperGemma4 26B uncensored `Q4_K_M` is similarly fast in HTTP load, but quality is much worse and it produced one parser-breaking `<|channel>thought`/`<channel|>` leak during GSM8K retry handling, so it is not a keeper.
3. **Best MLX quality mode:** MLX Gemma 4 26B OptiQ with thinking disabled. It scores well and handles short/coding prompts well, but long-context TTFT is much worse than llama.cpp/Ollama because prompt cache reuse appears weak.
4. **Most important eval fix:** disable thinking/reasoning and use the right model-specific stop token.
5. **Qwen status:** corrected Qwen3.6 35B-A3B `UD-IQ4_NL` is a strong non-Gemma candidate: IFEval 0.8800/0.9211, GSM8K 0.8800/0.9000, HTTP coding ~18.5 tok/s, and good long-context TTFT. Qwen3-Coder 30B-A3B `Q4_K_M` is faster and stronger on GSM8K (0.9200), but trails on IFEval (0.8600/0.8947). Both are interesting; Gemma 4 26B `UD-Q4_K_XL` remains the safer 32 GB daily default. Qwen3.6 27B `IQ4_NL` is dense and rejected for daily use: lm-eval took ~147m and HTTP coding speed was only ~3.2 tok/s.
6. **DiffusionGemma MLX status:** MLX-VLM DiffusionGemma 26B 4bit loads and passes a short sanity check, but the full `limit=50` IFEval+GSM8K suite failed at 28/100 after repeated 600s no-token timeouts. Treat full quality eval as a 32 GB practical reject; the sweep now defaults to a short smoke eval only.
7. **Workflow status:** one-off shell scripts were retired. Quality evals now run through `uv run llm-refinery lm-eval` or `uv run llm-refinery suite`, with eval settings stored in YAML when possible.

## Recommended non-rejected shortlist

All quality scores are limited `limit=50` runs. Speed is coding-assistant HTTP load when available.

| Model | Quality | Speed | Memory | Use when |
|---|---:|---:|---|---|
| Gemma 4 26B Unsloth `UD-Q4_K_XL` llama.cpp | GSM8K 0.820/0.820; IFEval 0.880/0.908 | ~21 tok/s coding; p95 7.8s; TTFT 0.73s | Acceptable on 32 GB; RSS ~14-17 GB in long sessions | Best daily default: balanced quality, latency, and prompt-cache behavior. |
| Qwen3.6 35B-A3B `UD-IQ4_NL` llama.cpp | GSM8K 0.880/0.900; IFEval 0.880/0.921 | ~18.5 tok/s coding; p95 9.6s; TTFT 0.99s | Swap-heavy on 32 GB | Strong non-Gemma quality candidate; better suited to the 128 GB Mac. |
| Qwen3-Coder 30B-A3B `Q4_K_M` llama.cpp | GSM8K 0.920/0.920; IFEval 0.860/0.895 | ~20.7 tok/s coding; p95 11.4s; TTFT 0.14s | Fits similarly to other 30B-class MoEs | Math/coding-focused tasks where strict instruction following matters slightly less. |
| Ollama Gemma 4 12B Q8 | GSM8K 0.880/0.880; IFEval 0.920/0.947 | ~7.2 tok/s coding token check; TTFT 1.32s | Compact/easy Ollama path | Best 12B instruction follower and lightweight fallback. |
| MLX Gemma 4 26B OptiQ | GSM8K 0.800/0.820; IFEval 0.880/0.921 | ~20.9 tok/s coding; p95 7.9s; TTFT 0.31s | Comfortable | MLX workflows and short/coding prompts; avoid long-context due poor TTFT. |
| Gemma 4 31B Google QAT `Q4_0` llama.cpp | GSM8K 0.960/0.960; IFEval 0.900/0.934 | ~3.6 tok/s coding; p95 46s; TTFT 4.8s | Tight on 32 GB | Highest math/quality fallback when latency does not matter. |
| Gemma 4 12B Unsloth QAT `UD-Q4_K_XL` llama.cpp | GSM8K 0.880/0.920; IFEval 0.840/0.895 | ~8.8 tok/s coding; p95 24.4s; TTFT 2.48s | Very comfortable; RSS ~9.2 GB after eval + HTTP load | Compact llama.cpp math-oriented fallback. |
| Gemma 4 12B Google QAT `Q4_0` llama.cpp | GSM8K 0.860/0.860; IFEval 0.860/0.895 | ~12.5 tok/s observed; HTTP TBD | Very comfortable | Compact llama.cpp fallback if footprint matters. |

Practical ordering:

```text
Daily default:          Gemma 4 26B Unsloth UD-Q4_K_XL
Best 12B fallback:      Ollama Gemma 4 12B Q8
Best math quality:      Gemma 4 31B QAT Q4_0
Best non-Gemma quality: Qwen3.6 35B-A3B UD-IQ4_NL
Best coding/math MoE:   Qwen3-Coder 30B-A3B Q4_K_M
Best MLX option:        Gemma 4 26B OptiQ
```

Reasoning/thinking-disabled settings:

| Runtime | Required setting |
|---|---|
| llama.cpp | `--reasoning off` |
| Ollama OpenAI-compatible API | `--gen-kwargs 'reasoning_effort="none"'` for `llm-refinery lm-eval`, or JSON field `"reasoning_effort": "none"` |
| MLX `mlx_lm.server` | `--chat-template-args '{"enable_thinking":false}'` |
| Qwen llama.cpp | `--reasoning off --reasoning-format deepseek` |
| Gemma lm-eval stop token | `<turn|>` |
| Qwen lm-eval stop token | `<|im_end|>` |

Without those fixes, the model often emits reasoning text and little/no final `content`, which makes the scores invalid. For Qwen3.6, visible empty `<think></think>` prefixes mean the server was probably launched with the wrong reasoning format; `llm-refinery suite` now fails preflight if reasoning tags appear in `content`.

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

Superseded variants are excluded here. Rows are ranked primarily by IFEval prompt strict, then IFEval inst strict, with practical speed/memory notes. Qwen runs are tracked separately below because this leaderboard is Gemma-only.

| Rank | Model | Role | IFEval prompt strict | IFEval inst strict | GSM8K strict | Practical speed |
|---:|---|---|---:|---:|---:|---:|
| 1 | Ollama `hf.co/ggml-org/gemma-4-12B-it-GGUF:Q8_0` | Best instruction-following quality | **0.9200** | **0.9474** | 0.8800 | ~7.2 tok/s coding |
| 2 | llama.cpp 31B Google QAT `Q4_0` | Best math / high-quality fallback | 0.9000 | 0.9342 | **0.9600** | ~3.6 tok/s coding |
| 3 | MLX `mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit` | Best MLX quality mode | 0.8800 | 0.9211 | 0.8000 | ~20.9 tok/s coding; poor long-ctx TTFT |
| 4 | llama.cpp 12B Unsloth `UD-Q4_K_XL` | Best 12B llama.cpp instruction result; slow | 0.8800 | 0.9211 | 0.8200 | ~4.2 tok/s coding; ~2.4h eval |
| 5 | llama.cpp 26B Unsloth `UD-Q4_K_XL` | Best practical llama.cpp daily default | 0.8800 | 0.9079 | 0.8200 | ~21.0 tok/s coding; excellent long-ctx TTFT |
| 6 | llama.cpp 12B Google QAT `Q4_0` | Compact llama.cpp fallback | 0.8600 | 0.8947 | 0.8600 | ~12.5 tok/s observed; HTTP TBD |
| 7 | llama.cpp 12B Unsloth QAT `UD-Q4_K_XL` | Compact math-strong llama.cpp 12B | 0.8400 | 0.8947 | 0.8800 | ~8.8 tok/s coding; comfortable memory |

Removed from the active leaderboard:

| Model | Reason |
|---|---|
| Ollama `gemma4:12b` | Superseded by Ollama 12B Q8: same prompt strict, lower inst/GSM8K, slower in token check. |
| Ollama `hf.co/ggml-org/gemma-4-12B-it-GGUF:Q4_K_M` | Superseded by Ollama 12B Q8: lower quality with similar measured token speed. |
| llama.cpp 26B ggml-org `Q4_K_M` | Superseded by llama.cpp 26B Unsloth `UD-Q4_K_XL`: lower GSM8K and IFEval; only slightly faster raw generation. |
| llama.cpp SuperGemma4 26B uncensored `Q4_K_M` | Rejected for coding-agent default: HTTP speed is good, but limited eval quality is weak and one GSM8K request triggered parser-breaking internal channel-token leakage. |

## Current Qwen / non-Gemma candidates

Rows are ranked by practical coding-agent usefulness on the current 32 GB Mac.

| Rank | Model | Role | IFEval prompt strict | IFEval inst strict | GSM8K strict | Practical speed | Verdict |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | llama.cpp Qwen3.6 35B-A3B `UD-IQ4_NL` | Best non-Gemma instruction candidate so far | 0.8800 | 0.9211 | 0.8800 | ~18.5 tok/s coding; good long-ctx TTFT | Strong quality candidate, but swap-heavy on 32 GB |
| 2 | llama.cpp Qwen3-Coder 30B-A3B `Q4_K_M` | Coding-focused MoE / math-strong candidate | 0.8600 | 0.8947 | 0.9200 | ~20.7 tok/s coding; very low TTFT | Fast and math-strong, but weaker IFEval |
| 3 | llama.cpp Qwen3.6 27B `IQ4_NL` | Dense Qwen comparison | 0.6200 | 0.7368 | 0.8600 | ~3.2 tok/s coding | Reject: much too slow |

## Valid historical Gemma lm-eval quality results

All rows below are `limit=50`, so they are useful for local comparison but not publication-grade metrics. This table intentionally keeps superseded but valid Gemma runs for audit/history; use the leaderboard above for current Gemma decisions.

| Runtime / model | Notes | GSM8K strict | GSM8K flexible | IFEval prompt strict | IFEval inst strict | Eval time |
|---|---|---:|---:|---:|---:|---:|
| llama.cpp 31B Google QAT `Q4_0` | q8 KV, reasoning off, ctx 8192; memory-tight | **0.9600** | **0.9600** | 0.9000 | 0.9342 | 123.5m |
| Ollama `hf.co/ggml-org/gemma-4-12B-it-GGUF:Q8_0` | `reasoning_effort="none"`, Q8 12B | 0.8800 | 0.8800 | **0.9200** | **0.9474** | 74.4m |
| llama.cpp 12B Unsloth `UD-Q4_K_XL` | q8 KV, reasoning off, ctx 8192; strong quality but slow HTTP/load | 0.8200 | 0.8600 | 0.8800 | 0.9211 | 143.3m |
| llama.cpp 12B Unsloth QAT `UD-Q4_K_XL` | q8 KV, reasoning off, ctx 8192; faster/math-stronger but lower IFEval than non-QAT | 0.8800 | 0.9200 | 0.8400 | 0.8947 | 52.9m |
| llama.cpp Huihui 12B abliterated `Q4_K` | q8 KV, reasoning off, ctx 8192; valid abliterated comparison | 0.8600 | 0.8800 | 0.8600 | 0.9079 | 58.7m |
| llama.cpp 12B Google QAT `Q4_0` | q8 KV, reasoning off, ctx 16384; memory-comfortable | 0.8600 | 0.8600 | 0.8600 | 0.8947 | 48.7m active |
| Ollama `gemma4:12b` | `reasoning_effort="none"`, Ollama library quant, ~7.6 GB | 0.8200 | 0.8200 | **0.9200** | 0.9342 | 90.3m progress; 123m shell wall |
| Ollama `hf.co/ggml-org/gemma-4-12B-it-GGUF:Q4_K_M` | `reasoning_effort="none"`, official Q4_K_M | 0.8000 | 0.8200 | 0.9000 | 0.9211 | 56.3m |
| llama.cpp 12B Unsloth `Q8_0` + MTP Q8 head | q8 KV, reasoning off, ctx 8192, prompt cache/checkpoints enabled | 0.8200 | 0.8400 | 0.8600 | 0.9079 | 65.9m |
| llama.cpp 12B ggml-org `Q8_0` | q8 KV, reasoning off | 0.8200 | 0.8200 | 0.8600 | 0.9079 | 81.6m |
| MLX `mlx-community/gemma-4-12B-it-4bit` | `mlx-vlm`, `enable_thinking=False` | 0.8800 | **0.9200** | 0.8000 | 0.8684 | 73.8m |
| MLX `mlx-community/gemma-4-12B-it-8bit` | `mlx-vlm`, `enable_thinking=False` | 0.8200 | 0.8200 | 0.8600 | 0.9079 | 77.6m clean rerun |
| MLX `mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit` | thinking disabled | 0.8000 | 0.8200 | 0.8800 | 0.9211 | 21.8m |
| Ollama `gemma4:26b` | `reasoning_effort="none"`; valid rerun | 0.8200 | 0.8400 | 0.8600 | 0.8947 | 22.2m |
| MLX `mlx-community/gemma-4-26B-A4B-it-qat-mxfp4` | `enable_thinking=False`; memory-comfortable | 0.6800 | 0.7200 | 0.8600 | 0.8947 | 18.8m |
| llama.cpp 26B Unsloth `UD-Q4_K_XL` | q8 KV, reasoning off, ctx 8192; best llama.cpp 26B quality so far | 0.8200 | 0.8200 | 0.8800 | 0.9079 | 27.7m |
| llama.cpp SuperGemma4 26B uncensored `Q4_K_M` | q8 KV, reasoning off, ctx 8192; fast but quality/reliability reject | 0.7800 | 0.8200 | 0.7800 | 0.8289 | 55.3m |
| llama.cpp 26B Google QAT `Q4_0` | q8 KV, reasoning off, ctx 8192; fast | 0.8000 | 0.8000 | 0.8400 | 0.8816 | 18.9m |
| Ollama 26B QAT `Q4_0` LM Studio mirror | `reasoning_effort="none"`; backend used 32k ctx + mmproj | 0.7200 | 0.7200 | 0.8400 | 0.8684 | 20.1m |
| llama.cpp 26B Unsloth `UD-Q5_K_M` | q8 KV, reasoning off | 0.8000 | 0.8000 | 0.8600 | 0.8947 | 29.3m |
| llama.cpp 26B ggml-org `Q4_K_M` | q8 KV, reasoning off | 0.8000 | 0.8000 | 0.8600 | 0.8816 | 20.5m |

Interpretation:

- The **31B QAT Q4_0 llama.cpp result is the strongest math result so far** and validates the new QAT weights, but its IFEval scores still trail Ollama 12B Q8 and runtime is much slower.
- The **12B Q8 Ollama result remains the strongest instruction-following result so far**, despite being a smaller model.
- llama.cpp Unsloth 12B `UD-Q4_K_XL` is the strongest llama.cpp 12B instruction result so far and matches MLX 26B OptiQ on limited IFEval, but runtime is poor: ~143m lm-eval and ~4.2 tok/s coding HTTP. Treat it as a compact quality diagnostic rather than a daily default.
- llama.cpp Unsloth 12B QAT `UD-Q4_K_XL` is a major speed improvement over non-QAT and improves GSM8K to 0.8800/0.9200, but IFEval drops to 0.8400/0.8947. It is a compact math-oriented fallback, not the best instruction-following 12B.
- Huihui 12B abliterated `Q4_K` is valid and reasonably fast, but it does not beat the kept 12B options overall: IFEval trails Ollama 12B Q8, GSM8K trails Unsloth 12B QAT, and HTTP latency is not better. Keep only if abliterated behavior is specifically desired.
- Ollama `gemma4:12b` is a strong lower-disk/memory 12B option: it matches Q8 on IFEval prompt strict but trails Q8 on GSM8K and instruction strict.
- Ollama official `Q4_K_M` is the fastest 12B Ollama eval so far, but gives up quality versus both `gemma4:12b` and Q8.
- llama.cpp 12B QAT Q4_0 is fast and memory-comfortable. It improves GSM8K over llama.cpp 12B Q8, but does **not** improve IFEval prompt strict and trails Ollama 12B Q4/Q8 on instruction following.
- llama.cpp 26B QAT Q4_0 is fast, but quality is disappointing: GSM8K matches the old 26B Q4 baseline, IFEval prompt strict is worse, and it does not replace 26B `Q4_K_M` as the default.
- Ollama 26B QAT Q4_0 did not rescue the model: it matched llama.cpp on IFEval prompt strict but regressed badly on GSM8K and used more memory due 32k ctx + mmproj.
- MLX 26B QAT MXFP4 is memory-comfortable and fast, but quality trails MLX 26B OptiQ on GSM8K, IFEval prompt strict, and IFEval inst strict. It is not a keeper unless needed for additional MXFP4-specific testing.
- Unsloth 26B `UD-Q4_K_XL` is the best llama.cpp 26B quality result so far: it improves over ggml-org 26B `Q4_K_M` on GSM8K, IFEval prompt strict, and IFEval inst strict while remaining practical in HTTP load. It is the likely llama.cpp daily-default replacement.
- SuperGemma4 26B uncensored `Q4_K_M` is fast but not useful for the coding-agent shortlist: GSM8K and IFEval both fall well below the kept 26B and 12B options, and one GSM8K retry exposed internal channel tokens (`<|channel>thought` / `<channel|>`) that llama.cpp could not parse.
- Ollama `gemma4:26b` is now valid after `reasoning_effort="none"`; it is fast and has better GSM8K than the old llama.cpp 26B `Q4_K_M` baseline, but instruction-following does not beat Ollama 12B Q8, MLX 26B OptiQ, or Unsloth 26B `UD-Q4_K_XL`.
- llama.cpp 12B Q8 is a strong result: it beats llama.cpp 26B Q4 on GSM8K and IFEval instruction strict, but does not match Ollama 12B Q8. The Unsloth Q8 + MTP cache-enabled run preserved the same IFEval scores and improved eval wall time vs the older llama.cpp Q8 run, but memory/swap increased and HTTP latency still trails Ollama Q8.
- MLX 12B 4bit is a mixed result: excellent GSM8K, but meaningfully worse IFEval prompt strict. For coding-agent use, the instruction-following drop matters more than the math gain.
- MLX 12B 8bit exactly matched llama.cpp 12B Q8 on this limited eval. A clean rerun took 77.6m, close to Ollama 12B Q8 and slightly faster than llama.cpp 12B Q8.
- The 12B Q8/8bit/library-quant runs were much slower in lm-eval than the 26B llama.cpp/MLX runs.

## Qwen exploratory lm-eval quality results

All rows are `limit=50`. Qwen runs use `eos_string: "<|im_end|>"`.

| Runtime / model | Notes | GSM8K strict | GSM8K flexible | IFEval prompt strict | IFEval inst strict | Eval time |
|---|---|---:|---:|---:|---:|---:|
| llama.cpp Qwen3.6 35B-A3B `UD-IQ4_NL` | q8 KV, ctx 8192; corrected rerun with clean sanity check and `--reasoning-format deepseek` | 0.8800 | 0.9000 | 0.8800 | 0.9211 | 23.2m |
| llama.cpp Qwen3-Coder 30B-A3B `Q4_K_M` | q8 KV, ctx 8192; clean sanity check, coding-focused MoE | **0.9200** | **0.9200** | 0.8600 | 0.8947 | 20.3m |
| llama.cpp Qwen3.6 27B `IQ4_NL` | q8 KV, ctx 8192; dense model; visible empty `<think>` tags remained in `content`; quality provisional but speed reject is clear | 0.8600 | 0.9000 | 0.6200 | 0.7368 | 147.2m |

Interpretation:

- Correctly served Qwen3.6 35B-A3B is a serious local candidate: it matches MLX 26B OptiQ on IFEval, beats it on GSM8K, and has much better long-context TTFT.
- Qwen3-Coder 30B-A3B is the strongest Qwen math result so far and is faster than Qwen3.6 35B-A3B on short/coding prompts, but its IFEval scores trail both Qwen3.6 35B-A3B and Gemma 4 26B `UD-Q4_K_XL`.
- Against Gemma 4 26B `UD-Q4_K_XL`, Qwen3.6 35B-A3B improves GSM8K and instruction-level strict accuracy while tying prompt-level strict accuracy, but it is slower and uses more memory/swap. Qwen3-Coder improves GSM8K further but gives up IFEval.
- Qwen3.6 27B is dense, not MoE. It was far slower than 35B-A3B and did not improve instruction following, so reject it for practical daily use on the 32 GB Mac.

## Invalid or superseded lm-eval results

These should not be used for model comparison:

| Run | Why invalid / superseded |
|---|---|
| Earlier Ollama 26B runs with `EOS_STRING=<end_of_turn>` | Wrong Gemma 4 stop token and reasoning not disabled; GSM8K was 0.0000. |
| Earlier llama.cpp Q4/Q5/q4-KV runs with `EOS_STRING=<end_of_turn>` | Same issue; measured thinking/output formatting failure more than quality. |
| llama.cpp Q5 first run | Thinking was enabled; model filled `reasoning_content`, produced empty final `content`, and hit length. |
| Ollama 12B Q8 first run without `reasoning_effort="none"` | Runtime was 3h23m; GSM8K 0.0200 and IFEval prompt 0.4400 because it mostly generated reasoning. |
| Qwen3.6 35B-A3B first run with `--reasoning-format none` | Visible empty `<think></think>` tags remained in content; superseded by corrected `--reasoning-format deepseek` rerun. |
| Qwen3.6 27B quality run with visible `<think></think>` tags | Quality may be undercounted, but speed/memory results are enough to reject it for daily use. |
| MLX-VLM DiffusionGemma 26B 4bit full `limit=50` IFEval+GSM8K run | Failed at 28/100 requests after ~5h51m. Three retries hit the MLX-VLM 600s no-token queue timeout, so there are no valid quality metrics; this is a practical 32 GB reject for full lm-eval. |

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

### llama.cpp 12B Unsloth UD-Q4_K_XL, concurrency 1

HTTP load was run with `sweeps/gemma4-12b-unsloth-ud-q4-k-xl-http-load.yaml` against `unsloth/gemma-4-12B-it-GGUF` / `gemma-4-12b-it-UD-Q4_K_XL.gguf`, q8 KV, reasoning off, ctx 8192. Sanity preview was clean: `Hello, how are you today?`.

| Scenario | latency p95 | TTFT p95 | completion tok/s | check pass | errors |
|---|---:|---:|---:|---:|---:|
| interactive-short | 32.074s | 3.851s | 4.066 | n/a | 0 |
| coding-assistant | 52.331s | 3.955s | 4.232 | 1.000 | 0 |
| long-context-recall | 10.644s | 0.212s | 4.150 | 1.000 | 0 |

### llama.cpp 12B Unsloth QAT UD-Q4_K_XL, concurrency 1

HTTP load was run with `sweeps/gemma4-12b-unsloth-qat-ud-q4-k-xl-http-load.yaml` against `unsloth/gemma-4-12B-it-qat-GGUF` / `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`, q8 KV, reasoning off, ctx 8192. Sanity preview was clean: `Hello, I am here now.`.

| Scenario | latency p95 | TTFT p95 | completion tok/s | check pass | errors |
|---|---:|---:|---:|---:|---:|
| interactive-short | 15.024s | 2.575s | 8.771 | n/a | 0 |
| coding-assistant | 24.376s | 2.478s | 8.778 | 1.000 | 0 |
| long-context-recall | 5.766s | 0.123s | 8.794 | 1.000 | 0 |

### llama.cpp Huihui Gemma 4 12B abliterated Q4_K, concurrency 1

HTTP load was run with `sweeps/huihui-gemma4-12b-abliterated-http-load.yaml` against `huihui-ai/Huihui-gemma-4-12B-it-qat-q4_0-unquantized-abliterated-GGUF` / `Huihui-gemma-4-12B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf`, q8 KV, reasoning off, ctx 8192, prompt cache 8192 MiB, and 32 context checkpoints. Sanity preview was clean: `Hello, I am here now.`.

| Scenario | latency p95 | TTFT p95 | completion tok/s | check pass | errors |
|---|---:|---:|---:|---:|---:|
| interactive-short | 17.104s | 2.436s | 7.688 | n/a | 0 |
| coding-assistant | 29.962s | 2.357s | 8.026 | 1.000 | 0 |
| long-context-recall | 6.131s | 0.244s | 7.647 | 1.000 | 0 |

### llama.cpp 12B Unsloth QAT UD-Q4_K_XL + Gemma 4 MTP head

MTP was tested after upgrading the local `llama` binary to GitHub release `b9553`, using `unsloth/gemma-4-12B-it-qat-GGUF` / `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` plus `unsloth/gemma-4-12B-it-GGUF` / `MTP/gemma-4-12B-it-MTP-Q8_0.gguf`, q8_0/q8_0 target KV, q8_0/q8_0 draft KV, ctx 8192, reasoning off, no mmproj, no prompt cache/checkpoints, and `--spec-draft-n-max 2`.

| Test | Predicted tokens | Draft tokens | Accepted draft tokens | Accept rate | Weighted decode tok/s | Wall tok/s | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| 9-prompt acceptance suite | 1350 | 1142 | 772 | 0.676 | 12.48 | 11.72 | Short microbench only; promising but not representative of full suite. |
| coding-assistant prompt x6 | 1260 | 996 | 762 | 0.765 | 10.63 | 10.05 | All 6 contained `def percentile`; mean latency 20.9s. |

Full suite result with the same MTP server and prompt cache/checkpoints disabled:

| Metric | Result |
|---|---:|
| GSM8K strict/flex | 0.8800 / 0.9200 |
| IFEval prompt/inst strict | 0.8400 / 0.8947 |
| Eval time | 342.5m |
| HTTP interactive | 4.051 tok/s, p95 33.004s, TTFT 2.856s |
| HTTP coding | 5.543 tok/s, p95 39.426s, TTFT 2.990s, pass 1.000 |
| HTTP long-context | 5.065 tok/s, p95 12.122s, TTFT 0.118s, pass 1.000 |

MTP is functional and short prompts can show good draft acceptance, but this full-suite run was worse than non-MTP 12B QAT (`52.9m` eval, ~8.8 tok/s HTTP). The likely culprit is a combination of MTP overhead and running the suite with prompt cache/checkpoints disabled. QAT MTP should remain experimental until rerun with prompt cache enabled and compared under the same suite.

### llama.cpp 12B Unsloth Q8_0 + Gemma 4 MTP head

A cache-enabled MTP suite was run with `sweeps/gemma4-12b-unsloth-q8-mtp-llama-sweep.yaml` and `sweeps/gemma4-12b-unsloth-q8-mtp-http-load.yaml` against `unsloth/gemma-4-12B-it-GGUF` / `gemma-4-12b-it-Q8_0.gguf` plus `MTP/gemma-4-12B-it-MTP-Q8_0.gguf`, q8_0/q8_0 target KV, q8_0/q8_0 draft KV, ctx 8192, reasoning off, no mmproj, prompt cache 8192 MiB, 32 context checkpoints, and `--spec-draft-n-max 2`.

| Metric | Result |
|---|---:|
| GSM8K strict/flex | 0.8200 / 0.8400 |
| IFEval prompt/inst strict | 0.8600 / 0.9079 |
| Eval time | 65.9m |
| HTTP interactive | 7.157 tok/s, p95 18.253s, TTFT 2.439s |
| HTTP coding | 8.259 tok/s, p95 23.899s, TTFT 2.237s, pass 1.000 |
| HTTP long-context | 8.383 tok/s, p95 7.490s, TTFT 0.192s, pass 1.000 |

This is much better than the cache-disabled QAT MTP full suite, and faster than the old non-MTP llama.cpp 12B Q8 eval, but it still does not beat Ollama 12B Q8 on instruction following or practical latency. It is useful as an MTP reference run, not a daily default.

### llama.cpp 26B Unsloth UD-Q4_K_XL, concurrency 1

HTTP load was run with `sweeps/gemma4-26b-qat-http-load.yaml` while the server was actually `unsloth/gemma-4-26B-A4B-it-GGUF` / `gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf`, q8 KV, reasoning off, ctx 8192. The stored target label is stale (`llama-gemma4-26b-a4b-qat-q4_0`).

| Scenario | latency p95 | TTFT p95 | completion tok/s | check pass | errors |
|---|---:|---:|---:|---:|---:|
| interactive-short | 7.016s | 0.732s | 19.694 | n/a | 0 |
| coding-assistant | 7.798s | 0.731s | 20.965 | 1.000 | 0 |
| long-context-recall | 2.222s | 0.065s | 19.483 | 1.000 | 0 |

### llama.cpp SuperGemma4 26B uncensored Q4_K_M, concurrency 1

HTTP load was run with `sweeps/supergemma4-26b-uncensored-http-load.yaml` against `Jiunsong/supergemma4-26b-uncensored-gguf-v2` / `supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf`, q8 KV, reasoning off, ctx 8192, prompt cache 8192 MiB, and 32 context checkpoints. Sanity preview was clean: `Hello to you, dear friend.`. During lm-eval, one GSM8K request failed once and retried after the model/server path exposed parser-breaking internal channel tokens (`<|channel>thought` / `<channel|>`).

| Scenario | latency p95 | TTFT p95 | completion tok/s | check pass | errors |
|---|---:|---:|---:|---:|---:|
| interactive-short | 6.900s | 1.539s | 19.208 | n/a | 0 |
| coding-assistant | 8.256s | 1.143s | 19.529 | 1.000 | 0 |
| long-context-recall | 2.550s | 0.104s | 18.796 | 1.000 | 0 |

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

### llama.cpp 31B QAT Q4_0 + Gemma 4 MTP head

After upgrading the local `llama` binary to GitHub release `b9553`, Gemma 4 MTP loaded successfully with `google/gemma-4-31B-it-qat-q4_0-gguf` plus `unsloth/gemma-4-31B-it-GGUF` / `MTP/gemma-4-31B-it-MTP-Q8_0.gguf`. The tested server used q8_0/q8_0 target KV, q8_0/q8_0 draft KV, ctx 8192, reasoning off, no mmproj, and `--spec-type draft-mtp`.

| Config | Predicted tokens | Draft tokens | Accepted draft tokens | Accept rate | Weighted decode tok/s | Wall tok/s | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `--spec-draft-n-max 3`, f16/f16 KV, Q8 MTP head | 765 | 1028 | 417 | 0.406 | 3.32 | 3.07 | Moderate acceptance; not enough speedup. |
| `--spec-draft-n-max 2`, q8_0/q8_0 KV, Q8 MTP head | 1309 | 1318 | 645 | 0.489 | 3.45 | 3.21 | Better acceptance; still near non-MTP speed. |
| `--spec-draft-n-max 2`, q8_0/q8_0 KV, F16 MTP head | 1309 | 1322 | 643 | 0.486 | 3.35 | 3.10 | No improvement over Q8 head; slightly slower in this run. |

MTP is functional on this build, and q8_0 KV does not collapse acceptance to zero, but 31B QAT remains around the old non-MTP speed range on the 32 GB M4. `n_max=2` looks better than `n_max=3`; the F16 MTP head did not improve acceptance or speed over the Q8 MTP head. `n_max=4` and `p_min` sweeps are only worth testing for completeness.

Interpretation:

- Unsloth 12B `UD-Q4_K_XL` is quality-strong but **too slow for daily coding-agent use** in llama.cpp on this machine: coding latency p95 ~52s and completion throughput ~4.2 tok/s, despite modest memory use.
- Unsloth 12B QAT `UD-Q4_K_XL` roughly doubles non-QAT HTTP throughput and cuts lm-eval time by more than half, but it gives up instruction-following quality. Gemma 4 MTP (`n_max=2`, q8_0 KV) looked better in short microbenchmarks, but the QAT full suite regressed badly when run with prompt cache/checkpoints disabled: eval took ~342.5m and HTTP coding throughput dropped to ~5.5 tok/s. The cache-enabled Q8 MTP suite was healthier (65.9m eval, ~8.3 tok/s HTTP coding) but still does not beat Ollama 12B Q8 or the 26B daily default.
- Huihui 12B abliterated `Q4_K` is an interesting behavioral variant, but the baseline run is not a new keeper for coding-agent use: ~8.0 tok/s coding, p95 coding latency ~30s, IFEval 0.860/0.9079, and GSM8K 0.860/0.880.
- Unsloth 26B `UD-Q4_K_XL` is slightly slower in completion tok/s than some 26B baselines, but coding latency is excellent because it produces concise successful answers; long-context TTFT is also excellent.
- SuperGemma4 26B uncensored `Q4_K_M` has good HTTP speed (~19.5 tok/s coding, p95 ~8.3s), but quality/reliability are not competitive: IFEval prompt strict is only 0.7800, GSM8K strict is only 0.7800, and one eval request leaked internal channel tokens that caused a server parse retry.
- 26B QAT Q4_0 has a good latency profile and excellent prompt-cache behavior, but its limited-eval instruction-following score is worse than the older 26B `Q4_K_M`.
- 31B QAT Q4_0 is quality-strong but **too slow for the daily coding-agent default** at ~3.5 completion tok/s and ~4.8s TTFT on short/coding prompts.
- Gemma 4 MTP works with the 31B QAT target on `b9553`, but on the current M4 it has not produced a meaningful practical speedup; `n_max=2` q8_0/q8_0 KV reached ~0.49 draft acceptance and ~3.45 weighted decode tok/s with the Q8 MTP head. The F16 MTP head was not better.
- It is useful as a high-quality local fallback when latency is less important.
- MLX 26B became viable after disabling thinking and passed coding/long-context checks.
- MLX 26B has a **large long-context TTFT penalty**: ~20–25s vs sub-second for llama.cpp/Ollama.
- Direct checks showed MLX was only reporting about `19 / 5915` prompt tokens cached on repeated long-context requests, so it appears to re-prefill almost the whole prompt. llama.cpp/Ollama appear to benefit from much stronger prompt/prefix cache reuse.

### llama.cpp Qwen3.6 35B-A3B `UD-IQ4_NL`, concurrency 1

HTTP load was run with `sweeps/qwen36-http-load.yaml` against `unsloth/Qwen3.6-35B-A3B-GGUF` / `Qwen3.6-35B-A3B-UD-IQ4_NL.gguf`, q8 KV, ctx 8192, `--reasoning off --reasoning-format deepseek`. Sanity preview was clean: `Hello, how are you today?`.

| Scenario | latency p95 | TTFT p95 | completion tok/s | check pass | errors |
|---|---:|---:|---:|---:|---:|
| interactive-short | 7.382s | 1.054s | 18.079 | n/a | 0 |
| coding-assistant | 9.556s | 0.989s | 18.540 | 1.000 | 0 |
| long-context-recall | 2.889s | 0.198s | 17.218 | 1.000 | 0 |

Memory pressure was high for this 32 GB machine after the run: swap climbed from ~4.48 GB before to ~6.65 GB after, with only ~517 MB free swap. That makes 35B-A3B a strong quality candidate but not obviously comfortable as an always-on default on the current Mac.

### llama.cpp Qwen3-Coder 30B-A3B `Q4_K_M`, concurrency 1

HTTP load was run with `sweeps/qwen3-coder-http-load.yaml` against `lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF` / `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`, q8 KV, ctx 8192, `--reasoning off --reasoning-format deepseek`. Sanity preview was clean: `Hello there everyone!`.

| Scenario | latency p95 | TTFT p95 | completion tok/s | check pass | errors |
|---|---:|---:|---:|---:|---:|
| interactive-short | 5.821s | 0.092s | 24.018 | n/a | 0 |
| coding-assistant | 11.409s | 0.140s | 20.718 | 1.000 | 0 |
| long-context-recall | 3.932s | 0.140s | 13.345 | 1.000 | 0 |

Verdict: fast and math-strong, but weaker instruction following than Qwen3.6 35B-A3B and Gemma 4 26B `UD-Q4_K_XL`. Needs the agentic coding harness before deciding whether coding-specific behavior offsets lower IFEval.

### llama.cpp Qwen3.6 27B `IQ4_NL`, concurrency 1

HTTP load was run with `sweeps/qwen36-27b-http-load.yaml` against `unsloth/Qwen3.6-27B-GGUF` / `Qwen3.6-27B-IQ4_NL.gguf`, q8 KV, ctx 8192. This is a dense model and was much slower than the 35B-A3B MoE.

| Scenario | latency p95 | TTFT p95 | completion tok/s | check pass | errors |
|---|---:|---:|---:|---:|---:|
| interactive-short | 41.694s | 3.376s | 3.104 | n/a | 0 |
| coding-assistant | 56.477s | 3.508s | 3.170 | 1.000 | 0 |
| long-context-recall | 14.148s | 0.924s | 3.249 | 1.000 | 0 |

Verdict: reject for daily use on the 32 GB Mac. It is ~5x slower than Qwen3.6 35B-A3B in HTTP coding throughput and does not improve IFEval.

## GeoAnalystBench agent/data baseline

A first GeoAnalystBench smoke baseline was run with `benchmarks/geoanalystbench-ollama-gemma4-12b-q8.yaml` against Ollama `hf.co/ggml-org/gemma-4-12B-it-GGUF:Q8_0`, `reasoning_effort: none`, five open-source tasks, and both `workflow` and `code` response types (`10` total requests). This is an automatic harness check, not a judged workflow-similarity score.

| Model / runtime | Tasks / requests | success rate | latency p50 | latency p95 | workflow step abs err avg | code syntax pass | completion tok/s | duration | memory note |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Ollama Gemma 4 12B Q8 | 5 / 10 | 1.000 | 70.419s | 160.988s | 2.200 | 0.800 | 6.435 | 921s | Swap did not grow: ~1.82 GB before, ~1.81 GB after; Ollama reported ~13 GB GPU allocation with 32k context. |

One code-generation request failed Python syntax parsing: task `3` produced an unmatched `)` in the generated burn-scar analysis code. Overall this is a useful low-swap baseline, but latency is high enough that future GeoAnalystBench comparisons should include Gemma 4 26B `UD-Q4_K_XL` and Qwen3-Coder 30B for practical agent behavior.

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
| llama.cpp 12B Unsloth `UD-Q4_K_XL` + q8 KV + 8k ctx | Very comfortable footprint. Clean sanity had no swap and llama RSS ~7.5 GB; after full eval + HTTP load, llama RSS was ~8.9 GB and swap was only ~313 MB. Runtime, not memory, is the blocker. |
| llama.cpp 12B Unsloth QAT `UD-Q4_K_XL` + q8 KV + 8k ctx | Very comfortable. Started with llama RSS ~7.1 GB and ~297 MB swap; after eval + HTTP load, llama RSS was ~9.2 GB and swap was ~281 MB. |
| llama.cpp 12B Unsloth Q8_0 + MTP Q8 head + q8 KV + 8k ctx | Fits, but heavier than non-MTP 12B. Cache-enabled run started with llama RSS ~13.2 GB and ~0.94 GB swap, ended with llama RSS ~15.2 GB and ~2.52 GB swap. |
| llama.cpp Huihui 12B abliterated `Q4_K` + q8 KV + 8k ctx | Fits. Run started with pre-existing ~2.06 GB swap and llama RSS ~7.6 GB; after eval + HTTP load llama RSS grew to ~16.6 GB with ~2.0 GB swap. Prompt cache/checkpoints increased footprint over the long run. |
| llama.cpp 26B Google QAT `Q4_0` + q8 KV + 8k ctx | Comfortable enough. During eval: green pressure, ~24.6 / 32 GB used, ~16.8 GB wired, ~1.0 GB compressed, ~2.9 GB swap. After HTTP load, llama RSS grew to ~16.8 GB and compressed memory rose. |
| llama.cpp 26B Unsloth `UD-Q4_K_XL` + q8 KV + 8k ctx | Similar model/RSS footprint to Google 26B QAT. During eval/HTTP reruns, pressure stayed acceptable with llama RSS around ~14.3–17.2 GB. Swap remained high after a long benchmarking session, so restart/quit other apps for clean daily-use memory readings. |
| llama.cpp SuperGemma4 26B uncensored `Q4_K_M` + q8 KV + 8k ctx | Fits but pressured the 32 GB machine. Run started with ~1.59 GB swap; after eval + HTTP load swap grew to ~4.35 GB. Process snapshot did not capture the llama server at the end, so use system swap/compressor rather than per-process RSS for this run. |
| Ollama 26B QAT `Q4_0` LM Studio mirror | Yellow pressure near end of eval. Ollama backend used 32k ctx + mmproj; ~29.9 / 32 GB used, ~18.5 GB wired, ~7.3 GB compressed, ~5.7 GB swap. |
| MLX 26B QAT `mxfp4` | Green and comfortable during eval; ~21.6 / 32 GB used, ~15.7 GB wired, ~0.2 GB compressed, ~4.7 GB swap. |
| llama.cpp 31B Google QAT `Q4_0` + q8 KV + 8k ctx | Fits, but tight. During/after eval: green pressure, ~30 / 32 GB used, ~20.5 GB wired, ~3 GB compressed, ~3.3 GB swap. Not ideal as an always-on default. |
| llama.cpp 31B Google QAT `Q4_0` + MTP Q8 head + q8 KV + 8k ctx | Fits, but very tight. Activity Monitor showed green pressure but ~30.1 / 32 GB used, ~20.8 GB wired, ~4.3 GB compressed, ~1.6 GB swap. Per-process “App Memory” showed llama at only ~1.65 GB, so watch system wired/compressed/swap rather than process RSS for Metal allocations. |
| llama.cpp Qwen3.6 35B-A3B `UD-IQ4_NL` + q8 KV + 8k ctx | Fits but swap-heavy on the 32 GB Mac. Corrected eval + HTTP load ended at ~6.65 GB swap used with only ~0.52 GB free swap. Strong quality, but not ideal as an always-on default. |
| llama.cpp Qwen3-Coder 30B-A3B `Q4_K_M` + q8 KV + 8k ctx | Fits similarly to other 30B-class MoEs. Swap was already high before the run (~3.88 GB) and ended around ~4.18 GB after eval + HTTP load. Less swap growth than Qwen3.6 35B-A3B in this session. |
| llama.cpp Qwen3.6 27B `IQ4_NL` + q8 KV + 8k ctx | Fits but not comfortable after a long eval. Swap was ~3.6 GB before and ~5.9 GB after; generation speed was too slow to justify the footprint. |

## Local cache cleanup

After 12B triage, rejected/soft-rejected local 12B caches were removed to save disk:

- Deleted HF caches: `mlx-community/gemma-4-12B-it-8bit`, `unsloth/gemma-4-12B-it-GGUF`.
- Already absent locally: `mlx-community/gemma-4-12B-it-4bit`, `ggml-org/gemma-4-12B-it-GGUF` HF cache.
- Ollama already had only the kept `hf.co/ggml-org/gemma-4-12B-it-GGUF:Q8_0` model installed.
- Kept HF cache: `unsloth/gemma-4-12B-it-qat-GGUF`.

Current HF model cache after cleanup:

```text
6.4G  models--unsloth--gemma-4-12B-it-qat-GGUF
17G   models--lmstudio-community--Qwen3-Coder-30B-A3B-Instruct-GGUF
17G   models--unsloth--gemma-4-26B-A4B-it-GGUF
18G   models--google--gemma-4-31B-it-qat-q4_0-gguf
18G   models--unsloth--Qwen3.6-35B-A3B-GGUF
```

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
uv run llm-refinery lm-eval llama_cpp 50 \
  --eos-string '<turn|>' \
  --max-length 8192

uv run llm-refinery suite sweeps/qwen36-35b-llama-sweep.yaml \
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

### Qwen3.6 35B-A3B quality candidate

Terminal 1:

```bash
kill $(lsof -tiTCP:8080 -sTCP:LISTEN) 2>/dev/null || true
uv run llm-refinery server sweeps/qwen36-35b-llama-sweep.yaml --index 0
```

Terminal 2:

```bash
uv run llm-refinery suite sweeps/qwen36-35b-llama-sweep.yaml \
  --http-load-config sweeps/qwen36-http-load.yaml \
  --target llama-qwen36-35b-a3b-ud-iq4-nl
```

### Qwen3-Coder 30B-A3B Q4_K_M candidate

Terminal 1:

```bash
kill $(lsof -tiTCP:8080 -sTCP:LISTEN) 2>/dev/null || true
uv run llm-refinery server sweeps/qwen3-coder-30b-llama-sweep.yaml --index 0
```

Terminal 2:

```bash
uv run llm-refinery suite sweeps/qwen3-coder-30b-llama-sweep.yaml \
  --http-load-config sweeps/qwen3-coder-http-load.yaml \
  --target llama-qwen3-coder-30b-a3b-q4km
```

### Ollama Gemma 4 12B Q8 quality eval

```bash
uv run llm-refinery lm-eval ollama 50 \
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

1. Add a lightweight deterministic agentic eval harness: patch applies, pytest repair, JSON/tool validity, multi-file edit, long-context repo task, retry behavior.
2. Use the agentic harness to decide between Gemma 4 26B `UD-Q4_K_XL`, Qwen3.6 35B-A3B `UD-IQ4_NL`, Qwen3-Coder 30B-A3B `Q4_K_M`, Ollama 12B Q8, MLX 26B OptiQ, and llama.cpp 31B QAT.
3. Add a dedicated HTTP target/sweep for Gemma 4 26B `UD-Q4_K_XL` so future results are not stored under stale QAT labels.
4. DiffusionGemma 26B MLX 4bit full lm-eval is already a practical reject on the 32 GB Mac due repeated 600s no-token timeouts. Use only the updated smoke defaults / HTTP load on this machine; defer full 4bit/5bit/6bit/8bit/mxfp8/bf16 evals to the 128 GB M5 Max.
5. On the 128 GB M5 Max, rerun the top candidates for clean memory/speed numbers and try higher-quality Qwen/Gemma quants.
6. Run a real multimodal smoke test for Ollama 12B Q8 with an image.
7. Capture `prompt_tokens_details.cached_tokens` in `http-load` so prompt-cache behavior is visible in DB comparisons.
8. On the 128 GB M5 Max, try DS4 / DeepSeek V4 Flash q2-imatrix as the high-memory local frontier-ish candidate.
