# llm-refinery

Small, local-first experiment harness for refining local LLM serving choices across llama.cpp, Ollama, MLX, and other OpenAI-compatible endpoints.

It gives you a `llm-refinery` command that can:

- expand YAML sweep files into concrete `llama bench` / `llama server` commands
- run llama.cpp benchmark trials with repetitions and token dimensions
- run lm-eval quality checks against local chat-completions endpoints
- run HTTP latency/TTFT/load checks against llama.cpp, Ollama, MLX, and similar servers
- store commands, stdout/stderr artifacts, params, and parsed metrics in DuckDB
- compare local model/server candidates in one workflow

## Install

```bash
uv sync
```

Then use:

```bash
uv run llm-refinery --help
```

Or install editable:

```bash
uv pip install -e .
llm-refinery --help
```

## Quick start

Plan the commands without running anything:

```bash
uv run llm-refinery plan sweeps/gemma-cache-sweep.yaml
```

Run the first few benchmark trials:

```bash
uv run llm-refinery bench sweeps/gemma-cache-sweep.yaml --limit 3
```

Benchmark runs show a Rich progress bar with current-trial elapsed time,
suite elapsed time, average trial time, and ETA once at least one trial has completed.
Tune or disable it with:

```bash
uv run llm-refinery bench sweeps/gemma-cache-sweep.yaml --progress-interval 1
uv run llm-refinery bench sweeps/gemma-cache-sweep.yaml --no-progress
```

Show recent runs:

```bash
uv run llm-refinery report results/llm_refinery.duckdb
```

Compare configs by prompt-processing and generation throughput:

```bash
uv run llm-refinery compare results/llm_refinery.duckdb
uv run llm-refinery compare results/llm_refinery.duckdb --prompt-tokens 512 --gen-tokens 128
```

Show top runs by a parsed metric:

```bash
uv run llm-refinery report results/llm_refinery.duckdb --metric tg128.tokens_per_second
```

Run OpenAI-compatible / Ollama HTTP load evals against already-running servers:

```bash
uv run llm-refinery http-load sweeps/gemma-http-load-ollama-compare.yaml --dry-run
uv run llm-refinery http-load sweeps/gemma-http-load-ollama-compare.yaml --target llama-f16-kv
uv run llm-refinery http-load sweeps/gemma-http-load-ollama-compare.yaml --target ollama-gemma
```

Compare HTTP load results by latency, TTFT, and throughput:

```bash
uv run llm-refinery compare results/llm_refinery.duckdb \
  --metric latency_p95_s \
  --metric ttft_p95_s \
  --metric completion_tokens_per_second \
  --sort latency_p95_s \
  --ascending \
  --param target \
  --param provider \
  --param scenario \
  --param concurrency
```

If the parser improved after old runs, refresh stored metrics from artifacts:

```bash
uv run llm-refinery reparse results/llm_refinery.duckdb
```

Launch the server for one expanded config:

```bash
uv run llm-refinery server sweeps/gemma-cache-sweep.yaml --index 0
```

## Config model

See [`sweeps/gemma-cache-sweep.yaml`](sweeps/gemma-cache-sweep.yaml).

Important notes:

- The default command style is the unified llama.cpp CLI: `llama bench` and `llama server`.
- If your install uses separate binaries, set:

  ```yaml
  commands:
    bench: ["llama-bench"]
    server: ["llama-server"]
  ```

- `defaults` are base llama.cpp flags shared by bench/server. Keep this to flags both commands accept.
- `sweep` is a cartesian product over shared values.
- Model-level `params` override `defaults` before `sweep` values are applied.
- `bench.params` / `server.params` add command-specific flags. This matters because `llama bench` does not accept server-only flags like `--ctx-size`, `--mlock`, `--perf`, `--parallel`, `--flash-attn auto`, or `--n-gpu-layers all`.
- `bench.omit_params` / `server.omit_params` remove shared flags for one command type.
- Snake-case keys become llama.cpp kebab-case flags. Example: `ctx_size` -> `--ctx-size`.
- Boolean `true` values become flags. Boolean `false` values are omitted.

## Suggested workflow

1. Start with `llm-refinery plan` to verify exact llama.cpp commands.
2. Run a small `--limit` first for low-level benchmark sweeps.
3. Launch candidates with `llm-refinery server` or an external Ollama/MLX server.
4. Run `llm-refinery suite` for lm-eval + HTTP load checks.
5. Compare parsed metrics with `llm-refinery compare`.

This scaffold intentionally avoids Make. YAML is the source of truth, and Python handles expansion, execution, parsing, and storage.
