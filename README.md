# llama-cpp-tune

Small, local-first experiment harness for tuning `llama.cpp` models and runtime flags.

It gives you a `llama-tune` command that can:

- expand YAML sweep files into concrete `llama bench` / `llama-bench` commands
- run `llama-bench` trials with repetitions and token dimensions
- store commands, stdout/stderr artifacts, params, and parsed metrics in DuckDB
- print a quick local report
- launch a `llama server` command from the same sweep config

## Install

```bash
uv sync
```

Then use:

```bash
uv run llama-tune --help
```

Or install editable:

```bash
uv pip install -e .
llama-tune --help
```

## Quick start

Plan the commands without running anything:

```bash
uv run llama-tune plan sweeps/gemma-cache-sweep.yaml
```

Run the first few benchmark trials:

```bash
uv run llama-tune bench sweeps/gemma-cache-sweep.yaml --limit 3
```

Benchmark runs show a Rich progress bar with current-trial elapsed time,
suite elapsed time, average trial time, and ETA once at least one trial has completed.
Tune or disable it with:

```bash
uv run llama-tune bench sweeps/gemma-cache-sweep.yaml --progress-interval 1
uv run llama-tune bench sweeps/gemma-cache-sweep.yaml --no-progress
```

Show recent runs:

```bash
uv run llama-tune report results/llama_tune.duckdb
```

Compare configs by prompt-processing and generation throughput:

```bash
uv run llama-tune compare results/llama_tune.duckdb
uv run llama-tune compare results/llama_tune.duckdb --prompt-tokens 512 --gen-tokens 128
```

Show top runs by a parsed metric:

```bash
uv run llama-tune report results/llama_tune.duckdb --metric tg128.tokens_per_second
```

Run OpenAI-compatible / Ollama HTTP load evals against already-running servers:

```bash
uv run llama-tune http-load sweeps/gemma-http-load-ollama-compare.yaml --dry-run
uv run llama-tune http-load sweeps/gemma-http-load-ollama-compare.yaml --target llama-f16-kv
uv run llama-tune http-load sweeps/gemma-http-load-ollama-compare.yaml --target ollama-gemma
```

Compare HTTP load results by latency, TTFT, and throughput:

```bash
uv run llama-tune compare results/llama_tune.duckdb \
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
uv run llama-tune reparse results/llama_tune.duckdb
```

Launch the server for one expanded config:

```bash
uv run llama-tune server sweeps/gemma-cache-sweep.yaml --index 0
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

1. Start with `llama-tune plan` to verify the exact commands.
2. Run a small `--limit` first.
3. Compare parsed metrics with `llama-tune report`.
4. Run final candidates under `llama-tune server` and measure real HTTP latency/load separately.

This scaffold intentionally avoids Make. The YAML is the source of truth, and Python handles expansion, execution, parsing, and storage.
