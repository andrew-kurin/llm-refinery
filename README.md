# llm-refinery

Small, local-first experiment harness for refining local LLM serving choices across llama.cpp, Ollama, MLX, and other OpenAI-compatible endpoints.

It gives you a `llm-refinery` command that can:

- expand YAML sweep files into concrete `llama bench` / `llama server` commands
- run llama.cpp benchmark trials with repetitions and token dimensions
- run lm-eval quality checks against local chat-completions endpoints
- run HTTP latency/TTFT/load checks against llama.cpp, Ollama, MLX, and similar servers
- run agent/data benchmarks such as GeoAnalystBench against OpenAI-compatible endpoints
- supervise and resume the official external-process DABStep baseline
- store commands, stdout/stderr artifacts, params, parsed metrics, and structured host metadata in DuckDB
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

For cross-host service comparisons, start with the 100-request shared-prefix and
cache-busted prompt-pool cells in
[`sweeps/local-http-load-recommended.yaml`](sweeps/local-http-load-recommended.yaml).
The runner warms every concurrency slot and warns when a cell is too small for useful
tail inspection. It records all-request latency (including failures), visible and
reasoning TTFT, TPOT, approximate streaming-event ITL, and explicit correctness failures.

Compare HTTP load results by latency, TTFT, throughput, and optionally host metadata:

```bash
uv run llm-refinery compare results/llm_refinery.duckdb \
  --metric observed_latency_p95_s \
  --metric visible_ttft_p95_s \
  --metric tpot_p95_s \
  --metric completion_tokens_per_second \
  --sort observed_latency_p95_s \
  --ascending \
  --param target \
  --param protocol \
  --param scenario \
  --param concurrency \
  --param system.hardware.model \
  --param system.hardware.memory_gb
```

If a benchmark parser improved, refresh stored metrics from typed artifacts:

```bash
uv run llm-refinery reparse results/llm_refinery.duckdb
```

Reparsing dispatches by `benchmark_kind`; llama-bench, lm-eval, HTTP-load,
agent-eval, and DABStep artifacts each use their own parser. Empty parser results
are preserved unless `--force` is supplied.

Run a broader lm-eval reasoning/knowledge scoreboard, including the fixed GPQA task override:

```bash
uv run llm-refinery lm-eval ollama all \
  --suite-name quality-reasoning \
  --tasks gsm8k,arc_challenge_chat,ifeval,truthfulqa_gen,gpqa_main_fixed_generative \
  --include-path evals/lm_eval_tasks
```

Parsed lm-eval aggregate metrics are stored in DuckDB and can be compared with `llm-refinery compare`. Arbitrary OpenAI-compatible targets are accepted with a custom name plus `--model` and `--base-url`; use `--api-key-env` for authenticated endpoints. Pin `--package-spec` when exact lm-eval reproducibility matters. See [`evals/README.md`](evals/README.md).

For model selection, use the reproducible local tiers rather than drawing conclusions
from a 50-item smoke run:

```bash
# Fast pipeline validation only
uv run llm-refinery suite sweeps/local-quality-smoke-suite.yaml

# Complete IFEval + IFBench + GPQA Diamond + generative MuSR
uv run llm-refinery suite sweeps/local-quality-core-suite.yaml

# Overnight/release tier, adding all MMLU-Pro domains
uv run llm-refinery suite sweeps/local-quality-expanded-suite.yaml
```

Suite quality runs retain lm-eval's item-level JSONL artifacts and normalized sample
rows in DuckDB. Aggregate standard errors are expanded to 95% intervals, and retained
sample correctness includes Wilson intervals for auditability and paired follow-up analysis.

Compare two quality runs on their exact shared items (including flip counts and an
exact McNemar test):

```bash
uv run llm-refinery quality-compare results/llm_refinery.duckdb \
  <baseline-run-id> <candidate-run-id> --task ifeval_pinned
```

`correct` is normalized to each task's primary binary score (IFBench uses loose prompt
accuracy; IFEval uses strict prompt accuracy). Use `--sample-metric` to compare another
retained binary item metric explicitly.

Run a GeoAnalystBench smoke benchmark against an already-running OpenAI-compatible server (see [`docs/geoanalystbench.md`](docs/geoanalystbench.md)):

```bash
uv run llm-refinery agent-eval benchmarks/geoanalystbench-smoke.yaml --dry-run
uv run llm-refinery agent-eval benchmarks/geoanalystbench-smoke.yaml --limit 5
```

Compare GeoAnalystBench runs:

```bash
uv run llm-refinery compare results/llm_refinery.duckdb \
  --suite geoanalystbench-smoke \
  --metric response_availability_rate \
  --metric workflow_step_abs_error_avg \
  --metric code_contract_pass_rate \
  --metric code_reference_call_recall_avg \
  --sort code_contract_pass_rate \
  --param target \
  --param system.hardware.memory_gb
```

Run the official DABStep baseline from a prepared Hugging Face Space checkout:

```bash
uv run llm-refinery dabstep benchmarks/dabstep-smoke.yaml --dry-run
uv run llm-refinery dabstep benchmarks/dabstep-smoke.yaml
uv run llm-refinery dabstep benchmarks/dabstep-smoke.yaml --resume <failed-run-id>
```

DABStep answers are checkpointed as task samples and a canonical leaderboard-ready
`answers.jsonl`. See [`docs/dabstep.md`](docs/dabstep.md) for upstream setup, scoring,
and resume behavior.

If old runs predate structured host metadata, backfill them by assuming they ran on the current machine:

```bash
uv run llm-refinery backfill-system-metadata results/llm_refinery.duckdb
```

Launch the server for one expanded config:

```bash
uv run llm-refinery server sweeps/gemma-cache-sweep.yaml --index 0
```

## Configuration models

Configuration is strict: unknown YAML fields fail instead of being silently ignored.
Each workflow has its own manifest shape. Existing users should read the
[architecture cutover migration guide](docs/migration.md).

### Llama sweep configuration

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
- For non-llama servers that use a different model-source flag, set `server.model_flag`. Example: `mlx_vlm.server` uses `--model` instead of llama.cpp `-hf` / `-m`.
- `bench.omit_params` / `server.omit_params` remove shared flags for one command type.
- Snake-case keys become llama.cpp kebab-case flags. Example: `ctx_size` -> `--ctx-size`.
- Boolean `true` values become flags. Boolean `false` values are omitted.
- Bench, lm-eval, HTTP-load, agent-eval, DABStep, and suite runs record structured host metadata in `runs.system_json` for cross-machine history: macOS version, hardware model, chip/CPU fields when available, memory size, Python path/version, project version, and git head/dirty state. `llm-refinery compare --param system.hardware.model --param system.hardware.memory_gb` can display it.
- Linux/DGX profiles also capture OS/DMI, NVIDIA GPU/driver, CUDA runtime/toolkit, and
  DGX release metadata best-effort. A hashed machine fingerprint keeps identical configs
  from a Mac and DGX Spark as distinct comparison rows while collapsing reruns only on
  the same host.
- Server params support an `mtp_head` helper for Gemma/Qwen MTP draft heads. It expands to `--model-draft <path>` and, in `llm-refinery server`, auto-downloads when `hf` + `file` or `url` is provided:

  ```yaml
  server:
    params:
      spec_type: draft-mtp
      spec_draft_n_max: 2
      mtp_head:
        hf: unsloth/gemma-4-12B-it-GGUF
        file: MTP/gemma-4-12B-it-MTP-Q8_0.gguf
  ```

  The default download location is `~/.local/share/llm-refinery/mtp/<filename>`. Set `path:` to override it.

### Endpoint configuration

HTTP-load, agent-eval, DABStep, and suite manifests use a shared endpoint shape.
`protocol` describes the wire protocol rather than the server vendor:

```yaml
endpoint:
  name: local
  protocol: openai_chat
  base_url: http://127.0.0.1:8080/v1
  model: local-model
```

HTTP-load also supports `ollama_chat`. Cerebras and other OpenAI-compatible vendors
use `openai_chat` with `api_key_env` when needed.

### Suite configuration

Suite manifests are separate from llama sweep manifests. See
[`sweeps/gemma4-31b-suite.yaml`](sweeps/gemma4-31b-suite.yaml). They contain an
`endpoint`, `quality`, optional `http_load`, and `preflight` section. Referenced
HTTP-load paths resolve relative to the suite manifest.

Successful suite preflight responses are retained as `preflight.json`, including the
model identifier returned by the endpoint. When an endpoint exposes a stable model id,
set `preflight.expected_response_model` to make an accidental model swap fail closed.

## Suggested workflow

1. Start with `llm-refinery plan` to verify exact llama.cpp commands.
2. Run a small `--limit` first for low-level benchmark sweeps.
3. Launch candidates with `llm-refinery server` or an external Ollama/MLX server.
4. Run `llm-refinery suite` with a suite manifest for recorded lm-eval + HTTP-load checks. Use `endpoint.model` in YAML or `--api-model` when the endpoint requires the real model id.
5. Use `llm-refinery agent-eval` for direct chat benchmarks like GeoAnalystBench.
6. Use `llm-refinery dabstep` for the external multi-step DABStep agent baseline.
7. Compare parsed metrics with `llm-refinery compare`.

This scaffold intentionally avoids Make. YAML is the source of truth, and Python handles expansion, execution, parsing, and storage. See [`docs/architecture.md`](docs/architecture.md) for module boundaries and extension points.
