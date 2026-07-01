# DABStep support notes

DABStep is a multi-step data-agent benchmark, so it does not fit the current
`llama bench` / `lm-eval` / HTTP-load shape exactly. The current harness is still
useful for serving and endpoint health, but needs a first-class external/agent
benchmark path before full DABStep runs are reproducible.

## Current state

- `bench` stores llama-bench trials, stdout/stderr artifacts, parsed metrics, and
  structured host metadata in DuckDB.
- `http-load` stores OpenAI/Ollama-compatible latency, TTFT, throughput, response
  artifacts, parsed metrics, and structured host metadata in DuckDB.
- `agent-eval` now provides a generic agent/data benchmark entry point backed by
  adapter modules under `llm_refinery.benchmarks.agent`. The initial
  `geoanalystbench` adapter stores per-request artifacts, aggregate metrics, and
  structured host metadata in DuckDB.
- `suite` orchestrates preflight, sanity, `lm-eval`, HTTP load, and comparison.
- `lm-eval` is invoked through Python and now records aggregate metrics in DuckDB
  while preserving the original `results/lm_eval/...` JSON artifact path.
- `suite` prints pre/post memory snapshots, but the structured machine profile is
  recorded only by bench/http-load/agent-eval runs. Full suite-level pre/post
  snapshots are not yet stored as artifacts.

## Gaps for DABStep

1. **Benchmark abstraction**: DABStep should be a first-class benchmark type, not
   squeezed through HTTP load. It needs task-level status, score, tool/runtime
   errors, retries, and per-task artifacts.
2. **Result ingestion**: parse DABStep JSON/CSV outputs into DuckDB metrics such
   as success rate, average score, timeout rate, task count, tool error rate,
   wall time, tokens, and cost-like counters when available.
3. **Task manifest/subsets**: record exact task ids, split, seed, limit, and any
   excluded tasks so smoke runs and full runs are comparable.
4. **Agent/runtime config**: record the agent command, Python environment,
   package lock/runtime, data directory, sandbox directory, per-task timeout,
   max steps, retries, and concurrency.
5. **Endpoint config**: record OpenAI-compatible base URL, model id, generation
   kwargs, reasoning/thinking disabling, max tokens, and timeout settings.
6. **Resume/keep-going**: long local runs need per-task checkpointing and resume;
   a single timed-out task should not discard a multi-hour run.
7. **Suite artifacts**: store pre/post system snapshots and benchmark stdout/stderr
   under a suite run directory, not only terminal scrollback.
8. **Cross-machine comparison**: include host metadata in all run records and make
   it visible in comparison tables.

## Proposed shape

Add a generic `external-eval` or `agent-eval` command first, then specialize it
with a DABStep preset:

```yaml
name: dabstep-smoke

database: results/llm_refinery.duckdb

agent_eval:
  kind: dabstep
  command:
    - uvx
    - --from
    - dabstep
    - dabstep
    - run
  output_dir: results/dabstep
  task_ids: [example-task-1, example-task-2]
  limit: 10
  seed: 42
  timeout_s: 900
  retries: 1
  concurrency: 1
  keep_going: true
  env:
    OPENAI_BASE_URL: http://127.0.0.1:8080/v1
    OPENAI_API_KEY: local
    OPENAI_MODEL: local-model
  gen_kwargs:
    temperature: 0
    max_tokens: 2048
    reasoning_effort: none
```

Store one DuckDB run for the overall DABStep invocation plus optional task-level
rows/artifacts when available. Minimum metrics to normalize:

- `task_count`
- `success_count`
- `success_rate`
- `timeout_count`
- `error_count`
- `avg_score`
- `avg_steps`
- `avg_latency_s`
- `wall_duration_s`

## Recommended execution plan

1. Extend `agent-eval` with a DABStep adapter once the output schema is confirmed
   locally.
2. Add structured suite-level system snapshot artifacts.
3. Split the large CLI and HTTP-load modules along the same adapter/lifecycle
   boundaries.
4. Run a 5-10 task DABStep smoke on the 32 GB Mac with Gemma 26B `UD-Q4_K_XL` and
   Ollama Gemma 12B Q8.
5. Run larger/full DABStep comparisons on the 128 GB M5 Max.
