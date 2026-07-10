# DABStep implementation notes

DABStep is an external, multi-step data-agent benchmark. It should be a top-level
benchmark slice rather than an `agent-eval` adapter, because the current agent
interface deliberately models direct chat requests and scoring.

## Infrastructure now available

The harness now provides the common pieces DABStep needs:

- complete `RunSpec` fingerprints and explicit `benchmark_kind`
- durable `RunSession` failure recording
- typed artifacts
- parent suite/child run links
- a task-level DuckDB `samples` table
- benchmark-specific artifact reparsing
- strict manifests and shared endpoint configuration
- suite-level before/after system snapshot artifacts

## Remaining DABStep work

1. Add `llm_refinery.benchmarks.dabstep` with config, command construction,
   execution, parsing, and metrics modules.
2. Confirm the upstream JSON/CSV output schema and define stable task IDs.
3. Write one `samples` row after every task, including status, score, tool/runtime
   errors, retries, steps, latency, and artifact path.
4. Add `--resume RUN_ID` semantics that skip completed sample IDs and continue the
   same run.
5. Normalize aggregate metrics such as success rate, timeout/error counts, average
   score, steps, latency, tokens, and wall duration.
6. Register the DABStep reparser in `benchmarks/registry.py`.
7. Add smoke and interruption/resume integration tests.

## Proposed manifest

```yaml
name: dabstep-smoke
database: results/llm_refinery.duckdb

endpoint:
  name: local
  protocol: openai_chat
  base_url: http://127.0.0.1:8080/v1
  model: local-model

dabstep:
  command: [uvx, --from, dabstep, dabstep, run]
  output_dir: results/dabstep
  task_ids: [example-task-1, example-task-2]
  limit: 10
  seed: 42
  timeout_s: 900
  retries: 1
  concurrency: 1
  keep_going: true
  max_steps: 30
  generation:
    temperature: 0
    max_tokens: 2048
    reasoning_effort: none
```

Do not store endpoint secrets in `config_json`; record only the API-key environment
variable name and sanitized endpoint identity.
