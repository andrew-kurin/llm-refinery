# GeoAnalystBench support

`llm-refinery agent-eval` supports an initial GeoAnalystBench adapter for local
OpenAI-compatible model endpoints.

GeoAnalystBench source: <https://github.com/GeoDS/GeoAnalystBench>

## What is measured

The adapter reads `dataset/GeoAnalystBench.csv`, selects tasks, builds workflow
and/or code-generation prompts, calls the configured endpoint, and stores
per-request JSONL artifacts, checkpoint-friendly rows in the DuckDB `samples` table,
and aggregate metrics.

Current normalized metrics include:

- `request_count`
- `response_count`
- `response_availability_rate` (transport succeeded and non-empty content was returned)
- `error_count`
- latency percentiles
- response character statistics
- token totals/throughput when the endpoint reports usage
- `workflow_step_count_avg`
- `workflow_step_abs_error_avg`
- `code_syntax_valid_rate`
- `code_model_function_rate`
- `code_contract_pass_rate` (valid Python with a top-level `model()` function)
- `code_reference_import_recall_avg` and `code_reference_call_recall_avg`

These are intentionally structural diagnostics, not correctness scores. Response
availability says nothing about task quality. The code contract does not execute the
program, and reference import/call recall only measures overlap with one human reference;
valid alternative solutions may have low overlap. These checks do not reproduce the
paper's human/judge workflow similarity scoring or validate GIS outputs.

## Smoke run

Start a local server first, then run:

```bash
uv run llm-refinery agent-eval benchmarks/geoanalystbench-smoke.yaml --dry-run
uv run llm-refinery agent-eval benchmarks/geoanalystbench-smoke.yaml --limit 5
```

Compare results:

```bash
uv run llm-refinery compare results/llm_refinery.duckdb \
  --suite geoanalystbench-smoke \
  --metric response_availability_rate \
  --metric workflow_step_abs_error_avg \
  --metric code_contract_pass_rate \
  --metric code_reference_call_recall_avg \
  --sort code_contract_pass_rate \
  --param target \
  --param response_types \
  --param system.hardware.model \
  --param system.hardware.memory_gb
```

## YAML shape

```yaml
name: geoanalystbench-smoke

database: results/llm_refinery.duckdb

benchmark:
  kind: geoanalystbench
  dataset: https://raw.githubusercontent.com/GeoDS/GeoAnalystBench/master/dataset/GeoAnalystBench.csv
  open_source_only: true
  limit: 5
  prompt_variants: [domain_and_dataset]
  response_types: [workflow, code]

request:
  temperature: 0
  max_tokens: 1024
  timeout_s: 600
  retries: 1
  seed: 42

targets:
  - name: local-llama
    protocol: openai_chat
    base_url: http://127.0.0.1:8080/v1
    model: local-model
```

## Next improvements

- Add optional judge-based workflow similarity scoring.
- Add code execution for tasks whose data can be downloaded locally.
- Add resume semantics on top of the existing task-level `samples` table.

DABStep uses the separate external-process adapter documented in
[`dabstep.md`](dabstep.md); it is intentionally not routed through this
chat-request interface.
