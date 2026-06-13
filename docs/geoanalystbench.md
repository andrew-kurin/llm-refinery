# GeoAnalystBench support

`llm-refinery agent-eval` supports an initial GeoAnalystBench adapter for local
OpenAI-compatible model endpoints.

GeoAnalystBench source: <https://github.com/GeoDS/GeoAnalystBench>

## What is measured

The adapter reads `dataset/GeoAnalystBench.csv`, selects tasks, builds workflow
and/or code-generation prompts, calls the configured endpoint, and stores
per-request JSONL artifacts plus aggregate metrics in DuckDB.

Current normalized metrics include:

- `request_count`
- `success_count`
- `success_rate`
- `error_count`
- latency percentiles
- response character statistics
- token totals/throughput when the endpoint reports usage
- `workflow_step_count_avg`
- `workflow_step_abs_error_avg`
- `code_syntax_pass_rate`

These are intentionally lightweight automatic checks. They do not fully reproduce
human/judge workflow similarity scoring from the paper, and code syntax success is
not the same as executing GIS outputs against the original data.

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
  --metric success_rate \
  --metric workflow_step_abs_error_avg \
  --metric code_syntax_pass_rate \
  --sort success_rate \
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
    provider: openai
    base_url: http://127.0.0.1:8080/v1
    model: local-model
```

## Next improvements

- Add optional judge-based workflow similarity scoring.
- Add code execution for tasks whose data can be downloaded locally.
- Add task-level table/schema if we want cross-run per-task comparisons rather
  than aggregate run metrics only.
- Reuse the `agent-eval` interface for DABStep.
