# Architecture

`llm-refinery` is a modular monolith organized around benchmark vertical slices.
The CLI is an adapter; benchmark execution and persistence do not depend on Click.

## Dependency direction

```text
commands ───────► workflows / benchmark services
                      │
application ──────────┤
  RunSession           ▼
                 core models
                      ▲
benchmarks ───────────┤
providers ────────────┤
storage ──────────────┘
```

Enforced rules:

- `core` has no dependency on application, benchmarks, providers, storage,
  workflows, or commands.
- storage depends only on core models.
- benchmarks, providers, and application code never import commands or workflows.
- workflows call benchmark services directly; they do not invoke the project's CLI
  as a subprocess.

`tests/test_architecture.py` verifies these boundaries.

## Run lifecycle

Every persisted execution is described by `core.runs.RunSpec`:

- `benchmark_kind` selects benchmark-specific behavior such as reparsing
- `spec_hash` fingerprints the complete sanitized executable configuration
- `trial_name` is a human label suffixed by `spec_hash`
- `parent_run_id` links suite and child runs

`application.run_session.RunSession` owns timing, run IDs, typed artifacts, host
metadata, and durable `running`/`ok`/`failed` transitions. Unexpected exceptions and
interruptions are recorded as failed runs rather than disappearing.

## Storage

DuckDB infrastructure lives under `llm_refinery.storage`:

- `runs`: execution identity, status, configuration, and system profile
- `metrics`: normalized scalar metrics
- `artifacts`: typed artifact roles, portable database-relative paths, and media types
- `samples`: task/request-level outcomes for checkpointing and future resume support
- `schema_migrations`: applied database schema versions

Legacy databases are migrated when opened. Legacy benchmark kinds are inferred and
stdout/stderr paths become typed artifacts.

## Benchmark slices

Each directory under `llm_refinery.benchmarks` owns its configuration, planning,
execution, parser, and metrics where applicable:

- `llama_bench`
- `http_load`
- `lm_eval`
- `agent`
- `dabstep`

`benchmarks.registry` maps `benchmark_kind` to the correct artifact reparser. A
reparser must never inspect artifacts owned by another benchmark kind.

Agent benchmark implementations have a second, agent-specific registry. This is
appropriate for request/scoring adapters such as GeoAnalystBench. DABStep has its
own top-level external-process slice, which preserves the upstream agent loop while
using the shared lifecycle, artifacts, samples, resume checks, and reparsing system.

## Endpoints and protocols

`core.endpoints.Endpoint` is shared by HTTP-load, lm-eval, suites, and agent-eval.
The `protocol` field represents the wire protocol:

- `openai_chat`
- `ollama_chat`

A vendor that exposes an OpenAI-compatible endpoint uses `openai_chat`; adding a new
vendor does not require a new execution branch. Secret header values are excluded
from stored configuration, while a hash keeps experiment identity sensitive to
header changes.

## Adding a benchmark

1. Create `benchmarks/<kind>/` with strict config loading, execution, and parsing.
2. Build a complete `RunSpec` and execute it inside `RunSession`.
3. Give every artifact a stable semantic role and media type.
4. Store request/task results in `samples` when the benchmark has sub-work.
5. Register artifact reparsing in `benchmarks/registry.py`.
6. Add public behavior tests, including failure persistence and reparsing.
7. Add a thin Click command only if the benchmark needs standalone CLI exposure.

## Adding a protocol

1. Add a protocol constant in `core.endpoints`.
2. Implement transport behavior under `providers` or the owning benchmark slice.
3. Register dispatch in the benchmark transport.
4. Add endpoint validation and transport contract tests.

Keep registries internal until third-party plugins are an actual requirement.
