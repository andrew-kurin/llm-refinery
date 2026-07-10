# Architecture cutover migration

This release intentionally makes a hard configuration and internal Python API
cutover.

## YAML manifests

- Replace endpoint `provider` with wire `protocol`:
  - `openai` / `cerebras` → `openai_chat`
  - `ollama` → `ollama_chat`
- Move quality settings out of llama sweep files into a dedicated suite manifest.
- Suite manifests now use `endpoint`, `quality`, `http_load`, and `preflight`.
- Place the referenced HTTP-load path under `http_load.config`; it resolves relative
  to the suite manifest.
- Unknown fields now fail validation instead of being ignored.

See `sweeps/gemma4-31b-suite.yaml` and `sweeps/qwen-suite.yaml`.

## DuckDB

Opening an existing result database automatically:

- adds benchmark kind, spec hash, parent run, and schema version fields
- creates typed `artifacts` and task-level `samples` tables
- migrates legacy stdout/stderr paths into artifact roles
- converts stored provider fields to protocol fields
- gives historical trial names a complete spec-hash suffix

Back up important databases before first opening them with the new version. Schema
migrations are forward-only.

## Internal Python imports

Transitional root façade modules were removed. Import directly from the owning
vertical slice, for example:

```python
from llm_refinery.benchmarks.http_load.runner import run_http_load
from llm_refinery.benchmarks.llama_bench.config import load_llama_config
from llm_refinery.storage.duckdb import ResultStore
```
