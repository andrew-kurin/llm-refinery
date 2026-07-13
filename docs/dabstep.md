# DABStep

`llm-refinery dabstep` runs Adyen's official DABStep baseline as an external
process and imports its task answers into the common run lifecycle.

DABStep is a top-level benchmark slice rather than an `agent-eval` adapter. The
official baseline owns its smolagents code-execution loop, dataset context, and
answer scorer; llm-refinery owns process supervision, durable artifacts,
retries, resume, metrics, and DuckDB records.

## Prepare the official baseline

The upstream baseline currently documents Python 3.10 and is distributed in its
Hugging Face Space, not as a `dabstep` command on PyPI:

```bash
git clone https://huggingface.co/spaces/adyen/DABstep vendor/DABstep
uv venv --python 3.10 vendor/DABstep/.venv
uv pip install \
  --python vendor/DABstep/.venv/bin/python \
  -r vendor/DABstep/baseline/requirements.txt
```

Pin the checkout commit when exact reproducibility matters; the adapter records the
workspace Git `HEAD` in the run specification when available. The baseline fetches
`adyen/DABstep` tasks and context files from Hugging Face, so its first run needs
network access. The upstream command has no dataset-revision argument and always loads
the official repository's `main` revision. To fail closed, llm-refinery loads a pinned
task manifest and verifies that the selected tasks still exactly match current `main`
before launching. A mismatch aborts the run. The adapter defaults
`OTEL_SDK_DISABLED=true` to avoid requiring the optional local Phoenix trace collector;
set that environment variable to `false` if you launch the collector described by
upstream.

The official baseline executes model-authored Python locally. Use a dedicated
checkout/environment and do not place secrets in its workspace. The adapter filters
ambient environment variables, but it is not a filesystem or OS sandbox.

## Configure

See [`benchmarks/dabstep-smoke.yaml`](../benchmarks/dabstep-smoke.yaml):

```yaml
name: dabstep-smoke
database: results/llm_refinery.duckdb

endpoint:
  name: local
  protocol: openai_chat
  base_url: http://127.0.0.1:8080/v1
  model: local-model
  # api_key_env: LOCAL_LLM_API_KEY

dabstep:
  workspace: ../vendor/DABstep
  command: [.venv/bin/python, baseline/run.py]
  dataset_repo: adyen/DABstep
  dataset_revision: e68a4553c079601b09131851f4b7c6be9680d560
  split: dev
  task_ids: [5, 49]
  limit: all
  concurrency: 1
  max_steps: 10
  timeout_s: 900
  retries: 1
```

Paths in `workspace` and `tasks_file` are resolved relative to the manifest. The
command runs with `workspace` as its current directory. This is required by the
official baseline's context-path logic.

The official `baseline/run.py` accepts only a split and task IDs; it cannot consume a
custom task file or repository revision. Accordingly, `tasks_file` is rejected unless
`tasks_file_arg` names a flag implemented by a compatible wrapper. For example:

```yaml
dabstep:
  command: [.venv/bin/python, compatible_wrapper.py]
  tasks_file: fixtures/custom-tasks.jsonl
  tasks_file_arg: --tasks-file
```

For wrapper runs, llm-refinery passes the canonical selected `tasks.jsonl` artifact to
that flag on every attempt. Merely pointing `tasks_file` at different questions without
a consuming wrapper is not allowed.

By default, an endpoint model such as `local-model` becomes the LiteLLM model ID
`openai/local-model`. Set `dabstep.model_id` when a different LiteLLM provider
prefix is required. The endpoint API key is copied from `api_key_env` to the
subprocess's `OPENAI_API_KEY`; its value is never placed in the command or stored
configuration. Without `api_key_env`, the adapter uses the non-secret placeholder
`local` rather than inheriting an ambient OpenAI key.

Supported DABStep settings are:

- `workspace` (required): official Space checkout
- `command`: Python executable plus `baseline/run.py`
- `tasks_file` plus `tasks_file_arg`: optional custom manifest consumed by a compatible
  wrapper; neither is supported by the official baseline command
- `dataset_repo`: must remain `adyen/DABstep`, which is hard-coded upstream
- `dataset_revision`: pinned official task-manifest commit; symbolic revisions such as
  `main` are rejected
- `split`: `dev` (10 public scored tasks) or `default` (450 hidden-answer tasks)
- `task_ids` and `limit`: deterministic task selection; without either, the safety
  default is 10 tasks, while explicit task IDs are all selected unless a limit is set
- `concurrency`, `max_steps`, whole-process `timeout_s`, and process-level `retries`
- `keep_going`: return a persisted failed outcome instead of raising
- `model_id`: optional explicit LiteLLM model ID
- `pass_env`: additional environment-variable names to forward, for example
  `[HTTPS_PROXY, HTTP_PROXY]`; values are neither configured nor stored

Configuration is strict; unknown fields fail validation. The subprocess receives a
small runtime/cache/certificate environment allowlist, the endpoint key, and names
explicitly listed in `pass_env`; unrelated ambient secrets are not forwarded.

## Run and resume

```bash
uv run llm-refinery dabstep benchmarks/dabstep-smoke.yaml --dry-run
uv run llm-refinery dabstep benchmarks/dabstep-smoke.yaml
```

If a process fails, times out, or is interrupted, answers already written by the
official baseline are checkpointed. Resume the same DuckDB run and skip completed
task IDs:

```bash
uv run llm-refinery dabstep benchmarks/dabstep-smoke.yaml \
  --resume <run-id>
```

The manifest must still match the original run specification. Resume refuses a
different config, a different benchmark kind, an unknown run ID, or an already
successful run. It also verifies that the current executor host matches the stored
host identity.

Older run records whose system-profile capture failed have no executor identity to
verify and therefore fail closed. After independently confirming that you are on the
original executor, recover that run explicitly:

```bash
uv run llm-refinery dabstep benchmarks/dabstep-smoke.yaml \
  --resume <legacy-run-id> --allow-unverified-executor
```

This escape hatch works only when the stored identity is absent; it cannot override a
known host mismatch. A successful recovery stores the current host profile so later
resumes use normal identity validation.

## Artifacts and metrics

Each run stores typed artifacts under the database's `artifacts/<run-id>/`
directory:

- `answers.jsonl`: canonical, merged official answer/submission file
- `tasks.jsonl`: exact selected task manifest
- `stdout.txt` and `stderr.txt`: external process output across attempts
- `upstream-logs.txt` and `upstream-configs.yaml`: copied official baseline files
- `measurement.json`: attempt durations, exit codes, retries, timeouts, and interruptions

Each selected task also has a DuckDB `samples` row. A completed response has
status `ok`; correctness is represented separately by its `score` metric. Samples
also record process attempt, retry, timeout, and interruption counts. Missing
responses have status `failed` and can be replaced on resume.

Aggregate metrics include task/answer/missing counts, completion rate, process
attempt/error/retry/timeout/interruption counts, and wall duration. The `dev`
split additionally reports score totals, correct count, average score, accuracy,
success rate, and per-level success rate. The hidden `default` split cannot be
scored locally, so it reports completion metrics and produces a leaderboard-ready
`answers.jsonl`.

The official answer schema does not expose per-task token counts, agent steps, or
latency. llm-refinery deliberately leaves those metrics absent rather than
fabricating estimates.

The run specification also records a task-source contract. Official runs include the
SHA-256 of both the selected pinned manifest and the matching current-main manifest;
wrapper runs identify that their canonical manifest was passed explicitly. This binds
the recorded task questions/guidelines to the invocation instead of merely assuming
that matching task IDs refer to matching content. The official baseline still downloads
its context files from `main`; pin the baseline checkout and review upstream dataset
changes when exact context-level reproducibility is required.

Metrics can be rebuilt from artifacts:

```bash
uv run llm-refinery reparse results/llm_refinery.duckdb
```
