# DGX Spark targets

DGX Spark support uses two independent, read-only discovery paths:

- OpenSSH collects serving-host inventory using the user's normal SSH config.
- The OpenAI-compatible HTTP endpoint reports vLLM health, version, and served
  models.

The harness never starts, stops, signals, or reconfigures the model service.

## Configure the target

Copy `targets/dgx-spark-vllm.yaml` and set:

- `host.destination` to an OpenSSH destination such as `dgx`.
- `endpoint.base_url` to the client-visible URL, including `/v1`.
- `model.id` with `selection: explicit` when the server exposes multiple IDs.
- `model.tokenizer` when lm-eval cannot load the served ID as a tokenizer.

The SSH destination and HTTP hostname intentionally do not have to match. For
example, SSH can use `dgx` while requests use
`http://aitopatom-41de.local:8000/v1`.

## Inspect while vLLM is offline

Host inventory remains useful when no model is being served:

```bash
uv run llm-refinery target inspect targets/dgx-spark-vllm.yaml \
  --allow-service-unavailable \
  --json
```

Another user can override only the SSH alias without editing the file:

```bash
uv run llm-refinery target inspect targets/dgx-spark-vllm.yaml \
  --ssh-destination my-spark \
  --allow-service-unavailable
```

The fixed probe is streamed to `python3 -I -` over SSH. It makes no writes and
does not require llm-refinery to be installed on the Spark. Missing optional
tools produce partial inventory rather than triggering installation.

## Model discovery

When vLLM is available, the resolver reads `/health`, `/version`, and
`/v1/models`. Selection is deliberately fail-closed:

- `single` succeeds only when exactly one model ID is returned.
- `explicit` requires `model.id` and verifies that exact ID.
- Empty or ambiguous model lists stop the suite before benchmark requests.

`/server_info` is optional and sanitized before storage. It is useful for
recording dtype, context, parallelism, cache, and tokenizer configuration, but
the suite does not depend on its version-specific shape.

Set `discovery.service_required: false` for a host profile that should inspect
successfully while vLLM is offline. Benchmark suites still require a healthy
service and concrete model. Set `discovery.metrics: false` when the server's
Prometheus endpoint should not be sampled.

Keep the vLLM port on a trusted network. Some observability endpoints, including
server configuration and metrics, may not be protected by the OpenAI API key.

## Run the smoke suite

After the user independently starts vLLM:

```bash
uv run llm-refinery target inspect targets/dgx-spark-vllm.yaml --json
uv run llm-refinery suite sweeps/dgx-spark-quality-smoke-suite.yaml
```

The suite resolves the target once, validates quality context length and each
selected HTTP scenario's output-token budget against the discovered model limit,
then gives the same concrete model and endpoint to quality and HTTP-load child
runs. HTTP-load scenarios are overlaid with the suite target, so a stale target
inside the scenario library cannot redirect load measurements.

## Recorded identity

- `runs.system_json` is the executor/client, such as the Mac running lm-eval.
- `runs.target_json` is the DGX host, vLLM service, selected model, and topology.
- `target-discovery.json` retains the complete sanitized discovery report.
- `server-before.json` and `server-after.json` retain best-effort host snapshots.
- `vllm-metrics-before.prom` and `vllm-metrics-after.prom` retain read-only
  Prometheus snapshots when `/metrics` is available.

Mac-to-Spark HTTP latency is labeled `remote_client_to_server`. To measure the
server over loopback, run the same harness directly on the Spark with a local
target and `http://127.0.0.1:8000/v1`; this is a different comparison topology.

DGX Spark uses unified CPU/GPU memory. The probe records `/proc/meminfo` as the
system capacity and labels NVIDIA's value as reported device memory rather than
assuming discrete VRAM.
