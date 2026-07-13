# DGX Spark targets

DGX Spark support uses two independent, read-only discovery paths:

- OpenSSH collects serving-host inventory using the user's normal SSH config.
- The OpenAI-compatible HTTP endpoint reports vLLM health, version, and served
  models.

The harness never starts, stops, signals, or reconfigures the model service.

## Configure the target

Copy `targets/dgx-spark-vllm.yaml` and set:

- `host.destination` to an OpenSSH destination such as `dgx`.
- `host.expected_fingerprint` after the first trusted inspection when the SSH
  alias should be pinned to the discovered hardware or OS installation.
- `endpoint.base_url` to the client-visible URL, including `/v1`.
- `model.id` with `selection: explicit` when the server exposes multiple IDs.
- `transport.trust_env` to `false` for a direct LAN connection. Certificate
  environment variables remain available when a logical host is covered by
  `NO_PROXY`, but active HTTP proxying is rejected for IP-pinned DGX routes.

The SSH destination and HTTP hostname intentionally do not have to match. For
example, SSH can use `dgx` while requests use
`http://aitopatom-41de.local:8000/v1`.

For an identity-pinned setup, first inspect the intended machine over a trusted
SSH connection. Copy `host.profile.host_fingerprint` into
`host.expected_fingerprint` only when `host_fingerprint_strength` is `hardware`
or `installation`. Linux machine-id is preferred for the primary fingerprint so
the same machine has the same identity when inspected locally or through SSH;
this installation identity also preserves existing local-run and resume keys.
When available, the separately reported `host_hardware_fingerprint` hashes a
canonical DMI product UUID and becomes the primary identity if machine-id is not
usable. Neither raw identifier is recorded. The hostname fallback is marked
`weak` and cannot satisfy a pin.
Strong hashes emitted by earlier versions of this discovery code are retained as
finite compatibility aliases, so an existing hardware or installation pin
continues to verify after the primary identity format changes.
Future inspections fail closed if the SSH alias resolves to a different identity
or inventory cannot provide a verifiable fingerprint. Reinstalling the OS can
change an installation fingerprint, while replacing hardware changes a hardware
fingerprint.
This binding works across SSH alias and username overrides and does not require
the SSH control-plane name to equal the HTTP data-plane hostname. vLLM does not
expose the probe fingerprint over its HTTP API, so the target configuration is
still the explicit association between that pinned host and the service URL.

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
The command timeout includes SSH connection establishment and all inventory
queries; the supplied target uses a 5-second connection timeout and a 30-second
overall budget so best-effort NVIDIA queries retain a safety margin.

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
successfully while vLLM is offline. This tolerates only service unavailability;
required-host inventory failures, fingerprint mismatches, malformed discovery
responses, and missing or ambiguous model selections still fail. Benchmark
suites require a healthy service and concrete model. Set
`discovery.metrics: false` when the server's Prometheus endpoint should not be
sampled.

Keep the vLLM port on a trusted network. Some observability endpoints, including
server configuration and metrics, may not be protected by the OpenAI API key.

Target discovery and metrics share the target's `transport` policy. It defaults
to `trust_env: true`, matching HTTP-load behavior. Remote DGX hostnames are
resolved and IP-pinned before requests; an active environment proxy is rejected
because it cannot preserve both that address binding and logical TLS SNI. For a
deterministic direct LAN path, set `transport.trust_env: false`. For an HTTPS
endpoint using a private CA, set `transport.ca_bundle` to a PEM bundle; a relative path is resolved from
the target YAML file and must name an existing file. TLS, proxy, and HTTP
protocol failures are configuration/transport errors and are not hidden by
`--allow-service-unavailable`.

For an authenticated endpoint, set `endpoint.api_key_env` to the name of an
environment variable containing the Bearer token. Discovery, preflight, load,
and lm-eval then use the same credential. The lm-eval child receives the token
only in its environment, never in command arguments. Because the pinned
lm-eval API adapter cannot safely receive arbitrary headers, quality runs reject
custom headers and non-Bearer `Authorization` values instead of silently
dropping them.

For a schema-v2 target suite, quality evaluation inherits the target's
`transport.trust_env` and `transport.ca_bundle` policy. `quality.trust_env` and
`quality.ca_bundle` can override that policy; legacy endpoint suites default to
a direct path. The harness gives lm-eval a temporary loopback URL and relays
requests to the already-resolved address. The relay preserves logical Host and
HTTPS SNI, applies the configured CA, and refuses upstream redirects. With
`trust_env: true`, both `uvx` and the lm-eval child retain the executor's
network environment for package, dataset, and tokenizer resolution; the runner
merges loopback entries into `NO_PROXY`/`no_proxy`. Model traffic itself must be
direct: when an environment proxy is configured, cover the logical model host
with `NO_PROXY` or set `quality.trust_env: false`. The runner fails before
starting the relay if an active proxy would receive model credentials. With
`trust_env: false`, the lm-eval child explicitly drops ambient proxy and CA
variables. In both modes, only the relay applies the target CA and pinned route.

The default `quality.model_backend: local-chat-completions` backend does not
perform client-side tokenization or token-aware truncation. The harness therefore
rejects a quality `tokenizer` setting with that backend rather than pretending it
is active. Set `quality.model_backend: local-completions` together with
`quality.tokenizer` only when the service actually exposes `/v1/completions` and
the selected tasks support it. The same pairing is available as
`suite --model-backend local-completions --tokenizer ID`. An explicit tokenizer
uses lm-eval's Hugging Face tokenizer backend; omit it to let lm-eval detect
vLLM's remote tokenizer. With lm-eval 0.4.12, remote tokenization is incompatible
with chat-template application: set `quality.apply_chat_template: false` in a
suite, or pass `--no-apply-chat-template` to the standalone command. The harness
rejects that unsafe combination instead of allowing vLLM `/tokenize` requests
containing message dictionaries. The relay forwards `/tokenizer_info`,
`/tokenize`, and `/detokenize` through the same pinned route and deadline policy.
Keep quality request sizes within the discovered `max_model_len`.

Every quality child run writes lm-eval output beneath a unique directory named
for its recorded run ID. Result and sample collection is restricted to that
directory, so overlapping runs of the same target cannot consume each other's
artifacts. All model requests, including route-less local targets, pass through
the loopback relay so `request_timeout_s` is an absolute per-request deadline;
the relay applies the configured CA and direct pinned-route policy without
forwarding model credentials through an environment proxy.

## Run the smoke suite

After the user independently starts vLLM:

```bash
uv run llm-refinery target inspect targets/dgx-spark-vllm.yaml --json
uv run llm-refinery suite sweeps/dgx-spark-quality-smoke-suite.yaml
```

The suite resolves the target once, validates quality context length and each
selected HTTP scenario's output-token budget against the discovered model limit,
then gives the same concrete model and endpoint to quality and HTTP-load child
runs. Exact prompt, system-message, chat-template, and output context fit cannot
be proven without the served tokenizer; the suite records a preflight warning
with the rendered character size and remaining input-token budget. HTTP-load
scenarios are overlaid with the suite target, so a stale target inside the
scenario library cannot redirect load measurements. Schema-v2 suites also apply
the target transport policy to preflight sanity, quality evaluation, and HTTP
load, keeping direct routing and private-CA trust consistent across every request
to the serving endpoint.

`preflight.require_clean` checks listening ports only for a loopback endpoint on
the executor. It cannot prove that a remote DGX has no competing model server,
so remote suites must set `require_clean: false` explicitly instead of silently
claiming a clean host.

## Recorded identity

- `runs.system_json` is the executor/client, such as the Mac running lm-eval.
- `runs.target_json` is the DGX host, vLLM service, selected model, and topology.
- `runs.target_json.route` records the logical origin and selected IP address
  reused by discovery, preflight, quality relay, metrics, and HTTP load.
- `target-discovery.json` retains the complete sanitized discovery report.
- `server-before.json` and `server-after.json` retain best-effort host snapshots.
- `vllm-metrics-before.prom` and `vllm-metrics-after.prom` retain read-only
  Prometheus snapshots when `/metrics` is available.

Mac-to-Spark HTTP latency is labeled `remote_client_to_server`. To measure the
server over loopback, run the same harness directly on the Spark with a local
target and `http://127.0.0.1:8000/v1`; this is a different comparison topology.

DGX Spark uses unified CPU/GPU memory. The probe records `/proc/meminfo` as the
system capacity and labels NVIDIA's value as reported device memory rather than
assuming discrete VRAM. NVIDIA-SMI's `CUDA Version` is recorded as
`cuda_driver_supported_version`, because it is the newest CUDA level supported
by the driver rather than an observed runtime; `cuda_runtime_version` remains a
schema-compatibility alias.
