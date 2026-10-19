# MLX cache and recall investigation

The DiffusionGemma sweep pins Python 3.11, `mlx-vlm==0.6.3`, `mlx-lm==0.31.3`, and
`mlx==0.31.2`. The first 4-bit checkpoint also pins its model revision. Run these
commands from the repository root.

Start the server:

```bash
uv run llm-refinery server sweeps/diffusiongemma-26b-mlx-vlm-sweep.yaml --index 0
```

`tools/mlx_runtime.py serve` launches the upstream server, forwards the configured
prefill size when the diffusion caller omits it, and observes its load, cache
factory, and diffusion entry point. It writes
`results/mlx-runtime-8082.json` atomically with:

- Actual serving package versions, arguments, PID, and server instance ID.
- Resolved checkpoint paths and revisions.
- Cache types, `keep`, and `max_size` from actual `model.make_cache()` calls.
- Received diffusion arguments, including defaults, and supported argument names.
- The launcher's prefill and stop-token policies, included in the benchmark identity.

The cache observer preserves the returned objects. The diffusion wrapper fills
an omitted or `None` prefill size from the server setting; an explicit size is
preserved. This corrects the missing forwarding in mlx-vlm 0.6.3. Native cache
types and attention implementations are unchanged. The wrapper also isolates
the tokenizer's stop-token list before each diffusion call. In this runtime,
`load_config` aliases that list to the generation configuration, then generation
appends the configuration's IDs to it. Without isolation it doubles per request,
causing CPU memory growth and increasingly slow stop checks. Copying the unique
IDs preserves the same stopping behavior and leaves the model configuration intact.
Report I/O occurs when
entering generation and constructing its cache, so this launcher is intended for
diagnosis and paired comparisons, not precise upstream latency measurements.
Reload mode is disabled because a child server would bypass the wrapper.

Cache observations appear after a request constructs a cache. `null` for an
argument listed in `diffusion_supported_arguments` means the default is `None`;
an absent name in that list means the entry point does not accept the argument.
These observations distinguish CLI requests from settings the diffusion path
actually receives. The unwrapped 0.6.3 HTTP server omits the CLI prefill size;
this launcher forwards it to the generation API. Model-owned sliding windows can differ from
`--max-kv-size`. Check the report before attributing a speed difference to either.
Do not interpret 4-bit checkpoint weights as quantized KV caches.

`MAX_KV_SIZE` also controls server admission: prompt tokens plus the requested
maximum output tokens must fit that limit. In the pinned server, `8192` therefore
still rejects oversized requests even though it does not resize DiffusionGemma's
model-owned caches. It is not a no-op. The cache-factory and diffusion-forwarding
checks below do not bypass or test this separate admission behavior.

Run the short smoke, which now has a single fact and an exact-answer check:

```bash
uv run llm-refinery http-load sweeps/diffusiongemma-26b-mlx-http-load.yaml \
  --target mlx-diffusiongemma-26b-4bit --scenario recall-smoke
```

The HTTP config requires the report from the reporting launcher. Each run keeps
stable serving configuration in `config_json.server_runtime` and complete before
and after reports in its `server_runtime` artifact. PIDs, timestamps, local cache
paths, and observations do not change the benchmark configuration hash. A changed
server instance fails the trial. Keep the
report paired with the local server on port 8082; it does not authenticate a
remote endpoint.

## Exact-length recall probes

Generate prompts using the pinned model's tokenizer and chat template. This
downloads tokenizer/configuration files if needed and does not load model weights:

```bash
uv run --no-project --python 3.11 --with mlx-vlm==0.6.3 --with mlx-lm==0.31.3 --with mlx==0.31.2 \
  python tools/mlx_runtime.py recall \
  --model mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --revision 0d2cee4aa112548df2fad35a7f017eeb1ac8f352
```

This writes six fixtures and an HTTP-load config under `results/mlx-recall/` for
4,096, 8,191, 8,192, 8,193, 12,288, and 16,384 prompt tokens. The fact appears once
near the end, followed by neutral text and the question. The generator mirrors
0.6.3's DiffusionGemma chat preprocessing, including special tokens. Every trial
requires the server to report the intended prompt length; missing usage or a
template mismatch fails the token check rather than silently changing the test.

Start with one boundary cell; long DiffusionGemma requests have previously timed
out on the 32 GB machine. Stop the ordinary server first and start the diagnostic
server below, which explicitly disables the admission limit with `--max-kv-size 0`.
This allows the probes above 8192 tokens to reach the model. The normal sweep
retains its 8192-token request budget. This diagnostic setting does not impose a
memory limit on the model's full-attention caches.

```bash
uv run --no-project --python 3.11 --with mlx-vlm==0.6.3 --with mlx-lm==0.31.3 --with mlx==0.31.2 \
  python tools/mlx_runtime.py serve \
  --runtime-report results/mlx-runtime-8082.json \
  --model mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --revision 0d2cee4aa112548df2fad35a7f017eeb1ac8f352 \
  --host 127.0.0.1 --port 8082 --prefill-step-size 2048 --max-kv-size 0
```

Then run the generated probe against that server:

```bash
uv run llm-refinery http-load results/mlx-recall/http-load.yaml --dry-run
uv run llm-refinery http-load results/mlx-recall/http-load.yaml \
  --scenario recall-near-end-8193
```

Use `--tokens` to generate a smaller set. For a fact at the beginning, generate a
separate config with `--position start --output results/mlx-recall-start`. A
failure there can reflect deliberate eviction by a bounded cache; compare it
with the near-end test and a configuration retaining the full context. Three
requests are a correctness smoke, not a reliable throughput or tail-latency study.

Compare correctness and measured lengths alongside performance:

```bash
uv run llm-refinery compare results/llm_refinery.duckdb --include-failed \
  --metric check_pass_rate --metric prompt_tokens_check_pass_rate \
  --metric prompt_tokens_min --metric prompt_tokens_max \
  --metric completion_tokens_per_second --metric visible_ttft_p95_s \
  --param scenario --param concurrency
```

Wrong answers, missing usage, wrong token counts, and server restarts fail the
trial. Failed trials remain excluded from normal comparisons. `check_pass_rate`
measures answer correctness; `prompt_tokens_check_pass_rate` measures whether the
requested workload was reached. Require both to pass before ranking throughput.

Only compare prefill sizes 512/2048/4096 or capped/uncapped cache settings after
the runtime report confirms those changes take effect. The upstream rotating
sink-mask issue concerns retained prefixes (`keep > 0`); a model's native
`keep=0` sliding window is a different path. If testing KV quantization, hold
chunk size and prompt length fixed when comparing cache precisions, and measure
peak Metal allocation separately from system RAM/swap. The HTTP runner does not
measure server Metal memory.

## Independent correctness probes

The [recorded results](experiments/mlx-cache-recall-2026-10-18.json) contain
17 greedy Qwen2.5-1.5B retrieval cases run on an M5 Max, using the pinned model
revision and MLX versions. Each cell was run once; these are correctness probes,
not statistical performance measurements. For the three retained-window cases
(1,577 tokens at steps 512 and 4096, and 12,077 tokens at step 4096):

| Cache | Exact answers |
| --- | --- |
| Unbounded | 3/3 |
| Stock rotating, `keep=4` | 0/3 |
| Supplied sink-aware patch, `keep=4` | 3/3 |

The long cell used `max_size=8192` and actually processed three prefill chunks.
Early-fact controls failed after eviction and passed when the prefix was pinned.
The sink-mask finding applies to this Qwen cache path; DiffusionGemma's native
sliding caches use `keep=0`. This PR does not change upstream attention behavior.

The supplied patch passes all 32 default-window visibility cases but fails the
explicit smaller-window case. The guarded implementation matching
[mlx-lm PR #1978](https://github.com/ml-explore/mlx-lm/pull/1978/files) passes all
33 checks. Neither experimental implementation is installed into the server.

Reproduce the mask checks without loading weights:

```bash
uv run --no-project --python 3.11 --with mlx-vlm==0.6.3 --with mlx-lm==0.31.3 --with mlx==0.31.2 \
  python tools/mlx_cache_probe.py --mask-only
```

Omit `--mask-only` to run the 17 retrieval cells. This downloads the pinned
Qwen2.5-1.5B 4-bit checkpoint if needed. JSON results and prompt fixtures are
written under `results/mlx-cache-probe/`; use `--output` to change that directory.

To check the actual DiffusionGemma cache factory, server forwarding method, and
10 context-admission cases without loading weights:

```bash
uv run --no-project --python 3.11 --with mlx-vlm==0.6.3 --with mlx-lm==0.31.3 --with mlx==0.31.2 \
  python tools/mlx_diffusion_probe.py
```

This downloads only the pinned DiffusionGemma configuration if needed. It checks
25 rotating caches with `keep=0`, window 1024, and five ordinary KV caches. The
server-method test uses stubs to observe forwarded arguments, not full model
inference. It also verifies that the ordinary 8192-token admission budget rejects
an 8193-token prompt, while the explicit diagnostic setting `0` accepts it. The
stop-token reproduction uses actual `load_config` and `StoppingCriteria` methods:
12 stock resets/appends grow the list from 3 to 12,288 entries, while 40 isolated
calls keep it at 6 entries and the configuration at 3.

## DiffusionGemma measurements with weights

Run the native generation comparison with the same model revision and packages:

```bash
uv run --no-project --python 3.11 --with mlx-vlm==0.6.3 --with mlx-lm==0.31.3 --with mlx==0.31.2 \
  python tools/mlx_diffusion_benchmark.py \
  --tokens 4096 8191 8192 8193 16384 --prefill-steps none 4096 2048 512
```

This loads the pinned 4-bit checkpoint, creates fresh native KV caches on each
call, and compares unchunked prefill with three explicit chunk sizes. It records
actual encoder call lengths, exact answers, generation metadata, wall times and
peak Metal allocation. The peak counter is reset per request after synchronizing
and clearing the allocation cache; reported bytes include resident model weights.
There is one warmup per cell followed by three measured repetitions, with cell
order rotated each round. The stop-token list is isolated before every call and
its size is recorded. This uses the direct generation API; HTTP overhead is measured
separately. No KV quantization is enabled.

The initial unisolated timing series was discarded after the stop-token aliasing
caused exponential CPU memory growth and increasing stop-check time. The
controlled run measures chunk size with that independent issue held fixed.

On Apple M5 Max with 128 GiB RAM, the 16,384-token fixture produced these medians
over three measured calls per cell. GB below means decimal gigabytes of peak
Metal allocation, including weights:

| Prefill size | Exact answers | Wall time | Peak Metal |
| --- | --- | --- | --- |
| Unchunked | 3/3 | 9.33 s | 30.52 GB |
| 4096 | 3/3 | 6.37 s | 20.13 GB |
| 2048 | 3/3 | 5.28 s | 18.50 GB |
| 512 | 3/3 | 5.79 s | 18.17 GB |

The sweep now starts at 2048: this reduced total peak allocation by 39.4% versus
unchunked prefill and by 8.1% versus chunks of 4096 in this 16K cell. A size of
512 saved another 0.33 GB but took longer. These small paired measurements are
exploratory; they do not establish a universal fastest size.

Across 4096, 8191, 8192, 8193 and 16384 tokens, all 45 measured chunked calls
returned the exact code. Unchunked calls returned only `7391` at 8191 tokens and
an empty answer at 8193 tokens (0/3 each), while the other three lengths passed.
The cause of those answer differences remains unestablished. The model's native
`keep=0` path does not trigger the retained-sink bug reproduced on Qwen.

All 80 calls, including warmups, held the model configuration's stop-token list
at 3 IDs and the per-call list at 6. Repetitions use the same prompt and seed;
they demonstrate consistency on these five fixtures, not broad model quality.

The corrected HTTP launcher then completed 53 requests in one server process:
six warmups and 47 measured calls (32 at 512 tokens, three each at 4096, 8191,
8192, 8193 and 16384). Every measured call passed both exact-answer and reported
prompt-length checks. The report confirmed `prefill_step_size=2048`, native
`keep=0` sliding caches, and unchanged full-attention cache types. These long
requests used the explicitly uncapped admission setting `0`; the ordinary sweep
keeps its 8192-token admission budget.
An actual HTTP request under that ordinary budget rejected 8193 prompt tokens plus
32 requested output tokens with status 400 and the expected 8225-token budget error.

The [recorded DiffusionGemma results](experiments/mlx-diffusion-2026-10-18.json)
include every native call, HTTP samples and metrics, and the no-weights probe.
The repeated single-fact workload is a correctness and stability smoke. IFEval,
GSM8K and diverse long-context workloads still need separate quality evaluation.
