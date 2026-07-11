# Quality eval suites

Use `llm-refinery lm-eval` for lm-eval quality suites. Results are parsed into DuckDB and can be compared with `llm-refinery compare`. The harness defaults to `lm-eval[api]==0.4.12`; keep that package and each task dataset revision fixed when reproducing a published run.

Release-oriented suite runs use `--log_samples`. The harness copies every matching
sample JSONL into the run artifact directory and records normalized per-item rows in
DuckDB. A requested sample log that produces no sample records fails closed.

## Local quality tiers

The included manifests have deliberately different meanings:

- `sweeps/local-quality-smoke-suite.yaml`: ten examples per task/subtask; pipeline check only.
- `sweeps/local-quality-core-suite.yaml`: complete IFEval, IFBench, GPQA Diamond, and MuSR generative tasks.
- `sweeps/local-quality-expanded-suite.yaml`: the core tier plus every MMLU-Pro domain.

All tasks in those packs use generation and work through an OpenAI-compatible chat
endpoint. IFBench uses its pinned official rule-based grader and dataset; the manifests
install its exact scorer dependencies in the isolated `uvx` environment. The packs use
local `ifeval_pinned` and `mmlu_pro_pinned` task names so their Hugging Face dataset
commits are part of the executable configuration. The MuSR override converts the
upstream multiple-choice task to deterministic number generation and pins dataset commit
`7c365b439a222150f317764d4f16ae6c96d7d94a`.

The first online run also needs access to the gated `Idavidrein/gpqa` dataset
(accept its Hugging Face terms and provide `HF_TOKEN` if it is not already cached).
After the uv, Hugging Face, and NLTK caches are populated, set `quality.offline: true`
in the release manifest for network-isolated repeats.

For long-context retrieval, RULER also works through this path, but its tokenizer must
match the deployed model. Run it separately so context length is an explicit axis:

```bash
uv run llm-refinery lm-eval local all \
  --model local-model \
  --base-url http://127.0.0.1:8080/v1 \
  --tasks ruler \
  --tokenizer /path/to/the/deployed-model-tokenizer \
  --metadata '{"max_seq_lengths":[4096,8192,16384,32768]}' \
  --max-length 32768 \
  --log-samples \
  --online
```

After the task assets are cached, use the default offline mode for repeatability. Do
not substitute a convenient tokenizer from a different model: that changes sequence
construction and invalidates the context-length comparison.

## Reasoning/knowledge scoreboard

```bash
uv run llm-refinery lm-eval ollama all \
  --suite-name quality-reasoning \
  --tasks gsm8k,arc_challenge_chat,ifeval,truthfulqa_gen,gpqa_main_fixed_generative \
  --include-path evals/lm_eval_tasks \
  --max-length 8192 \
  --gen-kwargs 'reasoning_effort="none"'
```

Then compare:

```bash
uv run llm-refinery compare results/llm_refinery.duckdb \
  --suite quality-reasoning \
  --metric gsm8k.strict-match.exact_match \
  --metric gsm8k.flexible-extract.exact_match \
  --metric arc_challenge_chat.remove_whitespace.exact_match \
  --metric ifeval.prompt_strict_acc \
  --metric truthfulqa_gen.bleu_acc \
  --metric truthfulqa_gen.rougeL_acc \
  --metric gpqa_main_fixed_generative.flexible-extract.exact_match \
  --metric gpqa_main_fixed_generative.strict-match.exact_match \
  --param target \
  --param model
```

## Chat-completions compatibility

The default `llm-refinery lm-eval` path uses lm-eval's `local-chat-completions` backend. Do not mix standard loglikelihood tasks such as `arc_challenge` into that run; lm-eval will finish all generation requests and then fail when it reaches loglikelihood. Use `arc_challenge_chat` for chat-completions runs.

## GPQA fix

`evals/lm_eval_tasks/gpqa_fixed` vendors a small lm-eval task override for GPQA generative runs:

- keeps the explicit `Question: {{Question}}` marker in `doc_to_text`
- removes `Question:` from `generation_kwargs.until`
- derives answer-choice order from a stable question hash rather than global randomness

The removed stop marker avoids truncating responses when chat models echo or restate `Question:` before answering, which can break strict/flexible answer extraction.
