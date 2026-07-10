# Quality eval suites

Use `llm-refinery lm-eval` for lm-eval quality suites. Results are parsed into DuckDB and can be compared with `llm-refinery compare`. Set `--package-spec` to a pinned `lm-eval[api]` version when reproducing a published run.

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

The removed stop marker avoids truncating responses when chat models echo or restate `Question:` before answering, which can break strict/flexible answer extraction.
