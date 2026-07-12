# IFBench lm-eval adapter

This task evaluates the 300-record
[`allenai/IFBench_test`](https://huggingface.co/datasets/allenai/IFBench_test)
dataset with Ai2's official rule-based grader. The paper's primary metric is
`prompt_level_loose_acc`; the adapter also records strict and instruction-level
accuracy.

Both inputs and scoring code are immutable for reproducibility:

- dataset revision: `2e8a48de45ff3bf41242f927254ca81b59ca3ae2`
- grader revision: `allenai/IFBench@1091c4c3de6c1f6ed12c012ed68f11ea450b0117`
- lm-eval version: `0.4.12`

The scorer verifies the installed grader's source hashes and dependency
versions and raises an error on a mismatch. Install the additional packages in
`requirements.txt` into the same environment as lm-eval. With `uv`, the
standalone validation command is:

```bash
uvx --from 'lm-eval[api]==0.4.12' \
  --with 'ifbench @ git+https://github.com/allenai/IFBench.git@1091c4c3de6c1f6ed12c012ed68f11ea450b0117' \
  --with 'emoji==2.15.0' \
  --with 'nltk==3.9.2' \
  --with 'setuptools==80.9.0' \
  --with 'syllapy==0.7.2' \
  lm_eval --model dummy --tasks ifbench \
  --include_path evals/lm_eval_tasks/ifbench --limit 1
```

The first run must be online: `uvx` fetches the pinned scorer and dependencies,
Hugging Face fetches the pinned dataset, and the official scorer downloads its
NLTK resources. After all three caches have been primed, the same command works
offline (for example with `UV_OFFLINE=1 HF_HUB_OFFLINE=1`) as long as those
caches remain available. A fresh machine or cleared cache requires network
access again.
