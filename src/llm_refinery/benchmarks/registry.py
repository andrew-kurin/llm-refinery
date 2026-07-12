from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm_refinery.benchmarks.agent.parser import reparse_agent_eval_run
from llm_refinery.benchmarks.dabstep.parser import reparse_dabstep_run
from llm_refinery.benchmarks.http_load.parser import reparse_http_load_run
from llm_refinery.benchmarks.llama_bench.parser import reparse_llama_bench_run
from llm_refinery.benchmarks.lm_eval.parser import reparse_lm_eval_run

RunReparser = Callable[[dict[str, Any]], dict[str, float]]

_REPARSERS: dict[str, RunReparser] = {
    "agent_eval": reparse_agent_eval_run,
    "dabstep": reparse_dabstep_run,
    "http_load": reparse_http_load_run,
    "llama_bench": reparse_llama_bench_run,
    "lm_eval": reparse_lm_eval_run,
}


class ReparseNotSupported(ValueError):
    pass


def reparse_run(run: dict[str, Any]) -> dict[str, float]:
    benchmark_kind = str(run.get("benchmark_kind") or "")
    reparser = _REPARSERS.get(benchmark_kind)
    if reparser is None:
        raise ReparseNotSupported(f"no reparser for benchmark kind {benchmark_kind!r}")
    return reparser(run)


def reparsable_kinds() -> tuple[str, ...]:
    return tuple(sorted(_REPARSERS))
