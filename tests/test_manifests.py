from pathlib import Path

from llm_refinery.benchmarks.agent.config import load_agent_eval_config
from llm_refinery.benchmarks.http_load.config import load_http_load_config
from llm_refinery.benchmarks.llama_bench.config import load_llama_config
from llm_refinery.workflows.suite_config import load_suite_config


def test_checked_in_manifests_match_their_strict_schemas():
    loaded: list[Path] = []
    for path in Path("sweeps").glob("*.yaml"):
        if path.name.endswith("-http-load.yaml") or "http-load-" in path.name:
            load_http_load_config(path)
        elif path.name.endswith("-suite.yaml"):
            load_suite_config(path)
        else:
            load_llama_config(path)
        loaded.append(path)

    for path in Path("benchmarks").glob("*.yaml"):
        load_agent_eval_config(path)
        loaded.append(path)

    assert loaded
