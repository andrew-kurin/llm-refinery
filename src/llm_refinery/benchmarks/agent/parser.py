from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_refinery.core.metrics import add_distribution_metrics


def reparse_agent_eval_run(run: dict[str, Any]) -> dict[str, float]:
    artifact = (run.get("artifacts") or {}).get("responses")
    if not artifact:
        raise FileNotFoundError("agent-eval run has no responses artifact")
    path = Path(artifact["path"])
    results = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    successes = [result for result in results if result.get("ok")]
    measurement = (run.get("artifacts") or {}).get("measurement")
    if measurement:
        measurement_data = json.loads(Path(measurement["path"]).read_text(encoding="utf-8"))
        wall_duration_s = float(measurement_data["wall_duration_s"])
    else:
        wall_duration_s = float(run.get("duration_s") or 0.0)
    metrics: dict[str, float] = {
        "request_count": float(len(results)),
        "success_count": float(len(successes)),
        "error_count": float(len(results) - len(successes)),
        "wall_duration_s": wall_duration_s,
    }
    if results:
        metrics["success_rate"] = len(successes) / len(results)
    add_distribution_metrics(
        metrics,
        "latency",
        [result.get("latency_s") for result in successes],
    )
    add_distribution_metrics(
        metrics,
        "response_chars",
        [float(len(str(result.get("response_text") or ""))) for result in successes],
        unit_suffix="",
    )

    completion_tokens = [
        int(result["completion_tokens"])
        for result in successes
        if result.get("completion_tokens") is not None
    ]
    if completion_tokens:
        total = sum(completion_tokens)
        metrics["completion_tokens_total"] = float(total)
        metrics["completion_tokens_avg"] = total / len(completion_tokens)
        duration = metrics["wall_duration_s"]
        metrics["completion_tokens_per_second"] = total / duration if duration else 0.0

    workflows = [
        result
        for result in successes
        if (result.get("request") or {}).get("response_type") == "workflow"
        and result.get("workflow_step_count") is not None
    ]
    if workflows:
        metrics["workflow_count"] = float(len(workflows))
        metrics["workflow_step_count_avg"] = sum(
            int(result["workflow_step_count"]) for result in workflows
        ) / len(workflows)
        metrics["workflow_step_abs_error_avg"] = sum(
            int(result.get("workflow_step_abs_error") or 0) for result in workflows
        ) / len(workflows)

    code_results = [
        result
        for result in successes
        if (result.get("request") or {}).get("response_type") == "code"
        and result.get("code_syntax_ok") is not None
    ]
    if code_results:
        syntax_pass = sum(1 for result in code_results if result["code_syntax_ok"])
        metrics["code_count"] = float(len(code_results))
        metrics["code_syntax_pass_count"] = float(syntax_pass)
        metrics["code_syntax_pass_rate"] = syntax_pass / len(code_results)
    return metrics
