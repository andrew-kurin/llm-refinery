from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_refinery.benchmarks.agent.geoanalystbench_scoring import (
    code_structure_diagnostics,
    extract_python_code,
)
from llm_refinery.core.metrics import add_distribution_metrics


def reparse_agent_eval_run(run: dict[str, Any]) -> dict[str, float]:
    artifact = (run.get("artifacts") or {}).get("responses")
    if not artifact:
        raise FileNotFoundError("agent-eval run has no responses artifact")
    path = Path(artifact["path"])
    results = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
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
        "response_count": float(len(successes)),
        "error_count": float(len(results) - len(successes)),
        "wall_duration_s": wall_duration_s,
    }
    if results:
        metrics["response_availability_rate"] = len(successes) / len(results)
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
        _with_code_diagnostics(result)
        for result in successes
        if (result.get("request") or {}).get("response_type") == "code"
    ]
    if code_results:
        syntax_valid = sum(1 for result in code_results if result["code_syntax_ok"])
        model_functions = sum(1 for result in code_results if result["code_model_function_present"])
        contract_pass = sum(1 for result in code_results if result["code_contract_ok"])
        metrics["code_count"] = float(len(code_results))
        metrics["code_syntax_valid_count"] = float(syntax_valid)
        metrics["code_syntax_valid_rate"] = syntax_valid / len(code_results)
        metrics["code_model_function_count"] = float(model_functions)
        metrics["code_model_function_rate"] = model_functions / len(code_results)
        metrics["code_contract_pass_count"] = float(contract_pass)
        metrics["code_contract_pass_rate"] = contract_pass / len(code_results)
        _add_optional_average(
            metrics,
            "code_reference_import_recall",
            [result.get("code_reference_import_recall") for result in code_results],
        )
        _add_optional_average(
            metrics,
            "code_reference_call_recall",
            [result.get("code_reference_call_recall") for result in code_results],
        )
    return metrics


def _with_code_diagnostics(result: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(result)
    diagnostics = code_structure_diagnostics(
        extract_python_code(str(result.get("response_text") or ""))
    )
    if normalized.get("code_syntax_ok") is None:
        normalized["code_syntax_ok"] = diagnostics.syntax_ok
    if normalized.get("code_model_function_present") is None:
        normalized["code_model_function_present"] = diagnostics.model_function_present
    if normalized.get("code_contract_ok") is None:
        normalized["code_contract_ok"] = bool(
            normalized["code_syntax_ok"] and normalized["code_model_function_present"]
        )
    return normalized


def _add_optional_average(
    metrics: dict[str, float],
    prefix: str,
    values: list[object],
) -> None:
    known = [float(value) for value in values if isinstance(value, (int, float))]
    if not known:
        return
    metrics[f"{prefix}_comparison_count"] = float(len(known))
    metrics[f"{prefix}_avg"] = sum(known) / len(known)
