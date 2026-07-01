from __future__ import annotations

import ast
import re
from dataclasses import replace

from llm_refinery.benchmarks.agent.base import AgentEvalResult
from llm_refinery.core.metrics import add_distribution_metrics


def score_geoanalyst_result(result: AgentEvalResult) -> AgentEvalResult:
    if not result.ok:
        return result
    if result.request.response_type == "workflow":
        step_count = extract_workflow_step_count(result.response_text)
        task_length = int(getattr(result.request.task, "task_length", 0))
        return replace(
            result,
            workflow_step_count=step_count,
            workflow_step_abs_error=abs(step_count - task_length),
        )
    if result.request.response_type == "code":
        code = extract_python_code(result.response_text)
        try:
            ast.parse(code)
            syntax_ok = True
        except SyntaxError:
            syntax_ok = False
        return replace(result, code_syntax_ok=syntax_ok)
    return result


def summarize_geoanalyst_results(
    results: list[AgentEvalResult], wall_duration_s: float
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "request_count": float(len(results)),
        "success_count": float(sum(1 for result in results if result.ok)),
        "error_count": float(sum(1 for result in results if not result.ok)),
        "wall_duration_s": float(wall_duration_s),
    }
    if results:
        metrics["success_rate"] = metrics["success_count"] / len(results)
    successes = [result for result in results if result.ok]
    add_distribution_metrics(metrics, "latency", [result.latency_s for result in successes])
    add_distribution_metrics(
        metrics,
        "response_chars",
        [float(len(result.response_text)) for result in successes],
        unit_suffix="",
    )

    known_completion_tokens = [
        result.completion_tokens
        for result in successes
        if result.completion_tokens is not None
    ]
    if known_completion_tokens:
        metrics["completion_tokens_total"] = float(sum(known_completion_tokens))
        metrics["completion_tokens_avg"] = sum(known_completion_tokens) / len(
            known_completion_tokens
        )
        metrics["completion_tokens_per_second"] = (
            sum(known_completion_tokens) / wall_duration_s if wall_duration_s else 0.0
        )

    workflow_results = [
        result
        for result in successes
        if result.request.response_type == "workflow" and result.workflow_step_count is not None
    ]
    if workflow_results:
        metrics["workflow_count"] = float(len(workflow_results))
        metrics["workflow_step_count_avg"] = sum(
            result.workflow_step_count or 0 for result in workflow_results
        ) / len(workflow_results)
        metrics["workflow_step_abs_error_avg"] = sum(
            result.workflow_step_abs_error or 0 for result in workflow_results
        ) / len(workflow_results)

    code_results = [
        result
        for result in successes
        if result.request.response_type == "code" and result.code_syntax_ok is not None
    ]
    if code_results:
        syntax_pass = sum(1 for result in code_results if result.code_syntax_ok)
        metrics["code_count"] = float(len(code_results))
        metrics["code_syntax_pass_count"] = float(syntax_pass)
        metrics["code_syntax_pass_rate"] = syntax_pass / len(code_results)

    return metrics


def extract_workflow_step_count(text: str) -> int:
    numbers: list[int] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d{1,2})[\.)]\s+", line)
        if match:
            numbers.append(int(match.group(1)))
    if numbers:
        return max(numbers)
    bullet_lines = [line for line in text.splitlines() if re.match(r"^\s*[-*]\s+", line)]
    return len(bullet_lines)


def extract_python_code(text: str) -> str:
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()
