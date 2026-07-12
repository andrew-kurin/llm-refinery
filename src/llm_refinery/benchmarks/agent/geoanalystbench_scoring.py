from __future__ import annotations

import ast
import re
from dataclasses import dataclass, replace

from llm_refinery.benchmarks.agent.base import AgentEvalResult
from llm_refinery.core.metrics import add_distribution_metrics


@dataclass(frozen=True)
class CodeStructureDiagnostics:
    syntax_ok: bool
    model_function_present: bool
    reference_import_recall: float | None
    reference_call_recall: float | None

    @property
    def contract_ok(self) -> bool:
        return self.syntax_ok and self.model_function_present


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
        reference_code = str(getattr(result.request.task, "code", ""))
        diagnostics = code_structure_diagnostics(code, reference_code=reference_code)
        return replace(
            result,
            code_syntax_ok=diagnostics.syntax_ok,
            code_model_function_present=diagnostics.model_function_present,
            code_contract_ok=diagnostics.contract_ok,
            code_reference_import_recall=diagnostics.reference_import_recall,
            code_reference_call_recall=diagnostics.reference_call_recall,
        )
    return result


def summarize_geoanalyst_results(
    results: list[AgentEvalResult], wall_duration_s: float
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "request_count": float(len(results)),
        "response_count": float(sum(1 for result in results if result.ok)),
        "error_count": float(sum(1 for result in results if not result.ok)),
        "wall_duration_s": float(wall_duration_s),
    }
    if results:
        metrics["response_availability_rate"] = metrics["response_count"] / len(results)
    successes = [result for result in results if result.ok]
    add_distribution_metrics(metrics, "latency", [result.latency_s for result in successes])
    add_distribution_metrics(
        metrics,
        "response_chars",
        [float(len(result.response_text)) for result in successes],
        unit_suffix="",
    )

    known_completion_tokens = [
        result.completion_tokens for result in successes if result.completion_tokens is not None
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

    code_results = [result for result in successes if result.request.response_type == "code"]
    if code_results:
        syntax_valid = sum(1 for result in code_results if result.code_syntax_ok)
        model_functions = sum(1 for result in code_results if result.code_model_function_present)
        contract_pass = sum(1 for result in code_results if result.code_contract_ok)
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
            [result.code_reference_import_recall for result in code_results],
        )
        _add_optional_average(
            metrics,
            "code_reference_call_recall",
            [result.code_reference_call_recall for result in code_results],
        )

    return metrics


def extract_workflow_step_count(text: str) -> int:
    numbers: list[int] = []
    for line in text.splitlines():
        match = re.match(r"^\s*(\d{1,2})[\.)]\s+", line)
        if match:
            numbers.append(int(match.group(1)))
    if numbers:
        return len(numbers)
    bullet_lines = [line for line in text.splitlines() if re.match(r"^\s*[-*]\s+", line)]
    return len(bullet_lines)


def extract_python_code(text: str) -> str:
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def code_structure_diagnostics(
    code: str,
    *,
    reference_code: str = "",
) -> CodeStructureDiagnostics:
    generated_tree = _parse_python(code)
    reference_tree = _parse_python(reference_code)
    generated_imports = _import_roots(generated_tree)
    generated_calls = _call_names(generated_tree)
    reference_imports = _import_roots(reference_tree)
    reference_calls = _call_names(reference_tree)
    return CodeStructureDiagnostics(
        syntax_ok=generated_tree is not None,
        model_function_present=_has_top_level_model_function(generated_tree),
        reference_import_recall=_set_recall(generated_imports, reference_imports),
        reference_call_recall=_set_recall(generated_calls, reference_calls),
    )


def _parse_python(code: str) -> ast.Module | None:
    try:
        return ast.parse(code)
    except (SyntaxError, ValueError):
        return None


def _has_top_level_model_function(tree: ast.Module | None) -> bool:
    if tree is None:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "model"
        for node in tree.body
    )


def _import_roots(tree: ast.Module | None) -> set[str]:
    roots: set[str] = set()
    if tree is None:
        return roots
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def _call_names(tree: ast.Module | None) -> set[str]:
    names: set[str] = set()
    if tree is None:
        return names
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def _set_recall(generated: set[str], reference: set[str]) -> float | None:
    if not reference:
        return None
    return len(generated & reference) / len(reference)


def _add_optional_average(
    metrics: dict[str, float],
    prefix: str,
    values: list[float | None],
) -> None:
    known = [value for value in values if value is not None]
    if not known:
        return
    metrics[f"{prefix}_comparison_count"] = float(len(known))
    metrics[f"{prefix}_avg"] = sum(known) / len(known)
