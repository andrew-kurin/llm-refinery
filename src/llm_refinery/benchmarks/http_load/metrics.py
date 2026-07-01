from __future__ import annotations

from llm_refinery.benchmarks.http_load.models import RequestResult
from llm_refinery.core.metrics import add_distribution_metrics


def summarize_request_results(
    results: list[RequestResult], *, wall_duration_s: float, concurrency: int, max_tokens: int
) -> dict[str, float]:
    successes = [result for result in results if result.ok]
    request_count = len(results)
    error_count = request_count - len(successes)
    metrics: dict[str, float] = {
        "request_count": float(request_count),
        "success_count": float(len(successes)),
        "error_count": float(error_count),
        "error_rate": float(error_count / request_count) if request_count else 0.0,
        "concurrency": float(concurrency),
        "max_tokens": float(max_tokens),
        "wall_duration_s": wall_duration_s,
        "requests_per_second": float(len(successes) / wall_duration_s)
        if wall_duration_s > 0
        else 0.0,
    }

    add_distribution_metrics(metrics, "latency", [result.latency_s for result in successes])
    add_distribution_metrics(
        metrics,
        "ttft",
        [result.ttft_s for result in successes if result.ttft_s is not None],
    )

    completion_chars_total = sum(result.completion_chars for result in successes)
    metrics["completion_chars_total"] = float(completion_chars_total)
    metrics["completion_chars_per_second"] = (
        float(completion_chars_total / wall_duration_s) if wall_duration_s > 0 else 0.0
    )

    known_completion_tokens = [
        result.completion_tokens for result in successes if result.completion_tokens is not None
    ]
    if known_completion_tokens:
        completion_tokens_total = sum(known_completion_tokens)
        metrics["completion_tokens_total"] = float(completion_tokens_total)
        metrics["completion_tokens_per_second"] = (
            float(completion_tokens_total / wall_duration_s) if wall_duration_s > 0 else 0.0
        )

    known_prompt_tokens = [
        result.prompt_tokens for result in successes if result.prompt_tokens is not None
    ]
    if known_prompt_tokens:
        metrics["prompt_tokens_total"] = float(sum(known_prompt_tokens))

    checked_results = [result for result in successes if result.check_passed is not None]
    if checked_results:
        check_pass_count = sum(1 for result in checked_results if result.check_passed)
        metrics["check_pass_count"] = float(check_pass_count)
        metrics["check_fail_count"] = float(len(checked_results) - check_pass_count)
        metrics["check_pass_rate"] = check_pass_count / len(checked_results)

    eval_tps = [
        result.completion_tokens / result.server_eval_duration_s
        for result in successes
        if result.completion_tokens is not None
        and result.server_eval_duration_s is not None
        and result.server_eval_duration_s > 0
    ]
    add_distribution_metrics(metrics, "server_eval_tps", eval_tps, unit_suffix="")

    prompt_eval_tps = [
        result.prompt_tokens / result.server_prompt_eval_duration_s
        for result in successes
        if result.prompt_tokens is not None
        and result.server_prompt_eval_duration_s is not None
        and result.server_prompt_eval_duration_s > 0
    ]
    add_distribution_metrics(metrics, "server_prompt_eval_tps", prompt_eval_tps, unit_suffix="")
    return metrics
